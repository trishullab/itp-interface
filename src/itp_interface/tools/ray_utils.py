#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import asyncio
import time
import ray
import typing
import logging
import gc

class RayUtils(object):

    @staticmethod
    def init_ray(num_of_cpus: int = 10, object_store_memory_in_gb: float = 25, memory_in_gb: float = 0.5, runtime_env: typing.Dict[str, str] = None):
        gb = 2**30
        object_store_memory = int(object_store_memory_in_gb * gb)
        memory = int(memory_in_gb * gb)
        return ray.init(num_cpus=num_of_cpus, object_store_memory=object_store_memory, _memory=memory, ignore_reinit_error=True, runtime_env=runtime_env)

    @staticmethod
    def ray_run_within_parallel_limits(
        max_parallel: int,
        num_objects: int,
        transform_outputs: typing.Callable[[typing.List[typing.Any]], None],
        prepare_next: typing.Callable[[int], typing.List[typing.Any]],
        create_remotes: typing.Callable[[typing.List[typing.Any]], typing.List[typing.Any]],
        logger: logging.Logger = None,
        turn_off_logging: bool = False
    ):
        logger = logger or logging.getLogger(__name__)
        idx = 0
        next_batch = prepare_next(max_parallel)
        if not turn_off_logging:
            logger.info(f"Loading next_batch: {len(next_batch)}, max_parallel: {max_parallel}")
        assert len(next_batch) <= max_parallel, f"next_batch: {len(next_batch)}, max_parallel: {max_parallel}"
        remotes = create_remotes(next_batch)
        if not turn_off_logging:
            logger.info(f"Created remotes: {len(remotes)}")
        diff_remotes = len(remotes)
        while idx < num_objects or len(remotes) > 0:
            idx += diff_remotes
            idx = min(idx, num_objects)
            if not turn_off_logging:
                logger.info(f"Waiting for idx: {idx}, num_objects: {num_objects}, len(remotes): {len(remotes)}")
            ready, remotes = ray.wait(remotes)
            if len(ready) > 0:
                if not turn_off_logging:
                    logger.info(f"Got ready: {len(ready)}")
                results = ray.get(ready)
                transform_outputs(results)
                next_batch = prepare_next(len(results))
                assert len(next_batch) <= len(results), f"next_batch: {len(next_batch)}, ready: {len(results)}"
                new_remotes = create_remotes(next_batch)
                remotes.extend(new_remotes)
                diff_remotes = len(new_remotes)
                # Delete results to free up memory
                del results
                if not turn_off_logging:
                    logger.info(f"Running GC collect")
                gc.collect()
            else:
                diff_remotes = 0

@ray.remote(num_cpus=1)
class RayResourcePoolActor(object):
    def __init__(self, resources: list):
        self.resources = resources
        self.available = list(resources)
        self.lock_event = asyncio.Lock()
        self.acquired = []

    async def acquire(self, num: int):
        async with self.lock_event:
            if len(self.available) < num:
                return None
            acquired = self.available[:num]
            self.available = self.available[num:]
            self.acquired.extend(acquired)
            return acquired
    
    async def release(self, resources: list):
        async with self.lock_event:
            for resource in resources:
                try:
                    self.acquired.remove(resource)
                    self.available.append(resource)
                except ValueError:
                    pass
            return True
    
    def get_acquired(self):
        return list(self.acquired)
    
    def get_available(self):
        return list(self.available)
    
    async def wait_and_acquire(self, num: int, timeout: typing.Optional[float] = None):
        polling_time = 0.1
        start_time = time.time()
        while True:
            acquired = await self.acquire(num)
            if acquired is not None:
                return acquired
            if timeout is not None and time.time() - start_time > timeout:
                return None
            await asyncio.sleep(polling_time)


class RayTimedException(Exception):
    pass

@ray.remote
class TimedRayExec(object):
    def __init__(self, func, args=None, kwargs=None):
        # Check that func is a remote function
        assert hasattr(func, "remote"), f"func: {func} is not a remote function"
        self.func = func
        self.args = args if args else []
        self.kwargs = kwargs if kwargs else {}
        self._is_cancelled = False

    def execute_with_timeout(self, timeout: float = 60):
        ray_id = self.func.remote(*self.args, **self.kwargs)
        finished, unfinished = ray.wait([ray_id], timeout=timeout)
        is_cancelled = False
        if len(unfinished) > 0:
            ray.cancel(ray_id, force=True)
            is_cancelled = True
        self._is_cancelled = is_cancelled
        if not is_cancelled:
            assert len(finished) == 1, f"len(finished): {len(finished)}"
            return_typ = ray.get(finished[0])
            return return_typ
        else:
            return None
    
    def is_cancelled(self):
        return self._is_cancelled

if __name__ == "__main__":
    import os
    import time
    import random
    import unittest
    class TestRayResourcePoolActor(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            ray.init(ignore_reinit_error=True)

        @classmethod
        def tearDownClass(cls):
            ray.shutdown()

        def setUp(self):
            resources = ["res1", "res2", "res3", "res4"]
            self.pool = RayResourcePoolActor.remote(resources)

        def test_acquire(self):
            result = ray.get(self.pool.acquire.remote(2))
            self.assertEqual(result, ["res1", "res2"])
            self.assertEqual(ray.get(self.pool.get_available.remote()), ["res3", "res4"])
            self.assertEqual(ray.get(self.pool.get_acquired.remote()), ["res1", "res2"])

        def test_acquire_more_than_available(self):
            result = ray.get(self.pool.acquire.remote(5))
            self.assertIsNone(result)

        def test_release(self):
            ray.get(self.pool.acquire.remote(2))
            ray.get(self.pool.release.remote(["res1"]))
            self.assertEqual(ray.get(self.pool.get_available.remote()), ["res3", "res4", "res1"])
            self.assertEqual(ray.get(self.pool.get_acquired.remote()), ["res2"])

        def test_release_non_acquired_resource(self):
            ray.get(self.pool.acquire.remote(2))
            result = ray.get(self.pool.release.remote(["res5"]))
            self.assertTrue(result)
            self.assertEqual(ray.get(self.pool.get_available.remote()), ["res3", "res4"])
            self.assertEqual(ray.get(self.pool.get_acquired.remote()), ["res1", "res2"])

        def test_wait_and_acquire_success(self):
            print("Available: " + str(ray.get(self.pool.get_available.remote())))
            print("Acquired: " + str(ray.get(self.pool.get_acquired.remote())))
            result = ray.get(self.pool.wait_and_acquire.remote(2))
            self.assertEqual(result, ["res1", "res2"])

        def test_wait_and_acquire_timeout(self):
            result = ray.get(self.pool.wait_and_acquire.remote(5, timeout=1))
            self.assertIsNone(result)
    
    unittest.main()


    log_folder = f".log/ray_utils"
    os.makedirs(log_folder, exist_ok=True)
    log_file = f"{log_folder}/ray_utils-{time.strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    logger = logging.getLogger(__name__)
    size = 1000
    example_cnt = 100000
    last_job_idx = 0
    total_sum = 0
    total_sum_serial = 0
    job_spec = [[random.random() for _ in range(example_cnt)] for _ in range(size)]

    @ray.remote
    def _do_job(job):
        idx, arr = job
        for i in range(size*10):
            # This is just to stress the CPUs
            sum_val = sum(arr)
        return sum_val, arr

    def _prepare_remotes(num: int):
        global last_job_idx
        job_list = job_spec[last_job_idx:last_job_idx+num]
        job_list = [(last_job_idx + idx, job) for idx, job in enumerate(job_list)]
        last_job_idx += len(job_list)
        return job_list

    def _create_remotes(job_list: typing.List[typing.Tuple[int, typing.List[float]]]):
        remotes = []
        for job in job_list:
            logger.info(f"Queuing job {job[0]}")
            job_ref = ray.put(job)
            remotes.append(_do_job.remote(job_ref))
        return remotes
    
    def _transform_output(results):
        global total_sum, total_sum_serial
        for sum_val, arr in results:
            total_sum += sum_val
            total_sum_serial += sum(arr)
        del results # This is important to free up memory
    parallel = 30
    RayUtils.init_ray(num_of_cpus=parallel)
    RayUtils.ray_run_within_parallel_limits(parallel, size, _transform_output, _prepare_remotes, _create_remotes, logger=logger)
    assert total_sum == total_sum_serial, f"total_sum: {total_sum}, total_sum_serial: {total_sum_serial}"
    logger.info(f"total_sum: {total_sum}, total_sum_serial: {total_sum_serial}")