#!/usr/bin/env python3

import copy
import typing
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.proof_state import ProofState
from itp_interface.tools.cache import SimpleLruCache
from itp_interface.rl.simple_proof_env import ProofEnv, ProofEnvInfo
from itp_interface.rl.simple_proof_env_ray import ProofEnvActor
from itp_interface.tools.proof_env_utils import CapturedException, replicate_proof_env


class ThreadProofEnvPool(object):
    """Thread-based implementation of ProofEnvPool (fallback when Ray is not available)"""

    def __init__(self,
            pool_size: int = 1,
            proof_env_actors: typing.List[ProofEnvActor] = None,
            proof_env: ProofEnv = None,
            logger: typing.Optional[logging.Logger] = None,
            timeout: float = 120,
            max_parallel_envs: int = None):
        """
        Thread-based pool of proof environments.
        Uses ThreadPoolExecutor for parallel execution instead of Ray.
        """
        assert pool_size > 0 or (proof_env_actors is not None and len(proof_env_actors) > 0), "Pool size must be greater than 0"
        self._current_index = 0
        self._callback = None
        self._logger = logger if logger else logging.getLogger(__name__)
        self._env_to_steps_map : typing.Dict[int, typing.List[ProofAction]] = {}
        self._nonactive_env_to_state_map : typing.Dict[int, ProofState] = {}
        self._nonactive_env_to_done_map : typing.Dict[int, bool] = {}
        self._env_args_map : typing.Dict[int, typing.List] = {}
        self._env_kwargs_map : typing.Dict[int, typing.Dict] = {}
        self._timeout = timeout
        self._pool_lock = threading.RLock()

        if proof_env_actors is None:
            self.pool_size = pool_size
            self._frozeen_env = replicate_proof_env(proof_env, self._logger)
            # Create thread-safe ProofEnvActor instances (non-Ray version)
            self._proof_env_pool : typing.List[ProofEnvActor] = [
                ProofEnvActor(
                    name=self._frozeen_env.name,
                    dynamic_proof_executor_callback=self._frozeen_env.dynamic_proof_executor_callback,
                    lemma_name=self._frozeen_env.lemma_name,
                    retrieval_strategy=self._frozeen_env.retrieve_strategy,
                    max_proof_depth=self._frozeen_env.max_proof_depth,
                    always_retrieve_thms=self._frozeen_env._always_retrieve_thms,
                    logger=None,
                    should_load_env=False
                )
                for _ in range(self.pool_size)
            ]
        else:
            self.pool_size = len(proof_env_actors)
            self._frozeen_env = None
            self._proof_env_pool : typing.List[ProofEnvActor] = proof_env_actors
            # Get args and kwargs from existing actors
            for i, proof_env_actor in enumerate(self._proof_env_pool):
                try:
                    self._env_args_map[i] = proof_env_actor.get_env_args()
                    self._env_kwargs_map[i] = proof_env_actor.get_env_kwargs()
                except Exception as e:
                    self._logger.error(f"Error getting arguments for proof environment {i}: {e}")
                    raise Exception(f"Error getting arguments for proof environment {i}: {e}")

        self._errd_envs = set()
        self._errd_envs_exceptions = {}
        self._is_initialized = False
        self._active_envs = set(list(range(self.pool_size)))
        self._max_parallel_envs = max_parallel_envs if max_parallel_envs is not None else self.pool_size
        self._env_cache = SimpleLruCache(max_size_in_bytes=self._max_parallel_envs)
        self._executor = ThreadPoolExecutor(max_workers=self._max_parallel_envs)

    def __enter__(self):
        self._is_initialized = True
        self.reset(list(range(self.pool_size)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._is_initialized = False
        try:
            self._try_cleanup_envs(list(range(self.pool_size)))
        except Exception as e:
            self._logger.error(f"Error cleaning up environments: {e}")
        finally:
            self._executor.shutdown(wait=True)

    def _parallel_execute(self, callables, timeout):
        """Execute multiple callables in parallel using ThreadPoolExecutor"""
        futures = [self._executor.submit(callable_fn) for callable_fn in callables]
        results = []
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except FutureTimeoutError:
                results.append(CapturedException(TimeoutError(f"Operation timed out after {timeout}s")))
            except Exception as e:
                results.append(CapturedException(e))
        return results

    def add_and_init_proof_envs(self, count: int = 1):
        with self._pool_lock:
            count_before = len(self._proof_env_pool)
            self.add_proof_envs(count=count)
            count_after = len(self._proof_env_pool)
            return self.reset(list(range(count_before, count_after)))

    def add_proof_envs(self, count: int = 1):
        with self._pool_lock:
            assert self._is_initialized, "Cannot add proof environments after initialization"
            assert self._frozeen_env is not None, "Frozen environment must be provided"
            new_envs = [
                ProofEnvActor(
                    name=self._frozeen_env.name,
                    dynamic_proof_executor_callback=self._frozeen_env.dynamic_proof_executor_callback,
                    lemma_name=self._frozeen_env.lemma_name,
                    retrieval_strategy=self._frozeen_env.retrieve_strategy,
                    max_proof_depth=self._frozeen_env.max_proof_depth,
                    always_retrieve_thms=self._frozeen_env._always_retrieve_thms,
                    logger=None,
                    should_load_env=False
                )
                for _ in range(count)
            ]
            self._proof_env_pool.extend(new_envs)
            self.pool_size += count

    def add_proof_env(self, proof_env: ProofEnv = None):
        with self._pool_lock:
            assert self._is_initialized, "Cannot add proof environments after initialization"
            if proof_env is None:
                assert self._frozeen_env is not None, "Frozen environment must be provided"
                new_env = ProofEnvActor(
                    name=self._frozeen_env.name,
                    dynamic_proof_executor_callback=self._frozeen_env.dynamic_proof_executor_callback,
                    lemma_name=self._frozeen_env.lemma_name,
                    retrieval_strategy=self._frozeen_env.retrieve_strategy,
                    max_proof_depth=self._frozeen_env.max_proof_depth,
                    always_retrieve_thms=self._frozeen_env._always_retrieve_thms,
                    logger=None,
                    should_load_env=False
                )
            else:
                new_env = ProofEnvActor(
                    name=proof_env.name,
                    dynamic_proof_executor_callback=proof_env.dynamic_proof_executor_callback,
                    lemma_name=proof_env.lemma_name,
                    retrieval_strategy=proof_env.retrieve_strategy,
                    max_proof_depth=proof_env.max_proof_depth,
                    always_retrieve_thms=proof_env._always_retrieve_thms,
                    logger=None
                )
            self._proof_env_pool.append(new_env)
            self.pool_size += 1
            try:
                args = self._proof_env_pool[-1].get_env_args()
                kwargs = self._proof_env_pool[-1].get_env_kwargs()
                self._env_args_map[self.pool_size-1] = args
                self._env_kwargs_map[self.pool_size-1] = kwargs
            except Exception as e:
                self._logger.error(f"Error getting arguments for proof environment {self.pool_size-1}: {e}")
                raise Exception(f"Error getting arguments for proof environment {self.pool_size-1}: {e}")

    def get_errd_envs(self):
        with self._pool_lock:
            return copy.deepcopy(self._errd_envs)

    def get_errd_envs_exceptions(self):
        with self._pool_lock:
            return copy.deepcopy(self._errd_envs_exceptions)

    def get_timeout(self):
        return self._timeout

    def step(self, actions: typing.List[ProofAction], idxs: typing.List[int] = None) -> typing.List[typing.Tuple[ProofState, ProofAction, ProofState, float, bool, ProofEnvInfo]]:
        with self._pool_lock:
            assert self._is_initialized, "Pool must be initialized before stepping"
            assert len(actions) == len(self._proof_env_pool) or (idxs is not None and len(actions) == len(idxs)), \
                "Number of actions must match the number of proof environments"
            if idxs is None:
                idxs = list(range(len(actions)))
            assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot step errored environments: {set(idxs).intersection(self._errd_envs)}"

            max_parallel_chunks = [(i, min(i+self._max_parallel_envs, len(idxs))) for i in range(0, len(idxs), self._max_parallel_envs)]
            all_step_res = []
            for chunk_start, chunk_end in max_parallel_chunks:
                all_step_res.extend(self._step_chunk(actions[chunk_start:chunk_end], idxs[chunk_start:chunk_end]))
            return all_step_res

    def get_pool(self, idxs: typing.List[int]) -> 'ThreadProofEnvPool':
        with self._pool_lock:
            assert self._is_initialized, "Pool must be initialized before getting"
            assert len(idxs) > 0, "Must provide at least one index"
            return ThreadProofEnvPool(
                proof_env_actors=[self._proof_env_pool[idx] for idx in idxs],
                logger=self._logger,
                timeout=self._timeout,
                max_parallel_envs=self._max_parallel_envs)

    def reset(self, idxs: typing.List[int]) -> typing.List[typing.Tuple[ProofState, ProofAction, ProofState, float, bool, ProofEnvInfo]]:
        assert self._is_initialized, "Pool must be initialized before resetting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot reset errored environments: {set(idxs).intersection(self._errd_envs)}"
        reset_chunks = [idxs[i:min(i+self._max_parallel_envs, len(idxs))] for i in range(0, len(idxs), self._max_parallel_envs)]
        all_reset_res = []
        for chunk in reset_chunks:
            all_reset_res.extend(self._reset_chunk(chunk))
        return all_reset_res

    def get_state(self, idxs: typing.List[int]) -> typing.List[ProofState]:
        with self._pool_lock:
            assert self._is_initialized, "Pool must be initialized before getting"
            assert len(idxs) > 0, "Must provide at least one index"
            assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get state of errored environments: {set(idxs).intersection(self._errd_envs)}"

            results = []
            for idx in idxs:
                if idx in self._active_envs:
                    try:
                        state = self._proof_env_pool[idx].get_state()
                        results.append(state)
                    except Exception as e:
                        raise Exception(f"Error getting state for proof environment {idx}: {e}")
                else:
                    results.append(self._nonactive_env_to_state_map.get(idx, None))
            return results

    def get_done(self, idxs: typing.List[int]) -> typing.List[bool]:
        with self._pool_lock:
            assert self._is_initialized, "Pool must be initialized before getting"
            assert len(idxs) > 0, "Must provide at least one index"
            assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get done of errored environments: {set(idxs).intersection(self._errd_envs)}"

            results = []
            for idx in idxs:
                if idx in self._active_envs:
                    try:
                        done = self._proof_env_pool[idx].get_done()
                        results.append(done)
                    except Exception as e:
                        raise Exception(f"Error getting done for proof environment {idx}: {e}")
                else:
                    results.append(self._nonactive_env_to_done_map.get(idx, None))
            return results

    def dump_proof(self, idxs: typing.List[int]):
        with self._pool_lock:
            assert self._is_initialized, "Pool must be initialized before dumping"
            assert len(idxs) > 0, "Must provide at least one index"
            assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot dump proof of errored environments: {set(idxs).intersection(self._errd_envs)}"

            for idx in idxs:
                try:
                    self._proof_env_pool[idx].dump_proof()
                except Exception as e:
                    raise Exception(f"Error dumping proof for proof environment {idx}: {e}")

    def _get_attr(self, attr_name: str, idxs: typing.List[int]):
        with self._pool_lock:
            assert self._is_initialized, "Pool must be initialized before getting"
            assert len(idxs) > 0, "Must provide at least one index"
            assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get attribute {attr_name} of errored environments: {set(idxs).intersection(self._errd_envs)}"

            # Create callables for parallel attribute retrieval
            callables = [lambda idx=idx: self._proof_env_pool[idx].getattr(attr_name) for idx in idxs]

            # Execute in parallel
            attrs = self._parallel_execute(callables, self._timeout)

            # Check for exceptions
            for i, attr in enumerate(attrs):
                if isinstance(attr, CapturedException):
                    raise Exception(f"Error getting attribute {attr_name} for proof environment {i}: {attr.exception}")
            return attrs

    def get_proof_search_res(self, idxs: typing.List[int]) -> typing.List[typing.Tuple[typing.List[ProofAction], float]]:
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get proof search results of errored environments: {set(idxs).intersection(self._errd_envs)}"
        return self._get_attr("proof_search_res", idxs)

    def _reset_chunk(self, idxs: typing.List[int]) -> typing.List[ProofState]:
        self._logger.info(f"Resetting environments: {idxs}")
        assert self._is_initialized, "Pool must be initialized before resetting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot reset errored environments: {set(idxs).intersection(self._errd_envs)}"

        # Create callables for parallel reset
        callables = [lambda idx=idx: self._proof_env_pool[idx].reset() for idx in idxs]

        # Execute resets in parallel
        env_init_stats = self._parallel_execute(callables, self._timeout)

        results = []
        envs_to_remove = []

        for i, (idx, env_init_stat) in enumerate(zip(idxs, env_init_stats)):
            if isinstance(env_init_stat, CapturedException):
                self._errd_envs.add(idx)
                self._errd_envs_exceptions[idx] = env_init_stat
                envs_to_remove.append(idx)
                self._logger.error(f"Error initializing proof environment {idx}: {env_init_stat.exception}")
                results.append((None, None, None, 0.0, True, None))
            else:
                envs_removed = self._env_cache.add_to_cache(str(idx), idx, 1)
                for env_removed in envs_removed:
                    if int(env_removed) not in idxs:
                        envs_to_remove.append(env_removed)
                self._active_envs.add(idx)
                results.append(env_init_stat)

        if len(envs_to_remove) > 0:
            self._try_cleanup_envs(envs_to_remove)
        self._logger.info(f"Reset environments: {idxs}")
        return results

    def _step_chunk(self, actions: typing.List[ProofAction], idxs: typing.List[int] = None) -> typing.List[typing.Tuple[ProofState, ProofAction, ProofState, float, bool, ProofEnvInfo]]:
        assert self._is_initialized, "Pool must be initialized before stepping"
        assert len(actions) == len(self._proof_env_pool) or (idxs is not None and len(actions) == len(idxs)), \
            "Number of actions must match the number of proof environments"
        assert len(idxs) <= self._max_parallel_envs, f"Number of environments to step must be less than or equal to {self._max_parallel_envs}"
        if idxs is None:
            idxs = list(range(len(actions)))
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot step errored environments: {set(idxs).intersection(self._errd_envs)}"

        removed_envs = []
        non_active_envs = []
        self._logger.info(f"Stepping environments: {idxs}")

        for idx in idxs:
            envs_removed = self._env_cache.add_to_cache(str(idx), idx, 1)
            if idx not in self._active_envs:
                non_active_envs.append(idx)
            for env in envs_removed:
                if int(env) not in idxs:
                    removed_envs.append(env)

        if len(removed_envs) > 0:
            self._try_cleanup_envs(removed_envs)
        if len(non_active_envs) > 0:
            self._activate_envs(non_active_envs)

        for i, idx in enumerate(idxs):
            actions_so_far = self._env_to_steps_map.get(idx, [])
            actions_so_far.append(actions[i])
            self._env_to_steps_map[idx] = actions_so_far

        return self._unsafe_step_chunk(actions, idxs)

    def _activate_envs(self, idxs: typing.List[int]):
        self._logger.info(f"Activating environments: {idxs}")
        for idx in idxs:
            if idx in self._active_envs:
                continue
            if self._frozeen_env is not None:
                self._proof_env_pool[idx] = ProofEnvActor(
                    name=self._frozeen_env.name,
                    dynamic_proof_executor_callback=self._frozeen_env.dynamic_proof_executor_callback,
                    lemma_name=self._frozeen_env.lemma_name,
                    retrieval_strategy=self._frozeen_env.retrieve_strategy,
                    max_proof_depth=self._frozeen_env.max_proof_depth,
                    always_retrieve_thms=self._frozeen_env._always_retrieve_thms,
                    logger=None,
                    should_load_env=False
                )
            else:
                # Recreate from saved args/kwargs
                self._proof_env_pool[idx] = ProofEnvActor(*self._env_args_map[idx], **self._env_kwargs_map[idx])

        self.reset(idxs)

        # Rerun the steps again on all the environments that were not active
        idxs_to_run = []
        actions_to_run = []
        last_action_idx = 0
        actions_added = True
        while actions_added:
            actions_added = False
            for idx in idxs:
                actions = self._env_to_steps_map.get(idx, [])
                if len(actions) > 0:
                    if last_action_idx < len(actions):
                        actions_added = True
                        idxs_to_run.append(idx)
                        actions_to_run.append(actions[last_action_idx])
            if actions_added:
                last_action_idx += 1
                self._unsafe_step_chunk(actions_to_run, idxs_to_run)
                idxs_to_run = []
                actions_to_run = []
        self._logger.info(f"Activated environments: {idxs}")

    def _unsafe_step_chunk(self, actions: typing.List[ProofAction], idxs: typing.List[int] = None) -> typing.List[typing.Tuple[ProofState, ProofAction, ProofState, float, bool, ProofEnvInfo]]:
        # Create callables for parallel execution
        callables = [lambda i=i, idx=idx: self._proof_env_pool[idx].step(actions[i]) for i, idx in enumerate(idxs)]

        # Execute in parallel
        results = self._parallel_execute(callables, self._timeout)

        # Process results and handle exceptions
        actual_returns = []
        for i, (idx, result) in enumerate(zip(idxs, results)):
            if isinstance(result, CapturedException):
                self._errd_envs.add(idx)
                self._errd_envs_exceptions[idx] = result
                actual_returns.append((None, None, None, 0.0, True, None))
                self._logger.error(f"Error stepping proof environment {idx}: {result.exception}")
            else:
                actual_returns.append(result)
        return actual_returns

    def _try_cleanup_envs(self, idxs: typing.Union[typing.List[int], typing.List[str]]):
        self._logger.info(f"Cleaning up environments: {idxs}")
        idxs = [int(idx) for idx in idxs]
        try:
            for env_idx in idxs:
                if env_idx in self._active_envs:
                    try:
                        state = self._proof_env_pool[env_idx].get_state()
                        done = self._proof_env_pool[env_idx].get_done()
                        self._nonactive_env_to_state_map[env_idx] = state
                        self._nonactive_env_to_done_map[env_idx] = done
                    except Exception as e:
                        self._logger.error(f"Error getting state/done for proof environment {env_idx}: {e}")

            for env_idx in idxs:
                if env_idx in self._active_envs:
                    try:
                        self._proof_env_pool[env_idx].cleanup()
                    except Exception as e:
                        self._logger.error(f"Error cleaning up proof environment {env_idx}: {e}")
        except Exception as e:
            self._logger.error(f"Error cleaning up proof environments: {e}")

        # No need to "kill" threads like Ray actors - just remove from active set
        for env_idx in idxs:
            if env_idx in self._active_envs:
                self._active_envs.remove(env_idx)
        self._logger.info(f"Removed environments: {idxs}")
