#!/usr/bin/env python3

import typing
import logging
import ray
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.proof_state import ProofState
from itp_interface.tools.cache import SimpleLruCache
from itp_interface.rl.simple_proof_env import ProofEnv, ProofEnvInfo
from itp_interface.rl.simple_proof_env_ray import ProofEnvActor
from itp_interface.tools.proof_env_utils import CapturedException, replicate_proof_env


@ray.remote
class CaptureExceptionActor:
    def __init__(self, func, timeout:typing.Optional[float]=None, args=None, kwargs=None):
        self.func = func
        self.args = args if args else []
        self.kwargs = kwargs if kwargs else {}
        self.timeout = timeout

    def try_capture_exception(self):
        try:
            ray_id = self.func.remote(*self.args, **self.kwargs)
            if self.timeout is None:
                return_typ = ray.get(ray_id)
            else:
                return_typ = ray.get(ray_id, timeout=self.timeout)
            return return_typ
        except Exception as e:
            return CapturedException(e)


def run_safely_on_actor(func, timeout, *args, **kwargs):
    capture_exception_actor = CaptureExceptionActor.remote(func, timeout=timeout, *args, **kwargs)
    return capture_exception_actor.try_capture_exception.remote()


class RayProofEnvPool(object):
    """Ray-based implementation of ProofEnvPool using process-based parallelism"""

    def __init__(self,
            pool_size: int = 1,
            proof_env_actors: typing.List[ProofEnvActor] = None,
            proof_env: ProofEnv = None,
            logger: typing.Optional[logging.Logger] = None,
            timeout: float = 120,
            max_parallel_envs: int = None):
        """
        Keeps a pool of proof environments to be used in parallel,
        and replenishes them as needed. It keeps extra environments
        in a garbage collection list to be used when the pool is
        replenished.
        """
        assert pool_size > 0 or len(proof_env_actors) > 0, "Pool size must be greater than 0"
        self._current_index = 0
        self._callback = None
        self._logger = logger if logger else logging.getLogger(__name__)
        self._env_to_steps_map : typing.Dict[int, typing.List[ProofAction]] = {}
        self._nonactive_env_to_state_map : typing.Dict[int, ProofState] = {}
        self._nonactive_env_to_done_map : typing.Dict[int, bool] = {}
        self._env_args_map : typing.Dict[int, typing.List] = {}
        self._env_kwargs_map : typing.Dict[int, typing.Dict] = {}
        self._timeout = timeout
        if proof_env_actors is None:
            self.pool_size = pool_size
            self._frozeen_env = replicate_proof_env(proof_env, self._logger) # This is like a frozen copy we never change it
            self._proof_env_pool : typing.List[ProofEnvActor] = [
                ProofEnvActor.remote(
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
            all_args = ray.get([run_safely_on_actor(proof_env_actor.get_env_args, self._timeout) for proof_env_actor in self._proof_env_pool])
            all_kwargs = ray.get([run_safely_on_actor(proof_env_actor.get_env_kwargs, self._timeout) for proof_env_actor in self._proof_env_pool])
            for i, (args, kwargs) in enumerate(zip(all_args, all_kwargs)):
                if isinstance(args, CapturedException) or isinstance(kwargs, CapturedException):
                    self._logger.error(f"Error getting arguments for proof environment {i}: {args}")
                    self._logger.error(f"Error getting keyword arguments for proof environment {i}: {kwargs}")
                    raise Exception(f"Error getting arguments for proof environment {i}: {args}")
                self._env_args_map[i] = args
                self._env_kwargs_map[i] = kwargs
        self._errd_envs = set()
        self._errd_envs_exceptions = {}
        self._is_initialized = False
        self._active_envs = set(list(range(self.pool_size)))
        self._max_parallel_envs = max_parallel_envs if max_parallel_envs is not None else self.pool_size
        self._env_cache = SimpleLruCache(max_size_in_bytes=self._max_parallel_envs)

    def __enter__(self):
        self._is_initialized = True
        # load all environments which are not loaded
        self.reset(list(range(self.pool_size)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._is_initialized = False
        try:
            self._try_cleanup_envs(list(range(self.pool_size)))
        except Exception as e:
            self._logger.error(f"Error cleaning up environments: {e}")

    def add_and_init_proof_envs(self, count: int = 1):
        count_before = len(self._proof_env_pool)
        self.add_proof_envs(count=count)
        count_after = len(self._proof_env_pool)
        return self.reset(list(range(count_before, count_after)))

    def add_proof_envs(self, count: int = 1):
        assert self._is_initialized, "Cannot add proof environments after initialization"
        assert self._frozeen_env is not None, "Frozen environment must be provided"
        self._proof_env_pool.extend([
            ProofEnvActor.remote(
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
        ])
        self.pool_size += count

    def add_proof_env(self, proof_env: ProofEnv = None):
        assert self._is_initialized, "Cannot add proof environments after initialization"
        if proof_env is None:
            assert self._frozeen_env is not None, "Frozen environment must be provided"
            self._proof_env_pool.append(
                ProofEnvActor.remote(
                    name=self._frozeen_env.name,
                    dynamic_proof_executor_callback=self._frozeen_env.dynamic_proof_executor_callback,
                    lemma_name=self._frozeen_env.lemma_name,
                    retrieval_strategy=self._frozeen_env.retrieve_strategy,
                    max_proof_depth=self._frozeen_env.max_proof_depth,
                    always_retrieve_thms=self._frozeen_env._always_retrieve_thms,
                    logger=None,
                    should_load_env=False
                )
            )
        else:
            self._proof_env_pool.append(
                ProofEnvActor.remote(
                    name=proof_env.name,
                    dynamic_proof_executor_callback=proof_env.dynamic_proof_executor_callback,
                    lemma_name=proof_env.lemma_name,
                    retrieval_strategy=proof_env.retrieve_strategy,
                    max_proof_depth=proof_env.max_proof_depth,
                    always_retrieve_thms=proof_env._always_retrieve_thms,
                    logger=None
                )
            )
        self.pool_size += 1
        args = ray.get(run_safely_on_actor(self._proof_env_pool[-1].get_env_args, self._timeout))
        kwargs = ray.get(run_safely_on_actor(self._proof_env_pool[-1].get_env_kwargs, self._timeout))
        if isinstance(args, CapturedException) or isinstance(kwargs, CapturedException):
            self._logger.error(f"Error getting arguments for proof environment {self.pool_size-1}: {args}")
            self._logger.error(f"Error getting keyword arguments for proof environment {self.pool_size-1}: {kwargs}")
            raise Exception(f"Error getting arguments for proof environment {self.pool_size-1}: {args}")
        self._env_args_map[self.pool_size-1] = args
        self._env_kwargs_map[self.pool_size-1] = kwargs

    def get_errd_envs(self):
        return copy.deepcopy(self._errd_envs)

    def get_errd_envs_exceptions(self):
        return copy.deepcopy(self._errd_envs_exceptions)

    def get_timeout(self):
        return self._timeout

    def step(self, actions: typing.List[ProofAction], idxs: typing.List[int] = None) -> typing.List[typing.Tuple[ProofState, ProofAction, ProofState, float, bool, ProofEnvInfo]]:
        assert self._is_initialized, "Pool must be initialized before stepping"
        assert len(actions) == len(self._proof_env_pool) or (idxs is not None and len(actions) == len(idxs)), \
            "Number of actions must match the number of proof environments"
        if idxs is None:
            idxs = range(len(actions))
        # Make sure we are not stepping an errored environment
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot step errored environments: {set(idxs).intersection(self._errd_envs)}"

        # Step the active environments
        max_parallel_chunks = [(i, i+self._max_parallel_envs) for i in range(0, len(idxs), self._max_parallel_envs)]
        all_step_res = []
        for chunk in max_parallel_chunks:
            all_step_res.extend(self._step_chunk(actions[chunk[0]:chunk[1]], idxs[chunk[0]:chunk[1]]))
        return all_step_res

    def get_pool(self, idxs: typing.List[int]) -> 'RayProofEnvPool':
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        return RayProofEnvPool(
            proof_env_actors=[self._proof_env_pool[idx] for idx in idxs],
            logger=self._logger,
            timeout=self._timeout,
            max_parallel_envs=self._max_parallel_envs)

    def reset(self, idxs: typing.List[int]) -> typing.List[typing.Tuple[ProofState, ProofAction, ProofState, float, bool, ProofEnvInfo]]:
        assert self._is_initialized, "Pool must be initialized before resetting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot reset errored environments: {set(idxs).intersection(self._errd_envs)}"
        reset_chunks = [idxs[i:i+self._max_parallel_envs] for i in range(0, len(idxs), self._max_parallel_envs)]
        all_reset_res = []
        for chunk in reset_chunks:
            all_reset_res.extend(self._reset_chunk(chunk))
        return all_reset_res

    def get_state(self, idxs: int) -> typing.List[ProofState]:
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get state of errored environments: {set(idxs).intersection(self._errd_envs)}"
        active_idxs = []
        nonactive_idxs = []
        list_used = []
        for idx in idxs:
            if idx in self._active_envs:
                active_idxs.append(idx)
                list_used.append(active_idxs)
            else:
                nonactive_idxs.append(idx)
                list_used.append(nonactive_idxs)
        active_states = ray.get([run_safely_on_actor(self._proof_env_pool[idx].get_state, self._timeout) for idx in active_idxs])
        for i, state in enumerate(active_states):
            if isinstance(state, CapturedException):
                raise Exception(f"Error getting state for proof environment {i}: {state}")
        nonactive_states = [self._nonactive_env_to_state_map.get(idx, None) for idx in nonactive_idxs]
        results = []
        active_idx = 0
        nonactive_idx = 0
        for i, idx in enumerate(idxs):
            list_to_use = list_used[i]
            if list_to_use == active_idxs:
                results.append(active_states[active_idx])
                active_idx += 1
            else:
                results.append(nonactive_states[nonactive_idx])
                nonactive_idx += 1
        return results

    def get_done(self, idxs: int) -> typing.List[bool]:
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get done of errored environments: {set(idxs).intersection(self._errd_envs)}"
        active_idxs = []
        nonactive_idxs = []
        list_used = []
        for idx in idxs:
            if idx in self._active_envs:
                active_idxs.append(idx)
                list_used.append(active_idxs)
            else:
                nonactive_idxs.append(idx)
                list_used.append(nonactive_idxs)
        active_dones = ray.get([run_safely_on_actor(self._proof_env_pool[idx].get_done, self._timeout) for idx in active_idxs])
        for i, done in enumerate(active_dones):
            if isinstance(done, CapturedException):
                raise Exception(f"Error getting done for proof environment {i}: {done}")
        nonactive_dones = [self._nonactive_env_to_done_map.get(idx, None) for idx in nonactive_idxs]
        results = []
        active_idx = 0
        nonactive_idx = 0
        for i, idx in enumerate(idxs):
            list_to_use = list_used[i]
            if list_to_use == active_idxs:
                results.append(active_dones[active_idx])
                active_idx += 1
            else:
                results.append(nonactive_dones[nonactive_idx])
                nonactive_idx += 1
        return results

    def dump_proof(self, idxs: int):
        assert self._is_initialized, "Pool must be initialized before dumping"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot dump proof of errored environments: {set(idxs).intersection(self._errd_envs)}"
        proofs = ray.get([run_safely_on_actor(self._proof_env_pool[idx].dump_proof, self._timeout) for idx in idxs])
        for i, proof in enumerate(proofs):
            if isinstance(proof, CapturedException):
                raise Exception(f"Error dumping proof for proof environment {i}: {proof}")

    def _get_attr(self, attr_name: str, idxs: typing.List[int]):
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get attribute {attr_name} of errored environments: {set(idxs).intersection(self._errd_envs)}"
        attrs = ray.get([run_safely_on_actor(self._proof_env_pool[idx].getattr, self._timeout, args = [attr_name]) for idx in idxs])
        for i, attr in enumerate(attrs):
            if isinstance(attr, CapturedException):
                raise Exception(f"Error getting attribute {attr_name} for proof environment {i}: {attr}")
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
        should_load_envs = [False for _ in range(len(idxs))]
        init_remotes = []
        for should_load_env, idx in zip(should_load_envs, idxs):
            if not should_load_env:
                init_remotes.append(run_safely_on_actor(self._proof_env_pool[idx].reset, self._timeout))
        env_init_stats = ray.get(init_remotes)
        results = []
        envs_to_remove = []
        for i, env_init_stat in enumerate(env_init_stats):
            if isinstance(env_init_stat, CapturedException):
                self._errd_envs.add(idxs[i])
                self._errd_envs_exceptions[idxs[i]] = env_init_stat
                envs_to_remove.append(idxs[i])
                self._logger.error(f"Error initializing proof environment {i}: {env_init_stat}")
                results.append((None, None, None, 0.0, True, None))
            else:
                envs_removed = self._env_cache.add_to_cache(str(idxs[i]), idxs[i], 1)
                for env_removed in envs_removed:
                    if int(env_removed) not in idxs:
                        envs_to_remove.append(env_removed)
                self._active_envs.add(idxs[i])
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
            idxs = range(len(actions))
        # Make sure we are not stepping an errored environment
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
                self._proof_env_pool[idx] = ProofEnvActor.remote(
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
                self._proof_env_pool[idx] = ProofEnvActor.remote(*self._env_args_map[idx], **self._env_kwargs_map[idx])
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
        remotes = []
        for i, idx in enumerate(idxs):
            remotes.append(run_safely_on_actor(self._proof_env_pool[idx].step, self._timeout, args=[actions[i]]))
        return_remotes = ray.get(remotes)
        actual_returns = []
        for i, return_remote in enumerate(return_remotes):
            if isinstance(return_remote, CapturedException):
                self._errd_envs.add(idxs[i])
                self._errd_envs_exceptions[idxs[i]] = return_remote
                actual_returns.append((None, None, None, 0.0, True, None))
                self._logger.error(f"Error stepping proof environment {i}: {return_remote}")
            else:
                actual_returns.append(return_remote)
        return actual_returns

    def _try_cleanup_envs(self, idxs: typing.Union[typing.List[int], typing.List[str]]):
        self._logger.info(f"Cleaning up environments: {idxs}")
        idxs = [int(idx) for idx in idxs]
        try:
            state_remotes = []
            done_remotes = []
            for env_idx in idxs:
                proof_env_actor = self._proof_env_pool[env_idx]
                if env_idx in self._active_envs:
                    state_remotes.append(run_safely_on_actor(proof_env_actor.get_state, self._timeout))
                    done_remotes.append(run_safely_on_actor(proof_env_actor.get_done, self._timeout))
            states = ray.get(state_remotes)
            dones = ray.get(done_remotes)
            state_idx = 0
            for env_idx in idxs:
                if env_idx in self._active_envs:
                    if isinstance(states[state_idx], CapturedException) or isinstance(dones[state_idx], CapturedException):
                        self._logger.error(f"Error getting state/done for proof environment {env_idx}: {states[state_idx]}")
                        ex = Exception(f"Error getting state/done for proof environment {env_idx}: {states[state_idx]}")
                        raise CapturedException(ex)
                    else:
                        self._nonactive_env_to_state_map[env_idx] = states[state_idx]
                        self._nonactive_env_to_done_map[env_idx] = dones[state_idx]
                    state_idx += 1
            cleanup_remotes = []
            for env_idx in idxs:
                proof_env_actor = self._proof_env_pool[env_idx]
                if env_idx in self._active_envs:
                    cleanup_remotes.append(run_safely_on_actor(proof_env_actor.cleanup, timeout=15))
            ray.get(cleanup_remotes)
        except CapturedException as e:
            raise
        except Exception as e:
            self._logger.error(f"Error cleaning up proof environments: {e}")
        # Kill all actors
        for env_idx in idxs:
            if env_idx in self._active_envs:
                proof_env_actor = self._proof_env_pool[env_idx]
                try:
                    ray.kill(proof_env_actor)
                except Exception as e:
                    self._logger.error(f"Error killing proof environment actor: {e}")
        for env_idx in idxs:
            if env_idx in self._active_envs:
                self._active_envs.remove(env_idx)
        self._logger.info(f"Removed environments: {idxs}")
