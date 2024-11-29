#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import copy
import typing
import logging
import ray
from itp_interface.tools.isabelle_executor import IsabelleExecutor, HammerMode
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.proof_state import ProofState
from itp_interface.rl.simple_proof_env import ProofEnv, ProofEnvActor, ProofEnvInfo, ProofEnvReRankStrategy, ProofExecutorCallback

def replicate_proof_env(proof_env: ProofEnv, logger: typing.Optional[logging.Logger] = None) -> ProofEnv:
    new_proof_env = copy.deepcopy(proof_env)
    new_proof_env.logger = logger if logger else logging.getLogger(__name__)
    return new_proof_env

class CapturedException(Exception):
    def __init__(self, exception: Exception):
        self.exception = exception
        super().__init__(str(exception))

    def __str__(self):
        return f"CapturedException: {str(self.exception)}"

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

class ProofEnvPool(object):
    def __init__(self, 
            pool_size: int = 1,
            proof_env_actors: typing.List[ProofEnvActor] = None,
            proof_env: ProofEnv = None, 
            logger: typing.Optional[logging.Logger] = None):
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
        self._timeout = None
        self._errd_envs = set()
        self._errd_envs_exceptions = {}
        self._is_initialized = False
    
    def __enter__(self):
        self._is_initialized = True
        # load all environments which are not loaded
        should_load_envs = ray.get([proof_env_actor.should_load_env.remote() for proof_env_actor in self._proof_env_pool])
        init_remotes = []
        for should_load_env, proof_env_actor in zip(should_load_envs, self._proof_env_pool):
            if not should_load_env:
                catch_exception_actor = CaptureExceptionActor.remote(proof_env_actor.reset)
                init_remotes.append(catch_exception_actor.try_capture_exception.remote())
        env_init_stats = ray.get(init_remotes)
        for i, env_init_stat in enumerate(env_init_stats):
            if isinstance(env_init_stat, CapturedException):
                self._errd_envs.add(i)
                self._errd_envs_exceptions[i] = env_init_stat
                self._logger.error(f"Error initializing proof environment {i}: {env_init_stat}")
        try:
            self._timeout = max([ray.get(self._proof_env_pool[i].get_timeout.remote()) for i in range(self.pool_size)])
        except Exception as e:
            self._timeout = 60
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._is_initialized = False
        try:
            cleanup_remotes = [proof_env_actor.cleanup.remote() for proof_env_actor in self._proof_env_pool]
            ray.get(cleanup_remotes, timeout=15)
        except Exception as e:
            self._logger.error(f"Error cleaning up proof environments: {e}")
        # Kill all actors
        for proof_env_actor in self._proof_env_pool:
            try:
                ray.kill(proof_env_actor)
            except Exception as e:
                self._logger.error(f"Error killing proof environment actor: {e}")

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
                logger=None
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
                    logger=None
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
        remotes = []
        for i, idx in enumerate(idxs):
            catch_exception_actor = CaptureExceptionActor.remote(self._proof_env_pool[idx].step, timeout=self._timeout, args=[actions[i]])
            remotes.append(catch_exception_actor.try_capture_exception.remote())
        return_remotes = ray.get(remotes, timeout=self._timeout)
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

    def get_pool(self, idxs: typing.List[int]) -> 'ProofEnvPool':
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        return ProofEnvPool( 
            proof_env_actors=[self._proof_env_pool[idx] for idx in idxs], 
            logger=self._logger)
    
    def reset(self, idxs: typing.List[int]) -> typing.List[ProofState]:
        assert self._is_initialized, "Pool must be initialized before resetting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot reset errored environments: {set(idxs).intersection(self._errd_envs)}"
        return ray.get([self._proof_env_pool[idx].reset.remote() for idx in idxs])

    def get_state(self, idxs: int) -> typing.List[ProofState]:
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get state of errored environments: {set(idxs).intersection(self._errd_envs)}"
        return ray.get([self._proof_env_pool[idx].get_state.remote() for idx in idxs])
    
    def get_done(self, idxs: int) -> typing.List[bool]:
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get done of errored environments: {set(idxs).intersection(self._errd_envs)}"
        return ray.get([self._proof_env_pool[idx].get_done.remote() for idx in idxs])
    
    def dump_proof(self, idxs: int):
        assert self._is_initialized, "Pool must be initialized before dumping"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot dump proof of errored environments: {set(idxs).intersection(self._errd_envs)}"
        return ray.get([self._proof_env_pool[idx].dump_proof.remote() for idx in idxs])
    
    def _get_attr(self, attr_name: str, idxs: typing.List[int]):
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get attribute {attr_name} of errored environments: {set(idxs).intersection(self._errd_envs)}"
        return ray.get([self._proof_env_pool[idx].getattr.remote(attr_name) for idx in idxs])
    
    def get_proof_search_res(self, idxs: typing.List[int]) -> typing.List[typing.Tuple[typing.List[ProofAction], float]]:
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        assert len(set(idxs).intersection(self._errd_envs)) == 0, f"Cannot get proof search results of errored environments: {set(idxs).intersection(self._errd_envs)}"
        return self._get_attr("proof_search_res", idxs)

if __name__ == "__main__":
    import os
    os.chdir(root_dir)

    print("Interactive Proof Environment")
    supported_actions = [x.name for x in ProofAction.ActionType]

    def scan_action(language):
        inp_action_type = input(f"Enter an action type from {supported_actions}: (default RUN_TACTIC)")
        if inp_action_type not in supported_actions:
            inp_action_type = ProofAction.ActionType.RUN_TACTIC.name
        action_type = ProofAction.ActionType[inp_action_type]
        if action_type == ProofAction.ActionType.RUN_TACTIC:
            inp = input("Enter tactic(s) (';' separated): ")
            inp = inp.split(';')
            return ProofAction(action_type, language, tactics=inp)
        elif action_type == ProofAction.ActionType.GET_DFNS_THMS or action_type == ProofAction.ActionType.BACKTRACK or action_type == ProofAction.ActionType.EXIT:
            return ProofAction(action_type, language)
        else:
            raise Exception(f"Invalid action type {action_type}")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    inp = input("Want to run coq, lean, or isabelle env? (Enter 'coq'/'lean'/'lean4'/'isabelle') ")
    language = ProofAction.Language.COQ
    if inp == 'coq':
        proof_exec_callback = ProofExecutorCallback(
            project_folder=".",
            file_path="data/test/SimpleAlgebra.v",
            enable_search=False
        )
        theorem_name = "algb_add_comm"
        language = ProofAction.Language.COQ
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.BM25
    elif inp == 'lean':
        proof_exec_callback = ProofExecutorCallback(
            project_folder="data/test/lean_proj",
            file_path="data/test/lean_proj/src/simple_solved.lean",
            language=ProofAction.Language.LEAN,
            always_use_retrieval=True,
            keep_local_context=True
        )
        theorem_name = "a_plus_b_a_minus_a"
        language = ProofAction.Language.LEAN
        always_retrieve_thms = True
        retrieval_strategy = ProofEnvReRankStrategy.BM25
        pass
    elif inp == 'lean4':
        proof_exec_callback = ProofExecutorCallback(
            project_folder="data/test/lean4_proj",
            file_path="data/test/lean4_proj/Lean4Proj/Basic.lean",
            language=ProofAction.Language.LEAN4,
            always_use_retrieval=False,
            keep_local_context=True
        )
        theorem_name = "test3"
        language = ProofAction.Language.LEAN4
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.NO_RE_RANK
    elif inp == 'isabelle':
        proof_exec_callback = ProofExecutorCallback(
            project_folder="data/test",
            file_path="data/test/SimpleAlgebra.thy",
            language=ProofAction.Language.ISABELLE,
            use_hammer=HammerMode.AUTO
        )
        theorem_name = "sqrt_comp"
        language = ProofAction.Language.ISABELLE
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.BM25
    else:
        raise Exception(f"Invalid input {inp} for choosing coq/lean/lean4 env")
    
    if language == ProofAction.Language.ISABELLE:
        IsabelleExecutor.start_server(port=13000)
    
    try:
        test_ray = True
        if test_ray:
            logger = logging.getLogger(__name__)
            ray.init()
            env_actors = [
            ProofEnvActor.remote("test", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms, logger=logger, should_load_env=False)
            for _ in range(4)]
            pool = ProofEnvPool(proof_env_actors=env_actors, logger=logger)
            with pool:
                dones = pool.get_done(list(range(4)))
                action = scan_action(language)
                while action.action_type != ProofAction.ActionType.EXIT and not all(dones):
                    step_res = pool.step([action]*4, list(range(4)))
                    dones = []
                    for i, (state, act, new_state, reward, done, info) in enumerate(step_res):
                        if done:
                            print(f"Environment {i} done")
                        else:
                            print(f"Environment {i} not done")
                        dones.append(done)
                        print(f"[{i}] Reward: {reward}")
                        print(f"[{i}] Done: {done}")
                        print(f"[{i}] Info: {info.to_json()}")
                    if not all(dones):
                        action = scan_action(language)

            # If you wish to explicitly kill the actor, do so after the cleanup
            for env_actor in env_actors:
                ray.kill(env_actor)
    finally:
        if language == ProofAction.Language.ISABELLE:
            IsabelleExecutor.stop_server()
    