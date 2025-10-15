#!/usr/bin/env python3

import sys
import logging
from itp_interface.rl.simple_proof_env_pool import ProofEnvPool
from itp_interface.rl.simple_proof_env_ray import ProofEnvActor, HAS_RAY
from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
from itp_interface.rl.proof_action import ProofAction
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.isabelle_executor import IsabelleExecutor, HammerMode

# Conditional Ray import
if HAS_RAY:
    import ray


def scan_action(language, supported_actions):
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


def main():
    if HAS_RAY:
        print("Interactive Proof Environment Pool (Ray - Process-based)")
    else:
        print("Interactive Proof Environment Pool (Thread-based)")

    supported_actions = [x.name for x in ProofAction.ActionType]

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    inp = input("Want to run coq, lean, or isabelle env? (Enter 'coq'/'lean'/'lean4'/'isabelle') ")
    language = ProofAction.Language.COQ

    if inp == 'coq':
        proof_exec_callback = ProofExecutorCallback(
            project_folder=".",
            file_path="src/data/test/SimpleAlgebra.v",
            enable_search=False
        )
        theorem_name = "algb_add_comm"
        language = ProofAction.Language.COQ
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.BM25
    elif inp == 'lean':
        proof_exec_callback = ProofExecutorCallback(
            project_folder="src/data/test/lean_proj",
            file_path="src/data/test/lean_proj/src/simple_solved.lean",
            language=ProofAction.Language.LEAN,
            always_use_retrieval=True,
            keep_local_context=True
        )
        theorem_name = "a_plus_b_a_minus_a"
        language = ProofAction.Language.LEAN
        always_retrieve_thms = True
        retrieval_strategy = ProofEnvReRankStrategy.BM25
    elif inp == 'lean4':
        proof_exec_callback = ProofExecutorCallback(
            project_folder="src/data/test/lean4_proj",
            file_path="src/data/test/lean4_proj/Lean4Proj/Basic.lean",
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
            project_folder="src/data/test",
            file_path="src/data/test/SimpleAlgebra.thy",
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
        logger = logging.getLogger(__name__)

        if HAS_RAY:
            # Ray-based implementation (process-based parallelism)
            ray.init()
            env_actors = [
                ProofEnvActor.remote("test", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms, logger=logger, should_load_env=False)
                for _ in range(4)]
            pool = ProofEnvPool(proof_env_actors=env_actors, logger=logger, max_parallel_envs=3)
            with pool:
                dones = pool.get_done(list(range(4)))
                action = scan_action(language, supported_actions)
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
                        action = scan_action(language, supported_actions)

            # Cleanup actors
            for env_actor in env_actors:
                ray.kill(env_actor)
        else:
            # Thread-based implementation (thread-safe, no Ray)
            env_actors = [
                ProofEnvActor("test", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms, logger=logger, should_load_env=False)
                for _ in range(4)]
            pool = ProofEnvPool(proof_env_actors=env_actors, logger=logger, max_parallel_envs=3)
            with pool:
                dones = pool.get_done(list(range(4)))
                action = scan_action(language, supported_actions)
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
                        action = scan_action(language, supported_actions)

            # Cleanup actors
            for env_actor in env_actors:
                env_actor.cleanup()
    finally:
        if language == ProofAction.Language.ISABELLE:
            IsabelleExecutor.stop_server()


if __name__ == "__main__":
    main()
