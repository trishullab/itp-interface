#!/usr/bin/env python3

import sys
import logging
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
        print("Interactive Proof Environment (Ray - Process-based)")
    else:
        print("Interactive Proof Environment (Thread-based)")

    supported_actions = [x.name for x in ProofAction.ActionType]

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    inp = input("Want to run coq, lean, or isabelle env? (Enter 'coq'/'lean'/'lean4'/'isabelle') ")
    language = ProofAction.Language.COQ

    if inp == 'coq':
        proof_exec_callback = ProofExecutorCallback(
            project_folder=".",
            file_path="src/data/test/SimpleAlgebra.v"
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
        raise Exception(f"Invalid input {inp} for choosing coq/lean/lean4/isabelle env")

    if language == ProofAction.Language.ISABELLE:
        IsabelleExecutor.start_server(port=13000)

    try:
        logger = logging.getLogger(__name__)

        if HAS_RAY:
            # Ray-based implementation (process-based parallelism)
            ray.init()
            env_actor = ProofEnvActor.remote("test", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms, logger=logger)

            done_id = env_actor.get_done.remote()
            done = ray.get(done_id)
            action = scan_action(language, supported_actions)
            while action.action_type != ProofAction.ActionType.EXIT and not done:
                step_id = env_actor.step.remote(action)
                state, _, _, reward, done, info = ray.get(step_id)
                print(f"Reward: {reward}")
                print(f"Done: {done}")
                print(f"Info: {info.to_json()}")
                ray.get(env_actor.render.remote())
                if not done:
                    action = scan_action(language, supported_actions)

            # Cleanup
            cleanup_future = env_actor.cleanup.remote()
            ray.get(cleanup_future)
            ray.kill(env_actor)
        else:
            # Thread-based implementation (thread-safe, no Ray)
            env_actor = ProofEnvActor("test", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms, logger=logger)

            done = env_actor.get_done()
            action = scan_action(language, supported_actions)
            while action.action_type != ProofAction.ActionType.EXIT and not done:
                state, _, _, reward, done, info = env_actor.step(action)
                print(f"Reward: {reward}")
                print(f"Done: {done}")
                print(f"Info: {info.to_json()}")
                env_actor.render()
                if not done:
                    action = scan_action(language, supported_actions)

            # Cleanup
            env_actor.cleanup()
    finally:
        if language == ProofAction.Language.ISABELLE:
            IsabelleExecutor.stop_server()


if __name__ == "__main__":
    main()
