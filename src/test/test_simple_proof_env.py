#!/usr/bin/env python3

import sys
import logging
from itp_interface.rl.simple_proof_env import ProofEnv, ProofEnvReRankStrategy
from itp_interface.rl.proof_action import ProofAction
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.isabelle_executor import IsabelleExecutor, HammerMode


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
    print("Interactive Proof Environment (Non-Ray)")
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
        with ProofEnv("test", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms) as env:
            done = env.done
            env.render()
            action = scan_action(language, supported_actions)
            while action.action_type != ProofAction.ActionType.EXIT and not done:
                state, _, _, reward, done, info = env.step(action)
                env.render()
                if not done:
                    action = scan_action(language, supported_actions)
    finally:
        if language == ProofAction.Language.ISABELLE:
            IsabelleExecutor.stop_server()


if __name__ == "__main__":
    main()
