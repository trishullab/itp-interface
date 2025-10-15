import unittest

class Helper():
    def __init__(self):
        self.current_switch = None
    
    def build_lean4_project(self, project_folder):
        import os
        # Build the project
        with os.popen(f"cd {project_folder} && lake exe cache get && lake build") as proc:
            print("Building Lean4 project...")
            print('-'*15 + 'Build Logs' + '-'*15)
            print(proc.read())
            print('-'*15 + 'End Build Logs' + '-'*15)

    def build_coq_project(self, project_folder):
        import os
        try:
            with os.popen("opam switch show") as proc:
                self.current_switch = proc.read().strip()
        except:
            self.current_switch = None
        # Check if the switch exists
        # opam switch create simple_grp_theory 4.14.2
        if os.system("opam switch simple_grp_theory") != 0:
            cmds = [
                'opam switch create simple_grp_theory 4.14.2',
                'opam switch simple_grp_theory', 
                'eval $(opam env)',
                'opam repo add coq-released https://coq.inria.fr/opam/released',
                'opam pin add -y coq-lsp 0.1.8+8.18'
            ]
            final_cmd = ' && '.join(cmds)
            os.system(final_cmd)
        # IMPORTANT NOTE: Make sure to switch to the correct switch before running the code.
        os.system("opam switch simple_grp_theory && eval $(opam env)")
        # Clean the project
        os.system(f"eval $(opam env) && cd {project_folder} && make clean")
        # Build the project
        with os.popen(f"eval $(opam env) && cd {project_folder} && make") as proc:
            print("Building Coq project...")
            print('-'*15 + 'Build Logs' + '-'*15)
            print(proc.read())
            print('-'*15 + 'End Build Logs' + '-'*15)

    def switch_to_current_switch(self):
        import os
        if self.current_switch is not None:
            try:
                proc = os.popen(f"opam switch {self.current_switch} && eval $(opam env)")
                print(proc.read())
            finally:
                proc.close()

class Lean4Test(unittest.TestCase):
    def test_simple_lean4(self):
        from itp_interface.rl.proof_state import ProofState
        from itp_interface.rl.proof_action import ProofAction
        from itp_interface.rl.simple_proof_env import ProofEnv
        from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
        from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
        project_folder = "src/data/test/lean4_proj"
        file_path = "src/data/test/lean4_proj/Lean4Proj/Basic.lean"
        # Build the project
        # cd src/data/test/lean4_proj && lake build
        helper = Helper()
        helper.build_lean4_project(project_folder)
        language = ProofAction.Language.LEAN4
        theorem_name = "test3"
        # theorem test3 (p q : Prop) (hp : p) (hq : q)
        # : p ∧ q ∧ p :=
        proof_exec_callback = ProofExecutorCallback(
            project_folder=project_folder,
            file_path=file_path,
            language=language,
            always_use_retrieval=False,
            keep_local_context=True
        )
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.NO_RE_RANK
        env = ProofEnv("test_lean4", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms)
        proof_steps = [
            '-- TODO',
            'apply And.intro',
            'exact hp',
            'apply And.intro',
            '--TODO',
            '-- This is just some extra comment',
            'exact hq',
            'exact hp'
        ]
        with env:
            proof_was_finished = False
            for proof_step in proof_steps:
                state, _, next_state, _, done, info = env.step(ProofAction(
                    ProofAction.ActionType.RUN_TACTIC, 
                    language, 
                    tactics=[proof_step]))
                if info.error_message is not None:
                    print(f"Error: {info.error_message}")
                # This prints StateChanged, StateUnchanged, Failed, or Done
                print(info.progress)
                print('-'*30)
                if done:
                    print("Proof Finished!!")
                    proof_was_finished = True
                else:
                    s1 : ProofState = state
                    s2 : ProofState = next_state
                    print(f"Current Goal:")
                    print('-'*30)
                    for goal in s1.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
                    print(f"Action: {proof_step}")
                    print(f"="*30)
                    print(f"Next Goal:")
                    print('-'*30)
                    for goal in s2.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
            assert proof_was_finished, "Proof was not finished"

    def test_lean4_backtracking(self):
        from itp_interface.rl.proof_state import ProofState
        from itp_interface.rl.proof_action import ProofAction
        from itp_interface.rl.simple_proof_env import ProofEnv
        from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
        from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
        import random
        project_folder = "src/data/test/lean4_proj"
        file_path = "src/data/test/lean4_proj/Lean4Proj/Basic.lean"
        # Build the project
        helper = Helper()
        helper.build_lean4_project(project_folder)
        language = ProofAction.Language.LEAN4
        theorem_name = "test3"
        # theorem test3 (p q : Prop) (hp : p) (hq : q)
        # : p ∧ q ∧ p :=
        proof_exec_callback = ProofExecutorCallback(
            project_folder=project_folder,
            file_path=file_path,
            language=language,
            always_use_retrieval=False,
            keep_local_context=True
        )
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.NO_RE_RANK
        env = ProofEnv("test_lean4", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms)
        proof_steps = [
            'apply And.intro',
            'exact hp',
            'apply And.intro',
            'exact hq',
            'exact hp'
        ]
        with env:
            prev_state = env.state
            proof_was_finished = False
            for idx, proof_step in enumerate(proof_steps):
                if idx > 0 and random.random() <= 0.5:
                    print(f"Backtracking at step {idx + 1} i.e. {proof_step}")
                    state, _, next_state, _, done, info = env.step(
                    ProofAction(
                        ProofAction.ActionType.BACKTRACK, 
                        language))
                    assert next_state == prev_state, "Backtracking failed"
                    # Replay the last action
                    last_proof_step = proof_steps[idx-1]
                    state, _, next_state, _, done, info = env.step(
                        ProofAction(
                            ProofAction.ActionType.RUN_TACTIC, 
                            language, 
                            tactics=[last_proof_step]))
                state, _, next_state, _, done, info = env.step(
                ProofAction(
                    ProofAction.ActionType.RUN_TACTIC, 
                    language, 
                    tactics=[proof_step]))
                prev_state = state
                if done:
                    print("Proof Finished!!")
                    proof_was_finished = True
            assert proof_was_finished, "Proof was not finished"

    def test_simple_coq(self):
        from itp_interface.rl.proof_state import ProofState
        from itp_interface.rl.proof_action import ProofAction
        from itp_interface.rl.simple_proof_env import ProofEnv
        from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
        from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
        project_folder = "src/data/test/coq/custom_group_theory/theories"
        file_path = "src/data/test/coq/custom_group_theory/theories/grpthm.v"
        # Build the project
        # cd src/data/test/coq/custom_group_theory/theories && make
        helper = Helper()
        helper.build_coq_project(project_folder)
        language = ProofAction.Language.COQ
        theorem_name = "algb_identity_sum"
        # Theorem algb_identity_sum : 
        # forall a, algb_add a e = a.
        proof_exec_callback = ProofExecutorCallback(
            project_folder=project_folder,
            file_path=file_path,
            language=language,
            always_use_retrieval=False,
            keep_local_context=True
        )
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.NO_RE_RANK
        env = ProofEnv("test_coq", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms)
        proof_steps = [
            'intros.',
            'destruct a.',
            '- reflexivity.',
            '- reflexivity.'
        ]
        with env:
            for proof_step in proof_steps:
                state, _, next_state, _, done, info = env.step(ProofAction(
                    ProofAction.ActionType.RUN_TACTIC, 
                    language, 
                    tactics=[proof_step]))
                if info.error_message is not None:
                    print(f"Error: {info.error_message}")
                # This prints StateChanged, StateUnchanged, Failed, or Done
                print(info.progress)
                print('-'*30)
                if done:
                    print("Proof Finished!!")
                else:
                    s1 : ProofState = state
                    s2 : ProofState = next_state
                    print(f"Current Goal:")
                    print('-'*30)
                    for goal in s1.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
                    print(f"Action: {proof_step}")
                    print(f"="*30)
                    print(f"Next Goal:")
                    print('-'*30)
                    for goal in s2.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
        helper.switch_to_current_switch()

    def test_simple_lean_calc(self):
        from itp_interface.rl.proof_state import ProofState
        from itp_interface.rl.proof_action import ProofAction
        from itp_interface.rl.simple_proof_env import ProofEnv
        from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
        from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
        project_folder = "src/data/test/lean4_proj"
        file_path = "src/data/test/lean4_proj/Lean4Proj/Basic.lean"
        # Build the project
        # cd src/data/test/lean4_proj && lake build
        helper = Helper()
        helper.build_lean4_project(project_folder)
        language = ProofAction.Language.LEAN4
        theorem_name = "test_calc"
        # theorem test_calc (n: Nat) : n^2 + 2*n + 1 = (n + 1)*(n + 1) := by
        proof_exec_callback = ProofExecutorCallback(
            project_folder=project_folder,
            file_path=file_path,
            language=language,
            always_use_retrieval=False,
            keep_local_context=True
        )
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.NO_RE_RANK
        env = ProofEnv("test_lean4", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms)
        proof_steps = [
"""calc
_ = n^2 + n*2 + 1 := by rw [Nat.mul_comm 2 n]
_ = n^2 + (n + n) + 1 := by rw [Nat.mul_two]
_ = n^2 + n + n + 1 := by rw [←Nat.add_assoc]
_ = n*n + n + n + 1 := by rw [Nat.pow_two]
_ = n*n + n*1 + n + 1 := by rw [Nat.mul_one n]
_ = n*(n + 1) + n + 1 := by rw [Nat.left_distrib n n 1]
_ = n*(n + 1) + (n + 1) := by rw [Nat.add_assoc]
_ = n*(n + 1) + 1*(n + 1) := by rw (config := { occs := .pos [2]}) [←Nat.mul_one (n + 1), Nat.mul_comm]""",
"_ = (n + 1)*(n + 1) := by \n   rw [Nat.right_distrib n 1 (n + 1)]"
]
        with env:
            proof_was_finished = False
            for proof_step in proof_steps:
                state, _, next_state, _, done, info = env.step(ProofAction(
                    ProofAction.ActionType.RUN_TACTIC, 
                    language, 
                    tactics=[proof_step]))
                if info.error_message is not None:
                    print(f"Error: {info.error_message}")
                # This prints StateChanged, StateUnchanged, Failed, or Done
                print(f"DONE: {done}")
                print(info.progress)
                print('-'*30)
                if done:
                    s1 : ProofState = state
                    print(f"Current Goal:")
                    print('-'*30)
                    for goal in s1.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
                    print(f"Action: {proof_step}")
                    print(f"="*30)
                    print("Proof Finished!!")
                    proof_was_finished = True
                else:
                    s1 : ProofState = state
                    s2 : ProofState = next_state
                    print(f"Current Goal:")
                    print('-'*30)
                    for goal in s1.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
                    print(f"Action: {proof_step}")
                    print(f"="*30)
                    print(f"Next Goal:")
                    print('-'*30)
                    for goal in s2.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
            assert proof_was_finished, "Proof was not finished"

    def test_simple_lean_enforce_done_test(self):
        from itp_interface.rl.proof_state import ProofState
        from itp_interface.rl.proof_action import ProofAction
        from itp_interface.rl.simple_proof_env import ProofEnv
        from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
        from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
        project_folder = "src/data/test/lean4_proj"
        file_path = "src/data/test/lean4_proj/Lean4Proj/Basic.lean"
        # Build the project
        # cd src/data/test/lean4_proj && lake build
        helper = Helper()
        helper.build_lean4_project(project_folder)
        language = ProofAction.Language.LEAN4
        theorem_name = "test_calc"
        # theorem test_calc (n: Nat) : n^2 + 2*n + 1 = (n + 1)*(n + 1) := by
        proof_exec_callback = ProofExecutorCallback(
            project_folder=project_folder,
            file_path=file_path,
            language=language,
            always_use_retrieval=False,
            keep_local_context=True,
            enforce_qed=True
        )
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.NO_RE_RANK
        env = ProofEnv("test_lean4", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms)
        proof_steps = [
"""calc
    _ = n^2 + n*2 + 1 := by rw [Nat.mul_comm 2 n]
    _ = n^2 + (n + n) + 1 := by rw [Nat.mul_two]
    _ = n^2 + n + n + 1 := by rw [←Nat.add_assoc]
    _ = n*n + n + n + 1 := by rw [Nat.pow_two]
    _ = n*n + n*1 + n + 1 := by rw [Nat.mul_one n]
    _ = n*(n + 1) + n + 1 := by rw [Nat.left_distrib n n 1]
    _ = n*(n + 1) + (n + 1) := by rw [Nat.add_assoc]
    _ = n*(n + 1) + 1*(n + 1) := by rw (config := { occs := .pos [2]}) [←Nat.mul_one (n + 1), Nat.mul_comm]""",
"    _ = (n + 1)*(n + 1) := by rw [Nat.right_distrib n 1 (n + 1)]",
"done"
]
        with env:
            proof_finished = False
            for proof_step in proof_steps:
                state, _, next_state, _, done, info = env.step(ProofAction(
                    ProofAction.ActionType.RUN_TACTIC, 
                    language, 
                    tactics=[proof_step]))
                if info.error_message is not None:
                    print(f"Error: {info.error_message}")
                # This prints StateChanged, StateUnchanged, Failed, or Done
                print(f"DONE: {done}")
                print(info.progress)
                print('-'*30)
                if done:
                    assert proof_step == "done", "Proof can only finish with done"
                    s1 : ProofState = state
                    print(f"Current Goal:")
                    print('-'*30)
                    for goal in s1.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
                    print(f"Action: {proof_step}")
                    print(f"="*30)
                    print("Proof Finished!!")
                    proof_finished = True
                else:
                    s1 : ProofState = state
                    s2 : ProofState = next_state
                    print(f"Current Goal:")
                    print('-'*30)
                    for goal in s1.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
                    print(f"Action: {proof_step}")
                    print(f"="*30)
                    print(f"Next Goal:")
                    print('-'*30)
                    for goal in s2.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
            assert proof_finished, "Proof was not finished"

    def test_simple_lean4_done_test(self):
        from itp_interface.rl.proof_state import ProofState
        from itp_interface.rl.proof_action import ProofAction
        from itp_interface.rl.simple_proof_env import ProofEnv
        from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
        from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
        project_folder = "src/data/test/lean4_proj"
        file_path = "src/data/test/lean4_proj/Lean4Proj/Basic.lean"
        # Build the project
        # cd src/data/test/lean4_proj && lake build
        helper = Helper()
        helper.build_lean4_project(project_folder)
        language = ProofAction.Language.LEAN4
        theorem_name = "test3"
        # theorem test3 (p q : Prop) (hp : p) (hq : q)
        # : p ∧ q ∧ p :=
        proof_exec_callback = ProofExecutorCallback(
            project_folder=project_folder,
            file_path=file_path,
            language=language,
            always_use_retrieval=False,
            keep_local_context=True,
            enforce_qed=True
        )
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.NO_RE_RANK
        env = ProofEnv("test_lean4", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms)
        proof_steps = [
            'apply And.intro',
            'exact hp',
            'apply And.intro',
            'exact hq',
            'done'
        ]
        with env:
            for proof_step in proof_steps:
                state, _, next_state, _, done, info = env.step(ProofAction(
                    ProofAction.ActionType.RUN_TACTIC, 
                    language, 
                    tactics=[proof_step]))
                if info.error_message is not None:
                    print(f"Error: {info.error_message}")
                # This prints StateChanged, StateUnchanged, Failed, or Done
                print(info.progress)
                print('-'*30)
                if done:
                    raise Exception("Proof should not have finished")
                else:
                    s1 : ProofState = state
                    s2 : ProofState = next_state
                    print(f"Current Goal:")
                    print('-'*30)
                    for goal in s1.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)
                    print(f"Action: {proof_step}")
                    print(f"="*30)
                    print(f"Next Goal:")
                    print('-'*30)
                    for goal in s2.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                    print(f"="*30)

    def test_simple_lean4_have_test(self):
        from itp_interface.rl.proof_state import ProofState
        from itp_interface.rl.proof_action import ProofAction
        from itp_interface.rl.simple_proof_env import ProofEnv
        from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
        from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
        project_folder = "src/data/test/lean4_proj"
        file_path = "src/data/test/lean4_proj/Lean4Proj/Basic.lean"
        # Build the project
        # cd src/data/test/lean4_proj && lake build
        helper = Helper()
        helper.build_lean4_project(project_folder)
        language = ProofAction.Language.LEAN4
        theorem_name = "imo_1959_p1"
        # theorem test3 (p q : Prop) (hp : p) (hq : q)
        # : p ∧ q ∧ p :=
        proof_exec_callback = ProofExecutorCallback(
            project_folder=project_folder,
            file_path=file_path,
            language=language,
            always_use_retrieval=False,
            keep_local_context=True,
            enforce_qed=True
        )
        always_retrieve_thms = False
        retrieval_strategy = ProofEnvReRankStrategy.NO_RE_RANK
        env = ProofEnv("test_lean4", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms)
        proof_steps = [
'rw [Nat.gcd_rec]',
'rw [Nat.mod_eq_of_lt (by linarith)]',
'rw [Nat.gcd_rec]',
'rw [Nat.gcd_rec]',
'have eq₂ : (21 * n + 4) % (14 * n + 3) = 7 * n + 1 := by',
'    have eq₁ : 21 * n + 4 = (14 * n + 3) + (7 * n + 1) := by ring',
'    rw [eq₁, Nat.add_mod, Nat.mod_self, zero_add]',
'    have h₂ : 7 * n + 1 < 14 * n + 3 := by linarith',
'    rw [Nat.mod_eq_of_lt]',
'    rw [Nat.mod_eq_of_lt]',
'    exact h₂',
'    rw [Nat.mod_eq_of_lt]',
'    exact h₂',
'    exact h₂',
'rw [eq₂]'
        ]
        with env:
            for proof_step in proof_steps:
                state, _, next_state, _, done, info = env.step(ProofAction(
                    ProofAction.ActionType.RUN_TACTIC, 
                    language, 
                    tactics=[proof_step]))
                if info.error_message is not None:
                    print(f"Error: {info.error_message}")
                # This prints StateChanged, StateUnchanged, Failed, or Done
                print(info.progress)
                print('-'*30)
                if done:
                    raise Exception("Proof should not have finished")
                else:
                    s1 : ProofState = state
                    s2 : ProofState = next_state
                    print(f"Current Goal:")
                    print('-'*30)
                    for goal in s1.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                        print(f'*'*30)
                    print(f"="*30)
                    print(f"Action: {proof_step}")
                    print(f"="*30)
                    print(f"Next Goal:")
                    print('-'*30)
                    for goal in s2.training_data_format.start_goals:
                        hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                        print(hyps)
                        print('|- ', end='')
                        print(goal.goal)
                        print(f'*'*30)
                    print(f"="*30)
                    print(f"DONE: {done}")
                    print('-'*30)

def main():
    unittest.main()

if __name__ == '__main__':
    main()