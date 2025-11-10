#!/usr/bin/env python3

import sys
import typing
import os
import copy
import enum
import logging
from itp_interface.tools.simple_lean4_sync_executor import SimpleLean4SyncExecutor
from itp_interface.tools.training_data_format import Goal, TheoremProvingTrainingDataFormat
from itp_interface.tools.lean_parse_utils import LeanLineByLineReader
from itp_interface.tools.lean_context_helper import Lean3ContextHelper
from itp_interface.tools.misc_defns import HammerMode
from itp_interface.tools.iter_helpers import IntertwinedIterator

class DynamicProofExecutor(SimpleLean4SyncExecutor):
    class RunState(object):
        def __init__(self):
            self.tactics_ran : typing.List[str] = []
            self.last_exception : typing.Optional[str] = None
            self.line_tactic_map = {}
            self.line_proof_context_map = {}
    UnfocussedGoalsDescription = "there are unsolved goals"
    ProofFinishedDescription = "no goals"
    NotInProofModeDescription = "not in proof mode"
    GoalDescriptionOrder = {
        UnfocussedGoalsDescription: 2, # This more hard
        ProofFinishedDescription: 1, # This is easier coz proof is almost done
        NotInProofModeDescription: 0 # This is the easiest as the proof is done
    }
    class ContextType(enum.Enum):
        NoContext = 0
        LocalContext = 1
        BestContext = 2

    def goal_description_compare(description1: str, descripton2: str) -> int:
        """
        Returns 1 if description1 < description2, 0 if description1 == description2, -1 if description1 > description2
        """
        # In case of no description it is much more harder as we have to do a lot of work
        # So None will have same value as unfocussed goals
        order1 = DynamicProofExecutor.GoalDescriptionOrder.get(description1, 2) if description1 is not None else 2
        order2 = DynamicProofExecutor.GoalDescriptionOrder.get(descripton2, 2) if descripton2 is not None else 2
        if order1 < order2:
            return 1
        elif order1 == order2:
            return 0
        else:
            return -1

    def __init__(self, coq_context_helper: Lean3ContextHelper, project_folder: str = None, proof_file: str = None, instruction_iter: typing.Optional[str] = None, use_hammer: typing.Union[bool, HammerMode] = False, timeout_in_seconds: int = 60, use_human_readable_proof_context: bool = True, suppress_error_log: bool = True, context_type: ContextType = ContextType.NoContext, keep_local_context = False, enforce_qed: bool = False):
        assert proof_file is None or os.path.exists(proof_file), f"Proof file {proof_file} does not exist"
        assert coq_context_helper is not None, "coq_context_helper must not be None"
        self.proof_file = proof_file
        self.context_type = context_type
        self.lean_file_iter = LeanLineByLineReader(proof_file, remove_comments=False, no_strip=True).instruction_step_generator() if proof_file is not None else instruction_iter
        self.tactic_switch_iterator = IntertwinedIterator(self.lean_file_iter)
        self.run_state = DynamicProofExecutor.RunState()
        self.logger = None
        self.lean_context_helper = coq_context_helper
        super().__init__(project_root=project_folder, proof_step_iter=self.tactic_switch_iterator, use_hammer=use_hammer, timeout_in_sec=timeout_in_seconds, use_human_readable_proof_context=use_human_readable_proof_context, suppress_error_log=suppress_error_log, keep_local_context=keep_local_context, enforce_qed=enforce_qed)

    def __enter__(self):
        self.lean_context_helper.__enter__()
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lean_context_helper.__exit__(exc_type, exc_val, exc_tb)
        super().__exit__(exc_type, exc_val, exc_tb)

    def set_logger(self, logger: logging.Logger):
        self.logger = logger
        pass

    def get_focussed_goals(self) -> typing.List[Goal]:
        if not self.is_in_proof_mode():
            return []
        if self.proof_context is None:
            return []
        return self.lean_context_helper.get_focussed_goals(self)
    
    def get_unfocussed_goals(self) -> typing.List[Goal]:
        if not self.is_in_proof_mode():
            return []
        if self.proof_context is None:
            return []
        return self.lean_context_helper.get_unfocussed_goals(self)

    def get_current_proof_state_as_training_data(self) -> TheoremProvingTrainingDataFormat:
        # get the current goal
        if self.needs_cut_close():
            current_goals = self.get_unfocussed_goals()
            training_data_format = TheoremProvingTrainingDataFormat(start_goals=current_goals)
            training_data_format.goal_description = DynamicProofExecutor.UnfocussedGoalsDescription
        elif not self.is_in_proof_mode():
            current_goals = self.get_focussed_goals()
            training_data_format = TheoremProvingTrainingDataFormat(start_goals=current_goals)
            training_data_format.goal_description = DynamicProofExecutor.NotInProofModeDescription
        elif self.needs_qed():
            current_goals = self.get_focussed_goals()
            assert len(current_goals) == 0, "There should be no goals when needs_qed is True"
            training_data_format = TheoremProvingTrainingDataFormat(start_goals=current_goals)
            training_data_format.goal_description = DynamicProofExecutor.ProofFinishedDescription
        else:
            current_goals = self.get_focussed_goals()
            training_data_format = TheoremProvingTrainingDataFormat(start_goals=current_goals)
            if len(self.lean_error_messages) > 0:
                training_data_format.goal_description = '\n'.join(self.lean_error_messages)
            else:
                training_data_format.goal_description = None
        return training_data_format

    def run_tactics(self, tactics: typing.List[str]) -> typing.Tuple[int, bool]:
        tactic_failed = False
        start_line_num = self.line_num
        self.run_state.line_tactic_map[self.line_num] = len(self.run_state.tactics_ran)
        self.run_state.line_proof_context_map[self.line_num] = copy.deepcopy(self.proof_context)
        for tactic in tactics:
            self.tactic_switch_iterator.set_next_instruction(tactic)
            self.run_next()
            self.run_state.tactics_ran.append(tactic)
            self.run_state.line_proof_context_map[self.line_num] = copy.deepcopy(self.proof_context)
        was_cancelled = False
        if len(self.lean_error_messages) > 0:
            current_thm_name = self.get_lemma_name_if_running()
            assert current_thm_name is not None, "current_thm_name must not be None"
            tactic_failed = True
            self.run_state.last_exception = '\n'.join(self.lean_error_messages)
            # Cancel the last tactic
            self.cancel_tactic_till_line(start_line_num, no_backtracking=True)
            was_cancelled = True
        if self._last_tactic_was_modified:
            assert self._last_modified_tactic is not None, "last_modified_tactic must not be None if last_tactic_was_modified is True"
            if not was_cancelled:
                assert len(self.run_state.tactics_ran) > 0, "There must be at least one tactic ran if last_tactic_was_modified is True" + \
                    f" but got len={len(self.run_state.tactics_ran)}\n" + \
                    f"modified tactic = \n{self._last_modified_tactic}\n" + \
                    f"len(tactics) = {len(tactics)}"
                original_last_tactic = tactics[-1]
                if self._last_modified_tactic != original_last_tactic:
                    self.run_state.tactics_ran[-1] = self._last_modified_tactic
                else:
                    self._last_modified_tactic = None
                    self._last_tactic_was_modified = False
        return start_line_num, not tactic_failed
    
    def get_last_tactic(self) -> typing.Optional[str]:
        if len(self.run_state.tactics_ran) == 0:
            return None
        if self.run_state.last_exception is not None:
            return None
        if self._last_tactic_was_modified:
            return self._last_modified_tactic
        else:
            return None

    def get_last_exception(self) -> typing.Optional[str]:
        last_exception = self.run_state.last_exception
        self.run_state.last_exception = None
        return last_exception
    
    def skip_to_theorem(self, theorem_name: str):
        self._skip_to_theorem(theorem_name)

    # [TODO] change this for bactracking
    def cancel_tactic_till_line(self, tactic_line_num: int, no_backtracking: bool = False) -> bool:
        assert tactic_line_num <= self.line_num, "tactic_line_num must be <= self.line_num"
        assert tactic_line_num >= 0, "tactic_line_num must be >= 0"
        cancelled_some_tactics = False
        if tactic_line_num < self.line_num:
            state_num = self.run_state.line_tactic_map[tactic_line_num]
            self.run_state.tactics_ran = self.run_state.tactics_ran[:state_num]
            line_tactic_map_keys = list(self.run_state.line_tactic_map.keys())
            for line_num in line_tactic_map_keys:
                if line_num >= tactic_line_num:
                    del self.run_state.line_tactic_map[line_num]
            line_proof_context_map_keys = list(self.run_state.line_proof_context_map.keys())
            if not no_backtracking:
                self.proof_context = self.run_state.line_proof_context_map[tactic_line_num]
                self.line_num = tactic_line_num
                cancelled_some_tactics = self._backtrack_tactic_line(tactic_line_num)
                self._proof_running = self.proof_context is not None
            for line_num in line_proof_context_map_keys:
                if line_num >= tactic_line_num:
                    del self.run_state.line_proof_context_map[line_num]
        return cancelled_some_tactics