#!/usr/bin/env python3

import sys, os
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
INT_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'INT'))
if INT_dir not in sys.path:
    sys.path.append(INT_dir)

import os
import random
import logging
import re
import time
import json
import typing
import uuid
import bisect
import shutil
from typing import Iterator, List, Optional, Tuple, OrderedDict, Generator, Dict, Any
from itp_interface.tools.int_context import ProofContext
from int_environment.proof_system.all_axioms import all_axioms_to_prove
from int_environment.proof_system.graph_seq_conversion import Parser
from int_environment.proof_system.prover import Prover

logger = logging.getLogger()

class INTExecutor:
    def __init__(self,
        timeout_in_sec: int = 5,
        proof_step_iter: Optional[Iterator[str]] = None,
        enable_search: bool = False,
        suppress_error_log: bool = False,
        logger: Optional[logging.Logger] = None):
        self.ticks = str(time.time()).replace(".", "")
        self.random_num = str(random.randint(0, 1000000000))
        self.timeout_in_sec = min(timeout_in_sec, 7)
        self.main_file_iter = proof_step_iter 
        self.current_stmt = None
        self.proof_context : ProofContext = None
        self.int_error_messages : List[str] = []
        self.suppress_error_log = False
        self._proof_running = False
        self.logger = logger or logging.getLogger(__name__)
        self.enable_search = enable_search
        self.parser = Parser()
        self.prover = None
        pass

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _update_proof_state(self, observation: Dict[str, List[Any]]):
        assert self._proof_running == False
        self.prover = Prover(
            axioms = all_axioms_to_prove,
            conditions = observation['observation']['ground_truth'],
            objectives = observation['observation']['objectives'],
            prove_direction = 'backward'
        )
        self.proof_context = ProofContext.from_dict({
            'fg_goals': [{
                'hypotheses': [''],
                'goal': self.parser.observation_to_source(observation['observation'])
        }]})
        self._proof_running = True

    def run_next(self) -> bool:
        assert self.prover is not None and self._proof_running
        try:
            stmt = next(self.main_file_iter)
        except StopIteration:
            self.execution_complete = True
            return False
        self.current_stmt = stmt
        try:
            output = self.prover.apply_theorem_seq_style(stmt)
            if self.prover.is_proved():
                self._proof_running = False
            else:
                self.proof_context = ProofContext.from_dict({
                    'fg_goals': [{
                        'hypotheses': [''],
                        'goal': self.parser.observation_to_source(self.prover.get_observation())
                }]})
        except:
            if not self.suppress_error_log:
                logger.error(f"Got an exception while running '{stmt}' on INT.")
                logger.exception(f"Exception Log")
            raise
        return True
    
    def is_in_proof_mode(self):
        return True if self._proof_running and not self.prover.is_proved() else False

    def needs_qed(self):
        return False #(George; 9/24/24): Unnecessary for INT
    
    def needs_cut_close(self):
        return False #(George; 9/24/24): Unnecessary for INT

    def run_next_without_exec(self) -> bool:
        raise NotImplementedError
    
    def find_all_theorems_names(self) -> List[Tuple[str, str]]:
        raise NotImplementedError
    
    def get_tokens_in_given_stmt(self, stmt: str, ignore_first_token: bool = False) -> Iterator[str]:
        raise NotImplementedError

    def tokenize(self, stmt: str) -> Iterator[str]:
        raise NotImplementedError

    def search_type_matching_defns(self, name: str) -> List:
        raise NotImplementedError

    def get_all_type_matching_defns(self, name: str) -> Iterator:
        raise NotImplementedError

    def search_exact(self, name: str) -> List:
        raise NotImplementedError

    def search_defn(self, name: str, match_until: Tuple[str], max_search_res: Optional[int] = None) -> List[Tuple[str, str, bool]]:
        raise NotImplementedError

    def run_without_executing(self, stmt: str):
        raise NotImplementedError
    
    def run_lemma_without_executing(self):
        raise NotImplementedError
    
    def run_till_next_lemma(self) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError
    
    def run_till_next_lemma_return_exec_stmt(self) -> Generator[str, None, None]:
        raise NotImplementedError
    
    def run_to_finish_lemma_return_exec(self) -> Generator[str, None, None]:
        raise NotImplementedError
    
    def run_to_finish_lemma(self) -> bool:
        raise NotImplementedError
    
    def run_till_line_num(self, line_num: int):
        raise NotImplementedError
    
    def run_to_finish(self):
        raise NotImplementedError

    def get_lemma_name_if_running(self) -> Optional[str]:
        raise NotImplementedError
    
    def get_lemma_stmt_if_running(self) -> Optional[str]:
        return self.proof_context.fg_goals[0].goal
    
    def get_current_lemma_name(self) -> Optional[str]:
        return ""
    
    def _check_if_thm_read(self, idx: int, full_stmt: str) -> bool:
        raise NotImplementedError
    
    def _parse_theorem_stmt(self, idx: int, stmt: str, do_full_check: bool = False, interesting_span: typing.Tuple[int, int] = None) -> str:
        raise NotImplementedError
    
    def _execute_till_last_theorem(self, idx: int, full_stmt: str):
        raise NotImplementedError

    def _stmt_has_lemma(self, idx: int, stmt: str, do_full_check: bool = False) -> bool:
        raise NotImplementedError
    
    def _get_env(self, idx) -> Optional[int]:
        raise NotImplementedError
    
    def _update_env(self, idx: int):
        raise NotImplementedError
    
    def _update_proof_state_idx(self, idx: int):
        raise NotImplementedError
    
    def _should_start_proof(self, stmt: str) -> bool:
        raise NotImplementedError
    
    def _remove_proof_add_sorry(self) -> str:
        raise NotImplementedError

    def _get_error_msg(self, msg_idx, msg) -> str:
        raise NotImplementedError #Note(George, 9/20/24): INT does have error messages stemming from _/logic.py but it is unclear if that is important here.

    def _errors_since_last_thm(self, idx, error_message: str):
        raise NotImplementedError

    def _add_last_tactic(self, idx: int, stmt: str):
        raise NotImplementedError
    
    def _get_cmd_tactic_mode(self, idx: int, stmt: str):
        raise NotImplementedError

    def _backtrack_tactic_line(self, idx: int):
        raise NotImplementedError
    
    def _clear_tactics(self):
        raise NotImplementedError
