#!/usr/bin/env python3

import sys

from itp_interface.lean_server.lean_utils import ProofContext
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import random
from itp_interface.tools.lean_parse_utils import LeanLineByLineReader
from itp_interface.lean_server.lean4_repl_interface import ProcessInterface
from typing import Iterator, List, Optional, Tuple, OrderedDict


class Lean4SyncExecutor:
    def __init__(self, 
        project_root: Optional[str] = None, 
        prefix: Optional[str] = None, 
        main_file: Optional[str] = None, 
        use_hammer: bool = False, 
        timeout_in_sec: int = 60, 
        use_human_readable_proof_context: bool = False, 
        proof_step_iter: Optional[Iterator[str]] = None, 
        suppress_error_log: bool = False, 
        mathlib_root: Optional[str] = None, 
        enable_search: bool = False, 
        namespaces: Optional[List[str]] = None, 
        keep_local_context: bool = False):
        assert proof_step_iter is None or isinstance(proof_step_iter, Iterator), \
            "proof_step_iter must be an iterator"
        assert main_file is not None or proof_step_iter is not None, \
            "Either main_file or proof_step_iter must be provided"
        assert main_file is None or proof_step_iter is None, \
            "Only one of main_file or proof_step_iter must be provided"
        assert main_file is None or (os.path.exists(main_file) and main_file.endswith(".lean")), \
            "main_file must be a valid path to a '.lean' file"
        assert project_root is None or (os.path.exists(project_root) and os.path.isdir(project_root)), \
            "project_root must be a valid path to a directory"
        assert not use_hammer, "Hammer is not supported for Lean4"
        self.use_human_readable_proof_context = use_human_readable_proof_context
        self.project_root = project_root if project_root is not None else "."
        self.main_file = main_file if main_file is not None else f"temp{random.randint(0, 100000000)}.lean"
        self.use_hammer = use_hammer
        self.timeout_in_sec = min(timeout_in_sec, 120) # Maximum 120s timeout
        self.current_stmt = None
        self.line_num = 0
        self.main_file_iter = proof_step_iter
        self.suppress_error_log = suppress_error_log
        self.process_interace : ProcessInterface = None
        self.execution_complete = False
        self._max_memory_in_mib = 40000 # 40 GiB is needed for mathlib to work seemlessly
        self._lines_executed = []
        self.proof_context : ProofContext = None
        self.curr_lemma_name : Optional[str] = None
        self.curr_lemma : Optional[str] = None
        self.lean_error_messages : List[Message] = []
        self._proof_running = False
        self._file_content = ""
        self.local_file_lemmas: OrderedDict[str, str] = OrderedDict()
        self.local_theorem_lemma_description: OrderedDict[str, str] = OrderedDict()
        self._proof_start_idx: Optional[int] = None
        self._import_end_idx: Optional[int] = None
        if mathlib_root is not None:
            self._mathlib_root = mathlib_root
        else:
            self._mathlib_root = os.path.join(self.project_root, "_target", "deps", "mathlib")
        self._mathlib_src_root = os.path.join(self._mathlib_root, "src")
        self._enable_search = enable_search
        if self._enable_search:
            pass
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def run_next(self) -> bool:
        pass

    def run_next_without_exec(self) -> bool:
        raise NotImplementedError

    def run_all_without_exec(self) -> bool:
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

    def run_lemma_without_executing(self) -> bool:
        raise NotImplementedError

    def run_till_next_lemma(self) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError

    def run_till_next_lemma_return_exec_stmt(self) -> Iterator[str]:
        raise NotImplementedError

    def run_to_finish_lemma_return_exec(self) -> Iterator[str]:
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
        raise NotImplementedError

    def get_current_lemma_name(self) -> Optional[str]:
        raise NotImplementedError
