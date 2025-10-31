#!/usr/bin/env python3

import os
import random
import logging
import re
import time
import json
import typing
import bisect
import subprocess
from itp_interface.tools.tactic_parser import TacticParser, ErrorInfo, LeanLineInfo, RequestType
from itp_interface.lean_server.lean_context import ProofContext
from itp_interface.lean_server.lean4_utils import Lean4Utils
from itp_interface.tools.lean_parse_utils import LeanLineByLineReader
from itp_interface.tools.theorem_details import TheoremDetails
from itp_interface.tools.misc_defns import HammerMode
from itp_interface.tools.iter_helpers import ClonableIterator
from typing import List, Optional, Tuple, OrderedDict, Dict

class SimpleLean4SyncExecutor:
    theorem_regex = r"((((theorem|lemma)[\s]+([^\s:]*))|example)([\S|\s]*?)(:=|=>)[\s]*?)[\s]+"
    theorem_match = re.compile(theorem_regex, re.MULTILINE)
    unsolved_message = "unsolved goals"
    no_goals = "No goals to be solved"
    def __init__(self, 
        project_root: Optional[str] = None, 
        prefix: Optional[str] = None, 
        main_file: Optional[str] = None, 
        use_hammer: typing.Union[bool, HammerMode] = False, 
        timeout_in_sec: int = 60, 
        use_human_readable_proof_context: bool = True, 
        proof_step_iter: Optional[ClonableIterator] = None, 
        suppress_error_log: bool = False,  
        enable_search: bool = False,
        keep_local_context: bool = False,
        enforce_qed: bool = False,
        logger: Optional[logging.Logger] = None):
        assert proof_step_iter is None or isinstance(proof_step_iter, ClonableIterator), \
            "proof_step_iter must be an iterator"
        assert main_file is not None or proof_step_iter is not None, \
            "Either main_file or proof_step_iter must be provided"
        assert main_file is None or proof_step_iter is None, \
            "Only one of main_file or proof_step_iter must be provided"
        assert main_file is None or (os.path.exists(main_file) and main_file.endswith(".lean")), \
            f"main_file must be a valid path to a '.lean' file ({main_file})"
        assert project_root is None or (os.path.exists(project_root) and os.path.isdir(project_root)), \
            "project_root must be a valid path to a directory"
        assert not use_hammer, "Hammer is not supported for Lean4"
        self.use_human_readable_proof_context = use_human_readable_proof_context
        self.project_root = project_root if project_root is not None else "."
        self.main_file = main_file
        self.ticks = str(time.time()).replace(".", "") # This ensures that the temp file name is unique and doesn't clash with other temp files
        # This helps in running parallel instances of prover
        self.random_num = str(random.randint(0, 100000000))
        self.temp_filename_suffix = f"temptodel{self.ticks}{self.random_num}.lean"
        self.temp_file = os.path.join(prefix, self.temp_filename_suffix) if prefix is not None else self.temp_filename_suffix
        self.temp_file_full_path = os.path.join(self.project_root, self.temp_file)
        self.temp_file_full_path = os.path.abspath(self.temp_file_full_path)
        self.use_hammer = use_hammer
        self.timeout_in_sec = min(timeout_in_sec, 120) # Maximum 120s timeout
        self.current_stmt = None
        self.line_num = 0
        self.main_file_iter = proof_step_iter
        self.suppress_error_log = suppress_error_log
        self.tactic_parser: TacticParser | None = None
        self.execution_complete = False
        self._enforce_qed = enforce_qed
        self._ready_to_accept_proof = not self._enforce_qed
        self._max_memory_in_mib = 40000 # 40 GiB is needed for mathlib to work seemlessly
        self._lines_executed = []
        self.proof_context : ProofContext| None = None
        self.curr_lemma_name : Optional[str] = None
        self.curr_lemma : Optional[str] = None
        self.lean_error_messages : List[str] = []
        self._proof_running = False
        self._file_content = ""
        self.local_file_lemmas: OrderedDict[str, str] = OrderedDict()
        self.local_theorem_lemma_description: OrderedDict[str, str] = OrderedDict()
        self._proof_start_idx: Optional[int] = None
        self._import_end_idx: Optional[int] = None
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.use_file = False
        self._enable_search = enable_search
        self._theorem_started = False
        self._content_till_last_theorem_stmt = None
        self._last_theorem = None
        self._anon_theorem_count = 0
        self._debug_traces = []
        self.debug_enabled = False
        self._last_tactics : dict[int, str] = {}
        self.possible_proof_tactics = ""
        self._last_tactic_line_idx = None
        self._error_messages_so_far = set()
        self._error_messages_since_last_thm = {}
        self._run_exactly = False
        if self._enable_search:
            pass
        pass
    
    def set_run_exactly(self):
        self._run_exactly = True
    
    def unset_run_exactly(self):
        self._run_exactly = False
    
    def run_exactly(self):
        return self._run_exactly

    def reset(self,
        proof_step_iter: Optional[ClonableIterator] = None):
        # Note: We CANNOT reset the main_file_iter as it is a generator
        assert (proof_step_iter is not None and isinstance(proof_step_iter, ClonableIterator)) or self.main_file is not None, \
            "Either proof_step_iter must be provided or main_file must be set"
        self.current_stmt = None
        self.line_num = 0
        self.main_file_iter = proof_step_iter if proof_step_iter is not None else self.main_file_iter
        self.tactic_parser: TacticParser | None = None
        self.execution_complete = False
        self._lines_executed = []
        self.proof_context : ProofContext | None = None
        self.curr_lemma_name : Optional[str] = None
        self.curr_lemma : Optional[str] = None
        self.lean_error_messages : List[str] = []
        self._proof_running = False
        self._file_content = ""
        self.local_file_lemmas: OrderedDict[str, str] = OrderedDict()
        self.local_theorem_lemma_description: OrderedDict[str, str] = OrderedDict()
        self._proof_start_idx: Optional[int] = None
        self._import_end_idx: Optional[int] = None
        self._theorem_started = False
        self._content_till_last_theorem_stmt: str|None = None
        self._content_till_after_theorem_stmt: str|None = None
        self._last_theorem = None
        self._anon_theorem_count = 0
        self._last_tactics : dict[int, str] = {}
        self.possible_proof_tactics = ""
        self._last_tactic_line_idx = None
        self._debug_traces = []
        self.debug_enabled = False
        self._error_messages_so_far = set()
        self._error_messages_since_last_thm = {}
        if self._enable_search:
            pass
        pass

    def __enter__(self):
        self.tactic_parser = TacticParser(project_path=self.project_root, logger=self.logger)
        if self.main_file_iter is None:
            assert self.main_file is not None, "main_file must be set if main_file_iter is None"
            self.main_file_iter = LeanLineByLineReader(self.main_file, remove_comments=True, no_strip=True).instruction_step_generator()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tactic_parser is not None:
            self.tactic_parser.close()
        if os.path.exists(self.temp_file_full_path):
            os.remove(self.temp_file_full_path)

    def is_in_proof_mode(self):
        return True if self.proof_context else (len(self.lean_error_messages) > 0) # It is still in proof mode if we encountered a wrong proof

    def needs_qed(self):
        return self.proof_context is not None and len(self.proof_context.all_goals) == 0

    def needs_cut_close(self):
        return self.proof_context is not None and len(self.proof_context.fg_goals) == 0 and len(self.proof_context.all_goals) > 0

    def run_next(self) -> bool:
        try:
            assert self.main_file_iter is not None, "main_file_iter should not be None"
            next_line = next(self.main_file_iter)
        except StopIteration:
            self.execution_complete = True
            return False
        self.current_stmt = next_line
        self.line_num += 1
        try:
            idx = len(self._lines_executed)
            self._run_stmt_on_lean_server(idx, self.current_stmt)
        except:
            if not self.suppress_error_log:
                self.logger.error(f"Got an exception while running '{self.current_stmt}' on lean. File name: {self.main_file}")
                self.logger.exception(f"Exception Log")
            raise
        if self.run_exactly():
            self._lines_executed.append(self.current_stmt)
        else:
            self._lines_executed.append("") # Add an empty line to keep the line numbers in sync
        return True
            
    def get_lemma_name_if_running(self) -> Optional[str]:
        if not self.is_in_proof_mode():
            return None
        else:
            try:
                return self.curr_lemma_name
            except:
                return None
    
    def get_lemma_stmt_if_running(self) -> Optional[str]:
        try:
            assert self.curr_lemma_name is not None, "Current lemma name should not be None"
            return self.local_theorem_lemma_description[self.curr_lemma_name]
        except:
            return None
        
    def get_current_lemma_name(self) -> Optional[str]:
        if self.curr_lemma_name is None:
            return None
        else:
            return self.curr_lemma_name
    
    def _add_last_tactic(self, idx: int, stmt: str):
        if idx not in self._last_tactics:
            self._last_tactics[idx] = stmt
            self._last_tactic_line_idx = idx

    def _get_lean_code_with_tactics(self, idx: int, stmt: str):
        assert self._last_theorem is not None, "Last theorem should not be None"
        self._add_last_tactic(idx, stmt)
        temp_tactics_so_far = [(k, v) for k, v in self._last_tactics.items()]
        temp_tactics_so_far = sorted(temp_tactics_so_far, key=lambda x: x[0])
        tactics_so_far = [v for _, v in temp_tactics_so_far]
        assert len(tactics_so_far) > 0, "There should be at least one tactic so far"
        last_tactic = tactics_so_far[-1]
        # Caclulate the space padding for the last tactic
        # see all the space tokens
        space_tokens = []
        for c in last_tactic:
            if c.isspace():
                space_tokens.append(c)
            else:
                break
        done = f"{''.join(space_tokens)}done"
        _ , _, theorem_stmt = self._last_theorem
        return theorem_stmt + "\n".join(tactics_so_far) + "\n" + done

    def _backtrack_tactic_line(self, idx: int):
        # identify the keys to remove
        idx_to_remove = []
        backtracked = False
        for k in self._last_tactics.keys():
            if k >= idx:
                idx_to_remove.append(k)
        for k in idx_to_remove:
            backtracked = True
            del self._last_tactics[k]
        idx_to_remove = []
        for k in self._error_messages_since_last_thm.keys():
            if k >= idx:
                idx_to_remove.append(k)
        for k in idx_to_remove:
            backtracked = True
            msg = self._error_messages_since_last_thm[k]
            if msg in self._error_messages_so_far:
                self._error_messages_so_far.remove(msg)
            del self._error_messages_since_last_thm[k]
        self._last_tactic_line_idx = max(self._last_tactics.keys(), default=None) 
        return backtracked

    def _clear_tactics(self):
        tactics_so_far = [(k, v) for k, v in self._last_tactics.items()]
        tactics_so_far = sorted(tactics_so_far, key=lambda x: x[0])
        tactics_so_far = [v for _, v in tactics_so_far]
        self.possible_proof_tactics += "\n".join(tactics_so_far)
        self._last_tactics : dict[int, str] = {}
        self._last_tactic_line_idx = None
        self._error_messages_since_last_thm = {}
        pass

    def _theorem_started_init(self):
        if self._theorem_started:
            assert self._last_theorem is not None, "Last theorem should not be None"
            theorem_name, theorem_stmt, full_stmt = self._last_theorem
            self.curr_lemma_name = theorem_name
            self.curr_lemma = theorem_stmt
            if len(theorem_name) == 0:
                self._anon_theorem_count += 1
                theorem_name = f"anon_theorem____{self._anon_theorem_count}"
            self.local_file_lemmas[theorem_name] = theorem_stmt
            self.local_theorem_lemma_description[theorem_name] = full_stmt

    def _format_error_message(self, error_info: ErrorInfo) -> str:
        return f"L {error_info.position.line}, C {error_info.position.column}: {error_info.message}"
    
    def _update_proof_context(self, idx, tactics: List[LeanLineInfo], errors: List[ErrorInfo]):
        proof_goal_message: str|None = None
        last_error_message: str|None = None
        assert len(tactics) >= 0, "Tactics should not be None"
        last_tactic: LeanLineInfo = tactics[-1]
        if not tactics and not errors:
            raise ValueError(f"Response is None cannot update proof context for line number {idx}")
        for error in errors:
            if error.message.startswith(SimpleLean4SyncExecutor.unsolved_message):
                proof_goal_message = error.message # Always take the last unsolved goals message
            elif error.message.startswith(SimpleLean4SyncExecutor.no_goals):
                proof_goal_message = error.message
            else:
                # Make sure that it is after the last tactic
                if error.position.line >= last_tactic.line:
                    last_error_message = error.message # Always take the last error message
                    self._error_messages_since_last_thm[idx] = self._format_error_message(error)
                else:
                    last_error_message = None
                self._error_messages_so_far.add(self._format_error_message(error))
        goal_texts = []
        proof_is_running = False
        if proof_goal_message is not None:
            if not proof_goal_message.startswith(SimpleLean4SyncExecutor.no_goals):
                assert proof_goal_message.startswith(SimpleLean4SyncExecutor.unsolved_message), \
                    f"Unexpected proof goal message: {proof_goal_message}"
                goal_text = proof_goal_message[len(SimpleLean4SyncExecutor.unsolved_message):]
                goal_texts.append(goal_text)
            proof_is_running = True
        error_messages = []
        if last_error_message is not None:
            error_messages.append(last_error_message)
        if proof_goal_message is None and len(error_messages) == 0:
            assert len(goal_texts) == 0, "If there is no proof goal message, there should be no goal texts"
            proof_is_running = True # It is about to finish
        if len(error_messages) == 0:
            assert proof_is_running, f"Proof is not running but no error message is present, errors:\n{errors}, \nlemma: \n{self.curr_lemma_name}, \nlemma_stmt: \n{self.curr_lemma}, \nline_num: \n{self.line_num}"
            self._proof_running = proof_is_running
            if self._proof_running:
                proof_goals = []
                if len(goal_texts) == 0:
                    proof_goals = []
                else:
                    proof_goals = [g_text for g_text in goal_texts 
                    if g_text is not None and len(g_text) > 0]
                self.proof_context = self._parse_proof_context(proof_goals)
                if self.proof_context == ProofContext.empty():
                    self._proof_running = False
                    self.proof_context : ProofContext | None = None
                    self.curr_lemma = None
                    self.curr_lemma_name = None
                    self._clear_tactics()
            else:
                self.proof_context : ProofContext | None = None
        else:
            self.lean_error_messages = error_messages

    def _run_stmt_on_lean_server(self, idx : int, stmt: str, theorem_started: bool = False):
        assert self.tactic_parser is not None, "Tactic parser is not initialized"
        assert self._content_till_last_theorem_stmt is not None, "Content till last theorem statement should not be None"
        if "sorry" in stmt and self._proof_running:
            # We don't need to run the sorry statements. This should be treated as a failed proof step
            self.lean_error_messages = ["The tactic 'sorry' was found in the statement, this is not allowed"]
            return
        elif len(stmt.strip()) == 0 and self._proof_running:
            # We don't need to run the empty statements. This should be treated as a failed proof step
            self.lean_error_messages = ["There is no tactic in the statement, it is just empty line or whitespace"]
            return
        elif self.proof_context == ProofContext.empty() and \
            self._proof_running and \
            stmt != "done":
            self.lean_error_messages = [
            "The proof is about to finish, please use 'done' to finish the proof."]
            return
        elif stmt == "done" and \
            self._proof_running and \
            self.proof_context != ProofContext.empty():
            self.lean_error_messages = [
                "The proof is not finished, please complete the proof before using 'done'."]
            return

        proof_should_run = False
        if theorem_started:
            # Load the theorem context at once
            self.tactic_parser.parse(
                self._content_till_last_theorem_stmt,
                fail_on_error=True,
                parse_type=RequestType.CHKPT_TACTICS
            )
        if theorem_started or not self._proof_running:
            proof_should_run = self._theorem_started
            self._theorem_started_init()
        if not self._proof_running and not proof_should_run:
            return
        code_was_executed = False
        while not code_was_executed:
            # Run the statement in tactic mode
            code = self._get_lean_code_with_tactics(idx, stmt)
            tactics, error_info = self.tactic_parser.parse(
                code,
                fail_on_error=False,
                parse_type=RequestType.PARSE_TACTICS)
            code_was_executed = True
            self._update_proof_context(idx, tactics, error_info)
        pass

    def _skip_to_theorem(self, theorem: str):
        # Skip to the given theorem
        found_theorem = False
        thm_namespace, given_theorem_name = parse_thm_name(theorem)
        # Scan the whole file first
        lines : list[str] = []
        assert self.main_file_iter is not None, "main_file_iter should not be None"
        assert self.tactic_parser is not None, "Tactic parser is not initialized"
        while True:
            try:
                stmt = next(self.main_file_iter)
            except StopIteration:
                break
            lines.append(stmt)
        full_file = "\n".join(lines) + "\n"
        # run the tactic parser in theorem parsing mode
        lean_line_infos, _ = self.tactic_parser.parse(
            full_file, 
            fail_on_error=False, 
            parse_type=RequestType.PARSE_THEOREM)
        # Filter out theorems and lemmas
        theorems = [info for info in lean_line_infos if info.decl_type == "theorem" or info.decl_type == "lemma"]
        found_theorem = False
        for thm in theorems:
            name = thm.name
            assert name is not None, "Theorem name should not be None"
            if given_theorem_name == name:
                actual_namespace = thm.namespace if thm.namespace is not None else ""
                if actual_namespace == thm_namespace:
                    # Found the theorem
                    found_theorem = True
                    line_num = thm.line
                    break
        if not found_theorem:
            raise ValueError(f"The theorem '{theorem}' was not found in the file '{self.main_file}'")
        assert line_num > 0, "Theorem line number should be greater than 0"
        self._lines_executed = lines[:line_num - 1]
        theorem_text = thm.text

        content_until_after_theorem = "\n".join(self._lines_executed) + "\n" + theorem_text

        # Parse out tactics now
        all_tactics_till_now, _ = self.tactic_parser.parse(content_until_after_theorem, fail_on_error=True, parse_type=RequestType.PARSE_TACTICS)
        # Find the first index which is after the theorem line
        first_idx_after_theorem = None
        for idx, tactic_info in enumerate(all_tactics_till_now):
            if tactic_info.line >= line_num:
                first_idx_after_theorem = idx
                break
        assert first_idx_after_theorem is not None, "Could not find the first tactic after the theorem"
        start_tactic = all_tactics_till_now[first_idx_after_theorem]
        tactic_start_line = start_tactic.line
        tactic_start_col = start_tactic.column
        assert tactic_start_line > 0, "Tactic start line should be greater than 0"
        content_until_after_theorem = "\n".join(lines[:tactic_start_line - 1] + [lines[tactic_start_line - 1][:tactic_start_col]])
        self._content_till_after_theorem_stmt = content_until_after_theorem
        self._content_till_after_theorem_stmt = self._content_till_after_theorem_stmt.strip()
        assert self._content_till_after_theorem_stmt.endswith(':='), "Content till last theorem statement should end with ':='"
        content_until_before_theorem = "\n".join(lines[:line_num - 1])
        self._content_till_last_theorem_stmt = content_until_before_theorem
        theorem_stmt = "\n".join(lines[line_num - 1:tactic_start_line - 1] + [lines[tactic_start_line - 1][:tactic_start_col]])
        theorem_stmt = theorem_stmt.strip()
        self._last_theorem = (given_theorem_name, theorem_stmt, theorem_stmt)
        self._theorem_started = True
        self._lines_executed.extend(lines[line_num - 1:tactic_start_line - 1] + [lines[tactic_start_line - 1][:tactic_start_col]])
        self._run_stmt_on_lean_server(tactic_start_line, "by", theorem_started=True)
        self._lines_executed.append('by')
        # Reset the iterator to the line of the theorem
        if lines[tactic_start_line - 1].strip().endswith("by"):
            self.main_file_iter.set_to_index(tactic_start_line)
        else:
            self.main_file_iter.set_to_index(tactic_start_line + 1)
        self.line_num = len(self._lines_executed)

    def _parse_proof_context(self, proof_goals: list) -> ProofContext:
        goals = []
        for proof_goal in proof_goals:
            if self.use_human_readable_proof_context:
                goals.extend(Lean4Utils.parse_proof_context_human_readable_as_goals(proof_goal))
            else:
                raise NotImplementedError("Parsing of non-human readable proof context is not implemented")
        if len(goals) == 0:
            return ProofContext.empty()
        else:
            return ProofContext(goals, [], [], [])

    def validate_proof_with_lake(self, theorem_name: Optional[str] = None, timeout_sec: int = 30, keep_temp_file: bool = True) -> typing.Dict[str, typing.Any]:
        """
        Validate the current proof state by running 'lake lean' on a temporary file.
        This provides an independent verification without relying on the REPL.

        Args:
            theorem_name: Name of the theorem to validate (optional, for logging only)
            timeout_sec: Timeout in seconds for the lake lean process
            keep_temp_file: If True, keeps the temporary file after validation (default: True)

        Returns:
            Dictionary with validation results:
            {
                'success': bool,  # True if proof is complete with no errors
                'compilation_ok': bool,  # True if file compiles
                'has_sorries': bool,  # True if there are unsolved goals (sorries)
                'error_message': str,  # Error message if any
                'errors': list,  # List of error details
                'lean_code': str,  # The code that was validated
                'theorem_name': str  # Name of theorem being validated
            }
        """

        # Get theorem name for logging/reporting, but don't require it
        if theorem_name is None:
            theorem_name = self.curr_lemma_name or "unknown"

        # Create the Lean code with all executed lines up to current point
        lines_executed_str = '\n'.join(self._lines_executed)

        if not lines_executed_str or not lines_executed_str.strip():
            return {
                'success': False,
                'compilation_ok': False,
                'has_sorries': False,
                'error_message': 'No code to validate',
                'errors': [],
                'lean_code': '',
                'theorem_name': theorem_name,
                'full_output': 'No code available to validate',
                'stdout': '',
                'stderr': '',
                'return_code': -1,
                'temp_filename': 'N/A',
                'temp_file_path': 'N/A',
                'temp_file_kept': False,
                'debug_traces': list(self._debug_traces),
                'possible_proof_tactics': self.possible_proof_tactics
            }

        # Build the complete Lean code with actual proof tactics
        # The proof tactics are accumulated in self.possible_proof_tactics
        actual_proof = ""  # Track the actual proof for sorry checking
        proof_tactics_source = self.possible_proof_tactics

        # If possible_proof_tactics is empty, try to use _last_tactics as fallback
        if not proof_tactics_source or not proof_tactics_source.strip():
            if self._last_tactics:
                # Extract tactics from _last_tactics (same logic as _clear_tacitcs)
                tactics_so_far = [(k, v) for k, v in self._last_tactics.items()]
                tactics_so_far = sorted(tactics_so_far, key=lambda x: x[0])
                tactics_so_far = [v for _, v in tactics_so_far]
                proof_tactics_source = "\n".join(tactics_so_far)

            # If both are empty, raise an error
            if not proof_tactics_source or not proof_tactics_source.strip():
                raise ValueError("No proof tactics available. Neither 'possible_proof_tactics' nor '_last_tactics' contain any proof steps.")

        # Now build the Lean code with the proof tactics
        if proof_tactics_source and proof_tactics_source.strip():
            # Find the last ':=' in lines_executed
            last_assign_idx = lines_executed_str.rfind(':=')
            if last_assign_idx != -1:
                # Take everything up to and including the last ':=' from lines_executed
                code_prefix = lines_executed_str[:last_assign_idx + 2]

                # In proof_tactics_source, find the first ':= by' (with flexible whitespace)
                # Use regex to find ':=' followed by optional whitespace and 'by'
                assign_by_match = re.search(r':=\s+by', proof_tactics_source)

                if assign_by_match:
                    # Extract everything after ':= <whitespace> by' (excluding the match itself)
                    match_end_idx = assign_by_match.end()
                    actual_proof = proof_tactics_source[match_end_idx:].strip()
                    lean_code = code_prefix + ' by\n' + actual_proof
                else:
                    # No ':= by' found, use proof_tactics_source as-is
                    actual_proof = proof_tactics_source.strip()
                    lean_code = code_prefix + ' by\n' + actual_proof
            else:
                # No ':=' found, just use lines_executed as-is
                lean_code = lines_executed_str
        else:
            # No proof tactics available, use lines_executed as-is
            lean_code = lines_executed_str

        # Create a unique temporary file
        temp_filename = f"validation_{self.ticks}_{self.random_num}.lean"
        temp_file_path = os.path.join(self.project_root, temp_filename)

        try:
            # Write the Lean code to the temporary file
            with open(temp_file_path, 'w') as f:
                f.write(lean_code)

            # Run lake lean on the file
            try:
                result = subprocess.run(
                    ['lake', 'lean', temp_filename],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec
                )

                stdout = result.stdout
                stderr = result.stderr
                output = stdout + '\n' + stderr

            except subprocess.TimeoutExpired:
                # Don't delete temp file on timeout so it can be inspected
                return {
                    'success': False,
                    'compilation_ok': False,
                    'has_sorries': False,
                    'error_message': f'Timeout after {timeout_sec} seconds',
                    'errors': [],
                    'lean_code': lean_code,
                    'theorem_name': theorem_name,
                    'full_output': f'Process timed out after {timeout_sec} seconds',
                    'stdout': '',
                    'stderr': '',
                    'return_code': -1,
                    'temp_filename': temp_filename,
                    'temp_file_path': temp_file_path,
                    'temp_file_kept': True,  # Keep file on timeout for debugging
                    'debug_traces': list(self._debug_traces),
                    'possible_proof_tactics': self.possible_proof_tactics
                }

            # Parse the output for errors and warnings
            errors = []
            error_pattern = re.compile(r'(\S+):(\d+):(\d+):\s*(warning|error):\s*(.+)')

            for line in output.split('\n'):
                match = error_pattern.match(line)
                if match:
                    filename, line_num, col_num, severity, message = match.groups()
                    errors.append({
                        'file': filename,
                        'line': int(line_num),
                        'column': int(col_num),
                        'severity': severity,
                        'message': message
                    })

            # Check for 'sorry' only in the actual proof we generated
            has_sorries = 'sorry' in actual_proof.lower()

            # Only fail on actual errors (not warnings)
            # Also check for "unsolved goals" in error messages
            theorem_has_error = False
            for error in errors:
                if error['severity'] == 'error':
                    theorem_has_error = True
                    # Also check if the error mentions unsolved goals
                    if 'unsolved goals' in error['message'].lower():
                        has_sorries = True

            # Determine success: compilation ok, no sorries in actual proof, no errors (ignore warnings)
            compilation_ok = result.returncode == 0
            success = compilation_ok and not has_sorries and not theorem_has_error

            error_message = ''
            if not compilation_ok:
                error_message = 'Compilation failed'
            elif has_sorries:
                error_message = 'Proof has unsolved goals (sorries)'
            elif theorem_has_error:
                error_message = 'Theorem has errors'
            else:
                error_message = 'Proof is complete'

            # Combine full raw output for debugging
            full_output = f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}"

            return {
                'success': success,
                'compilation_ok': compilation_ok,
                'has_sorries': has_sorries,
                'error_message': error_message,
                'errors': errors,
                'lean_code': lean_code,
                'return_code': result.returncode,
                'stdout': stdout,
                'stderr': stderr,
                'full_output': full_output,
                'theorem_name': theorem_name,
                'temp_filename': temp_filename,
                'temp_file_path': temp_file_path,
                'temp_file_kept': keep_temp_file,
                'debug_traces': list(self._debug_traces),
                'possible_proof_tactics': self.possible_proof_tactics
            }

        finally:
            # Clean up the temporary file only if requested
            if not keep_temp_file and os.path.exists(temp_file_path):
                os.remove(temp_file_path)


theorem_names_in_file_cache: Dict[str, List[TheoremDetails]] = {}
namespace_regex = r"^namespace[ ]+([\S]+)"
namespace_match = re.compile(namespace_regex, re.MULTILINE)
namespace_end_regex = r"^end[ ]+([\S]+)*"
namespace_end_match = re.compile(namespace_end_regex, re.MULTILINE)

def parse_thm_name(theorem_name: str) -> Tuple[str, str]:
    if theorem_name.startswith("{") and theorem_name.endswith("}"):
        thm_dict = json.loads(theorem_name)
        return thm_dict["namespace"], thm_dict["name"]
    else:
        return "", theorem_name

def process_namespaces(file_cotent: str, open_namespaces: List[str], is_full_content: bool=False):
    # Match the namespace regex
    # Break the content line by line and match the namespace and end namespace
    file_lines = file_cotent.split('\n')
    for line in file_lines:
        namespace_matches = namespace_match.findall(line)
        namespace_end_matches = namespace_end_match.findall(line)
        for ns in namespace_matches:
            if not is_full_content or ns not in open_namespaces:
                open_namespaces.append(ns)
        for ns in namespace_end_matches:
            try:
                open_namespaces.remove(ns)
            except ValueError:
                pass

def get_all_theorems_in_file(file_path: str, use_cache: bool=False) -> List[TheoremDetails]:
    if use_cache and file_path in theorem_names_in_file_cache:
        return theorem_names_in_file_cache[file_path]
    file_content = ""
    open_namespaces = []
    with open(file_path, "r") as f:
        file_content = f.read()
    line_by_line_reader = LeanLineByLineReader(file_content=file_content, remove_comments=True, no_strip=True)
    all_stmts = list(line_by_line_reader.instruction_step_generator())
    line_positions = [0] + [len(stmt) + 1 for stmt in all_stmts]
    # Cumulative sum of the line positions
    for i in range(1, len(line_positions)):
        line_positions[i] += line_positions[i - 1]
    full_content = '\n'.join(all_stmts)
    # all_matches = Lean4SyncExecutor.theorem_match.findall(full_content)
    all_matches = list(SimpleLean4SyncExecutor.theorem_match.finditer(full_content))
    all_theorems = []
    last_namespace_processed_idx = 0
    for match in all_matches:
        span_start, span_end = match.span()
        process_namespaces(full_content[last_namespace_processed_idx:span_start], open_namespaces)
        theorem_name = match.group(5)
        theorem_name = theorem_name if theorem_name is not None else f"\"{match.group(6).strip(': ')}\""
        theorem_namespace = '.'.join(open_namespaces) if len(open_namespaces) > 0 else ''
        line_number_start = bisect.bisect_left(line_positions, span_start)
        line_number_end = bisect.bisect_left(line_positions, span_end)
        theorem_pos = {
            'line_start': line_number_start + 1,
            'line_end': line_number_end + 1,
            'global_pos_start': span_start,
            'global_pos_end': span_end,
            'line_pos_start': span_start - line_positions[line_number_start] if line_number_start < len(line_positions) else 0,
            'line_pos_end': span_end - line_positions[line_number_end] if line_number_end < len(line_positions) else 0
        }
        theorem_details = TheoremDetails(
            theorem_name=theorem_name, 
            theorem_namespace=theorem_namespace, 
            theorem_file_path=file_path, 
            theorem_pos=theorem_pos)
        all_theorems.append(theorem_details)
        last_namespace_processed_idx = span_end
    if use_cache:
        theorem_names_in_file_cache[file_path] = all_theorems
    return all_theorems

def get_fully_qualified_theorem_name(theorem_details: TheoremDetails) -> str:
    if len(theorem_details.theorem_namespace) == 0:
        return theorem_details.theorem_name
    else:
        dict_thm = {"namespace": theorem_details.theorem_namespace, "name": theorem_details.theorem_name}
        return json.dumps(dict_thm)

def get_theorem_name_resembling(file_path: str, theorem_name: str, use_cache: bool=False) -> Optional[str]:
    all_theorems = get_all_theorems_in_file(file_path, use_cache=use_cache)
    all_theorems_name_unique_map : Dict[str, List[TheoremDetails]] = {}
    for thm in all_theorems:
        if thm.theorem_name in all_theorems_name_unique_map:
            all_theorems_name_unique_map[thm.theorem_name].append(thm)
        else:
            all_theorems_name_unique_map[thm.theorem_name] = [thm]
    all_parts = theorem_name.split('.')
    thm_start_idx = len(all_parts) - 1
    thm_found = False
    while not thm_found and thm_start_idx >= 0:
        full_name = '.'.join(all_parts[thm_start_idx:])
        # look for any theorems matching with full_name
        thm_found = full_name in all_theorems_name_unique_map
        thm_start_idx -= 1
    if not thm_found:
        full_name = '_root_.' + full_name
        # look for any theorems matching with the full_name
        thm_found = full_name in all_theorems_name_unique_map
        if not thm_found:
            raise ValueError(f"The theorem '{theorem_name}' was not found in the file '{file_path}'")
    assert thm_found, "The theorem was not found some code bug in finding the theorem names"
    theorem_name_matches = all_theorems_name_unique_map[full_name]
    if len(theorem_name_matches) == 1:
        if len(theorem_name_matches[0].theorem_namespace) == 0:
            return theorem_name_matches[0].theorem_name
        else:
            dict_thm = {"namespace": theorem_name_matches[0].theorem_namespace, "name": theorem_name_matches[0].theorem_name}
            return json.dumps(dict_thm)
    else:
        # We need to find the namespace which matches with the theorem_name
        for thm in theorem_name_matches:
            if theorem_name.endswith(thm.theorem_namespace + '.' + thm.theorem_name) or\
            (theorem_name.strip() == thm.theorem_name and len(thm.theorem_namespace) == 0):
                dict_thm = {"namespace": thm.theorem_namespace, "name": thm.theorem_name}
                return json.dumps(dict_thm)
        raise ValueError(f"The theorem '{theorem_name}' was not found in the file '{file_path}'")

def execute_lean_repl_on(file_path: str, project_root: str, theorem_name: str, logger: logging.Logger, with_print: bool=False):
    pprint = lambda msg: print(msg) if with_print else None
    with SimpleLean4SyncExecutor(main_file=file_path, project_root=project_root, logger=logger) as executor:
        executor.set_run_exactly()
        executor._skip_to_theorem(theorem_name)
        assert executor.proof_context is not None, "Proof context should be present"
        proof_exec = False
        while not executor.execution_complete:
            if executor.proof_context is not None:
                proof_exec = True
                for goal in executor.proof_context.all_goals:
                    for hyp in goal.hypotheses:
                        pprint(hyp)
                    pprint('-'*10)
                    pprint(goal.goal)
                pprint('-'*20)
            executor.run_next()
            pprint(f"Current statement: {executor.current_stmt}")
            if executor.proof_context is None and proof_exec:
                proof_exec = False
                pprint("Proof finished")
                break
            if executor.lean_error_messages:
                pprint(f"Error messages:\n{executor.lean_error_messages}")

if __name__ == "__main__":
    from itp_interface.tools.log_utils import setup_logger
    import datetime
    parent = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(parent))
    os.chdir(root)
    project_root = 'data/test/Mathlib/'
    file_path = 'data/test/Mathlib/.lake/packages/mathlib/Mathlib/Computability/TuringMachine.lean'
    assert os.path.exists(project_root), "Project root does not exist"
    assert os.path.exists(file_path), "File path does not exist"
    print("Finding all theorems in the file")
    all_theorems = get_all_theorems_in_file(file_path, use_cache=True)
    print(all_theorems)
    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    lean_exec_log_folder = f'.log/lean4_sync_executor/{date_time}'
    os.makedirs(lean_exec_log_folder, exist_ok=True)
    lean_exec_log_file = os.path.join(lean_exec_log_folder, "lean4_sync_executor.log")
    logger = setup_logger("Lean4SyncExecutor", lean_exec_log_file, level=logging.DEBUG, format='')
    theorems_similar_to_test = get_theorem_name_resembling(file_path, "Turing.TM1to1.tr_supports", use_cache=True)
    assert theorems_similar_to_test is not None, "Theorem similar to test should not be None"
    print("Theorem similar to ", "Turing.TM1to1.tr_supports", " is ", theorems_similar_to_test)
    project_root = 'data/test/lean4_proj/'
    file_path = 'data/test/lean4_proj/Lean4Proj/Basic.lean'
    theorem_name = "Lean4Proj2.test3"
    theorems_similar_to_test = get_theorem_name_resembling(file_path, theorem_name, use_cache=True)
    assert theorems_similar_to_test is not None, "Theorem similar to test should not be None"
    print("Theorem similar to ", "Lean4Proj2.test", " is ", theorems_similar_to_test)
    execute_lean_repl_on(file_path, project_root, theorems_similar_to_test, logger, with_print=True)
    mathlib_test_file = 'data/test/Mathlib/.lake/packages/mathlib/Mathlib/Data/Nat/Bits.lean'
    project_root = 'data/test/Mathlib'
    assert os.path.exists(mathlib_test_file), "Mathlib test file does not exist"
    assert os.path.exists(project_root), "Project root does not exist"
    theorems_similar_to_test = get_theorem_name_resembling(mathlib_test_file, "one_bits", use_cache=True)
    assert theorems_similar_to_test is not None, "Theorem similar to test should not be None"
    print("Theorem similar to ", "one_bits", " is ", theorems_similar_to_test)
    execute_lean_repl_on(mathlib_test_file, project_root, theorems_similar_to_test, logger, with_print=True)