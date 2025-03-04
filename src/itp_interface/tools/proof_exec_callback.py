#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import logging
import typing
from itp_interface.rl.proof_action import ProofAction
from itp_interface.tools.coq_context_helper import CoqContextHelper
from itp_interface.tools.lean_context_helper import Lean3ContextHelper
from itp_interface.tools.lean4_context_helper import Lean4ContextHelper
from itp_interface.tools.isabelle_context_helper import IsabelleContextHelper
from itp_interface.tools.coq_executor import CoqExecutor
from itp_interface.tools.lean_cmd_executor import Lean3Executor
from itp_interface.tools.lean4_sync_executor import Lean4SyncExecutor
from itp_interface.tools.isabelle_executor import IsabelleExecutor
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from itp_interface.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor
from itp_interface.tools.dynamic_lean4_proof_exec import DynamicProofExecutor as DynamicLean4ProofExecutor
from itp_interface.tools.dynamic_isabelle_proof_exec import DynamicProofExecutor as DynamicIsabelleProofExecutor
from itp_interface.tools.misc_defns import HammerMode

class ProofExecutorCallback(object):
    def __init__(self,
                project_folder: str,
                file_path: str,
                language: ProofAction.Language = ProofAction.Language.COQ,
                prefix: str = None,
                use_hammer: typing.Union[bool, HammerMode] = False,
                timeout_in_secs: int = 60,
                use_human_readable_proof_context: bool = True,
                suppress_error_log: bool = True,
                search_depth: int = 0,
                logger: logging.Logger = None,
                always_use_retrieval: bool = False,
                keep_local_context: bool = False,
                setup_cmds: typing.List[str] = [],
                port: typing.Optional[int] = None,
                enable_search: bool = True,
                enforce_qed: bool = False):
        self.project_folder = project_folder
        self.file_path = file_path
        self.language = language
        self.use_hammer = use_hammer
        self.timeout_in_secs = timeout_in_secs
        self.use_human_readable_proof_context = use_human_readable_proof_context
        self.suppress_error_log = suppress_error_log
        self.search_depth = search_depth
        self.logger = logger
        self.prefix = prefix
        self.always_use_retrieval = always_use_retrieval
        self.keep_local_context = keep_local_context
        self.setup_cmds = setup_cmds
        self.port = port
        self.enable_search = enable_search
        self.enforce_qed = enforce_qed
        pass

    def get_proof_executor(self) -> typing.Union[DynamicCoqProofExecutor, DynamicLeanProofExecutor, DynamicLean4ProofExecutor, DynamicIsabelleProofExecutor]:
        if self.language == ProofAction.Language.COQ:
            search_exec = CoqExecutor(self.project_folder, self.file_path, use_hammer=self.use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context, setup_cmds=self.setup_cmds)
            coq_context_helper = CoqContextHelper(search_exec, self.search_depth, logger=self.logger, enable_search=self.enable_search)
            return DynamicCoqProofExecutor(coq_context_helper, self.project_folder, self.file_path, context_type=DynamicCoqProofExecutor.ContextType.BestContext, use_hammer=self.use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context, setup_cmds=self.setup_cmds)
        elif self.language == ProofAction.Language.LEAN:
            search_exec = Lean3Executor(self.project_folder, self.prefix, self.file_path, use_hammer=self.use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context, enable_search=self.always_use_retrieval, keep_local_context=self.keep_local_context)
            lean_context_helper = Lean3ContextHelper(search_exec, self.search_depth, logger=self.logger)
            return DynamicLeanProofExecutor(lean_context_helper, self.project_folder, self.file_path, context_type=DynamicLeanProofExecutor.ContextType.NoContext, use_hammer=self.use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context, keep_local_context=self.keep_local_context)
        elif self.language == ProofAction.Language.LEAN4:
            search_exec = Lean4SyncExecutor(self.project_folder, self.prefix, self.file_path, use_hammer=self.use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context, enable_search=self.always_use_retrieval, keep_local_context=self.keep_local_context, enforce_qed=self.enforce_qed)
            lean4_context_helper = Lean4ContextHelper(search_exec, self.search_depth, logger=self.logger)
            return DynamicLean4ProofExecutor(lean4_context_helper, self.project_folder, self.file_path, context_type=DynamicLeanProofExecutor.ContextType.NoContext, use_hammer=self.use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context, keep_local_context=self.keep_local_context, enforce_qed=self.enforce_qed)
        elif self.language == ProofAction.Language.ISABELLE:
            search_exec = IsabelleExecutor(self.project_folder, self.file_path, use_hammer=self.use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context, port=self.port)
            isabelle_context_helper = IsabelleContextHelper(search_exec, self.search_depth, logger=self.logger)
            return DynamicIsabelleProofExecutor(isabelle_context_helper, self.project_folder, self.file_path, context_type=DynamicIsabelleProofExecutor.ContextType.BestContext, use_hammer=self.use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context, port=self.port)
        else:
            raise Exception(f"Unknown context type: {self.context_type}")