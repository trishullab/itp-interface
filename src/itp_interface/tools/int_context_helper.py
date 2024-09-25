#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import logging
import typing
from itp_interface.tools.int_executor import INTExecutor, ProofContext
from itp_interface.tools.training_data_format import Goal, LemmaRefWithScore, LemmaReferences, TrainingDataFormat
from typing import List, Optional

class INTContextHelper(object):
    def __init__(self, search_executor: INTExecutor, depth: Optional[int] = None, logger: logging.Logger = None) -> None:
        assert search_executor is not None, "Search executor cannot be None"
        assert depth is None or depth >= 0, "Depth should be greater than 0"
        self.search_executor = search_executor
        self.depth = depth if depth is not None else -1
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def get_focussed_goals(self, int_executor: INTExecutor) -> List[Goal]:
        return [Goal(hypotheses=goal.hypotheses, goal=goal.goal) for goal in int_executor.proof_context.fg_goals]

    def get_focussed_goals_from_proof_context(self, proof_context: ProofContext) -> List[Goal]:
        return [Goal(hypotheses=goal.hypotheses, goal=goal.goal) for goal in proof_context.fg_goals]
    
    def get_unfocussed_goals(self, int_executor: INTExecutor) -> List[Goal]:
        raise NotImplementedError
    
    def get_local_lemmas(self, int_executor: INTExecutor, logger: logging.Logger = None) -> List[typing.Tuple[str, str]]:
        raise NotImplementedError
    
    def set_relevant_defns_in_training_data_point(self, training_data_point: TrainingDataFormat, int_executor: INTExecutor, logger: logging.Logger = None, depth: int = None):
        raise NotImplementedError
    
    def set_all_type_matched_query_result(self, training_data_point: TrainingDataFormat, int_executor: INTExecutor, logger: logging.Logger = None, depth: int = None):
        raise NotImplementedError
    
    def set_useful_defns_theorems_for_training_data_generation(self, current_stmt: str, training_data_point: TrainingDataFormat, lean_executor: INTExecutor, logger: logging.Logger = None, depth: int = None, max_search_res: typing.Optional[int] = None):
        raise NotImplementedError
    
    def set_local_thms_dns(self, training_data_point: TrainingDataFormat, int_executor: INTExecutor, logger: logging.Logger = None):
        raise NotImplementedError