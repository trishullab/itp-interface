#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import uuid
from itp_interface.tools.lean_cmd_executor import Lean3Executor
from itp_interface.tools.lean_context_helper import Lean3ContextHelper
from itp_interface.tools.coq_training_data_generator import GenericTrainingDataGenerationTransform, TrainingDataGenerationType
from itp_interface.tools.training_data_format import Goal, MergableCollection, TrainingDataMetadataFormat, TrainingDataCollection, TrainingDataFormat
from itp_interface.tools.training_data import TrainingData

class LocalDataGenerationTransform(GenericTrainingDataGenerationTransform):
    def __init__(self,
                depth = None,
                max_search_results = None,
                buffer_size : int = 10000,
                logger = None,
                max_parallelism : int = 4):
        super().__init__(TrainingDataGenerationType.LOCAL, buffer_size, logger)
        self.depth = depth
        self.max_search_results = max_search_results
        self.max_parallelism = max_parallelism

    def get_meta_object(self) -> MergableCollection:
        return TrainingDataMetadataFormat(training_data_buffer_size=self.buffer_size)

    def get_data_collection_object(self) -> MergableCollection:
        return TrainingDataCollection()
    
    def load_meta_from_file(self, file_path) -> MergableCollection:
        return TrainingDataMetadataFormat.load_from_file(file_path)
    
    def load_data_from_file(self, file_path) -> MergableCollection:
        return TrainingDataCollection.load_from_file(file_path, self.logger)

    def __call__(self, training_data: TrainingData, project_id : str, lean_executor: Lean3Executor, print_coq_executor_callback: typing.Callable[[], Lean3Executor], theorems: typing.List[str] = None, other_args: dict = {}) -> TrainingData:
        print_lean_executor = print_coq_executor_callback()
        lean_context_helper = Lean3ContextHelper(print_lean_executor, self.depth, self.logger)
        lean_context_helper.__enter__()
        file_namespace = lean_executor.main_file.replace('/', '.')
        self.logger.info(f"=========================Processing {file_namespace}=========================")
        proof_running = False
        cmd_ran = lean_executor.run_next()
        cmd_exec = lean_executor.current_stmt
        prev_goal : typing.List[Goal] = lean_context_helper.get_focussed_goals(lean_executor) if lean_executor.is_in_proof_mode() else []
        line_number = lean_executor.line_num
        lemma_name = lean_executor.get_lemma_name_if_running()
        if lemma_name is None:
            lemma_name = "__NONE__"
        theorem_id = str(uuid.uuid4())
        proof_id = theorem_id # self.get_proof_id(theorem_id, file_namespace, line_number, lemma_name)
        local_lemma_refs_cnt = 0
        external_lemma_refs_cnt = 0
        theorems = set(theorems) if theorems is not None else None
        while cmd_ran:
            if lean_executor.is_in_proof_mode() and lemma_name != "__NONE__" and (theorems is None or lemma_name in theorems):
                proof_running = True
                prev_goal : typing.List[Goal] = [Goal(goal.hypotheses, goal.goal) for goal in prev_goal]
                next_goal : typing.List[Goal] = lean_context_helper.get_focussed_goals(lean_executor)
                if len(prev_goal) > 0:
                    training_data_format = TrainingDataFormat(
                        proof_id=proof_id,
                        all_useful_defns_theorems=[],
                        start_goals=prev_goal,
                        end_goals=next_goal,
                        proof_steps=[cmd_exec],
                        simplified_goals=[], 
                        addition_state_info={},
                        file_path=lean_executor.main_file,
                        theorem_name=lemma_name,
                        project_id=project_id)
                    assert len(training_data_format.proof_steps) > 0, f"Proof steps cannot be empty for {proof_id}"
                    for goal in training_data_format.start_goals:
                        lemma_cnt = len(training_data_format.all_useful_defns_theorems)
                        assert all([0 <= lemma_ref.lemma_idx < lemma_cnt for lemma_ref in goal.used_theorems_local]), f"Invalid lemma idx in {proof_id}"
                        assert all([0 <= lemma_ref.lemma_idx < lemma_cnt for lemma_ref in goal.used_theorems_external]), f"Invalid lemma idx in {proof_id}"
                        assert all([0 <= lemma_ref.lemma_idx < lemma_cnt for lemma_ref in goal.possible_useful_theorems_external]), f"Invalid lemma idx in {proof_id}"
                        assert all([0 <= lemma_ref.lemma_idx < lemma_cnt for lemma_ref in goal.possible_useful_theorems_local]), f"Invalid lemma idx in {proof_id}"
                    training_data.merge(training_data_format)
                    local_lemma_refs_cnt += sum([len(goal.used_theorems_local) for goal in training_data_format.start_goals])
                    external_lemma_refs_cnt += sum([len(goal.used_theorems_external) for goal in training_data_format.start_goals])
                prev_goal = next_goal
            else:
                prev_goal = []
            cmd_ran = lean_executor.run_next()
            cmd_exec = lean_executor.current_stmt
            line_number = lean_executor.line_num
            if proof_running and not lean_executor.is_in_proof_mode():
                proof_running = False
                self.logger.info(f"Finished processing lemma [{theorem_id}] {lemma_name}")
                theorem_id = str(uuid.uuid4())
            lemma_name = lean_executor.get_lemma_name_if_running()
            if lemma_name is None:
                lemma_name = "__NONE__"
            proof_id = theorem_id # self.get_proof_id(theorem_id, file_namespace, line_number, lemma_name)
            
        self.logger.info(f"===============Finished processing {file_namespace}=====================")
        try:
            lean_context_helper.__exit__(None, None, None)
        except:
            pass


if __name__ == "__main__":
    import os
    import logging
    import time
    os.chdir(root_dir)
    project_dir = "data/test/lean_proj"
    file_name = "data/test/lean_proj/src/simple_solved.lean"
    project_id = project_dir.replace('/', '.')
    time_str = time.strftime("%Y%m%d-%H%M%S")
    output_path = f".log/local_data_generation_transform/data/{time_str}"
    log_path = f".log/local_data_generation_transform/log/{time_str}"
    log_file = f"{log_path}/local_data_generation_transform-{time_str}.log"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    def _print_lean_executor_callback():
        search_lean_exec = Lean3Executor(project_root=project_dir, main_file=file_name, use_human_readable_proof_context=True, suppress_error_log=True)
        search_lean_exec.__enter__()
        return search_lean_exec
    transform = LocalDataGenerationTransform(0, buffer_size=1000)
    training_data = TrainingData(
        output_path, 
        "training_metadata.json",
        training_meta=transform.get_meta_object(), 
        logger=logger)
    with Lean3Executor(project_root=project_dir, main_file=file_name, use_human_readable_proof_context=True, suppress_error_log=True) as coq_exec:
        transform(training_data, project_id, coq_exec, _print_lean_executor_callback)
    save_info = training_data.save()
    logger.info(f"Saved training data to {save_info}")