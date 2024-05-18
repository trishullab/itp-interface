#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import uuid
import os
from itp_interface.tools.isabelle_executor import IsabelleExecutor
from itp_interface.tools.isabelle_context_helper import IsabelleContextHelper
from itp_interface.tools.coq_training_data_generator import GenericTrainingDataGenerationTransform, TrainingDataGenerationType
from itp_interface.tools.training_data_format import Goal, MergableCollection, TrainingDataMetadataFormat, TrainingDataCollection, TrainingDataFormat
from itp_interface.tools.training_data import TrainingData

# See this for running transformation on AFP
# https://github.com/albertqjiang/Portal-to-ISAbelle/blob/56def2c39f85d211e1f40cc5765581a567879106/src/main/python/legacy/demo.py#L2

class LocalDataGenerationTransform(GenericTrainingDataGenerationTransform):
    def __init__(self,
                depth = None,
                max_search_results = None,
                buffer_size : int = 10000,
                logger = None,
                max_parallelism : int = 4,
                **kwargs):
        super().__init__(TrainingDataGenerationType.LOCAL, buffer_size, logger)
        self.depth = depth
        self.max_search_results = max_search_results
        self.max_parallelism = max_parallelism
        self.ray_resource_pool = kwargs.get('resource_pool', None)
        self.pisa_severs = []

    def get_meta_object(self) -> MergableCollection:
        return TrainingDataMetadataFormat(training_data_buffer_size=self.buffer_size)

    def get_data_collection_object(self) -> MergableCollection:
        return TrainingDataCollection()
    
    def load_meta_from_file(self, file_path) -> MergableCollection:
        return TrainingDataMetadataFormat.load_from_file(file_path)
    
    def load_data_from_file(self, file_path) -> MergableCollection:
        return TrainingDataCollection.load_from_file(file_path, self.logger)

    def __call__(self, training_data: TrainingData, project_id : str, isabelle_executor: IsabelleExecutor, print_coq_executor_callback: typing.Callable[[], IsabelleExecutor], theorems: typing.List[str] = None, other_args: dict = {}) -> TrainingData:
        print_isabelle_executor = print_coq_executor_callback()
        isabelle_context_helper = IsabelleContextHelper(print_isabelle_executor, self.depth, self.logger)
        isabelle_context_helper.__enter__()
        file_namespace = isabelle_executor.main_file.replace('/', '.')
        self.logger.info(f"=========================Processing {file_namespace}=========================")
        proof_running = False
        cmd_ran = isabelle_executor.run_next(proof_search_mode=False)
        cmd_exec = isabelle_executor.current_stmt
        prev_goal : typing.List[Goal] = isabelle_context_helper.get_focussed_goals(isabelle_executor) if isabelle_executor.is_in_proof_mode() else []
        line_number = isabelle_executor.line_num
        lemma_name = isabelle_executor.get_lemma_name_if_running()
        if lemma_name is None:
            lemma_name = "__NONE__"
        theorem_id = str(uuid.uuid4())
        proof_id = theorem_id # self.get_proof_id(theorem_id, file_namespace, line_number, lemma_name)
        local_lemma_refs_cnt = 0
        external_lemma_refs_cnt = 0
        theorems = set(theorems) if theorems is not None else None
        while cmd_ran:
            if isabelle_executor.is_in_proof_mode() and lemma_name != "__NONE__" and (theorems is None or lemma_name in theorems):
                proof_running = True
                prev_goal : typing.List[Goal] = [Goal(goal.hypotheses, goal.goal) for goal in prev_goal]
                next_goal : typing.List[Goal] = isabelle_context_helper.get_focussed_goals(isabelle_executor)
                if len(prev_goal) > 0:
                    training_data_format = TrainingDataFormat(
                        proof_id=proof_id,
                        all_useful_defns_theorems=[],
                        start_goals=prev_goal,
                        end_goals=next_goal,
                        proof_steps=[cmd_exec],
                        simplified_goals=[], 
                        addition_state_info={},
                        file_path=isabelle_executor.main_file,
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
            cmd_ran = isabelle_executor.run_next(proof_search_mode=False)
            cmd_exec = isabelle_executor.current_stmt
            line_number = isabelle_executor.line_num
            if proof_running and not isabelle_executor.is_in_proof_mode():
                proof_running = False
                self.logger.info(f"Finished processing lemma [{theorem_id}] {lemma_name}")
                theorem_id = str(uuid.uuid4())
            lemma_name = isabelle_executor.get_lemma_name_if_running()
            if lemma_name is None:
                lemma_name = "__NONE__"
            proof_id = theorem_id # self.get_proof_id(theorem_id, file_namespace, line_number, lemma_name)
            
        self.logger.info(f"===============Finished processing {file_namespace}=====================")
        try:
            isabelle_context_helper.__exit__(None, None, None)
        except:
            pass


if __name__ == "__main__":
    import logging
    import time
    os.chdir(root_dir)
    project_dir = "data/test/isabelle/custom_hol"
    file_name = "data/test/isabelle/custom_hol/Basic_Logic.thy"
    project_id = project_dir.replace('/', '.')
    time_str = time.strftime("%Y%m%d-%H%M%S")
    output_path = f".log/local_data_generation_transform/data/{time_str}"
    log_path = f".log/local_data_generation_transform/log/{time_str}"
    log_file = f"{log_path}/local_data_generation_transform-{time_str}.log"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    def _print_isabelle_executor_callback():
        search_isabelle_exec = IsabelleExecutor(project_root=project_dir, main_file=file_name, use_human_readable_proof_context=True, suppress_error_log=True)
        search_isabelle_exec.__enter__()
        return search_isabelle_exec
    transform = LocalDataGenerationTransform(0, buffer_size=1000)
    training_data = TrainingData(
        output_path, 
        "training_metadata.json",
        training_meta=transform.get_meta_object(), 
        logger=logger)

    IsabelleExecutor.start_server(port=13000)
    
    try:
        with IsabelleExecutor(project_root=project_dir, main_file=file_name, use_human_readable_proof_context=True, suppress_error_log=True) as isabelle_exec:
            transform(training_data, project_id, isabelle_exec, _print_isabelle_executor_callback)
    finally:
        IsabelleExecutor.stop_server()
    
    save_info = training_data.save()
    logger.info(f"Saved training data to {save_info}")