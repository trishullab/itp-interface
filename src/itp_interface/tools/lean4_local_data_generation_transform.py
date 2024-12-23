#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import uuid
from itp_interface.tools.lean4_sync_executor import Lean4SyncExecutor
from itp_interface.tools.lean4_context_helper import Lean4ContextHelper
from itp_interface.tools.coq_training_data_generator import GenericTrainingDataGenerationTransform, TrainingDataGenerationType
from itp_interface.tools.training_data_format import MergableCollection, TrainingDataMetadataFormat, TrainingDataCollection, TrainingDataFormat
from itp_interface.tools.training_data import TrainingData

class Local4DataGenerationTransform(GenericTrainingDataGenerationTransform):
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

    def __call__(self, training_data: TrainingData, project_id : str, lean_executor: Lean4SyncExecutor, print_coq_executor_callback: typing.Callable[[], Lean4SyncExecutor], theorems: typing.List[str] = None, other_args: dict = {}) -> TrainingData:
        print_lean_executor = print_coq_executor_callback()
        lean_context_helper = Lean4ContextHelper(print_lean_executor, self.depth, self.logger)
        lean_context_helper.__enter__()
        file_namespace = lean_executor.main_file.replace('/', '.')
        self.logger.info(f"=========================Processing {file_namespace}=========================")
        theorem_id = str(uuid.uuid4())
        theorems = set(theorems) if theorems is not None else None
        assert len(theorems) == 1, "Only one theorem can be processed at a time"
        with lean_executor:
            lean_executor.set_run_exactly()
            lean_executor._skip_to_theorem(theorems.pop())
            start_goals = lean_context_helper.get_focussed_goals_from_proof_context(lean_executor.proof_context)
            ran_next = lean_executor.run_next()
            cmd_ran = lean_executor.current_stmt
            try:
                theorem_id = theorem_id + "/" + lean_executor.get_current_lemma_name()
            except:
                pass
            proof_id = theorem_id
            while not lean_executor.execution_complete and ran_next:
                if lean_executor.is_in_proof_mode():
                    end_goals = lean_context_helper.get_focussed_goals_from_proof_context(lean_executor.proof_context)
                else:
                    end_goals = []
                if len(start_goals) > 0 and \
                    (len(start_goals) != len(end_goals) or not all(s_g == e_g for s_g, e_g in zip(start_goals, end_goals))):
                    tdf = TrainingDataFormat(
                        proof_id=proof_id,
                        all_useful_defns_theorems=[],
                        start_goals=start_goals,
                        end_goals=end_goals,
                        proof_steps=[cmd_ran],
                        simplified_goals=[],
                        addition_state_info={},
                        file_path=lean_executor.main_file,
                        theorem_name=proof_id,
                        project_id=project_id)
                    training_data.merge(tdf)
                if lean_executor.is_in_proof_mode():
                    start_goals = end_goals
                    ran_next = lean_executor.run_next()
                    cmd_ran = lean_executor.current_stmt
                else:
                    ran_next = False
        training_data.meta.num_theorems += 1
        self.logger.info(f"===============Finished processing {file_namespace}=====================")
        self.logger.info(f"Total theorems processed in this transform: 1")
        try:
            lean_context_helper.__exit__(None, None, None)
        except:
            pass


if __name__ == "__main__":
    import os
    import logging
    import time
    os.chdir(root_dir)
    project_dir = 'data/test/lean4_proj/'
    file_name = 'data/test/lean4_proj/Lean4Proj/Basic.lean'
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
        search_lean_exec = Lean4SyncExecutor(main_file=file_name, project_root=project_dir)
        search_lean_exec.__enter__()
        return search_lean_exec
    transform = Local4DataGenerationTransform(0, buffer_size=1000)
    training_data = TrainingData(
        output_path, 
        "training_metadata.json",
        training_meta=transform.get_meta_object(), 
        logger=logger)
    with Lean4SyncExecutor(project_root=project_dir, main_file=file_name, use_human_readable_proof_context=True, suppress_error_log=True) as coq_exec:
        transform(training_data, project_id, coq_exec, _print_lean_executor_callback, theorems=['{"namespace": "Lean4Proj1", "name": "test2"}'])
    save_info = training_data.save()
    logger.info(f"Saved training data to {save_info}")