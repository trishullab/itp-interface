#!/usr/bin/env python3

import os
import sys
dir_name = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.path.abspath(dir_name)
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import uuid
from itp_interface.tools.simple_lean4_sync_executor import SimpleLean4SyncExecutor
from itp_interface.tools.coq_training_data_generator import GenericTrainingDataGenerationTransform, TrainingDataGenerationType
from itp_interface.tools.training_data_format import MergableCollection, TrainingDataMetadataFormat, ExtractionDataCollection, TheoremProvingTrainingDataFormat
from itp_interface.tools.training_data import TrainingData, DataLayoutFormat

class Local4DataExtractionTransform(GenericTrainingDataGenerationTransform):
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

    def get_meta_object(self) -> TrainingDataMetadataFormat:
        return TrainingDataMetadataFormat(
            training_data_buffer_size=self.buffer_size,
            data_filename_prefix="extraction_data_",
            lemma_ref_filename_prefix="extraction_lemma_refs_")

    def get_data_collection_object(self) -> MergableCollection:
        return ExtractionDataCollection()
    
    def load_meta_from_file(self, file_path) -> MergableCollection:
        return TrainingDataMetadataFormat.load_from_file(file_path)
    
    def load_data_from_file(self, file_path) -> MergableCollection:
        return ExtractionDataCollection.load_from_file(file_path, self.logger)

    def __call__(self, 
        training_data: TrainingData, 
        project_id : str, 
        lean_executor: SimpleLean4SyncExecutor, 
        print_coq_executor_callback: typing.Callable[[], SimpleLean4SyncExecutor], 
        theorems: typing.List[str] = None, 
        other_args: dict = {}) -> TrainingData:
        file_namespace = lean_executor.main_file.replace('/', '.')
        self.logger.info(f"=========================Processing {file_namespace}=========================")
        theorem_id = str(uuid.uuid4())
        if isinstance(theorems, list) and len(theorems) == 1 and theorems[0] == "*":
            theorems = None
        else:
            theorems = set(theorems) if theorems is not None else None
        cnt = 0
        temp_dir = os.path.join(training_data.folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        json_output_path = f"{temp_dir}/{file_namespace.replace('.', '_')}.lean.deps.json"
        file_dep_analyses = lean_executor.extract_all_theorems_and_definitions(json_output_path=json_output_path)
        self.logger.info(f"Extracted {len(file_dep_analyses)} FileDependencyAnalysis objects from {file_namespace}")
        self.logger.info(f"file_dep_analyses: {file_dep_analyses}")
        assert len(file_dep_analyses) == 1, "Expected exactly one FileDependencyAnalysis object"
        file_dep_analysis = file_dep_analyses[0]
        for decls in file_dep_analysis.declarations:
            line_info = decls.decl_info
            if theorems is not None and line_info.name not in theorems:
                continue
            training_data.merge(decls)
            cnt += 1
        training_data.meta.last_proof_id = theorem_id
        self.logger.info(f"===============Finished processing {file_namespace}=====================")
        self.logger.info(f"Total declarations processed in this transform: {cnt}")
        return training_data


if __name__ == "__main__":
    import os
    import logging
    import time
    os.chdir(root_dir)
    # project_dir = 'data/test/lean4_proj/'
    project_dir = 'data/test/Mathlib'
    # file_name = 'data/test/lean4_proj/Lean4Proj/Basic.lean'
    file_name = 'data/test/Mathlib/.lake/packages/mathlib/Mathlib/Algebra/Divisibility/Basic.lean'
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
        search_lean_exec = SimpleLean4SyncExecutor(main_file=file_name, project_root=project_dir)
        search_lean_exec.__enter__()
        return search_lean_exec
    transform = Local4DataExtractionTransform(0, buffer_size=1000)
    training_data = TrainingData(
        output_path, 
        "training_metadata.json",
        training_meta=transform.get_meta_object(), 
        logger=logger,
        layout=DataLayoutFormat.DECLARATION_EXTRACTION)
    with SimpleLean4SyncExecutor(project_root=project_dir, main_file=file_name, use_human_readable_proof_context=True, suppress_error_log=True) as coq_exec:
        transform(training_data, project_id, coq_exec, _print_lean_executor_callback, theorems=["*"])
    save_info = training_data.save()
    logger.info(f"Saved training data to {save_info}")