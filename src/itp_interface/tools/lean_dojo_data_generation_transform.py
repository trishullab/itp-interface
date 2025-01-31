#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import json
import uuid
import yaml
from itp_interface.lean_server.lean_context import ProofContext
from itp_interface.lean_server.lean4_utils import Lean4Utils
from itp_interface.tools.training_data import TrainingData
from itp_interface.tools.training_data_format import Goal, MergableCollection, TrainingDataCollection, TrainingDataFormat, TrainingDataMetadataFormat
from itp_interface.tools.coq_training_data_generator import GenericTrainingDataGenerationTransform, TrainingDataGenerationType

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
    
    def dump_theorems_from_file(self, file_path: str, output_path: str, output_filename: str, logger = None):
        assert file_path.endswith('.json'), f"Invalid file path {file_path}"
        assert output_filename.endswith('.yaml'), f"Invalid output filename {output_filename}"
        if logger is None:
            logger = self.logger
        logger.info(f"Dumping theorems from {file_path} to {output_path}/{output_filename}")
        # 1. Load the Lean Dojo file formatted json
        # 2. Extract theorems from the json
        # The Lean Dojo json format is as follows:
        # [
        #     {
        #         "url": "https://github.com/leanprover-community/mathlib",
        #         "commit": "8c1b484d6a214e059531e22f1be9898ed6c1fd47",
        #         "file_path": "src/linear_algebra/basis.lean",
        #         "full_name": "basis.sum_coords_self_apply",
        #         "traced_tactics":[
        #             {
        #                 "tactic": "simpa ...",
        #                 "state_before": "G : .....",
        #                 "state_after": "no goals"
        #             },
        #             {
        #                 "tactic": "simpa ...",
        #                 "state_before": "G : .....",
        #                 "state_after": "no goals"
        #             }, 
        #         ]
        #     }
        # ]
        with open(file_path, 'r') as f:
            dojo_json = json.load(f)
        output_filepath = os.path.join(output_path, output_filename)
        output_dict = {
            "name": output_filename[:-len('.yaml')],
            "num_files": 0,
            "language": "LEAN",
            "few_shot_data_path_for_retrieval": None,
            "few_shot_metadata_filename_for_retrieval": None,
            "dfs_data_path_for_retrieval": None,
            "dfs_metadata_filename_for_retrieval": None,
            "datasets": []
        }
        datasets : list = output_dict['datasets']
        projects = {}
        for dojo_entry in dojo_json:
            if dojo_entry['url'] not in projects:
                project_details = {}
                projects[dojo_entry['url']] = project_details
                project_details['files_dict'] = {}
                project_details['files'] = []
                # Just use the last part of the url as the project name
                datasets.append(
                {
                    "project": f"data/benchmarks/{dojo_entry['url'].strip('/').split('/')[-1]}",
                    "files": project_details['files'],
                })
                logger.info(f"Processing project {dojo_entry['url']}")
            else:
                project_details = projects[dojo_entry['url']]
            if dojo_entry['file_path'] not in project_details['files_dict']:
                theorems_in_file = []
                project_details['files_dict'][dojo_entry['file_path']] = theorems_in_file
                output_dict['num_files'] += 1
                project_details['files'].append(
                {
                    "path": dojo_entry['file_path'],
                    "theorems": theorems_in_file
                })
                logger.info(f"Processing file {dojo_entry['file_path']}")
            else:
                theorems_in_file = project_details['files_dict'][dojo_entry['file_path']]
            theorems_in_file.append(dojo_entry['full_name'])
        with open(output_filepath, 'w') as f:
            yaml.dump(output_dict, f, indent=2)
        pass

    def __call__(self, training_data: TrainingData, project_id : str, executor, print_executor_callback, theorems: typing.List[str] = None) -> TrainingData:
        assert isinstance(training_data, TrainingData)
        assert isinstance(project_id, str)
        json_path = project_id
        assert json_path.endswith('.json'), f"Invalid json path {json_path}"
        assert os.path.exists(json_path), f"Invalid json path {json_path}"
        self.logger.info(f"Generating training data for {project_id}")
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        # Go over all the theorems in the json file and generate training data for them
        for dojo_entry in json_data:
            project_path = f"data/benchmarks/{dojo_entry['url'].strip('/').split('/')[-1]}"
            file_path = dojo_entry['file_path']
            theorem_name = dojo_entry['full_name']
            theorem_id = str(uuid.uuid4())
            for tactic in dojo_entry['traced_tactics']:
                state_after = tactic['state_after']
                state_before = tactic['state_before']
                tactic = tactic['tactic']
                try:
                    start_goals: ProofContext = Lean4Utils.parse_proof_context_human_readable(state_before)
                    end_goals: ProofContext = Lean4Utils.parse_proof_context_human_readable(state_after)
                except Exception as e:
                    print(f"Error parsing proof context for {theorem_id} {theorem_name} \ntactic: {tactic}")
                    print("State before:")
                    print(state_before)
                    self.logger.error(f"Error parsing proof context for {theorem_id} {theorem_name} {tactic}")
                    self.logger.error(f"Error: \n{e}")
                    print("State after:")
                    print(state_after)
                    raise
                if len(start_goals.all_goals) > 0:
                    # Create a training data object
                    training_data_format = TrainingDataFormat(
                        proof_id=theorem_id,
                        start_goals=[Goal(goal.hypotheses, goal.goal) for goal in start_goals.all_goals],
                        end_goals=[Goal(goal.hypotheses, goal.goal) for goal in end_goals.all_goals],
                        proof_steps=[tactic],
                        file_path=file_path,
                        project_id=project_path,
                        theorem_name=theorem_name
                    )
                    training_data.merge(training_data_format)
        pass


if __name__ == "__main__":
    import os
    import logging
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default=f".log/benchmarks/leandojo_benchmark_4/leandojo_benchmark_4/random/val.json")
    parser.add_argument("--output_dir", type=str, default=f"<root>/data/proofsteps/leandojo/random/val")
    args = parser.parse_args()
    if "<root>" in args.output_dir:
        # Get root from os.environ
        args.output_dir = args.output_dir.replace("<root>/", os.environ.get("ROOT", "").trim('/') + "/")
    os.chdir(root_dir)
    project_dir = "data/test/lean_proj"
    file_name = "data/test/lean_proj/src/simple_solved.lean"
    project_id = project_dir.replace('/', '.')
    time_str = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(args.output_dir, time_str) # f".log/run_data_generation_transforms/data/{time_str}"
    log_path = f".log/run_data_generation_transforms/log/{time_str}"
    log_file = f"{log_path}/run_data_generation_transforms-{time_str}.log"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    # Dump the files and theorems from the Lean Dojo benchmark
    from itp_interface.tools.run_data_generation_transforms import RunDataGenerationTransforms
    transform = LocalDataGenerationTransform(0, buffer_size=1000, logger=logger)
    final_training_meta = transform.get_meta_object()
    final_training_meta.training_data_buffer_size = transform.buffer_size
    final_training_meta.data_filename_prefix = RunDataGenerationTransforms.get_data_filename_prefix(transform)
    final_training_meta.data_filename_suffix = RunDataGenerationTransforms.get_data_filename_suffix(transform)
    final_training_meta.lemma_ref_filename_prefix = RunDataGenerationTransforms.get_lemma_ref_filename_prefix(transform)
    final_training_meta.lemma_ref_filename_suffix = RunDataGenerationTransforms.get_lemma_ref_filename_suffix(transform)
    training_data = TrainingData(output_path, "local.meta.json", final_training_meta, logger=logger)
    transform(training_data, args.input_json, None, None, None)
    save_info = training_data.save()
    logger.info(f"Saved training data to {save_info}")
    # transform.dump_theorems_from_file(".log/benchmarks/leandojo_benchmark/random/test.json", "itp_interface/main/config/benchmark", "leandojo_random_test.yaml")
    # transform.dump_theorems_from_file(".log/benchmarks/leandojo_benchmark/random/train.json", "itp_interface/main/config/benchmark", "leandojo_random_train.yaml")
    # transform.dump_theorems_from_file(".log/benchmarks/leandojo_benchmark/random/val.json", "itp_interface/main/config/benchmark", "leandojo_random_val.yaml")
    # transform.dump_theorems_from_file(".log/benchmarks/leandojo_benchmark/novel_premises/test.json", "itp_interface/main/config/benchmark", "leandojo_novel_premises_test.yaml")
    # transform.dump_theorems_from_file(".log/benchmarks/leandojo_benchmark/novel_premises/train.json", "itp_interface/main/config/benchmark", "leandojo_novel_premises_train.yaml")
    # transform.dump_theorems_from_file(".log/benchmarks/leandojo_benchmark/novel_premises/val.json", "itp_interface/main/config/benchmark", "leandojo_novel_premises_val.yaml")