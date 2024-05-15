#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import argparse
import logging
import time
import os
import typing
import copy
import yaml
from itp_interface.tools.log_utils import setup_logger
from itp_interface.tools.training_data import TrainingData


def create_yaml(project_to_theorems, name, language, output_file):
    data = {
        "name": name,
        "num_files": 0,
        "language": str(language),
        "few_shot_data_path_for_retrieval": None,
        "few_shot_metadata_filename_for_retrieval": None,
        "dfs_data_path_for_retrieval": None,
        "dfs_metadata_filename_for_retrieval": "local.meta.json",
        "theorem_cnt": 0,
        "datasets": []
    }
    for project_root, file_dict in project_to_theorems.items():
        dataset = {"project": project_root, "files": []}
        for file_path, theorems in file_dict.items():
            data["num_files"] += 1
            data["theorem_cnt"] += len(theorems)
            dataset["files"].append({"path": file_path, "theorems": theorems}) 
        data["datasets"].append(dataset)

    with open(output_file, 'w') as yaml_file:
        yaml.dump(data, yaml_file, sort_keys=False)

def extract_project_to_theorems(training_data : TrainingData, project_to_theorems: typing.Dict[str, typing.Dict[str, typing.List[str]]]):
    for tdf in training_data:
        project_path = tdf.project_id
        file_path = tdf.file_path
        # Check if both project and file_path are the same absolute path
        both_abs_path = os.path.isabs(project_path) and os.path.isabs(file_path)
        if both_abs_path:
            file_path = os.path.relpath(file_path, project_path)
        if project_path not in project_to_theorems:
            project_to_theorems[project_path] = {}
        if file_path not in project_to_theorems[project_path]:
            project_to_theorems[project_path][file_path] = []
        if tdf.theorem_name not in project_to_theorems[project_path][file_path]:
            project_to_theorems[project_path][file_path].append(tdf.theorem_name)

def extract_benchmarks_from_datasets(datasets, metafilenames, name, language, output, max_parallelism=8, logger=None):
    """
    extract datasets
    """
    assert len(datasets) == len(metafilenames), "Length of datasets and metafilenames must be the same"
    assert max_parallelism > 0, "Max parallelism must be greater than 0"
    logger = logger or logging.getLogger("dataset_merge")
    tds : typing.List[TrainingData] = []
    for dataset, metafilename in zip(datasets, metafilenames):
        training_data = TrainingData(
            folder=dataset,
            training_meta_filename=metafilename,
            max_parallelism=max_parallelism
        )
        training_data.load_meta()
        tds.append(training_data)
        logger.info(f"Inited training data for {dataset}")
    project_to_theorems_map = {}
    for td in tds:
        logger.info(f"Start loading {td.folder} ...")
        td.load()
        logger.info(f"Loaded {td.folder}.")
        logger.info(f"Start extraction from {td.folder} ...")
        extract_project_to_theorems(td, project_to_theorems_map)
        logger.info(f"Finished extraction from {td.folder}.")
    logger.info(f"Saving the extracted theorems to {output} ...")
    create_yaml(project_to_theorems_map, name, language, output)
    logger.info(f"Saved the extracted theorems to {output}.")
    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--datasets", type=str, nargs="+", help="List of datasets to merge")
    args.add_argument("--output", type=str, help="Output file")
    args.add_argument("--metafilenames", type=str, nargs="+", help="List of metafilenames", default=None)
    args.add_argument("--name", type=str, help="Name of the merged dataset")
    args.add_argument("--language", type=str, help="Language of the merged dataset")
    args.add_argument("--max_parallelism", type=int, help="Max parallelism", default=8)
    args = args.parse_args()
    # Add root dir to python path
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = os.path.join(".log", "extract_benchmark", time_str)
    os.makedirs(log_folder, exist_ok=True)
    logger = setup_logger("extract_benchmark", os.path.join(log_folder, "extract.log"))
    if args.metafilenames is None:
        args.metafilenames = ['local.meta.json' for _ in range(len(args.datasets))]
    extract_benchmarks_from_datasets(
        args.datasets, args.metafilenames, args.name, args.language, args.output, args.max_parallelism, logger
    )