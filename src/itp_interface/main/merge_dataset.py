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
from itp_interface.tools.log_utils import setup_logger
from itp_interface.tools.training_data import TrainingData


def merge_datasets(datasets, metafilenames, output, max_parallelism=8, logger=None):
    """
    Merge datasets
    """
    assert len(datasets) == len(metafilenames), "Length of datasets and metafilenames must be the same"
    assert len(datasets) > 1, "At least 2 datasets are required to merge"
    assert os.path.exists(output), f"Output folder {output} does not exist"
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
    metadata = copy.deepcopy(tds[-1].meta)
    for td in tds[:-1]:
        metadata.total_proof_step_cnt += td.meta.total_proof_step_cnt
        metadata.last_training_data += td.meta.last_training_data
    logger.info(f"Merged metadata:\n {metadata}")
    logger.info(f"Start merging datasets")
    cloned_metadata = copy.deepcopy(metadata)
    cloned_metadata.total_proof_step_cnt = 0
    cloned_metadata.last_training_data = 0
    merged_td = TrainingData(
        folder=output,
        training_meta_filename="local.meta.json",
        training_meta=cloned_metadata,
        max_parallelism=max_parallelism,
        logger=logger
    )
    for td in tds:
        logger.info(f"Start loading {td.folder} ...")
        td.load()
        logger.info(f"Loaded {td.folder}.")
        logger.info(f"Start merging {td.folder} ...")
        merged_td.merge(td)
        logger.info(f"Merged {td.folder}")
    logger.info("Finished merging datasets.")
    logger.info(f"Start saving merged dataset to {output} ...")
    merged_td.save()
    logger.info(f"Saved merged dataset to {output}.")
    logger.info("Finished merging datasets.")
    logger.info("Verify the merged dataset ...")
    new_merged_td = TrainingData(
        folder=output,
        training_meta_filename="local.meta.json",
        max_parallelism=max_parallelism,
        logger=logger
    )
    new_merged_td.load()
    assert len(new_merged_td) == metadata.total_proof_step_cnt, "Merged dataset is not correct"
    assert new_merged_td.meta.last_training_data == metadata.last_training_data, "Merged dataset is not correct"
    assert new_merged_td.meta.last_proof_id == metadata.last_proof_id, "Merged dataset is not correct"
    logger.info("Merged dataset is correct.")
    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--datasets", type=str, nargs="+", help="List of datasets to merge")
    args.add_argument("--output", type=str, help="Output file")
    args.add_argument("--metafilenames", type=str, nargs="+", help="List of metafilenames", default=None)
    args.add_argument("--max_parallelism", type=int, help="Max parallelism", default=8)
    args = args.parse_args()
    # Add root dir to python path
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    os.makedirs(args.output, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = os.path.join(".log", "merge_dataset", time_str)
    os.makedirs(log_folder, exist_ok=True)
    logger = setup_logger("dataset_merge", os.path.join(log_folder, "merge.log"))
    if args.metafilenames is None:
        args.metafilenames = ['local.meta.json' for _ in range(len(args.datasets))]
    merge_datasets(args.datasets, args.metafilenames, args.output, args.max_parallelism, logger)
    print(args)