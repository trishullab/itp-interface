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
from itp_interface.tools.training_data import TrainingData, TrainingDataFormat

def filter_keyword(tdf: TrainingDataFormat, keywords: typing.List[str]) -> bool:
    """
    Filter out the training data format if it contains any of the keywords
    """
    for keyword in keywords:
        for proof_step in tdf.proof_steps:
            if keyword in proof_step:
                return True
    return False

def filter_dataset(dataset, metafilename, output, keywords, max_parallelism=8, logger=None):
    """
    filter dataset
    """
    assert os.path.exists(output), f"Output folder {output} does not exist"
    assert max_parallelism > 0, "Max parallelism must be greater than 0"
    logger = logger or logging.getLogger("dataset_filter")
    training_data = TrainingData(
        folder=dataset,
        training_meta_filename=metafilename,
        max_parallelism=max_parallelism
    )
    training_data.load_meta()
    logger.info(f"Inited training data for {dataset}")
    metadata = copy.deepcopy(training_data.meta)
    logger.info(f"Start filtering datasets")
    cloned_metadata = copy.deepcopy(metadata)
    cloned_metadata.total_proof_step_cnt = 0
    cloned_metadata.last_training_data = 0
    logger.info(f"Cloned metadata:\n {cloned_metadata}")
    filtered_td = TrainingData(
        folder=output,
        training_meta_filename="local.meta.json",
        training_meta=cloned_metadata,
        max_parallelism=max_parallelism,
        logger=logger
    )
    # Load the training data and each proof step
    logger.info(f"Start loading {dataset} ...")
    training_data.load()
    logger.info(f"Loaded {dataset}.")
    logger.info(f"Start filtering {dataset} ...")
    skipped_cnt = 0
    total_data_points = 0
    for tdf in training_data:
        if total_data_points % 100 == 0 and total_data_points != 0:
            logger.info(f"Skipped {skipped_cnt} data points.")
        if filter_keyword(tdf, keywords):
            skipped_cnt += 1
        else:
            filtered_td.merge(tdf)
        total_data_points += 1
    logger.info(f"In total, skipped {skipped_cnt} data points.")
    logger.info(f"Total data points: {total_data_points}")
    logger.info(f"Total proof steps: {metadata.total_proof_step_cnt}")
    logger.info(f"Lenght of filtered dataset: {len(filtered_td)}")
    logger.info(f"Length of training data: {len(training_data)}")
    logger.info("Finished filtering dataset.")
    logger.info(f"Start saving filtered dataset to {output} ...")
    filtered_td.save()
    logger.info(f"Saved filtered dataset to {output}.")
    logger.info("Finished filtered datasets.")
    logger.info("Verifying the filtered dataset ...")
    new_merged_td = TrainingData(
        folder=output,
        training_meta_filename="local.meta.json",
        max_parallelism=max_parallelism,
        logger=logger
    )
    new_merged_td.load()
    assert len(new_merged_td) == metadata.total_proof_step_cnt - skipped_cnt, "Filtered dataset is not correct"
    logger.info("Verified the filtered dataset.")
    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, help="Dataset to filter")
    args.add_argument("--output", type=str, help="Output file")
    args.add_argument("--metafilename", type=str, help="metafilename for dataset", default="local.meta.json")
    args.add_argument("--max_parallelism", type=int, help="Max parallelism", default=8)
    args.add_argument("--filter_keywords", type=str, nargs="+", help="List of keywords to filter", default=None)
    args = args.parse_args()
    # Add root dir to python path
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    os.makedirs(args.output, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = os.path.join(".log", "filter_dataset", time_str)
    os.makedirs(log_folder, exist_ok=True)
    logger = setup_logger("dataset_filter", os.path.join(log_folder, "filter.log"))
    filter_dataset(args.dataset, args.metafilename, args.output, args.filter_keywords, args.max_parallelism, logger)
    print(args)