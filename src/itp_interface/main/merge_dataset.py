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

def filter_training_data(training_data: TrainingData, max_distance_to_good: int = 10):

    def _reconstruct_proof_tree(prev_state_id_map: typing.Dict[int, typing.Set[int]], 
        good_state_ids: typing.Set[int], 
        bad_state_ids: typing.Set[int],
        distance_map: typing.Dict[int, int]):
        distance = 1
        while True:
            # The idea is at some point no new good state ids will be added
            # This is level order traversal but in reverse
            new_good_state_ids = set()
            for end_state, start_states in prev_state_id_map.items():
                if end_state in good_state_ids:
                    for start_state in start_states:
                        if start_state not in good_state_ids:
                            # To avoid infinite loop we need to not add the start_state if it is already in good_state_ids
                            new_good_state_ids.add(start_state)
                            dist = distance_map.get(start_state, distance)
                            if dist >= distance:
                                distance_map[start_state] = distance
            distance += 1
            if len(new_good_state_ids) == 0:
                break
            good_state_ids.update(new_good_state_ids)
        # Now identify the states which are not good
        for end_state in prev_state_id_map.keys():
            if end_state not in good_state_ids:
                bad_state_ids.add(end_state)
    
    def _reconstruct_prev_state_id_map(training_datas: typing.List[TrainingDataFormat]) -> typing.Tuple[typing.Dict[int, int], int]:
            prev_state_id_map = {}
            done_state = None
            for training_data in training_datas:
                if training_data.addition_state_info is not None and len(training_data.addition_state_info) > 0:
                    start_state = training_data.addition_state_info.get("start_goal_id", None)
                    end_state = training_data.addition_state_info.get("end_goal_id", None)
                    if start_state is not None and end_state is not None:
                        prev_to_end_state = prev_state_id_map.get(end_state, set())
                        prev_to_end_state.add(start_state)
                        prev_state_id_map[end_state] = prev_to_end_state
                    if done_state is None and training_data.addition_state_info.get("done", False):
                        done_state = end_state
            return prev_state_id_map, done_state
    
    filtered_training_data : typing.List[TrainingDataFormat] = []
    proof_id_maps : typing.Dict[str, typing.List[TrainingDataFormat]] = {}
    for idx in range(len(training_data)):
        example = training_data[idx]
        training_datas : typing.List[TrainingDataFormat] = proof_id_maps.get(example.proof_id, [])
        training_datas.append(example)
        proof_id_maps[example.proof_id] = training_datas
    for proof_id, training_datas in proof_id_maps.items():
        prev_state_id_map, done_state = _reconstruct_prev_state_id_map(training_datas)
        # Now we have the prev_state_id_map and done_state
        # Every state from where we can reach done_state is a should have a good value
        # Every other state should have a bad value
        # First figure out all state ids from where we can reach done_state
        good_state_ids = set()
        good_state_ids.add(done_state)
        bad_state_ids = set()
        distance_map = {done_state: 0}
        _reconstruct_proof_tree(prev_state_id_map, good_state_ids, bad_state_ids, distance_map)
        # Now we have the good_state_ids and bad_state_ids
        # Now annotate the training data with the value function
        for training_data in training_datas:
            if training_data.addition_state_info is not None and len(training_data.addition_state_info) > 0:
                end_state_id = training_data.addition_state_info.get("end_goal_id", None)
                if end_state_id is not None:
                    progress = training_data.addition_state_info.get("progress", "")
                    if end_state_id in good_state_ids and (progress == "StateChanged" or progress == "Done"):
                        distance = distance_map.get(end_state_id, max_distance_to_good)
                        distance = min(distance, max_distance_to_good)
                        progress = f"[GOOD] [{distance}] {progress}"
                    else:
                        progress = f"[BAD] {progress}"
                    training_data.addition_state_info["progress"] = progress
                    if "GOOD" in progress:
                        filtered_training_data.append(training_data)
            elif training_data.addition_state_info is None:
                filtered_training_data.append(training_data)
    
    return filtered_training_data
    

def merge_datasets(datasets, metafilenames, output, max_parallelism=8, logger=None, should_filter_data=False):
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

        filtered_training_data_points : typing.List[TrainingDataFormat] = None
        if should_filter_data and td.folder != datasets[0]:
            # TODO: move to the right location
            filtered_training_data_points = filter_training_data(td)

        if filtered_training_data_points is not None:
            logger.info(f"Filtered training data points to {len(filtered_training_data_points)}")
            logger.info(f"Start merging {td.folder} after filtering...")
            for data in filtered_training_data_points:
                merged_td.merge(data)
        else:
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
    if not should_filter_data:
        new_merged_td.load()
        assert len(new_merged_td) == metadata.total_proof_step_cnt, "Merged dataset is not correct"
        assert new_merged_td.meta.last_training_data == metadata.last_training_data, "Merged dataset is not correct"
        assert new_merged_td.meta.last_proof_id == metadata.last_proof_id, "Merged dataset is not correct"
        logger.info("Merged dataset is correct.")
    else:
        logger.info("Filtered and merged data, skipping verification")
    logger.info("Finished verifying the merged dataset.")
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