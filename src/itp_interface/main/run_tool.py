#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import hydra
import logging
import os
import time
from itp_interface.tools.coq_build_tool import CoqRepoBuilder
from itp_interface.tools.coq_local_data_generation_transform import LocalDataGenerationTransform
from itp_interface.tools.run_data_generation_transforms import RunDataGenerationTransforms
from itp_interface.tools.log_utils import setup_logger
from itp_interface.main.config import Experiments, EvalRunCheckpointInfo, TransformType, parse_config

def run_data_generation_pipeline(experiment: Experiments, log_dir: str, checkpoint_info: EvalRunCheckpointInfo, logger: logging.Logger = None):
    try:
        transforms = []
        if experiment.run_settings.transform_type == TransformType.LOCAL:
            transform = LocalDataGenerationTransform(
                experiment.run_settings.dep_depth, 
                max_search_results=experiment.run_settings.max_search_results, 
                buffer_size=experiment.run_settings.buffer_size, 
                logger=logger)
            builder : CoqRepoBuilder = CoqRepoBuilder.load_from_file(experiment.run_settings.info_file)
            builder.set_logger(logger)
            builder.enable_error_loggging()
            transforms.append(transform)
        else:
            raise ValueError(f"Unexpected transform_type: {experiment.run_settings.transform_type}")
        data_transform = RunDataGenerationTransforms(transforms, 
                log_dir, 
                save_intermidiat_transforms=len(transforms) > 1 or \
                experiment.run_settings.save_intermidiate_transforms, 
                logger=logger)
        for data_type in ["train", "test"]:
            new_output_dir = os.path.join(experiment.run_settings.output_dir, data_type)
            os.makedirs(new_output_dir, exist_ok=True)
            if experiment.run_settings.transform_type == TransformType.LOCAL:
                projects = list(builder.compilable_projects)
            else:
                projects = []
            files = []
            if data_type == "train":
                if experiment.run_settings.transform_type == TransformType.LOCAL:
                    files = builder.train_compilable_files + builder.train_uncompilable_files
                else:
                    raise ValueError(f"Unexpected transform_type: {experiment.run_settings.transform_type}")
            elif data_type == "test":
                if experiment.run_settings.transform_type == TransformType.LOCAL:
                    files = builder.test_compilable_files + builder.test_uncompilable_files
                else:
                    raise ValueError(f"Unexpected transform_type: {experiment.run_settings.transform_type}")
            else:
                raise ValueError(f"Unexpected data_type: {data_type}")
            if experiment.run_settings.transform_type == TransformType.LOCAL:
                if len(files) > 0:
                    data_transform.run_all_local_transforms(
                        experiment.run_settings.pool_size, 
                        builder.root, 
                        projects, 
                        files, 
                        use_human_readable=experiment.run_settings.use_human_readable, 
                        new_output_dir=new_output_dir, 
                        log_error=True)
    except Exception as e:
        logger.exception(e)
        raise e 


def run_data_generation(experiment: Experiments, log_dir: str, logger: logging.Logger = None):
    trial_cnt = 1
    run_settings = experiment.run_settings
    benchmark = experiment.benchmark
    checkpoint_dir = experiment.run_settings.checkpoint_dir
    run_settings.checkpoint_dir = os.path.join(checkpoint_dir, benchmark.name, run_settings.name)
    os.makedirs(run_settings.checkpoint_dir, exist_ok=True)
    # Load the checkpoint file if it exists
    checkpoint_file = os.path.join(run_settings.checkpoint_dir, "checkpoint_info.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint_info: EvalRunCheckpointInfo = EvalRunCheckpointInfo.from_json(f.read())
        checkpoint_info.logging_dirs.append(log_dir)
    else:
        checkpoint_info = EvalRunCheckpointInfo(
            checkpoint_file=checkpoint_file,
            proof_dump_dir='', 
            logging_dirs=[log_dir], 
            theorem_maps={})
    while trial_cnt > 0:
        try:
            logger = logger or logging.getLogger(__name__)
            logger.info(f"Running experiment: \n{experiment.to_json(indent=4)}")
            run_data_generation_pipeline(experiment, log_dir, checkpoint_info, logger=logger)
            trial_cnt = 0
        except:
            trial_cnt -= 1
            logger.exception(f"Exception occurred. Retrying {trial_cnt} more times.")
            time.sleep(10)
    logger.info(f"Finished running experiment: \n{experiment.to_json(indent=4)}")

@hydra.main(config_path="config", config_name="experiments", version_base="1.2")
def main(cfg):
    experiment = parse_config(cfg)
    top_level_dir = os.path.dirname(root_dir)
    top_level_dir = os.path.dirname(top_level_dir)
    os.chdir(top_level_dir)
    log_dir = ".log/data_generation/benchmark/{}/{}".format(experiment.benchmark.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "eval.log")
    logger = setup_logger(__name__, log_path, logging.INFO, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Pid: {os.getpid()}")
    logger.info(f"Running Experiment: {experiment.to_json(indent=4)}")
    run_data_generation(experiment, log_dir, logger=logger)
    pass

if __name__ == "__main__":
    # RayUtils.init_ray(num_of_cpus=20, object_store_memory_in_gb=50, memory_in_gb=1)
    main()