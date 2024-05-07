#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import hydra
import logging
import os
import time
import shutil
import ray
import typing
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.coq_local_data_generation_transform import LocalDataGenerationTransform as CoqLocalDataGenerationTransform
from itp_interface.tools.lean_local_data_generation_transform import LocalDataGenerationTransform as LeanLocalDataGenerationTransform
from itp_interface.tools.run_data_generation_transforms import RunDataGenerationTransforms
from itp_interface.tools.log_utils import setup_logger
from itp_interface.main.config import Experiments, EvalRunCheckpointInfo, TransformType, parse_config
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from itp_interface.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor
from itp_interface.tools.dynamic_lean4_proof_exec import DynamicProofExecutor as DynamicLean4ProofExecutor
from itp_interface.tools.coq_executor import get_all_lemmas_in_file as get_all_lemmas_coq
from itp_interface.tools.lean4_sync_executor import get_all_theorems_in_file as get_all_lemmas_lean4, get_fully_qualified_theorem_name as get_fully_qualified_theorem_name_lean4

@ray.remote
def get_all_lemmas(project_folder, 
        file_path, 
        language, 
        use_hammer, 
        timeout_in_secs,
        use_human_readable_proof_context, 
        suppress_error_log, 
        always_use_retrieval,
        setup_cmds: typing.List[str], 
        log_file: str):
    logger = setup_logger('get_all_lemmas', log_file, logging.INFO, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    proof_exec_callback = ProofExecutorCallback(
        project_folder=project_folder,
        file_path=file_path,
        language=language,
        use_hammer=use_hammer,
        timeout_in_secs=timeout_in_secs,
        use_human_readable_proof_context=use_human_readable_proof_context,
        suppress_error_log=suppress_error_log,
        always_use_retrieval=always_use_retrieval,
        logger=logger,
        setup_cmds=setup_cmds
    )
    lemmas_to_prove = []
    try:
        with proof_exec_callback.get_proof_executor() as main_executor:
            if language == ProofAction.Language.COQ:
                assert isinstance(main_executor, DynamicCoqProofExecutor)
                lemmas_to_prove = get_all_lemmas_coq(main_executor, logger)
            elif language == ProofAction.Language.LEAN:
                assert isinstance(main_executor, DynamicLeanProofExecutor)
                main_executor.run_all_without_exec()
                lemmas_to_prove = main_executor.find_all_theorems_names()
            elif language == ProofAction.Language.LEAN4:
                assert isinstance(main_executor, DynamicLean4ProofExecutor)
                theorem_details = get_all_lemmas_lean4(file_path)
                lemmas_to_prove = [get_fully_qualified_theorem_name_lean4(theorem) for theorem in theorem_details]
            else:
                raise ValueError(f"Unexpected language: {language}")
    except Exception as e:
        logger.error(f"Error occurred while getting lemmas from {file_path}")
        logger.exception(e)
    logger.info(f"Discovered {len(lemmas_to_prove)} lemmas")
    return lemmas_to_prove

def run_data_generation_pipeline(experiment: Experiments, log_dir: str, checkpoint_info: EvalRunCheckpointInfo, logger: logging.Logger = None):
    try:
        transforms = []
        str_time = time.strftime("%Y%m%d-%H%M%S")
        clone_dir = os.path.join(experiment.run_settings.output_dir, "clone{}".format(str_time))
        if experiment.run_settings.transform_type == TransformType.LOCAL:
            if experiment.benchmark.language == ProofAction.Language.LEAN:
                transform = LeanLocalDataGenerationTransform(
                    experiment.run_settings.dep_depth, 
                    max_search_results=experiment.run_settings.max_search_results, 
                    buffer_size=experiment.run_settings.buffer_size, 
                    logger=logger)
                os.makedirs(clone_dir, exist_ok=True)
            elif experiment.benchmark.language == ProofAction.Language.COQ:
                only_proof_state = experiment.env_settings.retrieval_strategy == ProofEnvReRankStrategy.NO_RE_RANK
                transform = CoqLocalDataGenerationTransform(
                    experiment.run_settings.dep_depth, 
                    max_search_results=experiment.run_settings.max_search_results, 
                    buffer_size=experiment.run_settings.buffer_size, 
                    logger=logger,
                    no_dfns=only_proof_state,
                    no_thms=only_proof_state)
                clone_dir = None
            else:
                raise ValueError(f"Unexpected language: {experiment.benchmark.language}")
            transforms.append(transform)
        else:
            raise ValueError(f"Unexpected transform_type: {experiment.run_settings.transform_type}")
        # Find all the lemmas to prove
        project_to_theorems = {}
        for idx, dataset in enumerate(experiment.benchmark.datasets):
            if dataset.project not in project_to_theorems:
                project_to_theorems[dataset.project] = {}
            file_to_theorems = project_to_theorems[dataset.project]
            lemma_discovery_remotes = [] 
            for file_idx, file in enumerate(dataset.files):
                if file.path not in file_to_theorems:
                    file_to_theorems[file.path] = []
                if isinstance(file.theorems, list):
                    file_to_theorems[file.path].extend(file.theorems)
                else:
                    discover_log_file = os.path.join(log_dir, f"discover{idx}_{file_idx}.log")
                    lemma_discovery_remotes.append(get_all_lemmas.remote(
                        project_folder=dataset.project,
                        file_path=os.path.join(dataset.project, file.path),
                        language=experiment.benchmark.language,
                        use_hammer=False,
                        timeout_in_secs=experiment.run_settings.timeout_in_secs,
                        use_human_readable_proof_context=experiment.run_settings.use_human_readable,
                        suppress_error_log=True,
                        always_use_retrieval=False,
                        setup_cmds=experiment.benchmark.setup_cmds,
                        log_file=discover_log_file))
                pass
            if len(lemma_discovery_remotes) > 0:
                lemmas = ray.get(lemma_discovery_remotes)
                for file, theorems in zip(dataset.files, lemmas):
                    file_to_theorems[file.path].extend(theorems)
            pass
        data_transform = RunDataGenerationTransforms(transforms, 
                log_dir, 
                save_intermidiat_transforms=len(transforms) > 1 or \
                experiment.run_settings.save_intermidiate_transforms, 
                logger=logger)
        new_output_dir = os.path.join(experiment.run_settings.output_dir, str_time)
        os.makedirs(new_output_dir, exist_ok=True)
        projects = list(project_to_theorems.keys())
        if experiment.run_settings.transform_type == TransformType.LOCAL:
            # clone the root directory
            if clone_dir is not None:
                for project in projects:
                    logger.info(f"==============================>Cloning root directory {project} to {clone_dir}<==============================")
                    temp_clone_dir = os.path.join(clone_dir, project)
                    shutil.copytree(project, temp_clone_dir, dirs_exist_ok=True)
                    # change the root_dir in files to clone_dir
                    project_to_theorems[temp_clone_dir] = project_to_theorems[project]
                    del project_to_theorems[project]
                logger.info(f"==============================>Cloned root directory {project} to {clone_dir}<==============================")
            try:
                data_transform.run_all_local_transforms(
                    experiment.run_settings.pool_size,
                    project_to_theorems, 
                    use_human_readable=experiment.run_settings.use_human_readable, 
                    new_output_dir=new_output_dir, 
                    log_error=True)
            finally:
                if clone_dir is not None:
                    for project in projects:
                        temp_clone_dir = os.path.join(clone_dir, project)
                        logger.info(f"==============================>Removing cloned root directory {temp_clone_dir}<==============================")
                        shutil.rmtree(temp_clone_dir, ignore_errors=True)
                        logger.info(f"==============================>Removed cloned root directory {temp_clone_dir}<==============================")
                        shutil.rmtree(clone_dir, ignore_errors=True)
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
    os.chdir(root_dir)
    # top_level_dir = os.path.dirname(root_dir)
    # top_level_dir = os.path.dirname(top_level_dir)
    # os.chdir(top_level_dir)
    log_dir = ".log/data_generation/benchmark/{}/{}".format(experiment.benchmark.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "eval.log")
    logger = setup_logger(__name__, log_path, logging.INFO, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Pid: {os.getpid()}")
    logger.info(f"Running Experiment: {experiment.to_json(indent=4)}")
    run_data_generation(experiment, log_dir, logger=logger)
    pass

if __name__ == "__main__":
    # from itp_interface.tools.ray_utils import RayUtils
    # RayUtils.init_ray(num_of_cpus=20, object_store_memory_in_gb=50, memory_in_gb=1, runtime_env={"working_dir": root_dir, "excludes": [".log", "data"]})
    main()