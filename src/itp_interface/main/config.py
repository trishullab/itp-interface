#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum
from itp_interface.rl.proof_tree import ProofSearchResult
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy

class SettingType(Enum):
    Agent = "Agent"
    GptF = "GptF"

    def __str__(self):
        return self.value

class PolicyName(Enum):
    # WARN: Don't make enums dataclasses because deserialization has some weird bug which matches the deserialized enum to all the enum values
    Dfs = "Dfs"
    FewShot = "FewShot"

    def __str__(self):
        return self.value
    
class TransformType(Enum):
    LOCAL = "LOCAL"
    GLOBAL = "GLOBAL"
    FULL = "FULL"

    def __str__(self):
        return self.value

@dataclass_json
@dataclass
class EnvSettings(object):
    name: str
    retrieval_strategy: ProofEnvReRankStrategy

@dataclass_json
@dataclass
class RunSettings(object):
    name: str
    use_human_readable: bool
    save_intermidiate_transforms: bool
    buffer_size: int
    pool_size: int
    transform_type: TransformType
    dep_depth: int
    output_dir: str
    setting_type: SettingType
    timeout_in_secs: int # coq tactic execution timeout
    proof_retries: int
    max_theorems_in_prompt: int
    max_number_of_episodes: int
    max_steps_per_episode: int
    render: bool
    checkpoint_dir: str
    should_checkpoint: bool
    max_search_results: typing.Optional[int] = None
    random_seed: int = 42
    random_split: bool = False
    train_eval_test_split: typing.List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])

@dataclass_json
@dataclass
class EvalFile(object):
    path: str
    theorems: typing.Union[str, typing.List[str]]

@dataclass_json
@dataclass
class EvalDataset(object):
    project: str
    files: typing.List[EvalFile]

@dataclass_json
@dataclass
class EvalBenchmark(object):
    name: str
    num_files: int
    language: ProofAction.Language
    datasets: typing.List[EvalDataset]
    few_shot_data_path_for_retrieval: str = None
    few_shot_metadata_filename_for_retrieval: str = None
    dfs_data_path_for_retrieval: str = None
    dfs_metadata_filename_for_retrieval: str = None
    setup_cmds: typing.List[str] = field(default_factory=list)

@dataclass_json
@dataclass
class Experiments(object):
    env_settings: EnvSettings
    run_settings: RunSettings
    benchmark: EvalBenchmark

@dataclass_json
@dataclass
class EvalRunCheckpointInfo(object):
    checkpoint_file: str
    logging_dirs: typing.List[str]
    proof_dump_dir: str
    theorem_maps: typing.Dict[str, typing.Dict[str, bool]]

    def add_path_to_maps(self, path: str):
        if path not in self.theorem_maps:
            self.theorem_maps[path] = {}

    def add_theorem_to_maps(self, path: str, theorem: str, success: bool):
        self.theorem_maps[path][theorem] = success
        with open(self.checkpoint_file, "w") as f:
            f.write(self.to_json(indent=4))
    
@dataclass_json
@dataclass
class EvalProofResults(object):
    path: str
    theorem_map: typing.Dict[str, typing.Dict[str, ProofSearchResult]]

    def add_path_to_maps(self, path: str):
        if path not in self.theorem_map:
            self.theorem_map[path] = {}
    
    def add_theorem_to_maps(self, path: str, theorem: str, proof_result: ProofSearchResult):
        self.theorem_map[path][theorem] = proof_result
        with open(self.path, "w") as f:
            f.write(self.to_json(indent=4))


def parse_config(cfg):
    env_settings_cfg = cfg["env_settings"]
    env_settings = EnvSettings(
        name=env_settings_cfg["name"],
        retrieval_strategy=ProofEnvReRankStrategy(env_settings_cfg["retrieval_strategy"]))
    run_settings_cfg = cfg["run_settings"]
    eval_settings = RunSettings(
        name=run_settings_cfg["name"],
        use_human_readable=run_settings_cfg["use_human_readable"],
        save_intermidiate_transforms=run_settings_cfg["save_intermidiate_transforms"],
        buffer_size=run_settings_cfg["buffer_size"],
        pool_size=run_settings_cfg["pool_size"],
        transform_type=TransformType(run_settings_cfg["transform_type"]),
        dep_depth=run_settings_cfg["dep_depth"],
        output_dir=run_settings_cfg["output_dir"],
        max_search_results=run_settings_cfg["max_search_results"],
        setting_type=SettingType(run_settings_cfg["setting_type"]),
        timeout_in_secs=run_settings_cfg["timeout_in_secs"],
        proof_retries=run_settings_cfg["proof_retries"],
        max_theorems_in_prompt=run_settings_cfg["max_theorems_in_prompt"],
        max_number_of_episodes=run_settings_cfg["max_number_of_episodes"],
        max_steps_per_episode=run_settings_cfg["max_steps_per_episode"],
        render=run_settings_cfg["render"],
        checkpoint_dir=run_settings_cfg["checkpoint_dir"],
        should_checkpoint=run_settings_cfg["should_checkpoint"],
        random_seed=run_settings_cfg["random_seed"],
        random_split=run_settings_cfg["random_split"],
        train_eval_test_split=run_settings_cfg["train_eval_test_split"])
    benchmark_cfg = cfg["benchmark"]
    datasets_cfg = benchmark_cfg["datasets"]
    eval_datasets = []
    for dataset_cfg in datasets_cfg:
        files_cfg = list(dataset_cfg["files"])
        eval_files = []
        for file_cfg in files_cfg:
            theorems = None
            if type(file_cfg["theorems"]) == str:
                theorems = file_cfg["theorems"]
            else:
                theorems = list(file_cfg["theorems"])
            eval_files.append(EvalFile(
                path=file_cfg["path"],
                theorems=theorems))
        eval_datasets.append(EvalDataset(
            project=dataset_cfg["project"],
            files=eval_files))
    language = ProofAction.Language(benchmark_cfg["language"])
    benchmark = EvalBenchmark(
        name=benchmark_cfg["name"],
        num_files=benchmark_cfg["num_files"],
        language=language,
        datasets=eval_datasets,
        few_shot_data_path_for_retrieval=benchmark_cfg["few_shot_data_path_for_retrieval"],
        few_shot_metadata_filename_for_retrieval=benchmark_cfg["few_shot_metadata_filename_for_retrieval"],
        dfs_data_path_for_retrieval=benchmark_cfg["dfs_data_path_for_retrieval"],
        dfs_metadata_filename_for_retrieval=benchmark_cfg["dfs_metadata_filename_for_retrieval"],
        setup_cmds=benchmark_cfg["setup_cmds"] if "setup_cmds" in benchmark_cfg else [])
    return Experiments(env_settings=env_settings, run_settings=eval_settings, benchmark=benchmark)