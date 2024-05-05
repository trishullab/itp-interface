#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import typing
import yaml

class FileFilter:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.filtered_files = []

    def filter(self, criteria_func: typing.Callable[[str, str], bool]):
        self._filter_recursive(self.root_dir, criteria_func)
        return self.filtered_files

    def _filter_recursive(self, directory: str, criteria_func: typing.Callable[[str, str], bool]):
        for root, dirs, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                if criteria_func(file, full_path):
                    relative_path = os.path.relpath(full_path, self.root_dir)
                    self.filtered_files.append(relative_path)

class DatasetCreator:
    def __init__(self, name: str, root_dir: str, langauge: str):
        self.root_dir = root_dir
        self.file_filter = FileFilter(root_dir)
        self.language = langauge
        self.name = name

    def create_yaml(self, output_file):
        files = self.file_filter.filter(self.is_part_of_dataset)
        data = {
            "name": self.name,
            "num_files": len(files),
            "language": self.language,
            "few_shot_data_path_for_retrieval": None,
            "few_shot_metadata_filename_for_retrieval": None,
            "dfs_data_path_for_retrieval": None,
            "dfs_metadata_filename_for_retrieval": "local.meta.json",
            "datasets": []
        }
        dataset = {"project": self.root_dir, "files": []}
        for file in files:
            dataset["files"].append({"path": file, "theorems": "*"})
        data["datasets"].append(dataset)

        with open(output_file, 'w') as yaml_file:
            yaml.dump(data, yaml_file, sort_keys=False)
    
    def is_part_of_dataset(self, file_name: str, full_path: str):
        raise NotImplementedError("This method should be implemented by the subclass.")

class CoqDatasetCreator(DatasetCreator):
    def is_part_of_dataset(self, file_name: str, full_path: str):
        if file_name.endswith(".v"):
            # Check if the file has .vo file
            vo_file = full_path.replace(".v", ".vo")
            if os.path.exists(vo_file):
                return True
        return False
    
class IsabelleDatasetCreator(DatasetCreator):
    def is_part_of_dataset(self, file_name: str, full_path: str):
        if file_name.endswith(".thy"):
            return True
        return False

class LeanDatasetCreator(DatasetCreator):
    def is_part_of_dataset(self, file_name: str, full_path: str):
        if file_name.endswith(".lean"):
            return True
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create a dataset yaml file.')
    parser.add_argument('--root_dir', type=str, help='The root directory of the project')
    parser.add_argument('--output_file', type=str, help='The output yaml file')
    parser.add_argument('--language', type=str, help='The language of the project')
    parser.add_argument('--name', type=str, help='The name of the dataset')
    args = parser.parse_args()

    if args.language == "coq":
        args.language = "COQ"
        creator = CoqDatasetCreator(args.name, args.root_dir, args.language)
    elif args.language == "isabelle":
        creator = IsabelleDatasetCreator(args.name, args.root_dir, args.language)
    elif args.language == "lean":
        args.language = "LEAN4"
        creator = LeanDatasetCreator(args.name, args.root_dir, args.language)
    else:
        raise ValueError("Language not supported.")
    
    creator.create_yaml(args.output_file)