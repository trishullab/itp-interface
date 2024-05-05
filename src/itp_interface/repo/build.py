#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import subprocess
import yaml
import random
from multiprocessing import Pool
from itp_interface.repo.clone import RepoCloner

class RepoBuilder:
    def __init__(self, folder, disable_build=False):
        self.folder = folder
        self.folder = os.path.abspath(self.folder)
        self.disable_build = disable_build
        self._check_git_installed()
        self.cloner = RepoCloner(folder)

    def _check_git_installed(self):
        try:
            subprocess.run(['git', '--version'], check=True, stdout=subprocess.PIPE)
        except subprocess.CalledProcessError:
            raise RuntimeError("Git is not installed on this machine.")

    def build_repos(self, yaml_file):
        with open(yaml_file, 'r') as stream:
            data = yaml.safe_load(stream)
            repos = data.get('repos', [])
        # Using a multiprocessing Pool for parallel building
        with Pool(processes=len(repos)) as pool:
            pool.map(self.clone_repo, repos)
        
        # Build cannot be parallelized because of opam locks
        for repo in repos:
            self.build_repo(repo)
    
    def clone_repo(self, repo):
        repo_name = next(iter(repo))
        repo_data = repo[repo_name]
        self.cloner.clone_repo(repo_data)
    
    def build_repo(self, repo):
        print(f"Building repo: {repo}")
        repo_name = next(iter(repo))
        repo_data = repo[repo_name]
        build_cmds = repo_data.get('build_cmds', [])
        build_path = os.path.join(self.folder, repo_name)
        self._build_repo(build_path, build_cmds, self.disable_build)
        print(f"Finished building repo: {repo}")

    def _build_repo(self, build_path, build_cmds, build_disabled):
        random_id = random.randint(0, 100000)
        os.chdir(build_path)
        # Make a temporary shell script to run all build commands
        with open(f"temptodelbuild{random_id}.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            for cmd in build_cmds:
                f.write(f"{cmd}\n")
        os.system(f"chmod 777 ./temptodelbuild{random_id}.sh")
        if not build_disabled:
            # Run the shell script
            os.system(f"./temptodelbuild{random_id}.sh")
            # Clean up the shell script
            os.remove(f"./temptodelbuild{random_id}.sh")

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_yaml", help="Path to the yaml file containing repo data")
    parser.add_argument("--disable_build", action="store_true", help="Disable building the repos")
    parser.add_argument("--repo_folder", default=".repo", help="Folder to clone and build the repos in")
    args = parser.parse_args()
    builder = RepoBuilder(args.repo_folder, args.disable_build)
    builder.build_repos(args.repo_yaml)
