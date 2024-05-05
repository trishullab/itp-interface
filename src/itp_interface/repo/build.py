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
                pool.map(self._clone_and_build_repo, repos)

    def _clone_and_build_repo(self, repo):
        repo_name = next(iter(repo))
        repo_data = repo[repo_name]
        build_cmds = repo_data.get('build_cmds', [])
        build_path = os.path.join(self.folder, repo_name)
        self.cloner.clone_repo(repo_data)
        self._build_repo(build_path, build_cmds, self.disable_build)

    def _build_repo(self, build_path, build_cmds, build_disabled):
        random_id = random.randint(0, 100000)
        os.chdir(build_path)
        # Make a temporary shell script to run all build commands
        with open(f"temptodelbuild{random_id}.sh", 'w') as f:
            for cmd in build_cmds:
                f.write(f"{cmd}\n")
        os.system(f"chmod +x temptodelbuild{random_id}.sh")
        if not build_disabled:
            # Run the shell script
            os.system(f"./temptodelbuild{random_id}.sh")
            # Clean up the shell script
            os.remove(f"temptodelbuild{random_id}.sh")

# Example usage
if __name__ == "__main__":
    builder = RepoBuilder(".repo", disable_build=True)
    builder.build_repos("/home/amthakur/Project/itp-interface/src/itp_interface/main/config/repo/coq_repos.yaml")
