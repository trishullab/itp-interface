#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import subprocess
import yaml

class RepoCloner:
    def __init__(self, folder):
        self.folder = folder
        self._check_git_installed()

    def _check_git_installed(self):
        try:
            subprocess.run(['git', '--version'], check=True, stdout=subprocess.PIPE)
        except subprocess.CalledProcessError:
            raise RuntimeError("Git is not installed on this machine.")

    def clone_repos(self, yaml_file):
        with open(yaml_file, 'r') as stream:
            data = yaml.safe_load(stream)
            repos = data.get('repos', [])
            for repo in repos:
                assert len(repo) == 1, "Each repo entry should have exactly one key-value pair"
                repo_name = list(repo.keys())[0]
                print(f"Cloning {repo_name}")
                repo = list(repo.values())[0]
                self.clone_repo(repo)
    
    def clone_repo(self, repo_dict):
        url = repo_dict.get('url')
        commit = repo_dict.get('commit', None)
        branch = repo_dict.get('branch', None)
        if not branch:
            branch = self._get_default_branch(url)
        if not commit:
            commit = self._get_latest_commit(url, branch)
        self._clone_repo(url, commit, branch)        

    def _clone_repo(self, url, commit, branch):
        repo_name = url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(self.folder, repo_name)
        if not os.path.exists(repo_path):
            os.makedirs(repo_path)
        os.chdir(repo_path)
        print(f"Cloning {url} at commit {commit} to branch {branch}")
        subprocess.run(['git', 'clone', '--single-branch', '--branch', branch, url, '.'])
        subprocess.run(['git', 'checkout', commit])
        print(f"Cloned {url} at commit {commit} to branch {branch}")

    def _get_default_branch(self, url):
        try:
            output = subprocess.run(['git', 'ls-remote', '--symref', url, 'HEAD'], stdout=subprocess.PIPE)
            default_branch = None
            for line in output.stdout.decode().split('\n'):
                if line.startswith('ref: refs/heads/'):
                    default_branch = line.split('/')[-1].split('\t')[0].strip()
                    break
            if not default_branch:
                raise RuntimeError("Unable to determine default branch for repository: {}".format(url))
            return default_branch
        except subprocess.CalledProcessError:
            raise RuntimeError("Unable to determine default branch for repository: {}".format(url))

    def _get_latest_commit(self, url, branch):
        try:
            output = subprocess.run(['git', 'ls-remote', url, 'refs/heads/' + branch], stdout=subprocess.PIPE)
            latest_commit = output.stdout.decode().split()[0]
            return latest_commit
        except subprocess.CalledProcessError:
            raise RuntimeError("Unable to fetch latest commit for repository: {}".format(url))

# Example usage
if __name__ == "__main__":
    cloner = RepoCloner(".repo")
    cloner.clone_repos("src/itp_interface/main/config/repo/coq_repos.yaml")
