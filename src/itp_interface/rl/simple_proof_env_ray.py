#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import ray
from itp_interface.rl.simple_proof_env import ProofEnv


@ray.remote
class ProofEnvActor(ProofEnv):
    def __init__(self, *args, **kwargs):
        self._should_load_env = kwargs.get("should_load_env", True)
        kwargs.pop("should_load_env", None)
        self._env_args = args
        self._env_kwargs = kwargs
        super().__init__(*args, **kwargs)
        if self._should_load_env:
            super().__enter__()
        pass

    def get_env_args(self):
        return self._env_args

    def get_env_kwargs(self):
        return self._env_kwargs

    def should_load_env(self):
        return self._should_load_env

    def get_timeout(self):
        return self.dynamic_proof_executor_callback.timeout_in_secs
