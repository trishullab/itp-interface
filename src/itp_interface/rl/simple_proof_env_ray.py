#!/usr/bin/env python3

import threading
from itp_interface.rl.simple_proof_env import ProofEnv

# Conditional Ray import
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ray = None


if HAS_RAY:
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
else:
    # Thread-safe fallback implementation when Ray is not available
    class ProofEnvActor(ProofEnv):
        def __init__(self, *args, **kwargs):
            self._should_load_env = kwargs.get("should_load_env", True)
            kwargs.pop("should_load_env", None)
            self._env_args = args
            self._env_kwargs = kwargs
            # Add thread safety lock
            self._actor_lock = threading.RLock()
            super().__init__(*args, **kwargs)
            if self._should_load_env:
                super().__enter__()

        def get_env_args(self):
            with self._actor_lock:
                return self._env_args

        def get_env_kwargs(self):
            with self._actor_lock:
                return self._env_kwargs

        def should_load_env(self):
            with self._actor_lock:
                return self._should_load_env

        def get_timeout(self):
            with self._actor_lock:
                return self.dynamic_proof_executor_callback.timeout_in_secs

        # Override methods that need thread safety
        def step(self, action):
            with self._actor_lock:
                return super().step(action)

        def reset(self):
            with self._actor_lock:
                return super().reset()

        def get_state(self):
            with self._actor_lock:
                return super().get_state()

        def get_done(self):
            with self._actor_lock:
                return super().get_done()

        def get_history(self):
            with self._actor_lock:
                return super().get_history()

        def render(self):
            with self._actor_lock:
                return super().render()

        def dump_proof(self, dump_file_name=None, additional_info=None):
            with self._actor_lock:
                return super().dump_proof(dump_file_name, additional_info)

        def cleanup(self):
            with self._actor_lock:
                return super().cleanup()

        def getattr(self, attr_name):
            with self._actor_lock:
                return super().getattr(attr_name)
