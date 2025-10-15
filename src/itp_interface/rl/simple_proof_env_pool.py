#!/usr/bin/env python3

import typing
import logging
from itp_interface.rl.simple_proof_env import ProofEnv
from itp_interface.rl.simple_proof_env_ray import ProofEnvActor, HAS_RAY
from itp_interface.rl.proof_action import ProofAction

# Conditional imports based on Ray availability
if HAS_RAY:
    from itp_interface.rl.ray_proof_env_pool import RayProofEnvPool

from itp_interface.rl.thread_proof_env_pool import ThreadProofEnvPool


class ProofEnvPool(object):
    """
    Facade class that creates either RayProofEnvPool or ThreadProofEnvPool
    based on Ray availability and user preference.
    """

    def __init__(self,
            pool_size: int = 1,
            proof_env_actors: typing.List[ProofEnvActor] = None,
            proof_env: ProofEnv = None,
            logger: typing.Optional[logging.Logger] = None,
            timeout: float = 120,
            max_parallel_envs: int = None,
            use_ray: bool = None):
        """
        Initialize ProofEnvPool with automatic or explicit backend selection.

        Args:
            pool_size: Number of proof environments in the pool
            proof_env_actors: Pre-created ProofEnvActor instances
            proof_env: Template ProofEnv to replicate
            logger: Logger instance
            timeout: Timeout for operations in seconds
            max_parallel_envs: Maximum number of parallel environments
            use_ray: Backend selection:
                    - None (default): Auto-detect (use Ray if available)
                    - True: Force Ray usage (raises error if Ray not available)
                    - False: Force thread-based implementation
        """
        # Determine which implementation to use
        if use_ray is None:
            should_use_ray = HAS_RAY
        else:
            should_use_ray = use_ray and HAS_RAY
            if use_ray and not HAS_RAY:
                raise ImportError("Ray is not installed but use_ray=True was specified. Please install Ray with: pip install ray")

        # Create appropriate implementation
        if should_use_ray:
            if logger:
                logger.info("ProofEnvPool: Using Ray-based implementation")
            self._impl = RayProofEnvPool(
                pool_size=pool_size,
                proof_env_actors=proof_env_actors,
                proof_env=proof_env,
                logger=logger,
                timeout=timeout,
                max_parallel_envs=max_parallel_envs
            )
        else:
            if logger:
                logger.info("ProofEnvPool: Using Thread-based implementation")
            self._impl = ThreadProofEnvPool(
                pool_size=pool_size,
                proof_env_actors=proof_env_actors,
                proof_env=proof_env,
                logger=logger,
                timeout=timeout,
                max_parallel_envs=max_parallel_envs
            )

    # Delegate all methods to the underlying implementation
    def __enter__(self):
        return self._impl.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self._impl.__exit__(exc_type, exc_value, traceback)

    def add_and_init_proof_envs(self, count: int = 1):
        return self._impl.add_and_init_proof_envs(count)

    def add_proof_envs(self, count: int = 1):
        return self._impl.add_proof_envs(count)

    def add_proof_env(self, proof_env: ProofEnv = None):
        return self._impl.add_proof_env(proof_env)

    def get_errd_envs(self):
        return self._impl.get_errd_envs()

    def get_errd_envs_exceptions(self):
        return self._impl.get_errd_envs_exceptions()

    def get_timeout(self):
        return self._impl.get_timeout()

    def step(self, actions: typing.List[ProofAction], idxs: typing.List[int] = None):
        return self._impl.step(actions, idxs)

    def get_pool(self, idxs: typing.List[int]):
        return self._impl.get_pool(idxs)

    def reset(self, idxs: typing.List[int]):
        return self._impl.reset(idxs)

    def get_state(self, idxs: typing.List[int]):
        return self._impl.get_state(idxs)

    def get_done(self, idxs: typing.List[int]):
        return self._impl.get_done(idxs)

    def dump_proof(self, idxs: typing.List[int]):
        return self._impl.dump_proof(idxs)

    def get_proof_search_res(self, idxs: typing.List[int]):
        return self._impl.get_proof_search_res(idxs)
