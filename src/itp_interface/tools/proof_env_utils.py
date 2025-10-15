#!/usr/bin/env python3

import copy
import typing
import logging
from itp_interface.rl.simple_proof_env import ProofEnv


class CapturedException(Exception):
    """Exception wrapper for capturing and propagating exceptions across different execution contexts"""

    def __init__(self, exception: Exception):
        self.exception = exception
        super().__init__(str(exception))

    def __str__(self):
        return f"CapturedException: {str(self.exception)}"


def replicate_proof_env(proof_env: ProofEnv, logger: typing.Optional[logging.Logger] = None) -> ProofEnv:
    """
    Create a deep copy of a proof environment with an optional new logger.

    Args:
        proof_env: The proof environment to replicate
        logger: Optional logger instance to use for the replicated environment

    Returns:
        A deep copy of the proof environment
    """
    new_proof_env = copy.deepcopy(proof_env)
    new_proof_env.logger = logger if logger else logging.getLogger(__name__)
    return new_proof_env
