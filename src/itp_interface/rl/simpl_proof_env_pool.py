#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
# This is a hack to fix the wrong file name in the import statement of other files
# This will be deprecated once all dependencies are moved to simple_proof_env_pool.py vs simpl_proof_env_pool.py
# Let's add deprecation warning here
import warnings
warnings.warn("simpl_proof_env_pool.py is deprecated. Use simple_proof_env_pool.py instead.", DeprecationWarning)
from itp_interface.rl.simple_proof_env_pool import (
    ProofEnv, ProofEnvActor, ProofEnvInfo, ProofEnvReRankStrategy, ProofExecutorCallback,
    IsabelleExecutor, HammerMode, ProofAction, ProofState, 
    replicate_proof_env, CapturedException, CaptureExceptionActor, ProofEnvPool
)