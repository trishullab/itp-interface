[![Build Status](https://github.com/trishullab/itp-interface/actions/workflows/github-build-actions.yaml/badge.svg)](https://github.com/trishullab/itp-interface/actions/workflows/github-build-actions.yaml)
[![PyPI version](https://img.shields.io/pypi/v/itp-interface.svg)](https://pypi.org/project/itp-interface/)
[![PyPI downloads](https://img.shields.io/pypi/dm/itp-interface.svg)](https://pypi.org/project/itp-interface/)

# itp-interface
Generic interface for hooking up to any Interactive Theorem Prover (ITP) and collecting data for training ML models for AI in formal theorem proving.

## 🎉 What's New

**Python 3.14 Free-Threading Support** (January 2025) - `itp-interface` now supports Python 3.14's experimental free-threading mode (GIL-free execution)! Experience true parallel proof search with up to 2.13x speedup on multi-core systems. The interface automatically detects your Python version and seamlessly falls back to thread-based parallelism when Ray is unavailable. See [Python 3.14 Free-Threading Support](#python-314-free-threading-support-optional) for details. 

## Quick Setup for Lean 4:
1. Install itp-interface using the following command:
```bash
pip install itp-interface
```

2. Run the following command to prepare the REPL for Lean 4. The default version is 4.24.0. You can change the version by setting the `LEAN_VERSION` environment variable. If no version is set, then 4.24.0 is used.
>NOTE: The Lean 4 version must match the version of the Lean 4 project you are working with.
```bash
install-lean-repl
# To use a different Lean version, set LEAN_VERSION before running:
# export LEAN_VERSION="4.17.0" && install-lean-repl
```

3. Run the following command to build the REPL for Lean 4. Make sure that `lean --version` returns the correct version before running the command below. If not then check if `$HOME/.elan/bin` is in your path. Recommended to run `source $HOME/.elan/env` before running the command below.
```bash
install-itp-interface
```

>NOTE: These steps are only tested on Linux. For Windows, you can use WSL. These steps will not setup the Coq interface.

# Full Setup for Coq and Lean:
1. Install OCaml first. Use the instructions here: https://opam.ocaml.org/doc/Install.html. Note that OCaml officially only supports Linux installations. One can use WSL on Windows machines.

2. Run the following to install Coq on Linux.
```
sudo apt install build-essential unzip bubblewrap
sh <(curl -sL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)
```

3. Add the following to your `.bashrc` file: (sometimes the path `~/.opam/default` might not exist, so use the directory with version number present in the `~/.opam` directory)
```
export PATH="/home/$USER/.opam/default/bin:$PATH"
```

4. Create a `Miniconda` environment and activate it.

### Python 3.14 Free-Threading Support (Optional)

For Python 3.14 with free-threading (GIL-free) support, create a conda environment using:
```bash
conda create -n py314-ft python=3.14 python-freethreading -c conda-forge
conda activate py314-ft
```

This enables true parallel execution for computational threads. You can verify free-threading is working by running:
```bash
python src/test/test_python314_threading.py
```

**Note**: When using Python 3.14 free-threading:
- Ray is not supported (Ray doesn't support Python 3.14 yet)
- The interface will automatically fall back to thread-based parallelism using `ThreadPoolExecutor`
- `psutil` is not available in free-threading builds, so memory logging is disabled
- **Isabelle/PISA is not supported** - grpcio and protobuf are not compatible with Python 3.14's free-threading mode. Use Python < 3.14 for Isabelle support
- The `run-itp-data-gen` command now auto-detects Python version and uses Hydra-free mode for Python 3.14+

5. Run the commands for installing the Lean 4 interface as mentioned in [Quick Setup for Lean 4](#quick-setup-for-lean-4).

6. Add the following to your `.bashrc` file for Lean:
```
export PATH="/home/$USER/.elan/bin:$PATH"
```

## Running Simple Interactions:
1. Simple example for Lean 4 interaction:
```python
import os
from itp_interface.rl.proof_state import ProofState
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.simple_proof_env import ProofEnv
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy

project_folder = "src/data/test/lean4_proj"
file_path = "src/data/test/lean4_proj/Lean4Proj/Basic.lean"
# Code for building the Lean project
# cd src/data/test/lean4_proj && lake build
with os.popen(f"cd {project_folder} && lake build") as proc:
    print("Building Lean4 project...")
    print('-'*15 + 'Build Logs' + '-'*15)
    print(proc.read())
    print('-'*15 + 'End Build Logs' + '-'*15)
# Skip the above code if the project is already built
language = ProofAction.Language.LEAN4
theorem_name = "test3"
# theorem test3 (p q : Prop) (hp : p) (hq : q)
# : p ∧ q ∧ p :=
proof_exec_callback = ProofExecutorCallback(
    project_folder=project_folder,
    file_path=file_path,
    language=language,
    always_use_retrieval=False,
    keep_local_context=True
)
always_retrieve_thms = False
retrieval_strategy = ProofEnvReRankStrategy.NO_RE_RANK
env = ProofEnv("test_lean4", proof_exec_callback, theorem_name, retrieval_strategy=retrieval_strategy, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms)
proof_steps = [
    'apply And.intro',
    'exact hp',
    'apply And.intro',
    'exact hq',
    'exact hp'
]
with env:
    for proof_step in proof_steps:
        action = ProofAction(
            ProofAction.ActionType.RUN_TACTIC, 
            language, 
            tactics=[proof_step])
        state, _, next_state, _, done, info = env.step(action)
        if info.error_message is not None:
            print(f"Error: {info.error_message}")
        # This prints StateChanged, StateUnchanged, Failed, or Done
        print(info.progress)
        print('-'*30)
        if done:
            print("Proof Finished!!")
        else:
            s1 : ProofState = state
            s2 : ProofState = next_state
            print(f"Current Goal:")
            print('-'*30)
            for goal in s1.training_data_format.start_goals:
                hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                print(hyps)
                print('|- ', end='')
                print(goal.goal)
            print(f"="*30)
            print(f"Action: {proof_step}")
            print(f"="*30)
            print(f"Next Goal:")
            print('-'*30)
            for goal in s2.training_data_format.start_goals:
                hyps = '\n'.join([hyp for hyp in goal.hypotheses])
                print(hyps)
                print('|- ', end='')
                print(goal.goal)
            print(f"="*30)
```

2. One can also backtrack the last proof action using the following code:
```python
action = ProofAction(ProofAction.ActionType.BACKTRACK, language)
state, _, next_state, _, done, info = env.step(action)
```

3. The code for Coq interaction is similar to the Lean 4 interaction. The only difference is the language used in the `ProofAction` object. The language for Coq is `ProofAction.Language.COQ`. We also need to make sure that the Coq project is **built** before running the code. Please note that it is important to install the **correct version of Coq and Coq LSP** for the Coq project. The following code snippet shows how to interact with Coq:
```python
project_folder = "src/data/test/coq/custom_group_theory/theories"
file_path = "src/data/test/coq/custom_group_theory/theories/grpthm.v"

# IMPORTANT NOTE: The Coq project must be built before running the code.
# Create a switch for building the Coq project
if os.system("opam switch simple_grp_theory") != 0:
    cmds = [
        'opam switch create simple_grp_theory 4.14.2',
        'opam switch simple_grp_theory', 
        'eval $(opam env)',
        'opam repo add coq-released https://coq.inria.fr/opam/released',
        'opam pin add -y coq-lsp 0.1.8+8.18'
    ]
    final_cmd = ' && '.join(cmds)
    os.system(final_cmd)
# IMPORTANT NOTE: Make sure to switch to the correct switch before running the code.
os.system("opam switch simple_grp_theory && eval $(opam env)")
# Clean the project
os.system(f"cd {project_folder} && make clean")
# Build the project
with os.popen(f"cd {project_folder} && make") as proc:
    print("Building Coq project...")
    print('-'*15 + 'Build Logs' + '-'*15)
    print(proc.read())
    print('-'*15 + 'End Build Logs' + '-'*15)
# Skip the above code if the project is already built
language = ProofAction.Language.COQ # IMPORTANT NOTE: The language will change here to COQ
theorem_name = "algb_identity_sum"
# ....

# IMPORTANT NOTE: As a result of language change, the `ProofExecutorCallback` object will also change.
proof_exec_callback = ProofExecutorCallback(
    project_folder=project_folder,
    file_path=file_path,
    language=language, # The language will change here to COQ
    always_use_retrieval=False,
    keep_local_context=True
)

# IMPORTANT NOTE: The proof steps will also change for Coq.
proof_steps = [
    'intros.',
    'destruct a.',
    '- reflexivity.',
    '- reflexivity.'
]

# IMPORTANT NOTE: As a result of language change, the `action` object will also change.
action = ProofAction(
    ProofAction.ActionType.RUN_TACTIC, 
    language, # The language will change here to COQ
    tactics=[proof_step]
)
```

4. See the file [src/test/simple_env_test.py](src/test/simple_env_test.py) for more examples for Lean 4 interaction and Coq interaction.

## Generating Proof Step Data:

>NOTE: Make sure that you have installed the `itp-interface` package before running the following commands.

1.a. You need to run the following command to generate sample proof step data for Lean 4:
```
run-itp-data-gen --config-dir src/itp_interface/main/configs  --config-name simple_lean_data_gen
```
Check the `simple_lean_data_gen.yaml` configuration in the `src/itp_interface/main/configs` directory for more details. These config files are based on the `hydra` library (see [here](https://hydra.cc/docs/intro/)).

1.b. You need to run the following command to generate sample proof step data for Coq:
```
run-itp-data-gen --config-dir src/itp_interface/main/configs --config-name simple_coq_data_gen
```
Check the `simple_coq_data_gen.yaml` configuration in the `src/itp_interface/main/configs` directory for more details about where the generated data is stored and where the different ITP (Coq and Lean) projects are located in the file system.

## Important Note:
The ITP projects must be built before running proof step data generation. Make sure that the switch is set correctly while generating data for Coq projects because the Coq projects can be using different versions of Coq. Instructions for Coq project setup are listed in `src/itp_interface/main/configs/repo/coq_repos.yaml` file.

## Our Paper:

For more details, please refer to our paper: [ProofWala: Multilingual Proof Data Synthesis and Theorem-Proving](https://arxiv.org/abs/2502.04671).

```bibtex
@misc{thakur2025proofwala,
      title={${\rm P{\small ROOF}W{\small ALA}}$: Multilingual Proof Data Synthesis and Theorem-Proving}, 
      author={Amitayush Thakur and George Tsoukalas and Greg Durrett and Swarat Chaudhuri},
      year={2025},
      eprint={2502.04671},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.04671}, 
}
```