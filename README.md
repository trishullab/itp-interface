# itp-interface
Generic interface for hooking up to any Interactive Theorem Prover (ITP) and collecting data for training ML models for AI in formal theorem proving. 

## Setup Steps:
1. Install OCaml first. Use the instructions here: https://opam.ocaml.org/doc/Install.html . The opam version used in this project is 2.1.3 (OCaml 4.14.0). Note that OCaml officially only supports Linux installations. One can use WSL on Windows machines.

2. Run the following to install Coq on Linux. The Coq version used in this project is <= 8.15. 
```
sudo apt install build-essential unzip bubblewrap
sh <(curl -sL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)
```

3. Add the following to your `.bashrc` file: (sometimes the path `~/.opam/default` might not exist, so use the directory with version number present in the `~/.opam` directory)
```
export PATH="/home/$USER/.opam/default/bin:$PATH"
```

4. Create a `Miniconda` environment and activate it.


5. Change to the project root directory, and run the setup script i.e. `./src/scripts/setup.sh` from the root directory.

6. Add the following to your `.bashrc` file for Lean:
```
export PATH="/home/$USER/.elan/bin:$PATH"
```

7. You need to run the following command to generate sample proof step data for Lean 4:
```
python src/itp_interface/main/run_tool.py --config-name simple_lean_data_gen
```
Check the `simple_lean_data_gen.yaml` configuration in the `src/itp_interface/configs` directory for more details. These config files are based on the `hydra` library (see [here](https://hydra.cc/docs/intro/)).

8. You need to run the following command to generate sample proof step data for Coq:
```
python src/itp_interface/main/run_tool.py --config-name simple_coq_data_gen
```
Check the `simple_coq_data_gen.yaml` configuration in the `src/itp_interface/configs` directory for more details.

## Important Notes:
The ITP projects must be built before running proof step data generation. Make sure that the switch is set correctly while generating data for Coq projects because the Coq projects can be using different versions of Coq. Instructions for Coq project setup are listed in `src/itp_interface/main/config/repo/coq_repos.yaml` file.