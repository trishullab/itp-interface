#!/bin/bash

if [[ ! -d "src/itp_interface/scripts" ]]; then
    # Raise an error if the scripts directory is not present
    echo "Please run this script from the root of the repository, cannot find src/scripts"
    exit 1
fi
# Default Lean version
lean3_version="3.42.1"
lean4_version="4.7.0-rc2" # "nightly" # "4.7.0" # "stable"
lean_type="lean4" # For Lean 3
lean_repo="leanprover-community/lean" # For Lean 3
# Check if lean_version is passed as an argument
if [[ $# -eq 1 ]]; then
    lean_type=$1
    afp_folder=$HOME
fi

if [[ $# -eq 2 ]]; then
    lean_type=$1
    afp_folder=$2
fi

if [[ $lean_type == "lean" ]]; then
    lean_version=$lean3_version
    lean_repo="leanprover-community/lean"
elif [[ $lean_type == "lean4" ]]; then
    lean_version=$lean4_version
    lean_repo="leanprover/lean4"
else
    echo "Invalid Lean version, please choose between lean or lean4"
    exit 1
fi

# Don't run without activating conda
# Check if Conda is activated
conda_status=$(conda info | grep "active environment" | cut -d ':' -f 2 | tr -d '[:space:]')
if [[ $conda_status == "None" ]] || [[ $conda_status == "base" ]]; then
    echo "Please activate conda environment before running this script"
    exit 1
fi
echo "Setting up ITP Interface ..."
echo "[NOTE] The installation needs manual intervention on some steps. Please choose the appropriate option when prompted."
conda install pip
conda_bin=$(conda info | grep "active env location" | cut -d ':' -f 2 | tr -d '[:space:]')
pip_exe="$conda_bin/bin/pip"
ls -l $pip_exe
echo "Installing dependencies..."
echo "Installing Elan (Lean version manager) ..."
# # For Lean:
# # https://leanprover-community.github.io/install/debian_details.html
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
echo "Installed Elan (Lean version manager) successfully!"
source $HOME/.elan/env
echo "Installing $lean_type ($lean_repo:$lean_version)..."
elan toolchain install $lean_repo:$lean_version
elan override set $lean_repo:$lean_version
echo "Installed $lean_type ($lean_repo:$lean_version) successfully!"
export PATH=$PATH:$HOME/.elan/bin
echo "Clone all git submodules..."
git submodule update --init --recursive
echo "Cloned all git submodules successfully!"
echo "Building Coq's CompCert project..."
pushd ./src/data/benchmarks
echo "Installing Coq switch..."
opam switch create CompCert 4.07.1
eval $(opam env --switch=CompCert --set-switch)
opam repo add coq-released https://coq.inria.fr/opam/released
eval `opam config env`
opam pin -yn add coq 8.10.2
opam pin -yn add menhir 20190626
opam install -y coq-serapi coq menhir
if [[ ! -f "CompCert/Makefile.config" ]]; then
     (cd CompCert && ./configure x86_64-linux)
fi
make -C CompCert -j `nproc`
popd
echo "Building Coq's Simple Benchmark..."
pushd ./src/data/test/coq/custom_group_theory
cd theories
make
cd ..
popd
echo "Building Coq's Simple Benchmark done!"
echo "Coq's Setup complete!"

# Python dependencies
echo "Installing Python dependencies..."
$pip_exe install --user -r requirements.txt

# Only build Lean's Simple Benchmark if Lean 3 is selected
if [[ $lean_type == "lean" ]]; then
    echo "Building Lean's projects ..."
    (
        # Build Lean's projects
        echo "Building miniF2F..."
        echo "This may take a while... (don't underestimate the time taken to build miniF2F, meanwhile you can take a coffee break!)"
        pushd ./src/data/benchmarks
        set -euv
        cd miniF2F
        leanpkg configure
        leanproject get-mathlib-cache # This allows us to use .olean files from mathlib without building them again
        leanproject build
        popd
        echo "miniF2F built successfully!"
    ) || exit 1
    echo "Building Lean's Simple Benchmark..."
    pushd ./src/data/test/lean_proj
    leanproject build
    popd
    echo "Building Lean's Simple Benchmark done!"
    echo "Building Lean's projects done!"
    echo "Lean's Setup complete!"
    echo "Downloading ReProver benchmarks..."
    (
        # Download ReProver benchmarks
        echo "Running download.sh from src/data/benchmarks/lean-dojo..."
        ./src/data/benchmarks/lean-dojo/download.sh
        echo "Downloaded ReProver benchmarks successfully!"
    ) || exit 1
fi
if [[ $lean_type == "lean4" ]]; then
    echo "Building Lean 4's projects ..."
    (
        # Build Lean 4's projects
        echo "Building Lean 4's Simple Benchmark..."
        pushd ./src/data/test/lean4_proj
        lake build lean4_proj
        popd
        echo "Building Lean 4's Simple Benchmark done!"
    ) || exit 1
    echo "Building Lean 4's interface REPL..."
    (
        # Build Lean 4's interface REPL
        pushd ./imports/repl
        lake build repl
        popd
        echo "Lean 4's interface REPL built successfully!"
    ) || exit 1
    echo "Building Lean 4's Mathlib..."
    (
        # Build Lean 4's Mathlib
        echo "Building Lean 4's Mathlib..."
        echo "This may take a while... (don't underestimate the time taken to build Lean 4's Mathlib, meanwhile you can take a coffee break!)"
        pushd src/data/test/Mathlib
        lake exe cache get > /dev/null
        lake build Mathlib
        popd
        echo "Lean 4's Mathlib built successfully!"
    ) || exit 1
    echo "Building Lean 4's projects done!"
fi

# Download Isabelle
# First check if Isabelle is already installed
# default_isabelle_dir="$HOME/Isabelle2022" # Don't change this, otherwise PISA will not work
# issabell_installed=false
# if [[ -d "$default_isabelle_dir" ]]; then
#     echo "Isabelle is already installed at $default_isabelle_dir"
#     echo "If you want to reinstall Isabelle, please delete the directory $default_isabelle_dir and run this script again"
#     issabell_installed=true
# fi
# # Else download Isabelle
# if [[ $issabell_installed == false ]]; then
#     echo "Installing SDKMAN..."
#     curl -s "https://get.sdkman.io" | bash
#     echo "Installed SDKMAN successfully!"
#     echo "Installing Java 11 and SBT..."
#     source $HOME/.sdkman/bin/sdkman-init.sh
#     sdk install java 11.0.11-open
#     echo "Installed Java 11 successfully!"
#     echo "Installing SBT..."
#     sdk install sbt
#     echo "Installed SBT successfully!"
#     echo "Downloading Isabelle..."
#     wget https://isabelle.in.tum.de/website-Isabelle2022/dist/Isabelle2022_linux.tar.gz
#     echo "Downloaded Isabelle successfully!"
#     echo "Extracting Isabelle..."
#     # Extract Isabelle in the $default_isabelle_dir
#     tar -xzf Isabelle2022_linux.tar.gz -C $HOME
#     echo "Installed Isabelle successfully!"
#     echo "Cleaning up..."
#     rm Isabelle2022_linux.tar.gz
#     echo "Cleaned up!"
#     echo "Adding Isabelle to PATH..."
#     export PATH=$PATH:$default_isabelle_dir/bin
#     echo "export PATH=$PATH:$default_isabelle_dir/bin" >> ~/.bashrc
#     isabelle build -b -D $default_isabelle_dir/src/HOL/ -j 20
# fi
# pushd src/itp_interface/pisa
# sbt compile
# sbt assembly
# popd
# echo "Isabelle Setup complete!"

# echo "ITP Interface Setup complete!"

# echo "Building AFP"
# afp_installed_built=false
# if [[ -d "$afp_folder" ]]; then
#     echo "AFP is already installed at $afp_folder"
#     echo "If you want to reinstall AFP, please delete the directory $afp_folder and run this script again"
#     afp_installed_built=true
# fi

# afp_version="2022-12-06"
# if [[ $afp_installed_built == false ]]; then
#     mkdir -p $afp_folder
#     echo "Downloading AFP..."
#     wget https://www.isa-afp.org/release/afp-$afp_version.tar.gz
#     echo "Downloaded AFP successfully!"
#     echo "Extracting AFP..."
#     # Extract AFP in the $afp_folder
#     tar -xzf afp-$afp_version.tar.gz -C $afp_folder
#     echo "Extracted AFP successfully!"
#     echo "Cleaning up..."
#     afp_name=$(ls $afp_folder)
#     export AFP="$afp_folder/$afp_name/thys"
#     echo "$AFP"
#     rm afp-$afp_version.tar.gz
#     echo "Cleaned up!"
#     echo "Building AFP..."
#     export ISABELLE_BUILD_OPTIONS="timeout=3600"
#     export ML_OPTIONS="--maxheap 10000"
#     isabelle build -b -D $AFP -j 35
# fi

# echo "AFP Setup complete!"