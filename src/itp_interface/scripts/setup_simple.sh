#!/bin/bash

# This setup script is for setting up itp-interface for INT. Use script.sh for other languages like Lean, Coq, Isabelle.

if [[ ! -d "src/itp_interface/scripts" ]]; then
    # Raise an error if the scripts directory is not present
    echo "Please run this script from the root of the repository, cannot find src/scripts"
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

# Python dependencies
echo "Installing Python dependencies..."
$pip_exe install --user -r requirements.txt

echo "Simple ITP Interface Setup complete!"