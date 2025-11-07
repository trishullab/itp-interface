import os
import random
import string
import logging
import traceback
from itp_interface.tools.tactic_parser import build_tactic_parser_if_needed

file_path = os.path.abspath(__file__)

logging.basicConfig(level=logging.INFO)
# Create a console logger
logger = logging.getLogger(__name__)

def generate_random_string(length, allowed_chars=None):
    if allowed_chars is None:
        allowed_chars = string.ascii_letters + string.digits
    return ''.join(random.choice(allowed_chars) for _ in range(length))


def install_itp_interface():
    print("Installing itp_interface")
    itp_dir = os.path.dirname(os.path.dirname(file_path))
    tools_dir = os.path.join(itp_dir, "tools")
    tactic_parser_dir = os.path.join(tools_dir, "tactic_parser")
    assert os.path.exists(tactic_parser_dir), f"tactic_parser_dir: {tactic_parser_dir} does not exist"
    assert os.path.exists(os.path.join(tactic_parser_dir, "lean-toolchain")), f"lean-toolchain does not exist in {tactic_parser_dir}, build has failed"
    print("tactic_parser_dir: ", tactic_parser_dir)
    with open(os.path.join(tactic_parser_dir, "lean-toolchain"), "r") as f:
        lean_toolchain_content = f.read().strip()
    print("Lean toolchain version for tactic_parser: ", lean_toolchain_content)
    print(f"LEAN_VERSION set: {os.environ.get('LEAN_VERSION', 'Not Set')}")
    print("Building itp_interface")
    try:
        build_tactic_parser_if_needed(logger)
    except Exception:
        # print the stack trace
        traceback.print_exc()
        raise

def install_lean_repl():
    print("Updating Lean")
    print("Checking if Lean version is set in environment variables as LEAN_VERSION")
    print("If not, defaulting to 4.24.0")
    lean_version = os.environ.get("LEAN_VERSION", "4.24.0")

    # Make sure that .elan is installed
    print("Checking if .elan is installed")
    if os.system("elan --version") == 0:
        print("[OK] .elan is installed")
    else:
        print("Installing .elan")
        elan_url = "https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"
        os.system(f"curl -sSL {elan_url} -sSf | sh -s -- -y")
        print("[OK] .elan installed")

        lean_repo = "leanprover/lean4"
        os.system("source $HOME/.elan/env")
        os.system(f"echo 'Installing Lean 4 ({lean_repo}:{lean_version})...'")
        os.system(f"elan toolchain install {lean_repo}:{lean_version}")
        os.system(f"elan override set {lean_repo}:{lean_version}")
        os.system(
            f"echo 'Installed Lean 4 ({lean_repo}:{lean_version}) successfully!'")
        os.system("export PATH=$PATH:$HOME/.elan/bin")

        os.system("ls -l $HOME/.elan/bin")

        assert os.system(
            "export PATH=$PATH:$HOME/.elan/bin && lean --version") == 0, "Lean 4 is not installed aborting"
        print("[OK] Lean 4 installed successfully")

    print("NOTE: Please run `install-itp-interface` to finish the installation")
