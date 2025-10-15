import os
import random
import string

file_path = os.path.abspath(__file__)


def generate_random_string(length, allowed_chars=None):
    if allowed_chars is None:
        allowed_chars = string.ascii_letters + string.digits
    return ''.join(random.choice(allowed_chars) for _ in range(length))


def install_itp_interface():
    print("Installing itp_interface")
    itp_dir = os.path.dirname(os.path.dirname(file_path))
    tools_dir = os.path.join(itp_dir, "tools")
    repl_dir = os.path.join(tools_dir, "repl")
    assert os.path.exists(repl_dir), f"repl_dir: {repl_dir} does not exist"
    assert os.path.exists(os.path.join(repl_dir, "lean-toolchain")
                          ), f"lean-toolchain does not exist in {repl_dir}, build has failed"
    print("repl_dir: ", repl_dir)
    print("Building itp_interface")
    os.system(f"cd {repl_dir} && lake build repl")


def install_lean_repl():
    print("Updating Lean")
    itp_dir = os.path.dirname(os.path.dirname(file_path))
    tools_dir = os.path.join(itp_dir, "tools")
    repl_dir = os.path.join(tools_dir, "repl")
    assert os.path.exists(repl_dir), f"repl_dir: {repl_dir} does not exist"
    assert os.path.exists(os.path.join(repl_dir, "lean-toolchain")
                          ), f"lean-toolchain does not exist in {repl_dir}, build has failed"
    print("repl_dir: ", repl_dir)
    assert os.system("git --version") == 0, "git is not installed"
    print("[OK] git is installed")
    print("Checking if Lean version is set in environment variables as LEAN_VERSION")
    print("If not, defaulting to 4.24.0")
    lean_version = os.environ.get("LEAN_VERSION", "4.24.0")
    github_repo = "https://github.com/amit9oct/repl.git"
    if lean_version.strip() == "4.24.0":
        print("Lean version is set to 4.24.0, not cloning the REPL")
    else:
        # Clone the repl fresh
        print("Cloning the REPL fresh")
        os.system(f"rm -rf {repl_dir}")
        os.system(f"git clone {github_repo} {repl_dir}")
        # escape the version number
        lean_version_esc = lean_version.replace(".", "\.")
        print("Switching to the right REPL version", lean_version_esc)
        cmd_to_run = f"cd {repl_dir} && git log --grep \"v{lean_version_esc}\" --pretty=\"%h %s\""
        print("Running: ", cmd_to_run)
        output = os.popen(cmd_to_run).read()
        print("Output: ", output)
        if output == "":
            print(
                f"Could not find a commit with message containing {lean_version}")
            print("Probably this version does not exist in the git history of the REPL")
            lean_version = "4.24.0"
            print("Switching to v4.24.0 (latest default)")
            os.system(f"cd {repl_dir} && git checkout main")
        else:
            # Split on first space
            for line in output.split("\n"):
                if line:
                    commit, message = line.split(" ", 1)
                    if lean_version in message:
                        print(f"Switching to commit {commit}")
                        os.system(f"cd {repl_dir} && git checkout {commit}")
                        break
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
