import os

file_path = os.path.abspath(__file__)
def install_itp_interface():
    print("Installing itp_interface")
    itp_dir = os.path.dirname(os.path.dirname(file_path))
    tools_dir = os.path.join(itp_dir, "tools")
    repl_dir = os.path.join(tools_dir, "repl")
    assert os.path.exists(repl_dir), f"repl_dir: {repl_dir} does not exist"
    assert os.path.exists(os.path.join(repl_dir, "lean-toolchain")), f"lean-toolchain does not exist in {repl_dir}, build has failed"
    print("repl_dir: ", repl_dir)
    print("Building itp_interface")
    os.system(f"cd {repl_dir} && lake build repl")