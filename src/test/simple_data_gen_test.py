import unittest
import os
import subprocess
try:
    import ray
    from itp_interface.tools.ray_utils import RayResourcePoolActor, TimedRayExec, RayUtils
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ray = None
    RayResourcePoolActor = None
    TimedRayExec = None
    RayUtils = None

def pretty_print_file_contents(dir_path):
    print(f"Printing all files in the directory: {dir_path}")
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        if os.path.isfile(file_path):
            print('-'*50)
            print(f"Contents of {file_path}:")
            with open(file_path, "r") as file:
                print(file.read())

class TestDataGen(unittest.TestCase):
    def test_proof_step_data_gen(self):
        """
        Test that the 'run-itp-data-gen' command runs successfully with the given configuration.
        """
        # Construct the command as a single string.
        command = (
            "run-itp-data-gen --config-dir=src/itp_interface/main/configs "
            "--config-name=simple_lean_data_gen.yaml"
        )

        try:
            # Run the command using shell=True so that the shell does the PATH lookup.
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=700
            )
        except subprocess.TimeoutExpired as e:
            self.fail(f"'run-itp-data-gen' command timed out: {e}")
        except Exception as e:
            self.fail(f"'run-itp-data-gen' failed with unknown exception: {e}")

        # Check that the command exited with a return code of 0.
        self.assertEqual(
            result.returncode, 0,
            msg=f"'run-itp-data-gen' failed with return code {result.returncode}. Stderr: {result.stderr}"
        )

        # Print all the files in the .log/data_generation/benchmark/simple_benchmark_lean
        # directory to see what was generated.
        # Do a list and pick the last folder in the list as per the sorted order
        dirs = sorted(os.listdir(".log/data_generation/benchmark/simple_benchmark_lean"))
        print("Directories:", dirs)
        last_dir = dirs[-1]
        # Print the directory contents
        last_dir_path = os.path.join(".log/data_generation/benchmark/simple_benchmark_lean", last_dir)
        print("Last Directory Contents:", os.listdir(last_dir_path))
        train_data = os.path.join(last_dir_path, "train")
        list_files = os.listdir(train_data)
        print("Train Directory Contents:", list_files)
        data_files = [f for f in list_files if f.endswith(".json") and f.startswith("local_data_")]
        print("Data Files:", data_files)
        if len(data_files) == 0:
            # Print the last directory contents again
            pretty_print_file_contents(last_dir_path)
            print('='*50)
            # Open all the files in the train directory and print their contents for debugging
            pretty_print_file_contents(train_data)
               
        assert len(data_files) == 1, f"No files found in the train directory. Expected one file. Found: {data_files}"
        print(data_files[0])
        data_gen_file = os.path.join(train_data, data_files[0])
        print("Data Gen File:", data_gen_file)
        with open(data_gen_file, "r") as f:
            print(f.read())

def main():
    unittest.main()

if __name__ == '__main__':
    if HAS_RAY:
        os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
        object_store_memory_in_gb = 0.15
        memory_in_gb = 0.25
        ray_dashboard = RayUtils.init_ray(
            num_of_cpus=2, 
            object_store_memory_in_gb=object_store_memory_in_gb, 
            memory_in_gb=memory_in_gb)
    main()