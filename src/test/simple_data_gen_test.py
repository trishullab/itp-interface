import unittest
import os
import subprocess

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
            self.fail(f"Error running 'proof-wala-search': {e}")

        # Check that the command exited with a return code of 0.
        self.assertEqual(
            result.returncode, 0,
            msg=f"'run-itp-data-gen' failed with return code {result.returncode}. Stderr: {result.stderr}"
        )

        # Print all the files in the .log/data_generation/benchmark/simple_benchmark_lean
        # directory to see what was generated.
        # Do a list and pick the last folder in the list as per the sorted order
        dirs = sorted(os.listdir(".log/data_generation/benchmark/simple_benchmark_lean"))
        print(dirs)
        last_dir = dirs[-1]
        train_data = os.path.join(".log/data_generation/benchmark/simple_benchmark_lean", last_dir, "train")
        data_gen_file = os.path.join(train_data, "local_data_0000000025.json")
        print("Data Gen File:", data_gen_file)
        with open(data_gen_file, "r") as f:
            print(f.read())

def main():
    unittest.main()

if __name__ == '__main__':
    main()