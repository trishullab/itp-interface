import os
import pty
import subprocess
import json
import select
import time
import logging

class ProcessInterface:
    buffer_size = 1024
    def __init__(self, command, cwd, logger: logging.Logger = None, log_level=logging.INFO):
        """
        Note: This class is not thread-safe. It is intended to be used in a single-threaded environment.
        """
        master, slave = pty.openpty()
        self.process = subprocess.Popen(
            command.split(),
            cwd=cwd,
            stdin=slave,
            stdout=slave,
            stderr=subprocess.STDOUT,
            text=True
        )
        os.close(slave)
        self.master = master
        self.buffer = ''  # Buffer to accumulate data from stdout
        self.sent_commands = ''  # Buffer to track sent commands
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def send_command(self, command_dict):
        json_command = json.dumps(command_dict) + '\n\n'
        normalized_command = json_command.replace('\r\n', '\n')  # Normalize newlines
        os.write(self.master, normalized_command.encode())
        self.logger.debug(f"Sent: {normalized_command}")
        self.sent_commands += normalized_command  # Keep track of normalized sent commands

    def read_response(self, timeout=10):
        sent_command = self.sent_commands
        end_time = time.time() + timeout
        while time.time() < end_time:
            readable, _, _ = select.select([self.master], [], [], timeout)
            if readable:
                data = os.read(self.master, ProcessInterface.buffer_size).decode()
                normalized_data = data.replace('\r\n', '\n')  # Normalize received data
                self.buffer += normalized_data
                # Clean buffer by removing echoed commands before parsing
                if self.buffer.startswith(self.sent_commands):
                    self.buffer = self.buffer[len(self.sent_commands):]  # Remove echoed commands
                    self.sent_commands = ''  # Clear sent commands buffer after removing

                try:
                    # Attempt to parse the clean buffer as JSON
                    response = json.loads(self.buffer.strip())
                    self.logger.debug(f"Received: {response}")
                    self.buffer = ''  # Clear buffer after successful parse
                    return response
                except json.JSONDecodeError:
                    continue  # Continue if JSON is incomplete or invalid
            else:
                self.logger.debug("Timeout: No response received.")
                raise TimeoutError(f"Could complete \"{sent_command}\" within {timeout} seconds.")

    def close(self):
        os.close(self.master)
        self.process.terminate()
        self.process.wait()

# Process interface test
if __name__ == "__main__":
    #.lake/bin/repl
    repl_path = "./imports/repl/.lake/build/bin/repl"
    lean4_proj_path = "./src/data/test/lean4_proj"
    abs_repl_path = os.path.abspath(repl_path)
    interface = ProcessInterface(f"lake env {abs_repl_path}", lean4_proj_path, log_level=logging.DEBUG)
    try:
        interface.send_command({"path": "Lean4Proj/Basic.lean", "allTactics": True})
        response = interface.read_response(1)
        print(response)
    finally:
        interface.close()