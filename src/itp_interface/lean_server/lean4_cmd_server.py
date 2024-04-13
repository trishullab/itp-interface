import os
import pty
import subprocess
import json

class ProcessInterface:
    def __init__(self, command, cwd):
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

    def send_command(self, command_dict):
        json_command = json.dumps(command_dict) + '\n\n'
        normalized_command = json_command.replace('\r\n', '\n')  # Normalize newlines
        os.write(self.master, normalized_command.encode())
        self.sent_commands += normalized_command  # Keep track of normalized sent commands

    def read_response(self):
        while True:
            data = os.read(self.master, 1024).decode()
            normalized_data = data.replace('\r\n', '\n')  # Normalize received data
            self.buffer += normalized_data
            # Clean buffer by removing echoed commands before parsing
            if self.buffer.startswith(self.sent_commands):
                self.buffer = self.buffer[len(self.sent_commands):]  # Remove echoed commands
                self.sent_commands = ''  # Clear sent commands buffer after removing

            try:
                # Attempt to parse the clean buffer as JSON
                response = json.loads(self.buffer.strip())
                print(f"Received: {response}")
                self.buffer = ''  # Clear buffer after successful parse
                return response
            except json.JSONDecodeError:
                continue  # Continue if JSON is incomplete or invalid

    def close(self):
        os.close(self.master)
        self.process.terminate()
        self.process.wait()

# Example usage:
if __name__ == "__main__":
    interface = ProcessInterface("lake exe repl", "/home/amthakur/Project/itp-interface/imports/repl")
    try:
        interface.send_command({"path": "/home/amthakur/Project/itp-interface/src/data/test/lean4_proj/Lean4Proj/Basic.lean", "allTactics": True})
        print(interface.read_response())
    finally:
        interface.close()
