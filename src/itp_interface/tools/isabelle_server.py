import sys
root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import signal
import subprocess
import time
import threading
import uuid
from itp_interface.tools.log_utils import setup_logger

# Conditional Ray import
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ray = None

class _IsabelleServerImpl(object):
    def __init__(self, log_filename: str, port: int = 8000):
        assert port > 0, "Port number must be greater than 0"
        assert port < 65536, "Port number must be less than 65536"
        self.log_filename = log_filename
        self.port = port
        logger_id = str(uuid.uuid4())
        with open(log_filename, "w") as f:
            f.write("")
        self.logger = setup_logger(f"isabelle_server_{logger_id}", log_filename)
        self.pid = None
        self.thread_id = None
        self.process_killed = False
        self.process = None

    def start_server(self):
        last_dir = os.path.dirname(root_dir)
        os.chdir(last_dir)

        jar_path = "itp_interface/pisa/target/scala-2.13/PISA-assembly-0.1.jar"
        assert os.path.exists(jar_path), "PISA jar file not found. Please build the project using 'sbt assembly' commnad"
        cmd = f"java -cp {jar_path} pisa.server.PisaOneStageServer{self.port}"
        # Start the server in a separate process
        cwd = os.getcwd()
        server_process = subprocess.Popen(
            cmd, 
            cwd=cwd,
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid) # Create a new process group so that we can kill the process and all its children
        pid = server_process.pid
        self.pid = pid
        # Log the process id
        self.logger.info(f"Server process id: {self.pid}")
        # Log the port number
        self.logger.info(f"Server port: {self.port}")
        self.logger.info("Waiting for server to start")
        time.sleep(5)
        server_running = self.check_server_running()
        if not server_running:
            self.logger.error(f"Server is not running on port {self.port}")
        else:
            self.process = server_process
            thread = threading.Thread(target=self._server_logging_thread)
            thread.start()
            thread_id = thread.ident
            self.thread_id = thread_id
        pass

    def check_server_running(self):
        # log the netsat result netstat -nlp | grep :{port}
        netstat_cmd = f"netstat -nlp | grep :{self.port}"
        self.logger.info(f"Netstat command: {netstat_cmd}")
        output = subprocess.run(netstat_cmd, shell=True, capture_output=True)
        self.logger.info(f"Netstat output: {output}")
        output_str = output.stdout.decode("utf-8")
        self.logger.info(f"Netstat output stdout: {output_str}")        
        server_running = "tcp" in output_str
        return server_running

    def _server_logging_thread(self):
        # Keep checking the server is running
        while not self.process_killed:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                self.logger.info(line)
            except:
                self.logger.info("Stdout is closed")
                time.sleep(1)
        self.logger.info("Server is shut down")
        # Print the environment variables
        self.logger.info(f"Environment variables at shutdown time: {os.environ}")
        time.sleep(1)
        pass

    def stop_server(self):
        self.process_killed = True
        try:
            # Kill the server process
            os.killpg(self.pid, signal.SIGTERM)
        except:
            pass
        thread_id = self.thread_id
        # Find the right thread object
        for thread in threading.enumerate():
            if thread.ident == thread_id:
                thread.join(5)
                break
        pass

# Create Ray remote version if Ray is available
if HAS_RAY:
    IsabelleServer = ray.remote(_IsabelleServerImpl)
else:
    IsabelleServer = _IsabelleServerImpl