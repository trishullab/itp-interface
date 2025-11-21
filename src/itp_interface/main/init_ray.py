def main():
    # Start the ray cluster
    from filelock import FileLock
    import json
    import os
    import ray
    import logging
    import time
    import sys
    import argparse
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--num_cpus", type=int, default=10)
    argument_parser.add_argument("--object_store_memory", type=int, default=150*2**30)
    argument_parser.add_argument("--memory", type=int, default=300*2**30)
    argument_parser.add_argument("--metrics_report_interval_ms", type=int, default=3*10**8)
    args = argument_parser.parse_args()
    root_dir = f"{os.path.abspath(__file__).split('itp_interface')[-2]}"
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    os.makedirs(".log/locks", exist_ok=True)
    os.makedirs(".log/ray", exist_ok=True)
    ray_was_started = False
    pid = os.getpid()
    print("Initializing Ray")
    print("PID: ", pid)
    ray_session_path = ".log/ray/session_latest" if os.environ.get("RAY_SESSION_PATH") is None else os.environ.get("RAY_SESSION_PATH")
    # Try to first acquire the lock
    file_path = ".log/locks/ray.lock"
    # set RAY_SESSION_PATH environment variable
    os.environ["RAY_SESSION_PATH"] = ray_session_path
    # set lock file path
    os.environ["RAY_LOCK_FILE_PATH"] = file_path
    temp_lock = FileLock(file_path)
    if os.path.exists(ray_session_path):
        # try to acquire the lock for reading
        try:
            temp_lock.acquire(timeout=10)
            temp_lock.release()
        except:
            with open(ray_session_path, "r") as f:
                ray_session = f.read()
                ray_session = json.loads(ray_session)
            ray_address = ray_session["address"]
            # ray.init(address=ray_address)
            print("Ray was already started")
            print("Ray session: ", ray_session)
            sys.exit(0)
    with FileLock(file_path):
        if os.path.exists(ray_session_path):
            # Remove the ray_session_path
            os.remove(ray_session_path)
        os.environ["RAY_INITIALIZED"] = "1"
        ray_session = ray.init(
            num_cpus=args.num_cpus, 
            object_store_memory=args.object_store_memory, 
            _memory=args.memory, 
            logging_level=logging.CRITICAL, 
            ignore_reinit_error=False, 
            log_to_driver=False, 
            configure_logging=False,
            _system_config={"metrics_report_interval_ms": args.metrics_report_interval_ms})
        ray_session = dict(ray_session)
        ray_session["main_pid"] = pid
        print("Ray session: ", ray_session)
        with open(ray_session_path, "w") as f:
            f.write(json.dumps(ray_session))
        ray_was_started = True
        print("Ray was started")
        print("Ray session: ", ray_session)
        # Flush the stdout buffer
        sys.stdout.flush()
        while ray_was_started:
            # Keep the ray cluster alive till killed
            time.sleep(10000)

if __name__ == "__main__":
    main()