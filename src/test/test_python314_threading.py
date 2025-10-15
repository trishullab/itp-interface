#!/usr/bin/env python3
"""
Test to verify Python 3.14 free-threading (GIL-free) performance.
This test checks if computational threads actually run in parallel and faster than sequential execution.
"""

import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib


def cpu_intensive_task(n: int, iterations: int = 1000000) -> int:
    """
    CPU-intensive task that performs actual computation.
    Uses cryptographic hashing to ensure it's CPU-bound, not memory-bound.

    Args:
        n: Task identifier
        iterations: Number of hash computations to perform

    Returns:
        Task identifier (for verification)
    """
    result = 0
    data = f"task_{n}".encode()

    for i in range(iterations):
        # Perform CPU-intensive hashing
        h = hashlib.sha256(data + str(i).encode())
        result ^= int.from_bytes(h.digest()[:4], byteorder='big')

    return n


def run_sequential(num_tasks: int, iterations: int) -> float:
    """Run tasks sequentially and measure time."""
    start_time = time.time()

    results = []
    for i in range(num_tasks):
        result = cpu_intensive_task(i, iterations)
        results.append(result)

    end_time = time.time()
    return end_time - start_time


def run_parallel(num_tasks: int, iterations: int, max_workers: int) -> float:
    """Run tasks in parallel using ThreadPoolExecutor and measure time."""
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(cpu_intensive_task, i, iterations) for i in range(num_tasks)]
        results = [f.result() for f in futures]

    end_time = time.time()
    return end_time - start_time


def test_gil_free_threading():
    """
    Test if Python 3.14 can run computational threads in parallel without GIL.

    Expected behavior:
    - Python < 3.13: Sequential and parallel times should be similar (GIL blocks parallelism)
    - Python >= 3.14 (free-threading): Parallel should be significantly faster than sequential
    """
    print("=" * 80)
    print("Python 3.14 Free-Threading (GIL-free) Performance Test")
    print("=" * 80)
    print(f"\nPython version: {sys.version}")
    print(f"Python version info: {sys.version_info}")

    # Check if running Python 3.14+
    is_python_314_plus = sys.version_info >= (3, 14)
    print(f"Is Python 3.14+: {is_python_314_plus}")

    # Check if GIL is disabled (Python 3.13+ with free-threading build)
    try:
        gil_disabled = sys._is_gil_enabled is not None and not sys._is_gil_enabled()
    except AttributeError:
        gil_disabled = False

    print(f"GIL disabled: {gil_disabled}")
    print()

    # Test parameters
    num_tasks = 4
    iterations = 500000  # Reduced for faster testing
    max_workers = 4

    print(f"Test configuration:")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Iterations per task: {iterations:,}")
    print(f"  Max workers (threads): {max_workers}")
    print()

    # Run sequential test
    print("Running sequential execution...")
    sequential_time = run_sequential(num_tasks, iterations)
    print(f"  Sequential time: {sequential_time:.3f} seconds")
    print()

    # Run parallel test
    print("Running parallel execution (ThreadPoolExecutor)...")
    parallel_time = run_parallel(num_tasks, iterations, max_workers)
    print(f"  Parallel time: {parallel_time:.3f} seconds")
    print()

    # Calculate speedup
    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")
    print()

    # Analysis
    print("=" * 80)
    print("Analysis:")
    print("=" * 80)

    if speedup >= 2.0:
        print("✓ EXCELLENT: Parallel execution is significantly faster!")
        print(f"  This indicates true parallel execution (likely GIL-free Python 3.14+)")
        print(f"  Speedup: {speedup:.2f}x")
        status = "PASS"
    elif speedup >= 1.3:
        print("✓ GOOD: Parallel execution shows moderate speedup")
        print(f"  This suggests some level of parallelism")
        print(f"  Speedup: {speedup:.2f}x")
        status = "PASS"
    elif speedup >= 0.8:
        print("⚠ WARNING: Parallel execution shows minimal or no speedup")
        print(f"  This is expected with GIL-enabled Python (< 3.14 or without free-threading)")
        print(f"  Speedup: {speedup:.2f}x")
        if is_python_314_plus:
            print(f"  Note: You're on Python {sys.version_info.major}.{sys.version_info.minor}, but GIL may still be enabled.")
            print(f"  Check if Python was built with --disable-gil flag")
            status = "WARNING"
        else:
            print(f"  Expected behavior for Python {sys.version_info.major}.{sys.version_info.minor}")
            status = "EXPECTED"
    else:
        print("✗ FAIL: Parallel execution is slower than sequential")
        print(f"  This suggests thread overhead without parallelism benefit")
        print(f"  Speedup: {speedup:.2f}x")
        status = "FAIL"

    print()
    print("=" * 80)
    print(f"Test Status: {status}")
    print("=" * 80)
    print()

    # Recommendations
    if not gil_disabled and is_python_314_plus:
        print("Recommendations:")
        print("  To enable free-threading in Python 3.14+:")
        print("  1. Build Python with: ./configure --disable-gil")
        print("  2. Or use: python3.14t (free-threading build)")
        print()
    elif not is_python_314_plus:
        print("Recommendations:")
        print(f"  You're using Python {sys.version_info.major}.{sys.version_info.minor}")
        print("  To test free-threading, upgrade to Python 3.14+ with GIL disabled")
        print()

    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "python_version": sys.version_info,
        "gil_disabled": gil_disabled,
        "status": status
    }


if __name__ == "__main__":
    result = test_gil_free_threading()

    # Exit with appropriate code
    if result["status"] in ["PASS", "EXPECTED"]:
        sys.exit(0)
    elif result["status"] == "WARNING":
        sys.exit(0)  # Warning is acceptable
    else:
        sys.exit(1)  # Fail
