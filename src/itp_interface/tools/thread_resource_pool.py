#!/usr/bin/env python3
"""
Thread-safe resource pool for managing resources (like ports) without Ray.
"""

import threading
import typing
import time


class ThreadResourcePool:
    """
    Thread-safe resource pool that mimics RayResourcePoolActor behavior.
    Used as a fallback when Ray is not available.
    """

    def __init__(self, resources: typing.List[typing.Any]):
        """
        Initialize the resource pool.

        Args:
            resources: List of resources to manage (e.g., port numbers)
        """
        self._available_resources = list(resources)
        self._acquired_resources = []
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def wait_and_acquire(self, count: int = 1) -> typing.List[typing.Any]:
        """
        Wait for and acquire the specified number of resources.

        Args:
            count: Number of resources to acquire

        Returns:
            List of acquired resources
        """
        with self._condition:
            # Wait until enough resources are available
            while len(self._available_resources) < count:
                self._condition.wait()

            # Acquire resources
            acquired = []
            for _ in range(count):
                resource = self._available_resources.pop(0)
                self._acquired_resources.append(resource)
                acquired.append(resource)

            return acquired

    def release(self, resources: typing.List[typing.Any]):
        """
        Release resources back to the pool.

        Args:
            resources: List of resources to release
        """
        with self._condition:
            for resource in resources:
                if resource in self._acquired_resources:
                    self._acquired_resources.remove(resource)
                    self._available_resources.append(resource)

            # Notify waiting threads that resources are available
            self._condition.notify_all()

    def get_available_count(self) -> int:
        """
        Get the number of available resources.

        Returns:
            Number of available resources
        """
        with self._lock:
            return len(self._available_resources)

    def get_acquired_count(self) -> int:
        """
        Get the number of acquired resources.

        Returns:
            Number of acquired resources
        """
        with self._lock:
            return len(self._acquired_resources)
