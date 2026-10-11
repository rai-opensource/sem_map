# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
CUDA Time Buffer for timestamp-based latest-item access across processes.
Copied from visual-perception-engine with minimal modifications.
"""
import os
import time
import socket
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Literal
from contextlib import contextmanager

import torch
# cuda-python 13.x API
from cuda.bindings import driver as cuda

from sem_map.data_pipeline.cuda_utils import CUDASharedMemorySlotWithIDAndTimestamp, checkCudaErrors


@contextmanager
def nonblocking(lock):
    """Non-blocking lock acquisition context manager. Yields True if lock acquired, False otherwise."""
    locked = lock.acquire(False)
    try:
        yield locked
    finally:
        if locked:
            lock.release()


class TimeBufferInterface(ABC):
    """Abstract interface for timestamp-based buffers."""
    @abstractmethod
    def put(self, item) -> None:
        """Put item in the buffer, replace the oldest element if needed and return the respective timestamp."""
        pass

    @abstractmethod
    def get(self, last_timestamp) -> None | tuple[float, torch.Tensor]:
        """Get the most recent item with later timestamp than last_timestamp."""
        pass


class CUDATimeBuffer(TimeBufferInterface):
    """Buffer that will always output the most recent item that is newer than requested timestep.
    Adding new element to the buffer will overwrite the oldest element.

    Designed for one-to-many scenario: one process writes, many processes can read.
    Multiple readers serialize on reads because they all access the same newest slot.
    The serialization is acceptable since CUDA memcopy is the performance bottleneck.

    THREAD SAFETY: Uses internal mp.Lock per slot. Safe for multiple producers/consumers with VPE's implementation.
    """

    def __init__(
        self,
        max_size: int,
        dtype: torch.dtype,
        input_device: Literal["cpu", "cuda"],
        output_device: Literal["cpu", "cuda"],
        data_signature: dict[str, tuple[int]],
        is_original_process: bool = True,
        device_id: int = 0,
        add_batch_dim: bool = False,
    ):
        self.max_size = max_size
        self.dtype = dtype
        self.input_device = input_device
        self.output_device = output_device
        self.data_signature = data_signature
        assert all([None not in v for v in data_signature.values()]), (
            "None (wildcard shape) is not allowed in this data structure"
        )
        self.is_original_process = is_original_process
        self.device_id = device_id
        self.add_batch_dim = add_batch_dim
        self.logger = mp.get_logger()

        self._buffer = [
            CUDASharedMemorySlotWithIDAndTimestamp(
                data_signature, dtype, is_original_process, device_id=device_id, add_batch_dim=add_batch_dim
            )
            for _ in range(max_size)
        ]

        self._output_slot = (
            None  # memory slot where the output will be written, initialized the first time it is needed
        )

        self._position_locks = [mp.Lock() for _ in range(max_size)]
        self._newest_idx = mp.Value("i", max_size - 1)
        self._oldest_idx = mp.Value("i", 0)

        self._socket_lock = (
            mp.Lock()
        )  # to ensure that child processes receive shared memory handles only once and in order
        parent_sock, child_sock = socket.socketpair()
        os.set_inheritable(parent_sock.fileno(), True)
        os.set_inheritable(child_sock.fileno(), True)
        self._parent_sock = parent_sock
        self._child_sock = child_sock

        self.writeLock = LockedBufferWriter(self)

    def __getstate__(self):
        """Return the state of the object for pickling. Intended to be used only when passing the object to a child process.
        The new state will not contain the memory pointers, nor the parent socket.
        Furhtermore, the new state will contain a flag indicating that this is not the original process.
        """
        state = self.__dict__.copy()
        state["is_original_process"] = False
        state.pop(
            "_parent_sock"
        )  # removing parent socket is not necessary but it is a good practice to separete original and child processes
        state.pop("_output_slot")  # the output slot will be initialized in the child process
        return state

    def __setstate__(self, state):
        """Set the state of the object after unpickling. Intended to be used only when passing the object to a child process.
        The new state will contain a flag indicating that this is not the original process.
        """
        self.__dict__.update(state)
        self.is_original_process = False
        self._parent_sock = None
        self._output_slot = None

    @property
    def device(self):
        """Get the device of the buffer slots."""
        return self._buffer[0].device

    def close(self):
        """Close all buffer slots and sockets."""
        for slot in self._buffer:
            slot.close()

        # the sockets should be closed after sending and receiving the shared memory handles
        # if the handles are not transmitted, the sockets will be closed in the destructor
        self._child_sock.close()
        if self._parent_sock is not None:
            self._parent_sock.close()

    def send_shareable_handles(self, destination_pids: list[int], close_sockets: bool = True):
        """Send the handles of the shared memory slots over a local connection to all child processes.
        This method should be called only from the original process."""
        transmit_error = True
        try:
            for pid in destination_pids:
                for slot in self._buffer:
                    slot.send_shareable_handles(self._parent_sock, pid)
            transmit_error = False
        finally:
            if close_sockets:
                self._parent_sock.close()
                self._child_sock.close()  # parent process also has reference to the child socket

        if transmit_error:
            raise RuntimeError("Failed to transmit the shared memory handles.")

    def recv_shareable_handles(self, close_sockets: bool = True, timeout: float = 120.0):
        """Receive the handles of the shared memory slots over a local connection.
        This method should be called only from the child process.
        It needs to be called after the object is unpickled.
        Otherwise the child process will not have access to the shared memory.

        A timeout is applied to the socket so a worker cannot block forever
        holding a CUDA context if the parent failed mid-handle-exchange."""
        transmit_error = True
        try:
            # There is multiple child processes, we need to ensure that each child process
            # receives the shared memory handles for all slots in the buffer only once and
            # that it happens in order.
            with self._socket_lock:
                self._child_sock.settimeout(timeout)
                for slot in self._buffer:
                    slot.receive_shareable_handles(self._child_sock)
            transmit_error = False
        except (socket.timeout, TimeoutError) as e:
            raise RuntimeError(
                f"Timed out ({timeout}s) receiving shared memory handles - "
                f"parent process likely failed during handle exchange."
            ) from e
        finally:
            if close_sockets:
                self._child_sock.close()

        if transmit_error:
            raise RuntimeError("Failed to receive the shared memory handles.")

    def initialize_output_slot(self):
        """Initialize persistent memory destination where data read from the buffer will be written.

        This creates a reusable output buffer to avoid reallocating memory on every read operation.
        The buffer is used as the destination when copying data from shared CUDA memory slots.
        """
        self._output_slot = {}
        for k, v in self._buffer[0].effective_data_signature.items():
            self._output_slot[k] = torch.empty(*v, dtype=self.dtype, device=self.device)

    def put(self, item: dict[str, torch.Tensor], item_id: float, sync=True) -> None:
        """Write item to buffer, overwriting oldest slot if full."""
        with self._oldest_idx.get_lock():
            oldest_idx = self._oldest_idx.value
            self._oldest_idx.value = (oldest_idx + 1) % self.max_size

        with self._position_locks[oldest_idx]:
            self._buffer[oldest_idx].write(item, item_id, sync)

        with self._newest_idx.get_lock():
            self._newest_idx.value = oldest_idx

    def get(self, last_timestamp: float, sync: bool = False) -> None | tuple[int, float, torch.Tensor]:
        """Get newest item with timestamp > last_timestamp. Returns None if no new data."""
        # initialize the output slot the first time it is needed
        if self._output_slot is None or self.output_device == "cpu":
            self._output_slot = self._buffer[0].get_non_shared_empty_memory_slot(self.output_device)

        newest_idx = self._newest_idx.value
        with self._position_locks[newest_idx]:
            timestamp = self._buffer[newest_idx].get_timestamp()
            if timestamp == float("-inf") or timestamp <= last_timestamp:
                return None
            return self._buffer[newest_idx].read(self._output_slot, sync)

    def get_nowait(self, last_timestamp: float, sync: bool = False) -> None | tuple[int, float, torch.Tensor]:
        """Non-blocking version of get(). Returns None if lock unavailable or no new data."""
        # initialize the output slot the first time it is needed
        if self._output_slot is None:
            self._output_slot = self._buffer[0].get_non_shared_empty_memory_slot(self.output_device)

        newest_idx = self._newest_idx.value
        with nonblocking(self._position_locks[newest_idx]) as locked:
            if locked:
                timestamp = self._buffer[newest_idx].get_timestamp()
                if timestamp == float("-inf") or timestamp <= last_timestamp:
                    return None
                return self._buffer[newest_idx].read(self._output_slot, sync)
            else:
                return None


class LockedBufferWriter:
    """Context manager for thread-safe writing to CUDATimeBuffer."""

    def __init__(self, buffer: CUDATimeBuffer):
        self.buffer = buffer
        self.current_lock = None
        self.oldest_idx = None
        self.identifier = None

    def __call__(self, identifier):
        """Set identifier for this write operation."""
        self.identifier = identifier
        return self

    def __enter__(self):
        """Acquire oldest slot for writing. Returns memory_dict for direct writes."""
        with self.buffer._oldest_idx.get_lock():
            self.oldest_idx = self.buffer._oldest_idx.value
            self.buffer._oldest_idx.value = (self.oldest_idx + 1) % self.buffer.max_size

        self.current_lock = self.buffer._position_locks[self.oldest_idx]
        self.current_lock.acquire()

        return self.buffer._buffer[self.oldest_idx].memory_dict

    def __exit__(self, exc_type, exc_value, traceback):
        """Release lock and update timestamp/newest_idx only if write succeeded."""
        if exc_type is None:
            # Write succeeded - update timestamp and newest_idx
            self.buffer._buffer[self.oldest_idx].timestamp.value = time.perf_counter()
            self.buffer._buffer[self.oldest_idx].id.value = self.identifier
            with self.buffer._newest_idx.get_lock():
                self.buffer._newest_idx.value = self.oldest_idx
        else:
            # Write failed - invalidate slot to prevent reading corrupted/partial data
            self.buffer._buffer[self.oldest_idx].timestamp.value = float("-inf")

        self.current_lock.release()
