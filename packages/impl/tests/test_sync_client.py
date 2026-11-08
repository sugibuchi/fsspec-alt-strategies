"""Tests of `SyncClientSupport`: a sync client's calls in a filesystem's own threads."""

from __future__ import annotations

import asyncio
import contextvars
import gc
import threading

from fsspec.implementations.memory import MemoryFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.sync_client import SyncClientSupport
from fsspec_alt_strategies_impl.wrapper import SyncTreeWrapper


class _Client(SyncClientSupport):
    def __init__(self, operation_concurrency: object = 2) -> None:
        self.operation_concurrency = operation_concurrency


async def test_calls_run_in_its_own_threads() -> None:
    client = _Client()
    names = await asyncio.gather(
        *(
            client._run_in_thread(lambda: threading.current_thread().name)
            for _ in range(4)
        )
    )
    assert all(n.startswith("_Client") for n in names)
    assert client._thread_pool()._max_workers == 2
    client.close()


async def test_a_shared_semaphore_leaves_the_count_to_the_default() -> None:
    client = _Client(asyncio.Semaphore(3))
    assert client._thread_count() is None
    await client._run_in_thread(lambda: None)
    client.close()


async def test_the_context_goes_with_the_call() -> None:
    var: contextvars.ContextVar[str] = contextvars.ContextVar("var")
    var.set("caller's")
    client = _Client()
    assert await client._run_in_thread(var.get) == "caller's"
    client.close()


async def test_close_shuts_the_threads_down_and_a_call_after_makes_them_again() -> None:
    client = _Client()
    await client._run_in_thread(lambda: None)
    first = client._thread_pool()
    client.close()
    assert client._executor is None
    assert first._shutdown
    await client._run_in_thread(lambda: None)
    assert client._thread_pool() is not first
    client.close()


async def test_collecting_it_shuts_the_threads_down() -> None:
    client = _Client()
    await client._run_in_thread(lambda: None)
    pool = client._thread_pool()
    del client
    gc.collect()
    assert pool._shutdown


async def test_a_sync_wrapper_calls_in_its_threads() -> None:
    memory = MemoryFileSystem()
    memory.pipe_file("/test_sync_client/a.txt", b"a")
    tree = SyncTreeWrapper(
        memory,
        operation_concurrency=3,
        io_concurrency=2,
        storage_profile=StorageProfile(),
        asynchronous=True,
    )
    try:
        assert await tree._cat_file("/test_sync_client/a.txt") == b"a"
        # Its requests and its data calls, each with threads enough.
        assert tree._thread_pool()._max_workers == 5
    finally:
        tree.close()
        memory.rm("/test_sync_client", recursive=True)


def test_with_closes_its_threads() -> None:
    memory = MemoryFileSystem()
    with SyncTreeWrapper(
        memory,
        operation_concurrency=1,
        io_concurrency=1,
        storage_profile=StorageProfile(),
    ) as tree:
        assert tree.ls("/") is not None
        pool = tree._thread_pool()
    assert pool._shutdown
    assert tree._executor is None
