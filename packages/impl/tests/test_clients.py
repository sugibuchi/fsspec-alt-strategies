"""Tests of `common.clients`: a sync client's calls in threads, and an async client on its loop."""

from __future__ import annotations

import asyncio
import contextvars
import gc
import threading
from typing import Any

from fsspec.asyn import get_loop, sync
from fsspec.implementations.memory import MemoryFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.clients import (
    AsyncClientSupport,
    SyncClientSupport,
)
from fsspec_alt_strategies_impl.wrapper.tree import (
    FsspecTreeHandler,
    TreeFileSystemWrapper,
)


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


async def test_aclose_shuts_the_threads_down_too() -> None:
    client = _Client()
    await client._run_in_thread(lambda: None)
    first = client._thread_pool()
    await client.aclose()
    assert client._executor is None
    assert first._shutdown


async def test_collecting_it_shuts_the_threads_down() -> None:
    client = _Client()
    await client._run_in_thread(lambda: None)
    pool = client._thread_pool()
    del client
    gc.collect()
    assert pool._shutdown


async def test_a_sync_filesystems_handler_calls_in_its_threads() -> None:
    memory = MemoryFileSystem()
    memory.pipe_file("/test_sync_client/a.txt", b"a")
    handler = FsspecTreeHandler(
        memory,
        operation_concurrency=3,
        io_concurrency=2,
        storage_profile=StorageProfile(),
    )
    tree = TreeFileSystemWrapper(handler, asynchronous=True)
    try:
        assert await tree._cat_file("/test_sync_client/a.txt") == b"a"
        # Its requests and its data calls, each with threads enough.
        assert handler._thread_pool()._max_workers == 5
    finally:
        tree.close()
        memory.rm("/test_sync_client", recursive=True)


def test_with_closes_its_threads() -> None:
    memory = MemoryFileSystem()
    handler = FsspecTreeHandler(
        memory,
        operation_concurrency=1,
        io_concurrency=1,
        storage_profile=StorageProfile(),
    )
    with TreeFileSystemWrapper(handler) as tree:
        assert tree.ls("/") is not None
        pool = handler._thread_pool()
    assert pool._shutdown
    assert handler._executor is None


# AsyncClientSupport: clients made on the running loop, and closed on theirs.


class _Session(AsyncClientSupport[dict[str, Any]]):
    """Clients noting the loop each was made on, and each closing, on its loop."""

    def __init__(self, fail_close: bool = False) -> None:
        self.made: list[dict[str, Any]] = []
        self.closed: list[tuple[dict[str, Any], asyncio.AbstractEventLoop]] = []
        self.fail_close = fail_close

    async def create_clients(self) -> dict[str, Any]:
        await asyncio.sleep(0)  # Others ask meanwhile.
        clients = {"loop": asyncio.get_running_loop()}
        self.made.append(clients)
        return clients

    async def close_clients(self, clients: dict[str, Any]) -> None:
        self.closed.append((clients, asyncio.get_running_loop()))
        if self.fail_close:
            raise OSError("secret-bearing message")


async def test_clients_are_made_once_on_the_running_loop() -> None:
    session = _Session()
    got = await asyncio.gather(*(session.ensure_clients() for _ in range(5)))
    assert len(session.made) == 1
    assert all(c is session.made[0] for c in got)
    assert session.made[0]["loop"] is asyncio.get_running_loop()


async def test_aclose_on_their_loop_awaits_the_closing() -> None:
    session = _Session()
    clients = await session.ensure_clients()
    await session.aclose()
    assert session.closed == [(clients, asyncio.get_running_loop())]
    # Made again by the next call.
    assert await session.ensure_clients() is not clients


async def test_aclose_from_another_loop_closes_them_on_theirs() -> None:
    session = _Session()
    io_loop = get_loop()  # fsspec's own loop, in a thread of its own.
    clients = await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(session.ensure_clients(), io_loop)
    )
    await session.aclose()
    assert session.closed == [(clients, io_loop)]


def test_clients_of_a_loop_ended_are_dropped_and_made_again() -> None:
    session = _Session()
    first = asyncio.run(session.ensure_clients())
    second = asyncio.run(session.ensure_clients())
    assert second is not first
    assert session.closed == []  # The first loop ended: nothing could close them.


def test_sync_close_sends_the_closing_to_their_loop() -> None:
    session = _Session()
    io_loop = get_loop()
    clients = sync(io_loop, session.ensure_clients)
    session.close()
    sync(io_loop, asyncio.sleep, 0.05)  # The closing runs there, unawaited.
    assert session.closed == [(clients, io_loop)]


async def test_a_failed_closing_is_dropped() -> None:
    session = _Session(fail_close=True)
    await session.ensure_clients()
    await session.aclose()  # No error, and no message of the storage's.
    assert len(session.closed) == 1
