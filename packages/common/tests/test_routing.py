"""Tests of `routing`, which names the routers a `Call` came through."""

from __future__ import annotations

import asyncio

import fsspec

from fsspec_alt_strategies_common import Call, current_routers, routing

_FS = fsspec.filesystem("memory")


def test_a_direct_call_comes_through_no_router() -> None:
    assert current_routers() == ()
    assert Call(_FS, "find", "a").via == ()


def test_routers_nest_the_outermost_first() -> None:
    outer, inner = object(), object()
    with routing(outer):  # type: ignore[arg-type]
        with routing(inner):  # type: ignore[arg-type]
            call = Call(_FS, "find", "a")
        assert current_routers() == (outer,)
    assert call.via == (outer, inner)
    assert current_routers() == ()


def test_a_call_keeps_the_routers_it_was_made_in() -> None:
    router = object()
    with routing(router):  # type: ignore[arg-type]
        call = Call(_FS, "copy", "a", "b")
    # Made within, the call keeps them after the block too, and compares by them.
    assert call.via == (router,)
    assert call != Call(_FS, "copy", "a", "b")


def test_tasks_inherit_the_routers() -> None:
    router = object()

    async def main() -> tuple[object, ...]:
        with routing(router):  # type: ignore[arg-type]
            return await asyncio.create_task(_routers_in_task())

    assert asyncio.run(main()) == (router,)


async def _routers_in_task() -> tuple[object, ...]:
    return current_routers()
