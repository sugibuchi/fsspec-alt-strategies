"""Tests of `operation`: `Operation`, and its stopping as `OperationHandler` declares it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from fsspec_alt_strategies.operation import Operation
from fsspec_alt_strategies_common.control import State

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def _counting(
    items: int, sent: list[int], *, fail_at: int | None = None
) -> Operation[int, State]:
    """Yields 0, 1, ..., "sending" each a step ahead, as strategies do.

    Multiples of 5 are yielded as errors.
    """

    def outcome(i: int) -> int | Exception:
        return i if i % 5 else ValueError(i)

    async def run(op: Operation[int, State]) -> AsyncGenerator[int | Exception]:
        in_flight: list[int] = []
        for i in range(items):
            if op.cancelling():
                break
            if i == fail_at:
                raise RuntimeError("the whole operation failed")
            sent.append(i)
            in_flight.append(i)
            if len(in_flight) > 1:
                op.state = State(current=str(in_flight[0]))
                yield outcome(in_flight.pop(0))
        for i in in_flight:
            yield outcome(i)

    return Operation(run, State())


async def test_runs_only_when_iterated() -> None:
    sent: list[int] = []
    op = _counting(10, sent)
    await asyncio.sleep(0)
    assert sent == []
    assert not op.done()
    assert [i async for i in op if not isinstance(i, Exception)][:3] == [1, 2, 3]
    assert op.done() and not op.cancelled()
    assert [i async for i in op] == []


async def test_cancel_yields_what_is_in_flight() -> None:
    sent: list[int] = []
    op = _counting(10, sent)
    seen = []
    async for i in op:
        seen.append(i)
        if i == 2:
            assert op.cancel()
            assert op.cancelling() == 1
            assert not op.cancelled()
    # 3 was sent before the stop: it is still yielded.
    assert seen[-2:] == [2, 3]
    assert sent == [0, 1, 2, 3]
    assert op.done() and op.cancelled()
    assert not op.cancel()


async def test_closing_stops_at_once() -> None:
    sent: list[int] = []
    async with _counting(10, sent) as op:
        async for i in op:
            if i == 2:
                break
    assert sent == [0, 1, 2, 3]
    assert op.done() and op.cancelled()


async def test_stopping_before_iterating_runs_nothing() -> None:
    sent: list[int] = []
    op = _counting(10, sent)
    await op.aclose()
    assert op.done() and op.cancelled()
    op = _counting(10, sent)
    op.cancel()
    assert [i async for i in op] == []
    assert op.done() and op.cancelled()
    assert sent == []


async def test_an_error_ending_it_is_raised() -> None:
    op = _counting(10, [], fail_at=3)
    with pytest.raises(RuntimeError, match="whole operation"):
        async for _ in op:
            pass
    assert op.done() and not op.cancelled()
