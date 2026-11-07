"""Tests of `common.monitors`: running an operation to its end."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fsspec_alt_strategies.common.monitors import _collect
from fsspec_alt_strategies.operation import Operation
from fsspec_alt_strategies_common.control import State

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


async def test_collect_splits_items_and_errors() -> None:
    async def run(op: Operation[int, State]) -> AsyncGenerator[int | Exception]:
        for i in range(7):
            # Multiples of 5 are yielded as errors.
            yield i if i % 5 else ValueError(i)

    op: Operation[int, State] = Operation(run, State())
    items, errors = await _collect(op)
    assert items == [1, 2, 3, 4, 6]
    assert [e.args for e in errors] == [(0,), (5,)]
    assert op.done()
