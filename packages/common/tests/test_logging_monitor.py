# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import fsspec

from fsspec_alt_strategies import (
    Call,
    CallCancelled,
    LoggingMonitor,
    OperationHandler,
    State,
)

if TYPE_CHECKING:
    import pytest


class _Handler(OperationHandler):
    def cancel(self) -> bool:
        return True

    def cancelling(self) -> int:
        return 0

    def cancelled(self) -> bool:
        return False

    def done(self) -> bool:
        return True


_LOGGER = logging.getLogger("test.storage")
_CALL = Call(fsspec.filesystem("memory"), "copy", "a", "b")


async def _run(error: BaseException | None) -> None:
    monitor = LoggingMonitor(_LOGGER)
    op = _Handler()
    await monitor.started(_CALL, State(), op)
    await monitor.yielded(_CALL, "a/x", State(current="a/x"), op)
    await monitor.ended(_CALL, error, State(errors=2, current="a/x"), op)


def test_logs_the_start_and_end_at_its_level(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, "test.storage"):
        asyncio.run(_run(None))
    started, ended = caplog.records
    assert started.levelno == ended.levelno == logging.INFO
    assert started.getMessage() == "copy of 'a' to 'b' started"
    assert ended.getMessage().startswith("copy of 'a' to 'b' ended, 2 errors")


def test_logs_each_item_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, "test.storage"):
        asyncio.run(_run(None))
    [item] = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert item.getMessage() == "copy of 'a' to 'b' yielded 'a/x'"


def test_logs_a_cancellation(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, "test.storage"):
        asyncio.run(_run(CallCancelled(_CALL, State())))
    assert "was cancelled" in caplog.records[-1].getMessage()


def test_logs_a_failure(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, "test.storage"):
        asyncio.run(_run(FileNotFoundError("a")))
    assert "failed with FileNotFoundError" in caplog.records[-1].getMessage()
