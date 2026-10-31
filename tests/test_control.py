"""Tests of `control`: `Stop`, how callbacks are called, and stopping."""

from __future__ import annotations

import dataclasses

import pytest

from fsspec_alt_strategies.control import (
    OperationStopped,
    State,
    Stop,
    _finish,
    _report,
)


def test_stop_defaults_keep_work_and_return() -> None:
    stop = Stop()
    assert (stop.cleanup, stop.raise_error, stop.reason) == (False, False, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stop.cleanup = True  # type: ignore[misc]


async def test_report_without_callback() -> None:
    assert await _report(None, State()) is None


async def test_report_plain_and_async_callbacks() -> None:
    stop = Stop(reason="enough")

    def plain(state: State) -> Stop | None:
        return stop if state.errors else None

    async def asynchronous(state: State) -> Stop | None:
        return stop if state.errors else None

    for callback in (plain, asynchronous):
        assert await _report(callback, State()) is None
        assert await _report(callback, State(errors=1)) is stop


@pytest.mark.parametrize("answer", [True, False, 0, "stop", Stop])
async def test_report_refuses_other_answers(answer: object) -> None:
    with pytest.raises(TypeError, match="None or a Stop"):
        await _report(lambda _: answer, State())


def test_finish_returns_unless_asked_to_raise() -> None:
    result = ([], [])
    for stop, errors in ((None, []), (Stop(), []), (None, [OSError()])):
        assert _finish(result, stop, errors) is result


def test_finish_raises_with_the_result_and_the_errors() -> None:
    error = PermissionError("a")
    result = ([], [error])
    stop = Stop(raise_error=True, reason="fail fast")
    with pytest.raises(OperationStopped, match="fail fast") as info:
        _finish(result, stop, [error])
    assert info.value.result is result
    assert info.value.stop is stop
    assert isinstance(info.value.__cause__, ExceptionGroup)
    assert info.value.__cause__.exceptions == (error,)


def test_finish_raises_without_errors() -> None:
    with pytest.raises(OperationStopped) as info:
        _finish(([], []), Stop(raise_error=True), [])
    assert info.value.__cause__ is None
