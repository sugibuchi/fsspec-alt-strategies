"""Tests of `control`: `Stop`, results, and how callbacks are called."""

from __future__ import annotations

import dataclasses

import pytest

from fsspec_alt_strategies.control import (
    OperationStopped,
    Result,
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


def test_result_complete() -> None:
    assert Result().complete
    assert not Result(stop=Stop()).complete
    assert not Result(errors={"a": OSError("a")}).complete


def test_raise_for_errors() -> None:
    Result().raise_for_errors()
    error = PermissionError("a")
    with pytest.raises(ExceptionGroup) as info:
        Result(errors={"a": error}).raise_for_errors()
    assert info.value.exceptions == (error,)


def test_finish_returns_unless_asked_to_raise() -> None:
    for result in (Result(), Result(stop=Stop()), Result(errors={"a": OSError()})):
        assert _finish(result) is result


def test_finish_raises_with_the_result_and_the_errors() -> None:
    error = PermissionError("a")
    result = Result(
        stop=Stop(raise_error=True, reason="fail fast"), errors={"a": error}
    )
    with pytest.raises(OperationStopped, match="fail fast") as info:
        _finish(result)
    assert info.value.result is result
    assert isinstance(info.value.__cause__, ExceptionGroup)
    assert info.value.__cause__.exceptions == (error,)


def test_finish_raises_without_errors() -> None:
    with pytest.raises(OperationStopped) as info:
        _finish(Result(stop=Stop(raise_error=True)))
    assert info.value.__cause__ is None
