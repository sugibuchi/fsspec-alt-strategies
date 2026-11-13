"""Tests of `_across` and `_cross_cp_file`: a file's copy or move between two handlers."""

from __future__ import annotations

from typing import Any, Self

import pytest

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.transfer import _across, _cross_cp_file
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView

_ATOMIC = StorageProfile(atomic_writes=True)


def _storages() -> tuple[SimulatedFileSystem, SimulatedFileSystem]:
    src = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    src.add_file("data/x", b"xx")
    dst = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    dst.add_file("out/old", b"old")
    return src, dst


def _files(fs: SimulatedFileSystem) -> dict[str, bytes]:
    return {path: f.data for path, f in fs._file_data.items()}


async def test_unknown_writes_a_temporary_file_then_moves_it() -> None:
    src, dst = _storages()
    await _cross_cp_file(
        TreeView(src),
        {"name": "data/x"},
        TreeView(dst),
        "out/x",  # type: ignore[arg-type]
    )
    assert _files(dst) == {"out/old": b"old", "out/x": b"xx"}
    assert dst.stats.calls["write"] == 1
    # The simulator moves a file by a copy and a deletion, as fsspec's _mv_file.
    assert (dst.stats.calls["copy"], dst.stats.calls["delete"]) == (1, 1)


async def test_atomic_writes_write_in_place() -> None:
    src, dst = _storages()
    view = TreeView(dst, storage_profile=_ATOMIC)
    await _cross_cp_file(TreeView(src), {"name": "data/x"}, view, "out/x")  # type: ignore[arg-type]
    assert _files(dst) == {"out/old": b"old", "out/x": b"xx"}
    assert dst.stats.calls["write"] == 1
    assert (dst.stats.calls["copy"], dst.stats.calls["delete"]) == (0, 0)


class _Short:
    """A source file ending before its size: two bytes of the five its entry tells."""

    def __init__(self) -> None:
        self.blocks = [b"xx"]

    async def read(self, size: int = -1) -> bytes:
        return self.blocks.pop() if self.blocks else b""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass


class _Shortened(TreeView):
    async def open_reader(self, entry: dict[str, Any], **kwargs: Any) -> Any:
        return _Short()


@pytest.mark.parametrize("profile", [None, _ATOMIC])
async def test_a_size_other_than_the_sources_leaves_nothing(
    profile: StorageProfile | None,
) -> None:
    src, dst = _storages()
    with pytest.raises(OSError, match="read 2 bytes of 5"):
        await _cross_cp_file(
            _Shortened(src),
            {"name": "data/x", "size": 5},
            TreeView(dst, storage_profile=profile),
            "out/old",  # type: ignore[arg-type]
        )
    # The file there is kept, and no temporary file is left.
    assert _files(dst) == {"out/old": b"old"}


async def test_a_failed_move_deletes_the_temporary_file() -> None:
    src, dst = _storages()

    async def failing(path1: str, path2: str) -> None:
        raise PermissionError(path2)

    with pytest.raises(PermissionError):
        await _cross_cp_file(
            TreeView(src),
            {"name": "data/x"},
            TreeView(dst, mv_file=failing),
            "out/x",  # type: ignore[arg-type]
        )
    assert _files(dst) == {"out/old": b"old"}


async def test_a_missing_source_writes_nothing() -> None:
    src, dst = _storages()
    with pytest.raises(FileNotFoundError):
        await _cross_cp_file(
            TreeView(src),
            {"name": "data/missing"},
            TreeView(dst),
            "out/x",  # type: ignore[arg-type]
        )
    assert _files(dst) == {"out/old": b"old"}


async def test_a_move_deletes_the_source_once_copied() -> None:
    src, dst = _storages()
    move = _across(TreeView(src), TreeView(dst, storage_profile=_ATOMIC), move=True)
    await move({"name": "data/x"}, "out/x")
    assert _files(src) == {}
    assert _files(dst) == {"out/old": b"old", "out/x": b"xx"}


async def test_a_move_keeps_a_source_whose_copy_failed() -> None:
    src, dst = _storages()
    move = _across(_Shortened(src), TreeView(dst), move=True)
    with pytest.raises(OSError, match="read 2 bytes"):
        await move({"name": "data/x", "size": 5}, "out/x")
    assert _files(src) == {"data/x": b"xx"}
