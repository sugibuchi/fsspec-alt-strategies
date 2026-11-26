"""Tests of the hints the primitives get: ``missing`` for a destination known empty, a parent's entry, and a write's ``size``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.common.transfer import _across
from fsspec_alt_strategies_impl.tree.cp import _tree_cp
from fsspec_alt_strategies_impl.tree.mv import _tree_mv
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem, TreeView

if TYPE_CHECKING:
    from pathlib import Path

_FILES = ["data/a/x", "data/a/b/y", "data/z"]


def _hinting(base: type) -> type:
    """A view noting the ``missing`` each primitive making an item is told, by path."""

    class Hinting(base):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.hints: dict[str, bool] = {}

        async def makedirs(
            self,
            path: Any,
            exist_ok: bool = False,
            *,
            missing: bool = False,
            parent: Any = None,
        ) -> None:
            self.hints[path] = missing
            await super().makedirs(path, exist_ok, missing=missing)

        async def cp_file(
            self, entry: Any, path2: Any, *, missing: bool = False, parent: Any = None
        ) -> None:
            self.hints[path2] = missing
            await super().cp_file(entry, path2, missing=missing)

        async def mv_file(
            self, entry: Any, path2: Any, *, missing: bool = False, parent: Any = None
        ) -> None:
            self.hints[path2] = missing
            await super().mv_file(entry, path2, missing=missing)

    return Hinting


def _tree(fs: SimulatedFileSystem) -> None:
    for path in _FILES:
        fs.add_file(path, path.encode())


@pytest.mark.parametrize("strategy", [_tree_cp, _tree_mv])
async def test_a_tree_copy_to_a_missing_destination_says_so(
    hierarchical_fs: SimulatedFileSystem, strategy: Any
) -> None:
    _tree(hierarchical_fs)
    handler = _hinting(TreeView)(hierarchical_fs)
    _, errors = await _collect_with_errors(strategy(handler, handler, "data", "new"))
    assert errors == []
    # Everything below a destination found missing is missing: each directory
    # made, and each file copied or moved there.
    assert handler.hints
    assert all(handler.hints.values())
    assert {"new/a", "new/a/b/y", "new/z"} <= set(handler.hints)


@pytest.mark.parametrize("strategy", [_tree_cp, _tree_mv])
async def test_a_tree_copy_into_an_existing_directory_says_nothing(
    hierarchical_fs: SimulatedFileSystem, strategy: Any
) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_file("old/keep", b"k")
    handler = _hinting(TreeView)(hierarchical_fs)
    # "old" exists: what goes into it may meet what is there.
    _, errors = await _collect_with_errors(strategy(handler, handler, "data", "old"))
    assert errors == []
    assert handler.hints
    assert not any(handler.hints.values())


@pytest.mark.parametrize("strategy", [_tree_cp, _tree_mv])
async def test_a_flat_copy_to_a_missing_destination_says_so(
    flat_fs: SimulatedFileSystem, strategy: Any
) -> None:
    _tree(flat_fs)
    # A rename for the move, so that it renames by mv_file, as a copy copies by cp_file.
    profile = StorageProfile(implicit_directories=True, renames_files=True)
    handler = _hinting(FlatView)(flat_fs, storage_profile=profile)
    _, errors = await _collect_with_errors(strategy(handler, handler, "data", "new"))
    assert errors == []
    assert set(handler.hints) == {"new/a/x", "new/a/b/y", "new/z"}
    assert all(handler.hints.values())


async def test_pipe_and_put_give_the_size_with_the_name(
    hierarchical_fs: SimulatedFileSystem, tmp_path: Path
) -> None:
    opened: list[dict[str, Any]] = []

    class Sizing(TreeView):
        async def open_async_file(
            self, entry: Any, mode: str = "rb", **kwargs: Any
        ) -> Any:
            if mode != "rb":
                opened.append(dict(entry))
            return await super().open_async_file(entry, mode, **kwargs)

    handler = Sizing(hierarchical_fs)
    await handler.pipe_file(handler.verified("piped"), b"12345")
    local = tmp_path / "local"
    local.write_bytes(b"123")
    await handler.put_file(str(local), handler.verified("put"))
    assert opened == [{"name": "piped", "size": 5}, {"name": "put", "size": 3}]


async def test_a_copy_between_storages_gives_the_source_size_and_says_missing() -> None:
    src_fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    dst_fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    src_fs.add_file("a", b"1234")
    written: list[dict[str, Any]] = []

    class Destination(_hinting(TreeView)):  # type: ignore[misc]
        async def open_async_file(
            self, entry: Any, mode: str = "rb", **kwargs: Any
        ) -> Any:
            if mode != "rb":
                written.append(dict(entry))
            return await super().open_async_file(entry, mode, **kwargs)

    src, dst = TreeView(src_fs), Destination(dst_fs)
    await _across(src, dst)({"name": "a"}, "b", missing=True)
    assert await dst_fs._cat_file("b") == b"1234"
    # The size, read from the source as the copy starts, goes with the write.
    assert [w.get("size") for w in written] == [4]
    # Written beside the destination, then moved into place: the move is told.
    assert dst.hints == {"b": True}


class _Parenting(TreeView):
    """A view whose ``makedirs`` answers the entry of the directory made, with an id, and noting each ``parent`` told."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.parents: dict[str, Any] = {}
        self.ids = 0

    async def makedirs(
        self,
        path: Any,
        exist_ok: bool = False,
        *,
        missing: bool = False,
        parent: Any = None,
    ) -> Any:
        self.parents[path] = parent
        await super().makedirs(path, exist_ok, missing=missing)
        self.ids += 1
        return {"name": path, "type": "directory", "id": self.ids}

    async def cp_file(
        self, entry: Any, path2: Any, *, missing: bool = False, parent: Any = None
    ) -> None:
        self.parents[path2] = parent
        await super().cp_file(entry, path2, missing=missing)

    async def mv_file(
        self, entry: Any, path2: Any, *, missing: bool = False, parent: Any = None
    ) -> None:
        self.parents[path2] = parent
        await super().mv_file(entry, path2, missing=missing)


@pytest.mark.parametrize("strategy", [_tree_cp, _tree_mv])
async def test_what_goes_into_a_directory_made_gets_its_entry(
    hierarchical_fs: SimulatedFileSystem, strategy: Any
) -> None:
    _tree(hierarchical_fs)
    # Parents first, each directory made by one makedirs, after its parent.
    profile = StorageProfile(parents_required=True, copy_parents_required=True)
    handler = _Parenting(hierarchical_fs, storage_profile=profile)
    _, errors = await _collect_with_errors(strategy(handler, handler, "data", "new"))
    assert errors == []
    names = {p: (e or {}).get("name") for p, e in handler.parents.items()}
    # Each item made below a directory made gets that directory's entry, with its id.
    assert names["new/a"] == "new"
    assert names["new/a/b"] == "new/a"
    assert names["new/a/b/y"] == "new/a/b"
    assert names["new/z"] == "new"
    assert all("id" in e for p, e in handler.parents.items() if p != "new")


async def test_what_goes_into_an_existing_directory_gets_its_info() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    fs.add_file("old/keep", b"k")
    handler = _Parenting(fs)
    # "old" exists: its info, which placing the copy asks, is the parent of what goes in.
    _, errors = await _collect_with_errors(_tree_cp(handler, handler, "data/z", "old"))
    assert errors == []
    assert handler.parents["old/z"]["name"] == "old"
    assert handler.parents["old/z"]["type"] == "directory"


async def test_a_write_between_storages_gets_the_parent_in_its_entry() -> None:
    src_fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    dst_fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    src_fs.add_file("a", b"1234")
    dst_fs.add_dir("d")
    written: list[dict[str, Any]] = []

    class Destination(_Parenting):
        async def open_async_file(
            self, entry: Any, mode: str = "rb", **kwargs: Any
        ) -> Any:
            if mode != "rb":
                written.append(dict(entry))
            return await super().open_async_file(entry, mode, **kwargs)

    src, dst = TreeView(src_fs), Destination(dst_fs)
    parent = {"name": "d", "type": "directory", "id": 7}
    await _across(src, dst)({"name": "a"}, "d/b", parent=parent)
    # The write, beside the destination, and the move into place both get it.
    assert [w.get("parent") for w in written] == [parent]
    assert dst.parents == {"d/b": parent}
