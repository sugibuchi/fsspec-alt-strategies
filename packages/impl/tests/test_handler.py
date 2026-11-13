"""Tests of `StorageHandler`'s paths: what a handler verifies, and how it builds paths."""

from __future__ import annotations

from typing import Any

import pytest

from fsspec_alt_strategies_impl.common.handler import (
    StorageHandler,
    _FileSystemHandler,
)
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.find import _tree_find
from fsspec_alt_strategies_impl.tree.rm import _tree_rm
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView


class _Account(StorageHandler):
    """Paths of one storage account, "account/container/...", refusing names holding ":"."""

    def __init__(self, account: str = "acct") -> None:
        super().__init__(operation_concurrency=1, io_concurrency=1)
        self.account = account
        self.verified: list[str] = []

    def verify_path(self, path: str) -> None:
        self.verified.append(path)
        if path.split("/", 1)[0] != self.account:
            raise ValueError(f"Not a path of {self.account!r}: {path!r}")
        if ":" in path:
            raise ValueError(f"A name holds ':': {path!r}")

    # The primitives, which these tests never call.
    def ils(self, entry: Any, **kwargs: Any) -> Any: ...
    async def info(self, entry: Any, **kwargs: Any) -> Any: ...
    async def rm_file(self, entry: Any) -> None: ...
    async def rmdir(self, entry: Any) -> None: ...
    async def makedirs(self, path: str, exist_ok: bool = False) -> None: ...
    async def cp_file(self, entry: Any, path2: str) -> None: ...
    async def mv_file(self, entry: Any, path2: str) -> None: ...


def test_verify_path_is_abstract() -> None:
    class NoVerify(_Account):
        verify_path = StorageHandler.verify_path  # type: ignore[assignment]

    with pytest.raises(TypeError, match="verify_path"):
        NoVerify()


def test_join_checks_the_name_then_verifies_the_path() -> None:
    handler = _Account()
    assert handler.join("acct/c", "x") == "acct/c/x"
    assert handler.verified == ["acct/c/x"]

    for name in ("", ".", "..", "a/b"):
        with pytest.raises(ValueError, match="Not a name"):
            handler.join("acct/c", name)
    with pytest.raises(ValueError, match="holds ':'"):
        handler.join("acct/c", "a:b")


def test_join_all_verifies_the_path_built_once() -> None:
    handler = _Account()
    assert handler.join_all("acct/c", "/a/b/") == "acct/c/a/b"
    assert handler.verified == ["acct/c/a/b"]
    assert handler.join_all("acct/c", "") == "acct/c"
    with pytest.raises(ValueError, match="Not a name"):
        handler.join_all("acct/c", "a/../../other")
    with pytest.raises(ValueError, match="Not a name"):
        handler.join_all("acct/c", "a//b")


def test_entry_verifies_the_path_it_wraps() -> None:
    handler = _Account()
    assert handler.entry("acct/c/x") == {"name": "acct/c/x"}
    assert handler.verified == ["acct/c/x"]
    with pytest.raises(ValueError, match="Not a path of 'acct'"):
        handler.entry("other/c/x")


def test_a_path_of_another_account_is_refused() -> None:
    handler = _Account()
    handler.verify_path("acct/c/x")
    with pytest.raises(ValueError, match="Not a path of 'acct'"):
        handler.verify_path("other/c/x")
    # Joining below the root of another account's path is refused too.
    with pytest.raises(ValueError, match="Not a path of 'acct'"):
        handler.join("other", "c")


class _Any(_Account):
    """Any path, as a storage refusing nothing."""

    def verify_path(self, path: str) -> None:
        self.verified.append(path)


@pytest.mark.parametrize(
    ("root", "path", "parent", "name"),
    [
        ("", "a/b", "a", "b"),
        ("", "a/b/", "a", "b"),
        ("", "a", "", "a"),
        ("/", "/a/b", "/a", "b"),
        ("/", "/a", "/", "a"),
    ],
)
def test_parent_and_name(root: str, path: str, parent: str, name: str) -> None:
    handler = _Any()
    handler.root = root
    assert (handler.parent(path), handler.name(path)) == (parent, name)
    assert handler.verified == [parent]


@pytest.mark.parametrize("root", ["", "/"])
def test_the_root_has_no_parent(root: str) -> None:
    handler = _Any()
    handler.root = root
    with pytest.raises(ValueError, match="no parent"):
        handler.parent(root)


def test_a_parent_the_storage_refuses() -> None:
    """Paths are "account/container/...": an account's root is "acct", whose parent is none."""
    handler = _Account()
    handler.root = "acct"
    assert handler.parent("acct/c/x") == "acct/c"
    assert handler.parent("acct/c") == "acct"
    with pytest.raises(ValueError, match="no parent"):
        handler.parent("acct")


async def test_names_a_listing_gave_are_never_verified_again() -> None:
    """A storage holding "a:b", which its handler refuses from a caller, still has it found and deleted."""

    class Strict(_FileSystemHandler):
        recursive_order = None  # A directory per listing, each by its entry.

        def verify_path(self, path: str) -> None:
            if ":" in path:
                raise ValueError(f"A name holds ':': {path!r}")

    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    fs.add_file("data/a:b/x", b"1")
    fs.add_file("data/y", b"2")
    view = TreeView(fs)
    strict = Strict(view)
    view.__dict__["_alt_strategies_handler"] = strict

    found, errors = await _collect_with_errors(_tree_find(view, "data"))
    assert (sorted(e["name"] for e in found), errors) == (["data/a:b/x", "data/y"], [])
    _, errors = await _collect_with_errors(_tree_rm(view, "data"))
    assert errors == []
    assert not fs._file_data
    with pytest.raises(ValueError, match="holds ':'"):
        strict.entry("data/a:b")
