"""Tests of `StorageHandler`'s paths: what a handler verifies, and how it builds paths."""

from __future__ import annotations

from typing import Any

import pytest

from fsspec_alt_strategies_impl.common.handler import StorageHandler


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


def test_a_path_of_another_account_is_refused() -> None:
    handler = _Account()
    handler.verify_path("acct/c/x")
    with pytest.raises(ValueError, match="Not a path of 'acct'"):
        handler.verify_path("other/c/x")
    # Joining below the root of another account's path is refused too.
    with pytest.raises(ValueError, match="Not a path of 'acct'"):
        handler.join("other", "c")


@pytest.mark.parametrize(
    ("root", "path", "parent", "name"),
    [
        ("", "a/b", "a", "b"),
        ("", "a", "", "a"),
        ("/", "/a/b", "/a", "b"),
        ("/", "/a", "/", "a"),
    ],
)
def test_parent_and_name(root: str, path: str, parent: str, name: str) -> None:
    handler = _Account()
    handler.root = root
    assert (handler.parent(path), handler.name(path)) == (parent, name)
