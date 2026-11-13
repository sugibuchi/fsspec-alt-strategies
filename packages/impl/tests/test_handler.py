"""Tests of `StorageHandler`'s paths: what a handler verifies, and how it builds paths."""

from __future__ import annotations

from typing import Any

import pytest

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.handler import StorageHandler
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.filesystem import TreeFileSystem
from fsspec_alt_strategies_impl.tree.find import _tree_find
from fsspec_alt_strategies_impl.tree.rm import _tree_rm
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView


class _Account(StorageHandler):
    """Paths of one storage account, "account/container/...", refusing names holding ":"."""

    def __init__(self, account: str = "acct") -> None:
        super().__init__(operation_concurrency=1, io_concurrency=1)
        self.account = account
        self.checked: list[str] = []

    def verify_path(self, path: str) -> None:
        self.checked.append(path)
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
    assert handler.checked == ["acct/c/x"]

    for name in ("", ".", "..", "a/b"):
        with pytest.raises(ValueError, match="Not a name"):
            handler.join("acct/c", name)
    with pytest.raises(ValueError, match="holds ':'"):
        handler.join("acct/c", "a:b")


def test_join_all_verifies_the_path_built_once() -> None:
    handler = _Account()
    assert handler.join_all("acct/c", "/a/b/") == "acct/c/a/b"
    assert handler.checked == ["acct/c/a/b"]
    assert handler.join_all("acct/c", "") == "acct/c"
    with pytest.raises(ValueError, match="Not a name"):
        handler.join_all("acct/c", "a/../../other")
    with pytest.raises(ValueError, match="Not a name"):
        handler.join_all("acct/c", "a//b")


def test_verified_is_a_callers_path_once_verified() -> None:
    handler = _Account()
    assert handler.verified("acct/c/x") == "acct/c/x"
    assert handler.checked == ["acct/c/x"]
    with pytest.raises(ValueError, match="Not a path of 'acct'"):
        handler.verified("other/c/x")


def test_entry_verifies_the_path_it_wraps() -> None:
    handler = _Account()
    assert handler.entry("acct/c/x") == {"name": "acct/c/x"}
    assert handler.checked == ["acct/c/x"]
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
        self.checked.append(path)


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
    assert handler.checked == [parent]


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

    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    fs.add_file("data/a:b/x", b"1")
    fs.add_file("data/y", b"2")
    strict = _Strict(fs)
    strict.recursive_order = None  # A directory per listing, each by its entry.

    found, errors = await _collect_with_errors(_tree_find(strict, "data"))
    assert (sorted(e["name"] for e in found), errors) == (["data/a:b/x", "data/y"], [])
    _, errors = await _collect_with_errors(_tree_rm(strict, "data"))
    assert errors == []
    assert not fs._file_data
    with pytest.raises(ValueError, match="holds ':'"):
        strict.entry("data/a:b")


class _Strict(TreeView):
    """A simulated storage refusing paths holding ":" from a caller, as some storages do."""

    def verify_path(self, path: str) -> None:
        if ":" in path:
            raise ValueError(f"A name holds ':': {path!r}")


def _storage() -> SimulatedFileSystem:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    fs.add_file("data/a/x", b"1")
    fs.add_file("data/y", b"22")
    return fs


async def test_a_filesystem_on_a_handler_calls_its_primitives() -> None:
    inner = _storage()
    fs = TreeFileSystem(_Strict(inner), asynchronous=True)
    assert fs.handler is not None
    assert await fs._find("data") == ["data/a/x", "data/y"]
    assert (await fs._info("data/y"))["size"] == 2
    await fs._copy("data", "copied", recursive=True)
    assert await inner._cat_file("copied/a/x") == b"1"
    await fs._rm("data", recursive=True)
    assert not await inner._exists("data")


async def test_a_filesystem_on_a_handler_verifies_a_callers_paths() -> None:
    fs = TreeFileSystem(_Strict(_storage()), asynchronous=True)
    with pytest.raises(ValueError, match="holds ':'"):
        await fs._info("data/a:b")
    with pytest.raises(ValueError, match="holds ':'"):
        await fs._makedirs("data/a:b")
    with pytest.raises(ValueError, match="holds ':'"):
        await fs._copy("data/y", "data/c:d")


def test_a_handler_class_takes_the_storage_options() -> None:
    made: list[dict[str, Any]] = []

    class Options(TreeView):
        protocol = "mine"

        def __init__(self, inner: SimulatedFileSystem, **options: Any) -> None:
            made.append(options)
            super().__init__(inner)

    class Mine(TreeFileSystem):
        handler_class = Options

    inner = _storage()
    fs = Mine(inner, region="here", asynchronous=True, skip_instance_cache=True)
    assert isinstance(fs.handler, Options)
    # fsspec's own options are the filesystem's: the handler gets the others.
    assert made == [{"region": "here"}]


def test_a_filesystem_needs_a_handler() -> None:
    with pytest.raises(TypeError, match="handler"):
        TreeFileSystem()  # type: ignore[call-arg]


def test_a_storages_class_makes_its_handler_and_is_cached() -> None:
    class Cached(TreeView):
        protocol = "mine-cached"

    class Mine(TreeFileSystem):
        handler_class = Cached

    inner = _storage()
    with pytest.raises(TypeError, match="makes its handler from its storage options"):
        Mine(handler=TreeView(inner))
    # Made from its storage options, it is cached as fsspec caches any.
    assert Mine(inner, asynchronous=True) is Mine(inner, asynchronous=True)
    # On a handler, the family's class is never cached.
    handler = TreeView(inner)
    assert TreeFileSystem(handler) is not TreeFileSystem(handler)
    with pytest.raises(TypeError, match="names no handler_class"):

        class Unnamed(TreeFileSystem):
            cachable = False


def test_a_storages_class_names_its_handler_only() -> None:
    with pytest.raises(TypeError, match="_info belongs to its handler"):

        class Overriding(TreeFileSystem):
            handler_class = TreeView

            async def _info(self, path: str, **kwargs: Any) -> Any: ...

    with pytest.raises(TypeError, match="storage_profile belongs to its handler"):

        class Declaring(TreeFileSystem):
            handler_class = TreeView
            storage_profile = StorageProfile()


def test_a_filesystems_paths_are_its_handlers() -> None:
    class Bucket(TreeView):
        protocol = ("bkt", "bucket")
        root = ""

    class BucketFileSystem(TreeFileSystem):
        cachable = False
        handler_class = Bucket

    # A class takes its paths from its handler class, before it is made.
    assert BucketFileSystem.protocol == ("bkt", "bucket")
    assert BucketFileSystem._strip_protocol("bucket://a/b/") == "a/b"
    assert BucketFileSystem._strip_protocol(["bkt://a", "bkt::b"]) == ["a", "b"]
    assert BucketFileSystem.root_marker == ""

    fs = BucketFileSystem(_storage(), asynchronous=True)
    assert fs.unstrip_protocol("a/b") == "bkt://a/b"
    assert fs.unstrip_protocol("bucket://a/b") == "bucket://a/b"

    # A filesystem on a handler serves the handler's paths too.
    on_handler = TreeFileSystem(Bucket(_storage()))
    assert on_handler.protocol == ("bkt", "bucket")
    assert on_handler._strip_protocol("bkt://x") == "x"
    assert on_handler.unstrip_protocol("x") == "bkt://x"
