"""Tests of the wrappers of s3fs, gcsfs and adlfs, on fakes answering as each does.

The fakes keep only what each wrapper calls, with the arguments it passes:
the copied package never imports s3fs, gcsfs or adlfs, nor do its tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.wrapper.adlfs import AdlfsWrapper
from fsspec_alt_strategies_impl.wrapper.gcsfs import GcsfsWrapper
from fsspec_alt_strategies_impl.wrapper.s3fs import S3fsWrapper

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_LIMITS: dict[str, Any] = {
    "operation_concurrency": 4,
    "io_concurrency": 2,
    "storage_profile": StorageProfile(),
}


class _Store(AsyncFileSystem):
    """Keys of buckets, or containers, as an object store holds them, and the requests made."""

    cachable = False  # Each test's own keys and calls.

    def __init__(self, keys: dict[str, int]) -> None:
        super().__init__(asynchronous=True)
        self.keys = keys
        self.calls: list[tuple[Any, ...]] = []

    def split_path(self, path: str) -> tuple[str, str, None]:
        bucket, _, key = path.partition("/")
        return bucket, key, None

    def _starting(self, bucket: str, start: str) -> list[str]:
        if not any(k.startswith(f"{bucket}/") for k in self.keys):
            raise FileNotFoundError(bucket)
        return sorted(k for k in self.keys if k.startswith(f"{bucket}/{start}"))


class _S3(_Store):
    async def _iterdir(
        self, bucket: str, delimiter: str = "/", prefix: str = ""
    ) -> AsyncIterator[dict[str, Any]]:
        self.calls.append(("_iterdir", bucket, delimiter, prefix))
        for key in self._starting(bucket, prefix):
            yield {"name": key, "type": "file", "size": self.keys[key]}

    async def _copy_basic(self, path1: str, path2: str, **kwargs: Any) -> None:
        self.calls.append(("_copy_basic", path1, path2))

    async def _copy_managed(
        self, path1: str, path2: str, size: int, **kwargs: Any
    ) -> None:
        self.calls.append(("_copy_managed", path1, path2, size))

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        self.calls.append(("_cp_file", path1, path2))

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        self.calls.append(("_rm_file", path))


class _Gcs(_Store):
    async def _find(
        self, path: str, detail: bool = False, prefix: str = "", **kwargs: Any
    ) -> dict[str, dict[str, Any]]:
        self.calls.append(("_find", path, prefix))
        bucket, key, _ = self.split_path(path)
        start = f"{key}/{prefix}" if key else prefix
        # Unordered, as gcsfs's listing in a dict may come.
        names = reversed(self._starting(bucket, start))
        return {k: {"name": k, "type": "file", "size": self.keys[k]} for k in names}


def _s3(keys: dict[str, int]) -> tuple[_S3, S3fsWrapper]:
    fs = _S3(keys)
    return fs, S3fsWrapper(fs, rm_batch_size=1000, **_LIMITS)


def _gcs(keys: dict[str, int]) -> tuple[_Gcs, GcsfsWrapper]:
    fs = _Gcs(keys)
    return fs, GcsfsWrapper(fs, rm_batch_size=100, **_LIMITS)


_KEYS = {
    "bkt/data/2023-12.csv": 1,
    "bkt/data/2024-01.csv": 2,
    "bkt/data/2024-02/a.csv": 3,
    "bkt/other.csv": 4,
}


async def test_s3fs_lists_a_prefix_in_one_streamed_listing() -> None:
    fs, flat = _s3(_KEYS)
    names = [e["name"] async for e in flat._ils("bkt/data", True, prefix="2024-")]
    assert names == ["bkt/data/2024-01.csv", "bkt/data/2024-02/a.csv"]
    assert fs.calls == [("_iterdir", "bkt", "", "data/2024-")]


async def test_s3fs_lists_a_bucket_whole() -> None:
    fs, flat = _s3(_KEYS)
    names = [e["name"] async for e in flat._ils("bkt", True)]
    assert names == sorted(_KEYS)
    assert fs.calls == [("_iterdir", "bkt", "", "")]


async def test_s3fs_lists_nothing_of_a_missing_bucket() -> None:
    _, flat = _s3(_KEYS)
    assert [e async for e in flat._ils("missing/data", True)] == []


async def test_s3fs_finds_by_its_listing() -> None:
    fs, flat = _s3(_KEYS)
    assert await flat._find("bkt/data") == [
        "bkt/data/2023-12.csv",
        "bkt/data/2024-01.csv",
        "bkt/data/2024-02/a.csv",
    ]
    assert [c[0] for c in fs.calls] == ["_iterdir"]


@pytest.mark.parametrize(
    ("size", "call"),
    [
        (5 * 1024**3, ("_copy_basic", "bkt/a", "bkt/b")),
        (5 * 1024**3 + 1, ("_copy_managed", "bkt/a", "bkt/b", 5 * 1024**3 + 1)),
        (None, ("_cp_file", "bkt/a", "bkt/b")),
    ],
)
async def test_s3fs_copies_by_the_listed_size(size: int | None, call: tuple) -> None:
    fs, flat = _s3({})
    await flat._cp_file_entry({"name": "bkt/a", "type": "file", "size": size}, "bkt/b")
    assert fs.calls == [call]


async def test_s3fs_moves_by_a_copy_then_a_delete() -> None:
    fs, flat = _s3({})
    await flat._mv_file_entry({"name": "bkt/a", "type": "file", "size": 1}, "bkt/b")
    assert fs.calls == [("_copy_basic", "bkt/a", "bkt/b"), ("_rm_file", "bkt/a")]


async def test_gcsfs_lists_a_prefix_by_its_find_in_name_order() -> None:
    fs, flat = _gcs(_KEYS)
    names = [e["name"] async for e in flat._ils("bkt/data", True, prefix="2024-")]
    assert names == ["bkt/data/2024-01.csv", "bkt/data/2024-02/a.csv"]
    assert fs.calls == [("_find", "bkt/data", "2024-")]


class _GcsBatches(_Gcs):
    """gcsfs's batch deletion: each object deleted answers its path, any other an OSError."""

    on_google = True

    def __init__(self, keys: dict[str, int], refused: set[str] = frozenset()) -> None:
        super().__init__(keys)
        self.refused = refused

    async def _rm_files(self, paths: list[str]) -> list[Any]:
        self.calls.append(("batch", len(paths)))
        out: list[Any] = []
        for path in paths:
            if path in self.refused:
                out.append(OSError('{"code": 403, "message": "retention"}'))
            elif path in self.keys:
                del self.keys[path]
                out.append(path)
            else:
                out.append(OSError('{"code": 404, "message": "No such object"}'))
        return out

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        if path not in self.keys:
            raise FileNotFoundError(path)
        return {"name": path, "type": "file", "size": self.keys[path]}

    async def _rm(self, path: Any, **kwargs: Any) -> None:
        self.calls.append(("_rm", len(path)))


async def test_gcsfs_deletes_up_to_100_per_batch_by_its_own_rm_files() -> None:
    fs = _GcsBatches({f"bkt/k{i}": 1 for i in range(150)} | {"bkt/kept": 1})
    flat = GcsfsWrapper(fs, rm_batch_size=1000, **_LIMITS)
    paths = [f"bkt/k{i}" for i in range(150)] + ["bkt/missing"]
    assert await flat._rm_files(paths) == {}
    assert fs.calls == [("batch", 100), ("batch", 51)]
    assert list(fs.keys) == ["bkt/kept"]


async def test_gcsfs_returns_the_keys_it_refused() -> None:
    fs = _GcsBatches({"bkt/a": 1, "bkt/b": 1}, refused={"bkt/b"})
    flat = GcsfsWrapper(fs, rm_batch_size=100, **_LIMITS)
    refused = await flat._rm_files(["bkt/a", "bkt/b", "bkt/missing"])
    assert list(refused) == ["bkt/b"]
    assert "retention" in str(refused["bkt/b"])


async def test_gcsfs_off_google_deletes_as_fsspec() -> None:
    fs = _GcsBatches({"bkt/a": 1})
    fs.on_google = False  # E.g. an emulator, where gcsfs sends no batch.
    flat = GcsfsWrapper(fs, rm_batch_size=100, **_LIMITS)
    await flat._rm_files(["bkt/a"])
    assert fs.calls == [("_rm", 1)]


async def test_gcsfs_lists_nothing_of_a_missing_bucket() -> None:
    _, flat = _gcs(_KEYS)
    assert [e async for e in flat._ils("missing", True)] == []


# adlfs: a container client of the Azure SDK, listing blobs in pages.


class _Blob(SimpleNamespace):
    name: str
    size: int | None


class _Pages:
    def __init__(self, blobs: list[_Blob], per_page: int) -> None:
        self.pages = [blobs[i : i + per_page] for i in range(0, len(blobs), per_page)]

    async def by_page(self) -> AsyncIterator[AsyncIterator[_Blob]]:
        for page in self.pages:
            yield _aiter(page)


async def _aiter(blobs: list[_Blob]) -> AsyncIterator[_Blob]:
    for blob in blobs:
        yield blob


class _Adl(_Store):
    """Blobs by container, a directory's blob of size None, as an HNS account lists it."""

    def __init__(self, keys: dict[str, int | None]) -> None:
        super().__init__(keys)  # type: ignore[arg-type]
        self.service_client = SimpleNamespace(get_container_client=self._client)

    def _client(self, container: str) -> Any:
        def list_blobs(include: list[str], name_starts_with: str) -> _Pages:
            self.calls.append(("list_blobs", container, name_starts_with))
            names = self._starting(container, name_starts_with)
            blobs = [
                _Blob(name=n.partition("/")[2], container=container, size=self.keys[n])
                for n in names
            ]
            return _Pages(blobs, per_page=2)

        return SimpleNamespace(list_blobs=list_blobs)

    async def _details(self, contents: list[_Blob]) -> list[dict[str, Any]]:
        return [
            {
                "name": f"{b.container}/{b.name}",
                "size": b.size,
                "type": "file" if b.size is not None else "directory",
            }
            for b in contents
        ]

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("_info", path))
        if path in self.keys:
            size = self.keys[path]
            kind = "file" if size is not None else "directory"
            return {"name": path, "size": size or 0, "type": kind}
        if any(k.startswith(f"{path}/") for k in self.keys):
            return {"name": path, "size": 0, "type": "directory"}
        raise FileNotFoundError(path)


_TREE: dict[str, int | None] = {
    "c/data": None,
    "c/data/a.csv": 1,
    "c/data/sub": None,
    "c/data/sub/b.csv": 2,
    "c/data/sub/c.csv": 3,
    "c/empty": None,
    "c/top.csv": 4,
}


def _adl(keys: dict[str, int | None]) -> tuple[_Adl, AdlfsWrapper]:
    fs = _Adl(keys)
    return fs, AdlfsWrapper(fs, **_LIMITS)


async def test_adlfs_lists_a_tree_in_one_listing_of_pages() -> None:
    fs, tree = _adl(_TREE)
    entries = [e async for e in tree._ils("c/data", True)]
    assert [(e["name"], e["type"], e["size"]) for e in entries] == [
        ("c/data/a.csv", "file", 1),
        ("c/data/sub", "directory", 0),
        ("c/data/sub/b.csv", "file", 2),
        ("c/data/sub/c.csv", "file", 3),
    ]
    assert fs.calls == [("list_blobs", "c", "data/")]


async def test_adlfs_finds_by_its_listing() -> None:
    fs, tree = _adl(_TREE)
    assert await tree._find("c/data") == [
        "c/data/a.csv",
        "c/data/sub/b.csv",
        "c/data/sub/c.csv",
    ]
    assert [c[0] for c in fs.calls if c[0] == "list_blobs"] == ["list_blobs"]


async def test_adlfs_lists_a_file_as_itself_and_an_empty_directory_as_nothing() -> None:
    _, tree = _adl(_TREE)
    assert [e["name"] async for e in tree._ils("c/top.csv", True)] == ["c/top.csv"]
    assert [e async for e in tree._ils("c/empty", True)] == []


@pytest.mark.parametrize("path", ["c/missing", "missing", "missing/data"])
async def test_adlfs_raises_on_a_missing_path(path: str) -> None:
    _, tree = _adl(_TREE)
    with pytest.raises(FileNotFoundError):
        [e async for e in tree._ils(path, True)]


async def test_adlfs_deletes_a_directory_by_its_blob_and_a_container_by_adlfs() -> None:
    calls: list[tuple[str, str]] = []

    class Adl(_Adl):
        async def _rm_file(self, path: str, **kwargs: Any) -> None:
            calls.append(("_rm_file", path))

        async def _rmdir(self, path: str) -> None:
            calls.append(("_rmdir", path))

    tree = AdlfsWrapper(Adl(_TREE), **_LIMITS)
    await tree._rmdir("c/data/sub")
    await tree._rmdir("c")
    assert calls == [("_rm_file", "c/data/sub"), ("_rmdir", "c")]


# Without the private methods each wrapper calls: as the generic wrapper.


async def test_s3fs_without_its_private_methods_lists_and_copies_as_fsspec() -> None:
    class Bare(_Store):
        async def _find(self, path: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("_find", path))
            return {
                k: {"name": k, "type": "file", "size": 1}
                for k in self._starting(*path.split("/", 1))
            }

        async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
            self.calls.append(("_cp_file", path1, path2))

    fs = Bare(_KEYS)
    flat = S3fsWrapper(fs, rm_batch_size=1000, **_LIMITS)
    names = [e["name"] async for e in flat._ils("bkt/data", True, prefix="2024-")]
    assert names == ["bkt/data/2024-01.csv", "bkt/data/2024-02/a.csv"]
    await flat._cp_file_entry({"name": "bkt/a", "type": "file", "size": 1}, "bkt/b")
    assert fs.calls == [("_find", "bkt/data"), ("_cp_file", "bkt/a", "bkt/b")]


async def test_gcsfs_whose_find_takes_no_prefix_lists_as_fsspec() -> None:
    class Bare(_Store):
        async def _find(self, path: str, detail: bool = False, **kwargs: Any) -> Any:
            assert "prefix" not in kwargs
            self.calls.append(("_find", path))
            return {
                k: {"name": k, "type": "file", "size": 1}
                for k in self._starting(*path.split("/", 1))
            }

    fs = Bare(_KEYS)
    flat = GcsfsWrapper(fs, rm_batch_size=100, **_LIMITS)
    names = [e["name"] async for e in flat._ils("bkt/data", True, prefix="2024-")]
    assert names == ["bkt/data/2024-01.csv", "bkt/data/2024-02/a.csv"]
    assert fs.calls == [("_find", "bkt/data")]


async def test_adlfs_without_its_client_lists_a_directory_at_a_time() -> None:
    class Bare(_Adl):
        def __init__(self, keys: dict[str, int | None]) -> None:
            super().__init__(keys)
            del self.service_client

        async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
            self.calls.append(("_ls", path))
            depth = path.count("/") + 1
            names = {
                "/".join(k.split("/")[: depth + 1])
                for k in self.keys
                if k.startswith(f"{path}/")
            }
            return [await self._info(n) for n in sorted(names)]

    fs = Bare(_TREE)
    tree = AdlfsWrapper(fs, **_LIMITS)
    names = sorted([e["name"] async for e in tree._ils("c/data", True)])
    assert names == [
        "c/data/a.csv",
        "c/data/sub",
        "c/data/sub/b.csv",
        "c/data/sub/c.csv",
    ]
    assert [c for c in fs.calls if c[0] == "_ls"] == [
        ("_ls", "c/data"),
        ("_ls", "c/data/sub"),
    ]
