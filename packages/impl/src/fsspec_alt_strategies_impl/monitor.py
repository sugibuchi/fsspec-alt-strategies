# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A filesystem whose calls the monitors see: as it is, or wrapped.

`monitored` returns a filesystem that tells the monitors of a `monitoring`
block of its operations. One inheriting the interface's `Monitored`, such as a
`TreeFileSystem` or a wrapper, already does. Any other is wrapped as its entry
in the known implementations says: none is chosen for a filesystem not there.

This module depends on `adlfs`, `flat`, `gcsfs`, `s3fs` and `tree` of `wrapper`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.monitor import Monitored
from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies.wrapper import known_implementations

from .wrapper.adlfs import AdlfsWrapper
from .wrapper.flat import AsyncFlatWrapper, SyncFlatWrapper
from .wrapper.gcsfs import GcsfsWrapper
from .wrapper.s3fs import S3fsWrapper
from .wrapper.tree import AsyncTreeWrapper, SyncTreeWrapper

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fsspec.spec import AbstractFileSystem

    from fsspec_alt_strategies.wrapper import KnownImplementation

__all__ = ["monitored"]

# The wrappers of a class of their own, each of the family it is written for.
_OWN_WRAPPERS: dict[tuple[str, str], Any] = {
    ("s3fs.core.S3FileSystem", "flat"): S3fsWrapper,
    ("gcsfs.core.GCSFileSystem", "flat"): GcsfsWrapper,
    ("adlfs.spec.AzureBlobFileSystem", "tree"): AdlfsWrapper,
}


def _wrapper(name: str, kind: str, asynchronous: bool) -> Any:
    """The wrapper of the class named, of a family, async or not: its own if it has one."""
    own = _OWN_WRAPPERS.get((name, kind))
    if own is not None and asynchronous:
        return own
    elif kind == "tree":
        return AsyncTreeWrapper if asynchronous else SyncTreeWrapper
    else:
        return AsyncFlatWrapper if asynchronous else SyncFlatWrapper


def monitored(
    fs: AbstractFileSystem,
    implementations: Mapping[str, KnownImplementation] | None = None,
    wrap_unknown_fs: bool = False,
) -> AbstractFileSystem:
    """The filesystem, as one whose calls the monitors see.

    A filesystem inheriting `Monitored` is returned as it is. Any other is
    wrapped, as the entry of its class says, or of the nearest class it
    inherits. Its profile's ``fs_type`` picks a `TreeFileSystem` or a
    `FlatFileSystem`, a tree if unknown, and the entry gives its limits. The
    wrapper is async if the filesystem is an `AsyncFileSystem`. It is new, and
    may be dropped once used.

    s3fs, gcsfs and adlfs, or a class inheriting one, get a wrapper of their
    own, e.g. `S3fsWrapper`, listing and copying as each does best. It is
    used where the entry's family is the one it is written for.

    A filesystem no entry names is wrapped only if ``wrap_unknown_fs``: as
    a `TreeFileSystem`, one request at a time, with an unknown profile. That
    is right on any storage, if slow.

    Parameters
    ----------
    fs : AbstractFileSystem
        The filesystem.
    implementations : mapping, optional
        How each filesystem class is wrapped, by the full name of the class,
        e.g. "s3fs.core.S3FileSystem". The interface's
        `known_implementations` if None.
    wrap_unknown_fs : bool, default False
        Whether to wrap a filesystem no entry names, the way right on any
        storage, rather than raise.

    Returns
    -------
    AbstractFileSystem
        ``fs``, or a wrapper of it.

    Raises
    ------
    ValueError
        If ``fs`` is not monitored, no entry names its class, and
        ``wrap_unknown_fs`` is False.

    Examples
    --------
    >>> with monitoring(Progress()):
    ...     monitored(fsspec.filesystem("file")).rm("data/2024", recursive=True)
    """
    if isinstance(fs, Monitored):
        return fs

    table = known_implementations if implementations is None else implementations
    for cls in type(fs).__mro__:
        name = f"{cls.__module__}.{cls.__qualname__}"
        entry = table.get(name)
        if entry is not None:
            break
    else:
        if not wrap_unknown_fs:
            name = f"{type(fs).__module__}.{type(fs).__qualname__}"
            raise ValueError(f"No known implementation says how to wrap {name}")
        entry = {
            "storage_profile": StorageProfile(fs_type="tree"),
            "operation_concurrency": 1,
            "io_concurrency": 1,
        }

    # A family unknown is a tree, the way right on any storage.
    fs_type = entry["storage_profile"].fs_type or "tree"
    wrapper = _wrapper(name, fs_type, isinstance(fs, AsyncFileSystem))
    options: dict[str, Any] = {
        "operation_concurrency": entry["operation_concurrency"],
        "io_concurrency": entry["io_concurrency"],
        "storage_profile": entry["storage_profile"],
    }
    if fs_type == "flat":
        options["rm_batch_size"] = entry["rm_batch_size"]
    return wrapper(fs, **options)
