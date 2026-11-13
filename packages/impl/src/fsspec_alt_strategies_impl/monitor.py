# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A filesystem whose calls the monitors see: as it is, or wrapped.

`monitored` returns a filesystem that tells the monitors of a `monitoring`
block of its operations. One inheriting the interface's `Monitored`, such as a
`TreeFileSystem`, already does. Any other is wrapped by a handler, as its entry
in the known implementations says, and a `TreeFileSystemWrapper` or
`FlatFileSystemWrapper` made on it: none is chosen for a filesystem not there.

This module depends on `adlfs`, `flat`, `gcsfs`, `s3fs` and `tree` of
`wrapper`, and `handler` of `tree` and of `flat`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.monitor import Monitored
from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies.wrapper import known_implementations

from .flat.handler import FlatHandler
from .tree.handler import TreeHandler
from .wrapper.adlfs import AdlfsHandler
from .wrapper.flat import FlatFileSystemWrapper, FsspecFlatHandler
from .wrapper.gcsfs import GcsfsHandler
from .wrapper.s3fs import S3fsHandler
from .wrapper.tree import FsspecTreeHandler, TreeFileSystemWrapper

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fsspec.spec import AbstractFileSystem

    from fsspec_alt_strategies.wrapper import KnownImplementation

__all__ = ["monitored"]

_BUILT_IN: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        FsspecTreeHandler,
        FsspecFlatHandler,
        S3fsHandler,
        GcsfsHandler,
        AdlfsHandler,
    )
}
"""The handlers an entry names by their bare class name: every copy has them."""


def _named(name: str) -> type:
    """The handler class an entry names: built in by a bare name, else imported by its full name."""
    module, _, qualname = name.rpartition(".")
    if not module:
        if name not in _BUILT_IN:
            raise ValueError(f"No built-in handler is named {name}")
        return _BUILT_IN[name]
    else:
        cls = getattr(importlib.import_module(module), qualname, None)
        if not isinstance(cls, type):
            raise ValueError(f"{module} has no class {qualname}")
        return cls


def _handler_class(entry: KnownImplementation, kind: str) -> type:
    """The handler class of an entry: the one it names, or its family's generic one."""
    if "handler" not in entry:
        return FsspecTreeHandler if kind == "tree" else FsspecFlatHandler

    cls = _named(entry["handler"])
    family = TreeHandler if kind == "tree" else FlatHandler
    if not issubclass(cls, family):
        # A handler of another family would answer as a profile it does not have.
        raise ValueError(f"{entry['handler']} is not a {family.__name__}")
    return cls


def monitored(
    fs: AbstractFileSystem,
    implementations: Mapping[str, KnownImplementation] | None = None,
    wrap_unknown_fs: bool = False,
) -> AbstractFileSystem:
    """The filesystem, as one whose calls the monitors see.

    A filesystem inheriting `Monitored` is returned as it is. Any other is
    wrapped, as the entry of its class says, or of the nearest class it
    inherits: by a handler, and a filesystem made on it, serving the wrapped
    one's paths. Its profile's ``fs_type`` picks a `TreeFileSystemWrapper` or
    a `FlatFileSystemWrapper`, a tree if unknown, and the entry gives its
    limits. The handler is the family's
    generic one, which calls the filesystem's async methods where its
    ``async_impl`` says it has them, else its sync ones in threads. The
    filesystem returned is new, and may be dropped once used.

    An entry's ``handler`` names another class, of the same family:

    - a bare class name is a handler built in, e.g. "S3fsHandler", as the
      known entries of s3fs, gcsfs and adlfs name theirs;
    - a full name is a class of the caller's own, e.g.
      "mypkg.handlers.MyHandler", imported here.

    A filesystem no entry names is wrapped only if ``wrap_unknown_fs``: as a
    tree, one request at a time, with an unknown profile. That is right on
    any storage, if slow.

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
        ``fs``, or a `TreeFileSystemWrapper` or `FlatFileSystemWrapper` on a
        handler of it.

    Raises
    ------
    ValueError
        If ``fs`` is not monitored, no entry names its class, and
        ``wrap_unknown_fs`` is False. If the entry's ``handler`` names no
        class, or one of another family.
    ImportError
        If the module of the entry's ``handler`` cannot be imported.

    Examples
    --------
    >>> with monitoring(Progress()):
    ...     monitored(fsspec.filesystem("file")).rm("data/2024", recursive=True)
    """
    if isinstance(fs, Monitored):
        return fs

    table = known_implementations if implementations is None else implementations
    for cls in type(fs).__mro__:
        entry = table.get(f"{cls.__module__}.{cls.__qualname__}")
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
    handler_class: Any = _handler_class(entry, fs_type)
    options: dict[str, Any] = {
        "operation_concurrency": entry["operation_concurrency"],
        "io_concurrency": entry["io_concurrency"],
        "storage_profile": entry["storage_profile"],
    }
    if fs_type == "flat":
        options["rm_batch_size"] = entry["rm_batch_size"]
    handler = handler_class(fs, **options)

    # The filesystem runs on the wrapped one's loop, sync or not as it is.
    family = TreeFileSystemWrapper if fs_type == "tree" else FlatFileSystemWrapper
    loop: dict[str, Any] = {}
    if getattr(fs, "async_impl", False) and not fs.asynchronous:
        loop["loop"] = fs.loop
    return family(handler, asynchronous=getattr(fs, "asynchronous", False), **loop)
