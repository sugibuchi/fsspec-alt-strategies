# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""How to wrap a known filesystem, to monitor its calls.

`known_implementations` maps a filesystem's class, by its full name, to how a
copy of the strategies wraps it: as a `TreeFileSystem` or a `FlatFileSystem`,
async or not, with its limits and its `StorageProfile`. The strategies'
``monitored`` reads it, or a table of the caller's own.

It is declared, never detected: an entry says what the developer knows of the
storage. A project adds its own, or changes one, before wrapping.

This module depends on `profile`.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from .profile import StorageProfile

__all__ = ["KnownImplementation", "known_implementations"]

KnownImplementation = TypedDict(
    "KnownImplementation",
    {
        # The family of the wrapper: a TreeFileSystem, or a FlatFileSystem.
        "type": Literal["tree", "flat"],
        # Whether the filesystem is an AsyncFileSystem, awaited, or a sync one.
        "async": bool,
        "operation_concurrency": int,
        "io_concurrency": int,
        "storage_profile": NotRequired[StorageProfile | None],
        # A flat wrapper's batch of deletions, e.g. 1000 on S3.
        "rm_batch_size": NotRequired[int],
    },
)
"""How a filesystem class is wrapped, as an entry of `known_implementations`."""

known_implementations: dict[str, KnownImplementation] = {
    "fsspec.implementations.local.LocalFileSystem": {
        "type": "tree",
        "async": False,
        "operation_concurrency": 8,
        "io_concurrency": 4,
        # Its mv is a rename, of a folder too; a file's folder must exist.
        "storage_profile": StorageProfile(
            parents_required=True, renames_files=True, renames_folders=True
        ),
    },
    "s3fs.core.S3FileSystem": {
        "type": "flat",
        "async": True,
        "operation_concurrency": 32,
        "io_concurrency": 8,
        "rm_batch_size": 1000,
        "storage_profile": StorageProfile(implicit_folders=True),
    },
}
"""The filesystems known, by the full name of their class, and how each is wrapped."""
