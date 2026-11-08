# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""How to wrap a known filesystem, to monitor its calls.

`known_implementations` maps a filesystem's class, by its full name, to how a
copy of the strategies wraps it: with its `StorageProfile`, whose ``fs_type``
says a `TreeFileSystem` or a `FlatFileSystem`, async or not, with its limits. The strategies'
``monitored`` reads it, or a table of the caller's own.

It is declared, never detected: an entry says what the developer knows of the
storage. A project adds its own, or changes one, before wrapping.

This module depends on `profile`.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from .profile import StorageProfile

__all__ = ["KnownImplementation", "known_implementations"]

KnownImplementation = TypedDict(
    "KnownImplementation",
    {
        # What the storage does: StorageProfile() if it is unknown. Its fs_type
        # is the family of the wrapper, a TreeFileSystem or a FlatFileSystem:
        # a tree, the way right on any storage, if unknown.
        "storage_profile": StorageProfile,
        # Whether the filesystem is an AsyncFileSystem, awaited, or a sync one.
        "async": bool,
        "operation_concurrency": int,
        "io_concurrency": int,
        # A flat wrapper's batch of deletions, e.g. 1000 on S3.
        "rm_batch_size": NotRequired[int],
    },
)
"""How a filesystem class is wrapped, as an entry of `known_implementations`."""

known_implementations: dict[str, KnownImplementation] = {
    "fsspec.implementations.local.LocalFileSystem": {
        # Its mv is a rename, of a folder too; a file's folder must exist.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=True,
            renames_files=True,
            renames_folders=True,
        ),
        "async": False,
        "operation_concurrency": 8,
        "io_concurrency": 4,
    },
    "s3fs.core.S3FileSystem": {
        "storage_profile": StorageProfile(fs_type="flat", implicit_folders=True),
        "async": True,
        "operation_concurrency": 32,
        "io_concurrency": 8,
        "rm_batch_size": 1000,
    },
    "gcsfs.core.GCSFileSystem": {
        # A bucket with a hierarchical namespace has folders of its own: unknown.
        "storage_profile": StorageProfile(fs_type="flat"),
        "async": True,
        "operation_concurrency": 32,
        "io_concurrency": 8,
        # As GCS's batch requests take at most 100 calls.
        "rm_batch_size": 100,
    },
    "adlfs.spec.AzureBlobFileSystem": {
        # Its ls lists one folder per request, with walk_blobs and a delimiter,
        # with a hierarchical namespace or without. Its folders are real with a
        # hierarchical namespace, prefixes without, and one class serves both:
        # unknown.
        "storage_profile": StorageProfile(fs_type="tree"),
        "async": True,
        "operation_concurrency": 32,
        "io_concurrency": 8,
    },
    "fsspec.implementations.memory.MemoryFileSystem": {
        "storage_profile": StorageProfile(fs_type="tree"),
        "async": False,
        "operation_concurrency": 4,
        "io_concurrency": 4,
    },
    "fsspec.implementations.sftp.SFTPFileSystem": {
        # Its mv is a rename, of a folder too; a file's folder must exist.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=True,
            renames_files=True,
            renames_folders=True,
        ),
        "async": False,
        # One SSH channel, which paramiko does not share between threads.
        "operation_concurrency": 1,
        "io_concurrency": 1,
    },
    "fsspec.implementations.arrow.HadoopFileSystem": {
        # Its mv is HDFS's rename, of a folder too.
        "storage_profile": StorageProfile(
            fs_type="tree", renames_files=True, renames_folders=True
        ),
        "async": False,
        "operation_concurrency": 8,
        "io_concurrency": 4,
    },
}
"""The filesystems known, by the full name of their class, and how each is wrapped."""
