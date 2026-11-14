# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""How to wrap a known filesystem, to monitor its calls.

`known_implementations` maps a filesystem's class, by its full name, to how a
copy of the strategies wraps it:

- its handler class, which runs the strategies that suit the storage: a
  generic one, e.g. "FsspecTreeHandler" or "FsspecFlatHandler", or one of a
  library, e.g. "S3fsHandler";
- for a generic handler, its `StorageProfile`, its concurrency limits and its
  batch sizes. A dedicated handler declares its own.

A generic handler calls the filesystem's async methods where its
``async_impl`` says it has them. The strategies' ``monitored`` reads the
table, or a table of the caller's own.

It is declared, never detected: an entry says what the developer knows of the
storage. A field left out of a profile is None, unknown: a storage's behaviour
that varies, or is not known. A project adds its own entries, or changes
one, before wrapping.

This module depends on `profile`.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from .profile import StorageProfile

__all__ = ["KnownImplementation", "known_implementations"]


class KnownImplementation(TypedDict):
    """How a filesystem class is wrapped, as an entry of `known_implementations`."""

    handler: str
    """The handler class wrapping it, which runs the strategies that suit the storage.

    The other fields are a generic handler's options, "FsspecTreeHandler" or
    "FsspecFlatHandler", each left out at its default, the one right on any
    storage. A dedicated handler, e.g. "S3fsHandler", or a class of the
    caller's own, declares its own as class attributes, and takes the
    filesystem alone: an entry naming one gives none.

    A bare class name, e.g. "FsspecTreeHandler" or "S3fsHandler", is one of
    the handlers that every copy of the strategies has. A full name, e.g.
    "mypkg.handlers.MyHandler", is a class of the caller's own, imported when
    wrapping. A storage of unknown kind takes "FsspecTreeHandler", the way
    right on any storage.
    """

    storage_profile: NotRequired[StorageProfile]
    """What the storage does, for a generic handler: unknown if left out."""

    operation_concurrency: NotRequired[int]
    """A generic handler's requests of the strategies at a time: 1 if left out."""

    io_concurrency: NotRequired[int]
    """A generic handler's reads and writes of data at a time: 1 if left out."""

    rm_batch_size: NotRequired[int]
    """A generic handler's batch of deletions, e.g. 1000 on S3: 1 if left out."""

    cp_batch_size: NotRequired[int]
    """A generic handler's batch of copies within the storage: 1 if left out."""

    mv_batch_size: NotRequired[int]
    """A generic handler's batch of moves within the storage: 1 if left out."""


known_implementations: dict[str, KnownImplementation] = {
    "fsspec.implementations.local.LocalFileSystem": {
        # Its mv is a rename, of a directory too, and its recursive rm one call,
        # shutil.rmtree. A file's directory must exist, as auto_mkdir is off by
        # default. Whether names are case-sensitive depends on the disk, e.g. not
        # on Windows or macOS by default: unknown.
        "handler": "FsspecTreeHandler",
        "storage_profile": StorageProfile(
            parents_required=True,
            copy_parents_required=True,
            implicit_directories=False,
            renames_files=True,
            renames_directories=True,
            deletes_directories=True,
            copies_directories=False,
            case_sensitive=None,
            recursive_order="depth_first",
        ),
        "operation_concurrency": 8,
        "io_concurrency": 4,
    },
    "s3fs.core.S3FileSystem": {
        # Its handler declares its profile, limits and batches: see S3fsHandler.
        "handler": "S3fsHandler",
    },
    "gcsfs.core.GCSFileSystem": {
        # For buckets without a hierarchical namespace. One with it has folders of
        # their own, which the flat strategies leave after a recursive rm or mv:
        # wrap it as a tree. Its handler declares the rest: see GcsfsHandler.
        "handler": "GcsfsHandler",
    },
    "adlfs.spec.AzureBlobFileSystem": {
        # Its handler declares its profile and limits: see AdlfsHandler.
        "handler": "AdlfsHandler",
    },
    "fsspec.implementations.memory.MemoryFileSystem": {
        # Its directories are made by mkdir, or implied by the files in them: a
        # mix, so unknown. Its mv is a copy, then a delete.
        "handler": "FsspecTreeHandler",
        "storage_profile": StorageProfile(
            parents_required=False,
            copy_parents_required=False,
            implicit_directories=None,
            renames_files=False,
            renames_directories=False,
            deletes_directories=False,
            copies_directories=False,
            case_sensitive=True,
            recursive_order="depth_first",
        ),
        "operation_concurrency": 4,
        "io_concurrency": 4,
    },
    "fsspec.implementations.sftp.SFTPFileSystem": {
        # Its mv is a rename, of a directory too. A file's directory must exist. Its
        # recursive rm walks the directory over the connection. Whether names are
        # case-sensitive depends on the server's disk: unknown.
        "handler": "FsspecTreeHandler",
        "storage_profile": StorageProfile(
            parents_required=True,
            copy_parents_required=True,
            implicit_directories=False,
            renames_files=True,
            renames_directories=True,
            deletes_directories=False,
            copies_directories=False,
            case_sensitive=None,
            recursive_order="depth_first",
        ),
        # One SSH channel, which paramiko does not share between threads.
        "operation_concurrency": 1,
        "io_concurrency": 1,
    },
    "fsspec.implementations.arrow.HadoopFileSystem": {
        # Its mv is HDFS's rename, of a directory too, and its recursive rm one
        # delete. HDFS creates a file's missing parents.
        "handler": "FsspecTreeHandler",
        "storage_profile": StorageProfile(
            parents_required=False,
            copy_parents_required=False,
            implicit_directories=False,
            renames_files=True,
            renames_directories=True,
            deletes_directories=True,
            copies_directories=False,
            case_sensitive=True,
            recursive_order="depth_first",
        ),
        "operation_concurrency": 8,
        "io_concurrency": 4,
    },
}
"""The filesystems known, by the full name of their class, and how each is wrapped."""
