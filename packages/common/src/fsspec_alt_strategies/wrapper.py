# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""How to wrap a known filesystem, to monitor its calls.

`known_implementations` maps a filesystem's class, by its full name, to how a
copy of the strategies wraps it:

- its `StorageProfile`, whose ``fs_type`` picks a tree or a flat handler;
- its concurrency limits, and a flat handler's batch of deletions;
- optionally, its handler class, where the generic one of its family is not
  the best.

Without a handler class, the family's generic handler is used, calling the
filesystem's async methods where its ``async_impl`` says it has them. The strategies' ``monitored`` reads the
table, or a table of the caller's own.

It is declared, never detected: an entry says what the developer knows of the
storage. Each entry sets every field of its profile, None where the storage's
behaviour varies or is unknown. A project adds its own entries, or changes
one, before wrapping.

This module depends on `profile`.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from .profile import StorageProfile

__all__ = ["KnownImplementation", "known_implementations"]


class KnownImplementation(TypedDict):
    """How a filesystem class is wrapped, as an entry of `known_implementations`."""

    storage_profile: StorageProfile
    """What the storage does: its ``fs_type`` is the family of the handler.

    A family left unknown is a tree, the way right on any storage.
    """

    operation_concurrency: int
    """The handler's requests of the strategies at a time."""

    io_concurrency: int
    """The handler's reads and writes of data at a time."""

    rm_batch_size: NotRequired[int]
    """A flat handler's batch of deletions, e.g. 1000 on S3."""

    handler: NotRequired[str]
    """The handler class wrapping it, of the family the profile's ``fs_type`` says.

    A bare class name, e.g. "S3fsHandler", is one of the handlers that every
    copy of the strategies has. A full name, e.g. "mypkg.handlers.MyHandler",
    is a class of the caller's own, imported when wrapping. Without it, the
    generic handler of the family is used.
    """


known_implementations: dict[str, KnownImplementation] = {
    "fsspec.implementations.local.LocalFileSystem": {
        # Its mv is a rename, of a directory too, and its recursive rm one call,
        # shutil.rmtree. A file's directory must exist, as auto_mkdir is off by
        # default. Whether names are case-sensitive depends on the disk, e.g. not
        # on Windows or macOS by default: unknown.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=True,
            copy_parents_required=True,
            implicit_directories=False,
            renames_files=True,
            renames_directories=True,
            deletes_directories=True,
            copies_directories=False,
            case_sensitive=None,
        ),
        "operation_concurrency": 8,
        "io_concurrency": 4,
    },
    "s3fs.core.S3FileSystem": {
        # Directories are prefixes. Its mv is a copy, then a delete.
        "storage_profile": StorageProfile(
            fs_type="flat",
            parents_required=False,
            copy_parents_required=False,
            implicit_directories=True,
            renames_files=False,
            renames_directories=False,
            deletes_directories=False,
            copies_directories=False,
            case_sensitive=True,
        ),
        "operation_concurrency": 32,
        "io_concurrency": 8,
        "rm_batch_size": 1000,
        # Streams its listing by prefix, and copies by the listed size.
        "handler": "S3fsHandler",
    },
    "gcsfs.core.GCSFileSystem": {
        # For buckets without a hierarchical namespace. One with it has folders of
        # their own, which the flat strategies leave after a recursive rm or mv:
        # wrap it as a tree. Both kinds served are a router, a filesystem per
        # bucket. The handler moves by _mv_file, which is fsspec's copy, then a
        # delete.
        "storage_profile": StorageProfile(
            fs_type="flat",
            parents_required=False,
            copy_parents_required=False,
            implicit_directories=None,
            renames_files=False,
            renames_directories=False,
            deletes_directories=False,
            copies_directories=False,
            case_sensitive=True,
        ),
        "operation_concurrency": 32,
        "io_concurrency": 8,
        # As GCS's batch requests take at most 100 calls.
        "rm_batch_size": 100,
        # Lists by prefix, and deletes 100 objects per batch request.
        "handler": "GcsfsHandler",
    },
    "adlfs.spec.AzureBlobFileSystem": {
        # Its ls lists one directory per request, with walk_blobs and a delimiter,
        # with a hierarchical namespace or without. Its directories are real with
        # one, prefixes without, and one class serves both: unknown. It has no
        # move of its own, so its mv is fsspec's copy, then a delete, even with
        # a hierarchical namespace.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=False,
            copy_parents_required=False,
            implicit_directories=None,
            renames_files=False,
            renames_directories=False,
            deletes_directories=False,
            copies_directories=False,
            case_sensitive=True,
        ),
        "operation_concurrency": 32,
        "io_concurrency": 8,
        # Lists a tree by one listing without a delimiter.
        "handler": "AdlfsHandler",
    },
    "fsspec.implementations.memory.MemoryFileSystem": {
        # Its directories are made by mkdir, or implied by the files in them: a
        # mix, so unknown. Its mv is a copy, then a delete.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=False,
            copy_parents_required=False,
            implicit_directories=None,
            renames_files=False,
            renames_directories=False,
            deletes_directories=False,
            copies_directories=False,
            case_sensitive=True,
        ),
        "operation_concurrency": 4,
        "io_concurrency": 4,
    },
    "fsspec.implementations.sftp.SFTPFileSystem": {
        # Its mv is a rename, of a directory too. A file's directory must exist. Its
        # recursive rm walks the directory over the connection. Whether names are
        # case-sensitive depends on the server's disk: unknown.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=True,
            copy_parents_required=True,
            implicit_directories=False,
            renames_files=True,
            renames_directories=True,
            deletes_directories=False,
            copies_directories=False,
            case_sensitive=None,
        ),
        # One SSH channel, which paramiko does not share between threads.
        "operation_concurrency": 1,
        "io_concurrency": 1,
    },
    "fsspec.implementations.arrow.HadoopFileSystem": {
        # Its mv is HDFS's rename, of a directory too, and its recursive rm one
        # delete. HDFS creates a file's missing parents.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=False,
            copy_parents_required=False,
            implicit_directories=False,
            renames_files=True,
            renames_directories=True,
            deletes_directories=True,
            copies_directories=False,
            case_sensitive=True,
        ),
        "operation_concurrency": 8,
        "io_concurrency": 4,
    },
}
"""The filesystems known, by the full name of their class, and how each is wrapped."""
