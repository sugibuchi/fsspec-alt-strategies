# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""How to wrap a known filesystem, to monitor its calls.

`known_implementations` maps a filesystem's class, by its full name, to how a
copy of the strategies wraps it:

- its `StorageProfile`, whose ``fs_type`` picks a tree or a flat wrapper;
- its concurrency limits, and a flat wrapper's batch of deletions;
- optionally, its wrapper class, where the generic one of its family is not
  the best.

Without a wrapper class, the family's generic wrapper is used, async if the
filesystem is an ``AsyncFileSystem``. The strategies' ``monitored`` reads the
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
    """What the storage does: its ``fs_type`` is the family of the wrapper.

    A family left unknown is a tree, the way right on any storage.
    """

    operation_concurrency: int
    """The wrapper's requests of the strategies at a time."""

    io_concurrency: int
    """The wrapper's reads and writes of data at a time."""

    rm_batch_size: NotRequired[int]
    """A flat wrapper's batch of deletions, e.g. 1000 on S3."""

    wrapper: NotRequired[str]
    """The wrapper class, of the family the profile's ``fs_type`` says.

    A bare class name, e.g. "S3fsWrapper", is one of the wrappers that every
    copy of the strategies has. A full name, e.g. "mypkg.wrappers.MyWrapper",
    is a class of the caller's own, imported when wrapping. Without it, the
    generic wrapper of the family is used.
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
        "wrapper": "S3fsWrapper",
    },
    "gcsfs.core.GCSFileSystem": {
        # For buckets without a hierarchical namespace. One with it has folders of
        # their own, which the flat strategies leave after a recursive rm or mv:
        # wrap it as a tree. Both kinds served are a router, a filesystem per
        # bucket. The wrapper moves by _mv_file, which is fsspec's copy, then a
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
        "wrapper": "GcsfsWrapper",
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
        "wrapper": "AdlfsWrapper",
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
