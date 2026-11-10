# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""How to wrap a known filesystem, to monitor its calls.

`known_implementations` maps a filesystem's class, by its full name, to how a
copy of the strategies wraps it:

- its `StorageProfile`, whose ``fs_type`` picks a tree or a flat wrapper;
- its concurrency limits, and a flat wrapper's batch of deletions;
- the recursive methods passed to the filesystem as they are, if any.

Whether the wrapper is async is not declared: it is async if the filesystem
is an ``AsyncFileSystem``. The strategies' ``monitored`` reads the table, or a
table of the caller's own.

It is declared, never detected: an entry says what the developer knows of the
storage. Each entry sets every field of its profile, None where the storage's
behaviour varies or is unknown. A project adds its own entries, or changes
one, before wrapping.

This module depends on `profile`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

from .profile import StorageProfile

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    native: NotRequired[Sequence[str]]
    """The recursive methods the wrapper passes to the filesystem as they are.

    Each is "mv", "rm", "copy", "find", "glob" or "du": the others run the
    family's strategies. One suits a filesystem whose own method uses the
    storage better than any strategy can, e.g. gcsfs's ``mv``, which renames a
    folder where the bucket allows it. Such a call is not streamed by the
    strategies, and a monitor sees no step of it.
    """


known_implementations: dict[str, KnownImplementation] = {
    "fsspec.implementations.local.LocalFileSystem": {
        # Its mv is a rename, of a folder too, and its recursive rm one call,
        # shutil.rmtree. A file's folder must exist, as auto_mkdir is off by
        # default.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=True,
            implicit_folders=False,
            renames_files=True,
            renames_folders=True,
            deletes_folders=True,
        ),
        "operation_concurrency": 8,
        "io_concurrency": 4,
    },
    "s3fs.core.S3FileSystem": {
        # Folders are prefixes. Its mv is a copy, then a delete.
        "storage_profile": StorageProfile(
            fs_type="flat",
            parents_required=False,
            implicit_folders=True,
            renames_files=False,
            renames_folders=False,
            deletes_folders=False,
        ),
        "operation_concurrency": 32,
        "io_concurrency": 8,
        "rm_batch_size": 1000,
    },
    "gcsfs.core.GCSFileSystem": {
        # A bucket with a hierarchical namespace has folders of its own: one
        # class serves both kinds of bucket, so that is unknown. Its _mv renames
        # a folder in such a bucket, but the wrapper moves by _mv_file, which
        # is fsspec's copy, then a delete.
        "storage_profile": StorageProfile(
            fs_type="flat",
            parents_required=False,
            implicit_folders=None,
            renames_files=False,
            renames_folders=False,
            deletes_folders=False,
        ),
        "operation_concurrency": 32,
        "io_concurrency": 8,
        # As GCS's batch requests take at most 100 calls.
        "rm_batch_size": 100,
    },
    "adlfs.spec.AzureBlobFileSystem": {
        # Its ls lists one folder per request, with walk_blobs and a delimiter,
        # with a hierarchical namespace or without. Its folders are real with
        # one, prefixes without, and one class serves both: unknown. It has no
        # move of its own, so its mv is fsspec's copy, then a delete, even with
        # a hierarchical namespace.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=False,
            implicit_folders=None,
            renames_files=False,
            renames_folders=False,
            deletes_folders=False,
        ),
        "operation_concurrency": 32,
        "io_concurrency": 8,
    },
    "fsspec.implementations.memory.MemoryFileSystem": {
        # Its folders are made by mkdir, or implied by the files in them: a
        # mix, so unknown. Its mv is a copy, then a delete.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=False,
            implicit_folders=None,
            renames_files=False,
            renames_folders=False,
            deletes_folders=False,
        ),
        "operation_concurrency": 4,
        "io_concurrency": 4,
    },
    "fsspec.implementations.sftp.SFTPFileSystem": {
        # Its mv is a rename, of a folder too. A file's folder must exist. Its
        # recursive rm walks the folder over the connection.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=True,
            implicit_folders=False,
            renames_files=True,
            renames_folders=True,
            deletes_folders=False,
        ),
        # One SSH channel, which paramiko does not share between threads.
        "operation_concurrency": 1,
        "io_concurrency": 1,
    },
    "fsspec.implementations.arrow.HadoopFileSystem": {
        # Its mv is HDFS's rename, of a folder too, and its recursive rm one
        # delete. HDFS creates a file's missing parents.
        "storage_profile": StorageProfile(
            fs_type="tree",
            parents_required=False,
            implicit_folders=False,
            renames_files=True,
            renames_folders=True,
            deletes_folders=True,
        ),
        "operation_concurrency": 8,
        "io_concurrency": 4,
    },
}
"""The filesystems known, by the full name of their class, and how each is wrapped."""
