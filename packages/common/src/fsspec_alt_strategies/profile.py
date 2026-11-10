# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What a storage does that fsspec's API leaves open, as the developer declares it.

Storages answer the same request differently. Some create a file's missing
parent directories, others raise FileNotFoundError. A strategy works on any of
them, but sends fewer requests, or copies less data, when it knows.
`StorageProfile` tells it.

A filesystem has one profile, which never changes. A strategy reads it from the
filesystem it works on, or from the destination's for a copy or move.
`StorageProfile()` knows nothing, and is right on any storage.

The library never detects a storage's behaviour: the developer declares it. A
field left None is unknown, and a strategy then takes the way that is right on
any storage. So a profile only saves requests or data copied, and never
changes the outcome, unless it is wrong.

A fact gets a field only once a strategy saves requests or data copied with
it. Facts that cost nothing to handle as they come, such as a write refused
because the file exists, have none.

This module depends on nothing else in the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["StorageProfile"]


@dataclass(frozen=True, kw_only=True)
class StorageProfile:
    """What a storage does that fsspec's API leaves open.

    Each strategy's docstring says the fields it reads, and the requests each
    saves.

    Parameters
    ----------
    fs_type : {"tree", "flat"}, optional
        The family of strategies the storage's filesystem answers with.
        "tree" is a hierarchical namespace listed a directory per request, with
        ``_ils``, as on a `TreeFileSystem`. "flat" is keys listed by prefix,
        with ``_iexpand``, ``_rm_files`` and ``rm_batch_size``, as on a
        `FlatFileSystem`. A router copying or moving between two storages
        picks the source's strategies with it. Unknown by default: the router
        then copies with fsspec's own copy.
    parents_required : bool, optional
        Whether writing a file into a missing directory fails, rather than
        creating its parents. Unknown by default, taken as True: directories are
        created before files are written into them. It tells of writes only:
        a rename's parent is always created first, as ``_mv_file`` is called
        only where the destination's parent exists.
    implicit_directories : bool, optional
        Whether directories exist only through the files under them, with no
        entry of their own, as on object stores without a hierarchical
        namespace. A directory then goes with its last file, and a file written
        creates its parents. Unknown by default, taken as False: a directory
        emptied is deleted.
    renames_files : bool, optional
        Whether moving a file is one request that copies no data, a rename,
        rather than a copy and a delete. Unknown by default, taken as False: a
        move copies, and deletes in batches where it can.
    renames_directories : bool, optional
        Whether ``mv_file`` moves a directory too, with all in it, in one request,
        as a storage's rename does. Unknown by default, taken as False: a
        directory's files are moved one by one, as ``mv_file`` is called on files
        only. True holds between any two paths of the filesystem: a storage
        whose rename stops at a boundary, e.g. containers, is several
        filesystems under a router.
    deletes_directories : bool, optional
        Whether the filesystem's ``_rm_tree`` deletes a directory with all in
        it, in one request, as a storage's recursive delete does. It deletes a
        file too. Unknown by default, taken as False: a directory is walked, and
        what is in it deleted one by one. Declared True, a failed ``_rm_tree``
        is the directory's failure, never walked.

    Raises
    ------
    ValueError
        If ``fs_type`` is neither "tree" nor "flat", or if
        ``implicit_directories`` and ``parents_required`` are both True, as a
        file's parents then exist as soon as it does.

    Examples
    --------
    >>> class MyFileSystem(TreeFileSystem):
    ...     storage_profile = StorageProfile(implicit_directories=True)
    """

    fs_type: Literal["tree", "flat"] | None = None
    parents_required: bool | None = None
    implicit_directories: bool | None = None
    renames_files: bool | None = None
    renames_directories: bool | None = None
    deletes_directories: bool | None = None

    def __post_init__(self) -> None:
        if self.fs_type not in (None, "tree", "flat"):
            raise ValueError(f"fs_type must be 'tree' or 'flat': {self.fs_type!r}")
        if self.implicit_directories and self.parents_required:
            raise ValueError(
                "parents_required cannot be True with implicit_directories: a file's"
                " parents exist as soon as it does"
            )
