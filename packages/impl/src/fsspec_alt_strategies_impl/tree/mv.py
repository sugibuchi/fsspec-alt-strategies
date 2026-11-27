# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive move that walks the tree and moves while it walks.

`_tree_mv` is an alternative to fsspec's recursive ``mv``. fsspec's copies the
whole tree, then deletes it, each stage listing the tree first. `_tree_mv`
moves each directory's files as soon as the directory is listed, each by one
``mv_entry``, and deletes each source directory once it is empty.

Where the handler's ``mv_tree`` moves a directory in one request, as a
storage's rename does, the profile says so. `_tree_mv` then moves each root
whole with it where it can, and walks only what it cannot move so.

This module depends on `cp` and `walk`, and `paths` and `transfer` of `common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.paths import _check_maxdepth
from ..common.transfer import _moves_of
from .cp import _transfers
from .walk import _MAX_RUNNING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fsspec_alt_strategies.states import MvState

    from ..common.handler import StorageHandler, StoragePath
    from ..common.operation import Operation
    from ..common.transfer import _Transfer
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


def _tree_mv(
    src: TreeFileOperationHandler,
    dst: StorageHandler,
    path1: StoragePath | Sequence[StoragePath],
    path2: StoragePath,
    *,
    cp_entry: _Transfer | None = None,
    mv_entry: _Transfer | None = None,
    same_storage: bool | None = None,
    maxdepth: int | None = None,
    path1_entry: dict[str, Any] | None = None,
    ils_kwargs: dict[str, Any] | None = None,
    max_running: int = _MAX_RUNNING,
) -> Operation[tuple[str, str], MvState]:
    """Move a directory and everything in it, moving while the tree is walked.

    It places the paths as fsspec's ``mv`` with ``recursive=True`` does. If
    ``path2`` is an existing directory, or ends with "/", and ``path1`` does not
    end with "/", ``path1`` goes into it: ``path1/x`` becomes
    ``path2/<name of path1>/x``. Otherwise ``path1`` becomes ``path2``:
    ``path1/x`` becomes ``path2/x``. A file moves the same way. Existing files
    are overwritten, as fsspec's ``mv`` does.

    A glob pattern moves each path it matches, and all in it, placed as
    fsspec's ``mv`` places them. Their paths below the matches' common
    prefix are kept under ``path2``, but a single file matched goes into
    ``path2``. The pattern is matched as `_tree_glob` matches it, then the
    directories matched are walked at once.

    A list of paths and patterns moves each path it names or matches into
    ``path2``, under its name, with all in it, as `_tree_cp` copies it. Unlike
    fsspec's ``mv``, it keeps the directories' trees. A path whose name another
    path takes first yields a FileExistsError, and stays where it is.

    The tree is walked as `_tree_walk` walks it. Each directory's files are
    moved as soon as the directory is listed, each by one ``mv_entry``. A
    source directory is deleted once everything in it is moved, so the deepest
    go first. A directory is created at the destination before files are moved
    into it, and for each empty directory.

    The destination is described by ``dst`` and its ``profile``,
    unknown if it has none. Within one storage, ``dst`` is ``src``, and
    ``same_storage`` is True. Between two, ``mv_entry`` copies a file to the
    other storage, then deletes it: by default ``cp_entry``, then the source's
    ``rm_file``. A path may then go to the same path, or under it, and nothing
    is moved whole.

    Unlike fsspec's ``mv``, it keeps going past a failure, and reports each. A
    file that failed to move stays where it was, and so do the directories above
    it. A storage refusing to overwrite yields a FileExistsError for each
    file already at its destination, as fsspec's ``mv`` raises one.

    It wins over fsspec's ``mv`` on storages that list one directory per request
    and rename nothing in one request. The moves then start after the first
    listing, not after the whole tree is listed twice.

    Within one storage, with the profile's ``renames_directories=True``,
    ``mv_tree`` moves a directory with all in it, in one request. Each
    directory root is then moved whole by one ``mv_tree``, where it can be, as
    a job among the files' moves. That wins by far on storages renaming a
    directory at once, such as Azure Data Lake Storage Gen2, HDFS or a local
    disk. A root goes whole only to a destination that does not exist, as a
    rename onto an existing path differs among storages. Into an existing
    directory, its contents merge, and its files are moved one by one. So are
    those of ``path1`` ending with "/", whose contents go onto ``path2``. A
    filesystem is one place renames work in: a storage where they stop, e.g.
    at containers, is several handlers under a router. A root whose move
    fails yields the error, with its cause, and stays where it is. A root
    moved whole is yielded once, as ``(root, target)``: what is in it is
    neither listed nor yielded.

    With the profile's ``case_sensitive=False``, within one storage, paths
    are compared folded. A single path moved onto itself in another case,
    e.g. "Dir" to "DIR", is renamed by one ``mv_tree``, as a root going to a
    missing destination: the destination found would be the root itself. It
    needs ``renames_directories=True``, for a file too: else, iterating raises
    ValueError, as no file by file move renames a directory. One into itself
    is refused, as in the same case.

    With ``maxdepth``, it moves the files `_tree_cp` copies with it, and
    nothing is moved whole. A source directory is deleted once emptied, so those
    keeping a directory at the limit stay, and empty ones stay too. fsspec's
    ``mv`` copies as its ``copy`` does with ``maxdepth``, but then deletes the
    whole source tree, deeper files too. This keeps them.

    Once cancelled, it starts no new listing or move, and yields the outcomes
    of the moves already sent. Closed, it waits for those moves without
    yielding them, as their outcome would otherwise be unknown. Iterating it
    raises FileNotFoundError if ``path1`` does not exist, or matches nothing,
    and ValueError if ``path2`` is inside it within one storage.

    Requests: as `_tree_cp`, with one ``mv_entry`` per file instead of its
    ``cp_entry``, and one ``rm_dir`` per source directory. With
    ``renames_directories``, each directory root first costs a ``dst_info`` of
    its target, unless ``path2`` was found missing or existing already. One moving whole
    then costs a ``dst_makedirs`` of its parent, and its ``mv_tree``: nothing
    more. Each request holds a permit of the source's ``operation_limit``
    while it runs.

    A destination's parent is made before any rename, as ``mv_entry`` and
    ``mv_tree`` are called only where the parent exists. The destination's
    profile saves the ``dst_makedirs`` of each directory with files only where
    ``mv_entry`` writes or copies, creating the directory: between two storages,
    by ``parents_required=False``, or within one with ``renames_files=False``,
    by ``copy_parents_required=False``. Its
    ``implicit_directories=True`` saves those, and within one storage the
    ``rm_dir`` of each source directory too, as a directory goes with its last
    file. It alone saves the ``dst_makedirs`` before a directory is moved
    whole.

    The primitives are named above as the handlers' own: ``ils`` and
    ``mv_tree`` are the source's, ``rm_dir`` its ``rmdir``, and ``dst_info``
    and ``dst_makedirs`` the destination's ``info`` and ``makedirs``.

    ``src`` is the handler moved from, and ``dst`` the one moved to,
    ``src`` again within one storage. ``path1`` is a directory, a file, a
    glob pattern, or a list of them, as the source's ``ils`` takes paths.
    ``mv_entry(entry, destination)`` moves a file by its source entry, as its
    listing gave it: by default, the source's ``mv_file`` within one storage.
    A storage renaming a file in one request implements ``mv_file`` so. Within
    one storage, unless ``mv_entry`` is given, the files go to the source's
    ``mv_files`` in batches of its ``mv_batch_size``, as `_tree_cp` sends its
    copies to ``cp_files``: by default each by ``mv_file``.
    Between two, it defaults to ``cp_entry(entry, destination)``, then the
    source's ``rm_file``. ``cp_entry`` copies a file, as `_tree_cp` takes it:
    by default `transfer._across`. Given ``mv_entry``, a move uses no
    ``cp_entry``. ``same_storage`` is whether ``dst`` is ``src``,
    unless given.
    ``maxdepth`` of 1 moves only the files in ``path1``. ``path1_entry`` is as
    `_tree_cp` takes it: a single file moved so costs the ``dst_info`` of
    ``path2``, a ``dst_makedirs`` unless its parent is known to exist, and its
    ``mv_entry``. ``ils_kwargs`` go to each of the source's ``ils``, a
    pattern's glob's too.

    The operation yields ``(source, destination)`` for each file and directory
    moved, or an error, and its state is an `MvState`. The call raises
    ValueError if ``maxdepth`` is less than 1. `_tree_cp` does the same, keeping the source.
    """
    _check_maxdepth(maxdepth)
    given = mv_entry
    _, mv_entry, same_storage = _moves_of(src, dst, cp_entry, mv_entry, same_storage)
    return _transfers(
        src,
        dst,
        mv_entry,
        move=True,
        path1=path1,
        path2=path2,
        same_storage=same_storage,
        maxdepth=maxdepth,
        path1_entry=path1_entry,
        batching=same_storage and given is None,
        ils_kwargs=ils_kwargs,
        max_running=max_running,
    )
