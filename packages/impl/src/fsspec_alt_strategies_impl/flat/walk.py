# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A walk of the keys under a prefix, from one listing streamed as it comes.

`_flat_walk` is the counterpart of `_tree_walk` for flat storages. fsspec's
``walk`` lists one folder per request, even on object stores, which can list
all the keys under a prefix at once. `_flat_walk` makes that one listing, and
derives the folders from the keys. As the keys come in order, those in a
folder come together: the folder is done once the listing has passed them.

This module depends on `_keys`, and `operation` and `paths` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import WalkState

from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim, _triples
from ._keys import _close, _folder, _folders

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

__all__: list[str] = []


def _flat_walk(
    fs: Any,
    path: str,
    *,
    maxdepth: int | None = None,
    topdown: bool = True,
    descend: Callable[[dict[str, Any]], bool] | None = None,
) -> Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState]:
    """Walk the keys under a prefix, from one listing streamed as it comes.

    It yields each folder as ``(path, dirs, files)``, as `_tree_walk` does:
    ``path`` itself, then each folder below it. ``dirs`` and ``files`` map the
    names in the folder to their entries. The files are the keys, as
    ``_iexpand`` gives them. The folders are derived from the keys, as fsspec's filesystems
    derive them for object stores, and a key ending with "/" is a folder's
    marker. A derived folder's entry has only its ``name``, ``size`` 0 and
    ``type``.

    A folder is done once the listing has passed its keys. Without
    ``topdown``, each folder comes as soon as it is done, after all those
    below it, as fsspec's ``walk`` orders them. With ``topdown``, a folder
    comes before those below it, but ``path`` is done only once the listing
    ends: so all of them come then. A walk that is cancelled leaves out the
    folders not done.

    ``maxdepth`` and ``descend`` leave folders out, but save no request: the
    listing gives all the keys under ``path`` anyway.

    It wins over fsspec's ``walk`` on flat storages, with many folders: one
    listing, of as many pages as there are keys, rather than a listing per
    folder. It loses on storages with a hierarchical namespace, whose empty
    folders a listing by prefix does not see: `_tree_walk` suits those.

    Once cancelled, it takes nothing more from the listing. Iterating it raises
    these errors, which end it:

    - FileNotFoundError, if ``path`` does not exist;
    - NotADirectoryError, if ``path`` is a file;
    - the error of the listing, as there is no other to go on with.

    Requests: the pages of one ``_iexpand`` of ``path``, one after another.
    One ``_info`` for ``path``, if nothing is under it, holding a permit of
    the filesystem's ``operation_limit``. The listing holds none.

    Parameters
    ----------
    fs : FlatFileSystem
        The filesystem walked. Its ``_iexpand`` lists every key starting with a
        prefix, in order of their names, each key once, as an object store
        lists them.
    path : str
        The folder to start from, as ``_iexpand`` takes paths.
    maxdepth : int, optional
        The deepest level to walk: 1 yields only ``path``. No limit by default.
    topdown : bool
        Whether a folder comes before those below it, rather than after.
    descend : callable, optional
        Called with the entry of each subfolder found. It is walked only if
        this returns True. All by default.

    Returns
    -------
    Operation
        Yielding ``(path, dirs, files)`` for each folder walked. Its state is a
        `WalkState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    _tree_walk : The same, for storages listing one folder per request.
    _flat_find : The files under a prefix, as fsspec's ``_find`` finds them.
    """
    _check_maxdepth(maxdepth)

    async def run(
        op: Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState],
    ) -> AsyncGenerator[tuple[str, dict[str, Any], dict[str, Any]]]:
        folders = _folders_done(
            fs,
            path,
            maxdepth=maxdepth,
            descend=descend,
            stopping=op.cancelling,
        )

        # Top down, the folders done, held until the listing ends.
        held: list[tuple[str, list[dict[str, Any]]]] = []

        async with aclosing(folders):
            async for folder, entries in folders:
                if topdown:
                    held.append((folder, entries))
                    continue

                op.state = WalkState(
                    current=folder, found=op.state.found + len(entries)
                )
                yield _triples(folder, entries, detail=True)

        # By name, so that each folder comes before those below it.
        for folder, entries in sorted(held):
            op.state = WalkState(current=folder, found=op.state.found + len(entries))
            yield _triples(folder, entries, detail=True)

    return Operation(run, WalkState())


async def _folders_done(
    fs: Any,
    path: str,
    *,
    maxdepth: int | None,
    descend: Callable[[dict[str, Any]], bool] | None,
    stopping: Callable[[], object],
) -> AsyncGenerator[tuple[str, list[dict[str, Any]]]]:
    """Each folder walked with its entries, as soon as it is done: bottom up.

    The folders open, those the listing is in, are ``path`` and its subfolders
    down to the key listed last. A key not under the deepest one closes it: the
    keys come in order, so none under it comes later. A walk stopped leaves
    out the folders still open.
    """
    root = _trim(path)
    prefix = f"{root}/" if root else ""
    under = False

    # The folders open, the root first, each with its entries so far.
    open_: list[tuple[str, list[dict[str, Any]]]] = [(root, [])]

    keys = fs._iexpand(prefix)
    try:
        async for key in keys:
            under = True
            name = key["name"]
            while len(open_) > 1 and not name.startswith(open_[-1][0] + "/"):
                yield open_.pop()

            _add(key, name[len(prefix) :], prefix, open_, maxdepth, descend)
            if stopping():
                return
    finally:
        await _close(keys)

    if not under and root:
        # Nothing under it: a file, or missing, as _tree_walk raises for them.
        async with fs.operation_limit:
            entry = await fs._info(root)
        if entry["type"] != "directory":
            raise NotADirectoryError(root)

    while open_:
        yield open_.pop()


def _add(
    key: dict[str, Any],
    relative: str,
    prefix: str,
    open_: list[tuple[str, list[dict[str, Any]]]],
    maxdepth: int | None,
    descend: Callable[[dict[str, Any]], bool] | None,
) -> None:
    """Add a key to the walk: the folders it implies first seen, then itself.

    The folders already open are the key's parents. Each new one goes into
    its parent's entries, and is opened unless ``maxdepth`` or ``descend``
    leaves out what is in it. A key in a folder left out adds nothing more.
    """
    if not relative:
        return  # The root's own marker.

    for folder in _folders(relative)[len(open_) - 1 :]:
        full = prefix + folder
        entries = open_[-1][1]
        last = entries[-1] if entries else None
        if last is not None and last["name"] == full and last["type"] == "directory":
            return  # In a folder left out, seen with a key before.

        entry = _folder(full)
        entries.append(entry)

        depth = folder.count("/") + 1
        if (maxdepth is not None and depth >= maxdepth) or (
            descend is not None and not descend(entry)
        ):
            return

        open_.append((full, []))

    if not relative.endswith("/"):
        open_[-1][1].append(key)
