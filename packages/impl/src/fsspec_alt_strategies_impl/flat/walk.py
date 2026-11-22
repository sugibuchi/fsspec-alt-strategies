# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A walk of the keys under a prefix, from one listing streamed as it comes.

`_flat_walk` is the counterpart of `_tree_walk` for flat storages. fsspec's
``walk`` lists one directory per request, even on object stores, which can list
all the keys under a prefix at once. `_flat_walk` makes that one listing, and
derives the directories from the keys. As the keys come in order, those in a
directory come together: the directory is done once the listing has passed them.

This module depends on `_keys`, and `handler`, `operation` and `paths` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal

from fsspec_alt_strategies.states import WalkState

from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim, _triples
from ._keys import _close, _dir, _dirs, _listing

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from ..common.handler import StoragePath
    from .handler import FlatFileOperationHandler

__all__: list[str] = []


def _flat_walk(
    handler: FlatFileOperationHandler,
    path: StoragePath,
    *,
    maxdepth: int | None = None,
    topdown: bool = True,
    on_error: Literal["omit", "raise"] | Callable[[Exception], object] | None = None,
    ils_kwargs: dict[str, Any] | None = None,
) -> Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState]:
    """Walk the keys under a prefix, from one listing streamed as it comes.

    It yields each directory as ``(path, dirs, files)``, as `_tree_walk` does:
    ``path`` itself, then each directory below it. ``dirs`` and ``files`` map
    the names in the directory to their entries. The files are the keys, as the
    recursive ``ils`` gives them. The directories are derived from the keys,
    as fsspec's filesystems derive them for object stores, and a key ending
    with "/" is a directory's marker. A derived directory's entry has only its
    ``name``, ``size`` 0 and ``type``.

    A directory is done once the listing has passed its keys. Without
    ``topdown``, each directory comes as soon as it is done, after all those
    below it, as fsspec's ``walk`` orders them. With ``topdown``, a directory
    comes before those below it, but ``path`` is done only once the listing
    ends: so all of them come then. A walk that is cancelled leaves out the
    directories not done.

    ``maxdepth`` leaves directories out, but saves no request: the
    listing gives all the keys under ``path`` anyway.

    It wins over fsspec's ``walk`` on flat storages, with many directories: one
    listing, of as many pages as there are keys, rather than a listing per
    directory. It loses on storages with a hierarchical namespace, whose empty
    directories a listing by prefix does not see: `_tree_walk` suits those.

    A failure of the listing ends the walk, as there is no other listing to go
    on with. It yields its error as ``path``'s, after the directories done,
    and leaves out those not done, as the storage raised it: the state's
    ``current`` is then ``path``. ``on_error``, below, may handle it instead,
    as with fsspec's ``walk``.

    Once cancelled, it takes nothing more from the listing. Iterating it raises
    these errors, which end it:

    - FileNotFoundError, if ``path`` does not exist;
    - NotADirectoryError, if ``path`` is a file;
    - with ``on_error="raise"``, the error of the listing.

    Requests: the pages of one recursive ``ils`` of ``path``, one after
    another. One ``info`` for ``path``, if nothing is under it, holding a
    permit of the handler's ``operation_limit``. The listing holds none.

    ``path`` is the directory the walk starts from on ``handler``. Its recursive ``ils``
     lists every key below a directory, in name order, each key once,
    as an object store lists them. ``maxdepth`` of 1 yields only ``path``.
    ``topdown`` puts a directory before those below it. ``ils_kwargs`` go to the recursive ``ils``, as fsspec's
    ``walk`` passes its own on.

    ``on_error`` says what becomes of the error of the listing:

    - None yields it, as every strategy does;
    - "omit" leaves it out;
    - "raise" raises it;
    - a callable is called with it, and it is left out.

    The operation yields ``(path, dirs, files)`` for each directory walked, or
    the listing's error, and its state is a `WalkState`. The call raises
    ValueError if ``maxdepth`` is less than 1, or ``on_error`` is none of the
    above.
    """
    _check_maxdepth(maxdepth)
    if not (on_error in (None, "omit", "raise") or callable(on_error)):
        raise ValueError(f"on_error must be 'omit', 'raise' or callable: {on_error!r}")

    async def run(
        op: Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState],
    ) -> AsyncGenerator[tuple[str, dict[str, Any], dict[str, Any]] | Exception]:
        directories = _dirs_done(
            handler,
            path,
            maxdepth=maxdepth,
            ils_kwargs=ils_kwargs,
            stopping=op.cancelling,
        )

        # Top down, the directories done, held until the listing ends.
        held: list[tuple[str, list[dict[str, Any]]]] = []

        # The listing's error, yielded after the directories done.
        failed: Exception | None = None

        async with aclosing(directories):
            async for item in directories:
                if isinstance(item, Exception):
                    failed = item
                    break

                directory, entries = item
                if topdown:
                    held.append((directory, entries))
                    continue

                op.state = WalkState(
                    current=directory, found=op.state.found + len(entries)
                )
                yield _triples(directory, entries, detail=True)

        if failed is not None and on_error == "raise":
            raise failed

        # By name, so that each directory comes before those below it.
        for directory, entries in sorted(held):
            op.state = WalkState(current=directory, found=op.state.found + len(entries))
            yield _triples(directory, entries, detail=True)

        if failed is None or on_error == "omit":
            return
        elif callable(on_error):
            on_error(failed)
        else:
            # Its path is the state's current, for a caller yielding it.
            op.state = WalkState(current=_trim(path), found=op.state.found)
            yield failed

    return Operation(run, WalkState())


async def _dirs_done(
    handler: FlatFileOperationHandler,
    path: str,
    *,
    maxdepth: int | None,
    ils_kwargs: dict[str, Any] | None,
    stopping: Callable[[], object],
) -> AsyncGenerator[tuple[str, list[dict[str, Any]]] | Exception]:
    """Each directory walked with its entries, as soon as it is done: bottom up.

    The directories open, those the listing is in, are ``path`` and its
    subdirectories down to the key listed last. A key not under the deepest one closes it: the
    keys come in order, so none under it comes later. A walk stopped leaves
    out the directories still open.

    A failure of the listing comes last, the directories still open left out.
    """
    root = _trim(path)
    prefix = f"{root}/" if root else ""
    under = False

    # The directories open, the root first, each with its entries so far.
    open_: list[tuple[str, list[dict[str, Any]]]] = [(root, [])]

    keys = _listing(handler, prefix, ils_kwargs=ils_kwargs)
    try:
        while True:
            # Only the listing's own error is the walk's to handle.
            try:
                key = await anext(keys)
            except StopAsyncIteration:
                break
            except Exception as e:
                yield e
                return

            under = True
            name = key["name"]
            while len(open_) > 1 and not name.startswith(open_[-1][0] + "/"):
                yield open_.pop()

            _add(key, name[len(prefix) :], prefix, open_, maxdepth)
            if stopping():
                return
    finally:
        await _close(keys)

    if not under and root:
        # Nothing under it: a file, or missing, as _tree_walk raises for them.
        async with handler.operation_limit:
            entry = await handler.info(handler.entry(root))
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
) -> None:
    """Add a key to the walk: the directories it implies first seen, then itself.

    The directories already open are the key's parents. Each new one goes into
    its parent's entries, and is opened unless ``maxdepth`` leaves out what is
    in it. A key in a directory left out adds nothing more.
    """
    if not relative:
        return  # The root's own marker.

    for directory in _dirs(relative)[len(open_) - 1 :]:
        full = prefix + directory
        entries = open_[-1][1]
        last = entries[-1] if entries else None
        if last is not None and last["name"] == full and last["type"] == "directory":
            return  # In a directory left out, seen with a key before.

        entry = _dir(full)
        entries.append(entry)

        depth = directory.count("/") + 1
        if maxdepth is not None and depth >= maxdepth:
            return

        open_.append((full, []))

    if not relative.endswith("/"):
        open_[-1][1].append(key)
