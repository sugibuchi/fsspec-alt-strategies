# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy that walks the tree and copies while it walks.

`_tree_cp` is an alternative to fsspec's recursive ``copy``. fsspec's lists the
whole tree first, then copies each path. `_tree_cp` copies each file as soon as
its directory's listing finds it, on `common.tree`'s traversal, by
`_Sending`. The destination may be another storage: within one, it is the
same handler given twice.

Where the handler's ``cp_tree`` copies a directory in one request, the
profile says so. `_tree_cp` then copies each root whole with it where it can,
and traverses only what it cannot copy so.

`_transfers` runs both `_tree_cp` and `_tree_mv`. `common.placement._place_roots`
places the roots, as fsspec's ``copy`` does. `_Sending` places each path below
them under its directory's map value, a `(source, destination)` pair, then copies or moves it.

This module depends on `glob`, `rm` and `walk`, and `glob`, `handler`,
`jobs`, `operation`, `paths` and `tree` of `common`.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from fsspec_alt_strategies.states import CpState, MvState

from ..common.glob import PathAction
from ..common.handler import FileTransfer
from ..common.jobs import _transfer_of
from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim
from ..common.placement import _place_roots, _Placed, _Root
from ..common.tree import PathTask, TreeTraversal, _write
from .glob import _tree_glob
from .rm import _bottom_up, _Call, _consumed, _events_for, _Key
from .walk import _MAX_RUNNING, _root_task

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Awaitable,
        Callable,
        Sequence,
    )

    from ..common.handler import Entry, StorageHandler, StoragePath
    from ..common.jobs import _Transfer
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


def _tree_cp(
    src: TreeFileOperationHandler,
    dst: StorageHandler,
    path1: StoragePath | Sequence[StoragePath],
    path2: StoragePath,
    *,
    cp_entry: _Transfer | None = None,
    same_storage: bool | None = None,
    maxdepth: int | None = None,
    path1_entry: dict[str, Any] | None = None,
    ils_kwargs: dict[str, Any] | None = None,
    max_running: int = _MAX_RUNNING,
) -> Operation[tuple[str, str], CpState]:
    """Copy a directory and everything in it, copying while the tree is walked.

    It places the paths as fsspec's ``copy`` with ``recursive=True`` does. If
    ``path2`` is an existing directory, or ends with "/", and ``path1`` does not
    end with "/", ``path1`` goes into it: ``path1/x`` becomes
    ``path2/<name of path1>/x``. Otherwise ``path1`` becomes ``path2``:
    ``path1/x`` becomes ``path2/x``. A file is copied the same way. Existing
    files are overwritten, as fsspec's ``copy`` does.

    A glob pattern copies each path it matches, and all in it, placed as
    fsspec's ``copy`` places them. Their paths below the matches' common
    prefix are kept under ``path2``, but a single file matched goes into
    ``path2``. The pattern is matched as `_tree_glob` matches it, then the
    directories matched are traversed at once.

    A list of paths and patterns copies each path it names or matches into
    ``path2``, under its name, with all in it. That is how ``cp -r`` places
    them. fsspec's ``copy`` places the files named as this does. It also
    flattens the directories' trees, though: each file in them goes straight into
    ``path2``. Two paths of the same name would go to the same place. The
    first, in path order, is copied, and each other yields a FileExistsError
    instead, as GNU's ``cp`` reports it. fsspec's ``copy`` overwrites one
    with the other. A path inside another one in the list is copied with that
    one only.

    The tree is traversed as `_tree_walk` traverses it. Each file is copied as
    soon as its directory's listing finds it, by a ``cp_entry``. A directory
    is created at the destination before files are copied into it, and for
    each empty directory.

    With ``maxdepth``, it copies the files fsspec's ``copy`` copies with it:
    those at most ``maxdepth`` levels below ``path1``, or below a pattern's
    matches one level less. Directories deeper are not listed. As fsspec's, it
    then makes no empty directory. Unlike fsspec's, it places the files as
    without ``maxdepth``: fsspec places them by the common prefix of the files
    found, which moves them when all are in one subdirectory.

    The destination is described by ``dst`` and its ``profile``,
    unknown if it has none. Within one storage, ``dst`` is ``src``, and
    ``same_storage`` is True: a path copied onto itself, or under itself, is
    refused. Between two, ``cp_entry`` copies from one to the other, by default
    `transfer._across`, its data through the caller. A
    path may then go to the same path, or under it. The copies run
    concurrently, so the destination's paths are created in another order than
    the source's. A storage listing in creation order shows it.

    A storage may list two files of one name in a directory, as Google Drive
    holds them. Both would go to one place, the second replacing the first:
    the first goes, and each other yields a FileExistsError, as GNU's ``cp``
    refuses two sources onto one destination. A move leaves it, and its
    directory, where they are.

    Unlike fsspec's ``copy``, it keeps going past a failure, and reports each,
    as the storage raised it: the state's ``current`` is then its path. A copy
    cancelled or failed leaves what was copied so far: it deletes nothing.

    It wins over fsspec's ``copy`` on storages that list one directory per
    request. The copies then start after the first listing, not after the
    whole tree is listed. Between two storages, it wins over fsspec's way too,
    a ``get`` to local files then a ``put``: the data needs no local disk.

    Within one storage, with the profile's ``copies_directories=True``,
    ``cp_tree`` copies a directory with all in it, in one request. Each
    directory root is then copied whole by one ``cp_tree``, where it can be,
    as `_tree_mv` moves one by ``mv_tree``: only to a destination that does
    not exist, and never with ``maxdepth``. That wins by far on storages
    copying a folder at once, such as Dropbox. A root whose copy fails yields
    the error. A root copied whole is yielded once, as ``(root, target)``:
    what is in it is neither listed nor yielded.

    With the profile's ``case_sensitive=False``, within one storage, paths
    are compared folded. A path onto itself in another case is left as it
    is, as one in the same case is, and one into itself is refused.

    Once cancelled, it starts no new listing or copy, and yields the outcomes
    of the copies already sent. Closed, it waits for those copies without
    yielding them. Iterating it raises FileNotFoundError if ``path1`` does not
    exist, or matches nothing, and ValueError if ``path2`` is inside it within
    one storage.

    Requests: one ``dst_info`` for ``path2``, unless it ends with "/" or
    ``path1`` is a pattern or a list. Patterns, or a list, add the requests of
    their `_tree_glob`. The listings of `_tree_walk` bottom up, per root. One
    ``cp_entry`` per file, and one ``dst_makedirs`` per destination directory
    with files or empty. A destination refusing to overwrite costs a
    ``dst_info``, a ``dst_rm_file`` and a ``cp_entry`` more per file it
    refused. The ``dst_info`` checks it is a file, as ``dst_rm_file`` is given
    only paths known to be files. Each request holds a permit of the source's
    ``operation_limit`` while it runs, a ``cp_entry`` between two storages
    too. With ``copies_directories``, each directory root first costs a
    ``dst_info`` of its target, unless ``path2`` was found missing or existing
    already. One copied whole then costs a ``dst_makedirs`` of its parent, and
    its ``cp_tree``: nothing more.

    The destination's profile's ``parents_required=False`` between two
    storages, ``copy_parents_required=False`` within one, or
    ``implicit_directories=True``, saves the ``dst_makedirs`` of each directory
    with files, as their copies create it. Its ``parents_required=True``
    makes every directory, each before what is in it, as such a storage
    makes a missing parent by more requests. That is one ``dst_makedirs``
    per directory, those holding only directories too, unless ``maxdepth``
    is given. A directory failing to be made then leaves out what goes in it,
    reported by its error alone. A directory known to exist needs
    none: the ``dst_info`` of ``path2`` shows it or its parent, and within one
    storage, ``path1``'s parent exists.

    ``path1_entry``, the entry of ``path1`` as the caller found it, saves the
    listing of a single file: it is copied as the entry says.

    The primitives are named above as the handlers' own: ``ils`` and
    ``cp_tree`` are the source's, and ``dst_info``, ``dst_makedirs`` and
    ``dst_rm_file`` the destination's ``info``, ``makedirs`` and ``rm_file``.

    ``src`` is the handler copied from, and ``dst`` the one copied
    to, ``src`` again within one storage. ``path1`` is a directory, a file,
    a glob pattern, or a list of them, as the source's ``ils`` takes paths.
    ``cp_entry(entry, destination)`` copies a file: by default, the source's
    ``cp_file`` within one storage, and `transfer._across` between two. ``same_storage`` is
    whether ``dst`` is ``src``, unless given. ``maxdepth`` of 1 copies
    only the files in ``path1``. ``ils_kwargs`` go to each of the source's
    ``ils``, a pattern's glob's too. ``max_running`` bounds the directories in
    flight.

    Within one storage, unless ``cp_entry`` is given, the files go to the
    source's ``cp_files`` in batches of its ``cp_batch_size``: by default each
    by ``cp_file``. The traversal emits each file to send, and the operation
    reads them in rounds: those emitted while the last round was sent, up to
    the larger of 16 and the batch size. A batch holds files of any
    directory. A move deletes each source directory after all below it,
    siblings at once.

    The operation yields ``(source, destination)`` for each file copied, or an
    error, and its state is a `CpState`. The call raises ValueError if
    ``maxdepth`` is less than 1. `_tree_mv` does the same, deleting the source as it goes.
    """
    _check_maxdepth(maxdepth)
    given = cp_entry
    cp_entry, same_storage = _transfer_of(src, dst, cp_entry, same_storage)
    return _transfers(
        src,
        dst,
        cp_entry,
        move=False,
        path1=path1,
        path2=path2,
        same_storage=same_storage,
        maxdepth=maxdepth,
        path1_entry=path1_entry,
        batching=same_storage and given is None,
        ils_kwargs=ils_kwargs,
        max_running=max_running,
    )


def _transfers(
    src: TreeFileOperationHandler,
    dst: StorageHandler,
    transfer: _Transfer,
    *,
    move: bool,
    path1: StoragePath | Sequence[StoragePath],
    path2: StoragePath,
    same_storage: bool,
    maxdepth: int | None,
    path1_entry: dict[str, Any] | None,
    batching: bool,
    ils_kwargs: dict[str, Any] | None,
    max_running: int,
) -> Operation[tuple[str, str], Any]:
    """The operation of a copy, or of a move: ``transfer`` sends each file.

    `common.placement._place_roots` places the roots, as fsspec's ``copy``
    does. Below them, each directory's map value places what is in it.
    """
    state: CpState | MvState = MvState(directories_moved=0) if move else CpState()

    def glob(pattern: StoragePath | Sequence[StoragePath]) -> Operation[Any, Any]:
        return _tree_glob(src, pattern, maxdepth=maxdepth, ils_kwargs=ils_kwargs)

    async def run(op: Operation[tuple[str, str], Any]) -> AsyncGenerator[Any]:
        placed = await _place_roots(
            move=move,
            handler=src,
            dst=dst,
            glob=glob,
            path1=path1,
            path2=path2,
            same_storage=same_storage,
            maxdepth=maxdepth,
            path1_entry=path1_entry,
            batching=batching,
            ils_kwargs=ils_kwargs,
        )
        if placed.whole:
            op.state = replace(
                op.state, **{"roots_renamed" if move else "roots_copied": 0}
            )
        for error in placed.errors:
            op.state = replace(op.state, errors=op.state.errors + 1)
            yield error
        if not placed.directories and not placed.files:
            return

        engine = TreeTraversal(
            max_running=max_running,
            max_events=_events_for(placed.batch_size),
            split=src.segments,
        )
        op._on_cancel = engine.cancel
        action = _Sending(
            engine._emit_event,
            op,
            placed,
            src,
            dst,
            transfer,
            move=move,
            maxdepth=maxdepth,
        )
        # A path ending with "/" goes as its contents, never whole.
        whole = placed.whole and not (isinstance(path1, str) and path1.endswith("/"))
        tasks = await _tasks(engine, action, whole, maxdepth, ils_kwargs)

        try:
            outcomes = _consumed(engine, tasks, action.consume)
            async with aclosing(outcomes):
                async for event in outcomes:
                    yield action.outcome(event)
        except NotADirectoryError:
            # The one root, a file: it goes alone.
            if op.cancelling() or not isinstance(path1, str):
                return
            for event in await action.file_root(placed.directories[0]):
                yield action.outcome(event)

    return Operation(run, state)


async def _tasks(
    engine: TreeTraversal,
    action: _Sending,
    whole: bool,
    maxdepth: int | None,
    ils_kwargs: dict[str, Any] | None,
) -> list[PathTask]:
    """A task per root: one going whole, a traversal, or a file's.

    A root goes whole to a target found missing. One whose target exists
    merges into it, so it is traversed. One whose target could not be asked
    yields that error, and stays where it is. Below a pattern's match,
    ``maxdepth`` counts one less.
    """
    directories = action.placed.directories
    goes: list[bool | Exception] = [False] * len(directories)
    if whole:
        goes = list(await asyncio.gather(*(action.goes_whole(d) for d in directories)))

    tasks: list[PathTask] = []
    for root, outcome in zip(directories, goes, strict=True):
        if isinstance(outcome, Exception):
            continue  # Yielded by the last task, as found.
        if outcome:
            tasks.append(_Call(engine, root.entry, _on(action.plan_whole, root)))
            continue
        depth = maxdepth
        if depth is not None and root.matched:
            depth -= 1
        tasks.append(
            _root_task(
                engine,
                {**root.entry, "type": "directory"},
                action,
                action.src,
                maxdepth=depth,
                ils_kwargs=ils_kwargs,
            )
        )
    tasks.extend(
        _Call(engine, root.entry, _on(action.plan_root, root))
        for root in action.placed.files
    )
    errors = [o for o in goes if isinstance(o, Exception)]
    if errors:
        tasks.append(_Call(engine, {"name": ""}, lambda _: action.emit_all(errors)))
    return tasks


def _on(
    call: Callable[[_Root], Awaitable[Any]], root: _Root
) -> Callable[[Entry], Awaitable[Any]]:
    """A `_Call`'s call on a placed root, rather than on its entry."""

    async def run(_: Entry) -> Any:
        return await call(root)

    return run


class _Sending(PathAction):
    """A copy's or a move's action, planning what goes, and `consume` sending it.

    The traversal plans, and writes nothing. `map` places each path under its
    parent's value: it returns ``(source, destination)``, or None for a name
    the destination refuses, its error emitted. A root's is as
    `_place_roots` placed it. A file's `reduce` returns its entry and
    destination. A directory's `reduce` emits ``("mkdir", source,
    destination)`` where the directory is needed, then ``("file", entry,
    destination, alone)`` for each file in its values. An empty one is made
    too, and for a move, ``("dir", entry, destination)`` deletes its source
    once all in it went. A directory's value is whether all below was
    planned.

    A directory is made where the storage requires parents, before its first
    file. Where it makes one directory per request, its parent existing, it
    is made first, before its values are read: so before all below it.

    `consume` sends a batch of those events, as `_transfers` reads them: the
    directories first, each once, then the files by the source's
    ``cp_files`` or ``mv_files``, in batches of its batch size. A move
    deletes its source directories at once with them, each after all below
    it. One holding anything that stayed is kept, and so is each above it.

    Each request is a `common.tree._write` under a permit of the source's
    ``operation_limit``, none starting once cancelled. Each path gone is an
    outcome ``("done", source, destination)``, each error ``(entry,
    error)``, as `common.glob`'s tasks emit a listing's.
    """

    def __init__(
        self,
        emit_event: Any,
        op: Operation[Any, Any],
        placed: _Placed,
        src: TreeFileOperationHandler,
        dst: StorageHandler,
        transfer: _Transfer,
        *,
        move: bool,
        maxdepth: int | None,
    ) -> None:
        super().__init__(emit_event)
        self.op = op
        self.placed = placed
        self.src = src
        self.dst = dst
        self.transfer = transfer
        self.move = move
        self.maxdepth = maxdepth
        self.limit = src.operation_limit
        self.batch_size = placed.batch_size
        self.batching = placed.batching
        self.implicit_directories = placed.implicit_directories
        self.parents_required = placed.parents_required
        # One directory per request, its parent existing: each is made before
        # all below it. Not with maxdepth, as fsspec's copy makes no empty
        # directory then.
        self.parents_first = (
            placed.parents_required
            and placed.profile.parents_required is True
            and maxdepth is None
        )

        # The directory roots, by their segments: the map values start there.
        self.roots = {self.key(r.entry["name"]): r for r in placed.directories}
        # The names of the files gone into each source directory: a second of
        # one name, as Google Drive lists, would go where the first went.
        self.names: dict[_Key, set[str]] = {}
        # The source directories holding something that stayed, until they come.
        self.kept: set[_Key] = set()
        # The roots going whole: their "dir" event sends them, with all in them.
        self.wholes: set[_Key] = set()

        # The consumer's alone. The destination directories made, or known to
        # exist, each with its entry, if known: the parent hint of what goes in
        # it. Those failing to be made, and those found missing, with all below.
        self.made: dict[str, Entry | None] = {}
        self.unmade: set[str] = set()
        self.missing: set[str] = set()
        for root in (*placed.directories, *placed.files):
            if root.missing:
                self.missing.add(root.target)
            if root.made:
                self.made[root.target] = root.directory
            if root.parent_exists and (parent := self.parent(root.target)):
                self.made[parent] = root.parent
        # The outcomes of the batch consumed.
        self.out: list[Any] = []

    # The traversal's calls: they plan.

    async def map(self, entry: Entry, match: bool, parent_value: Any) -> Any:
        name = _trim(entry["name"])
        if parent_value is None:
            root = self.roots.get(self.key(name))
            return None if root is None else (name, root.target)
        try:
            target = self.dst.join(parent_value[1], self.key(name)[-1])
        except ValueError as error:
            await self._emit_event((entry, error))
            return None
        if entry["type"] != "directory" and (
            exists := self.dst_entry_exists(name, target)
        ):
            await self._emit_event((entry, exists))
            return None
        return name, target

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        if entry["type"] != "directory":
            self.count(files_found=1)
            if value is None:
                return False  # Refused: its error, or its directory's, tells.
            # The entry goes with its copy, so that cp_entry need not ask it.
            return {**entry, "name": value[0]}, value[1]
        if value is None:
            return False  # Refused, with all in it.
        return await self.plan_directory(entry, value, values)

    async def plan_directory(
        self, entry: Entry, value: tuple[str, StoragePath], values: AsyncGenerator[Any]
    ) -> bool:
        """Emit a directory's mkdir, where needed, and each file in its values: whether all below was planned."""
        name, target = value
        made = False
        if self.parents_first:
            await self._emit_event(("mkdir", name, target))
            made = True
        items = 0
        planned = True
        async for item in values:
            items += 1
            if isinstance(item, tuple):
                if not made and self.parents_required:
                    await self._emit_event(("mkdir", name, target))
                    made = True
                await self._emit_event(("file", *item, False))
            else:
                planned &= item is True
        if not items:
            # An empty directory is made, whatever the storage. With maxdepth,
            # fsspec's copy leaves out directories, and copies files only: a
            # move then leaves it where it is, as one at maxdepth, not listed.
            if self.maxdepth is not None:
                return False
            if not made:
                await self._emit_event(("mkdir", name, target))
        if not planned:
            return False  # Kept, with what failed in it.
        if self.move:
            await self._emit_event(("dir", entry, target))
        return True

    async def failed(self, entry: Entry) -> Any:
        return False

    # The roots.

    async def goes_whole(self, root: _Root) -> bool | Exception:
        """Whether a root can go whole, as its target is missing: an error asking is returned."""
        if root.missing:
            return True
        if root.existing:
            return False
        try:
            async with self.limit:
                await self.dst.info(self.dst.entry(root.target))
        except FileNotFoundError:
            return True
        except Exception as e:
            return e
        return False

    async def plan_whole(self, root: _Root) -> None:
        """Plan a root going whole, by the storage's one request: a directory whose contents are not traversed, after its parent."""
        name = _trim(root.entry["name"])
        self.wholes.add(self.key(name))
        parent = self.parent(root.target)
        if parent and (not self.implicit_directories or self.parents_required):
            await self._emit_event(("mkdir", name, parent))
        await self._emit_event(("dir", root.entry, root.target))

    async def plan_root(self, root: _Root) -> None:
        """Plan a root that is a file: it goes alone, into its directory made if needed."""
        for event in self.plan_root_file(root):
            await self._emit_event(event)

    async def file_root(self, root: _Root) -> list[Any]:
        """The outcomes of the one root, found a file once the traversal ended: it goes alone."""
        entry = {**root.entry, "type": "file"}
        return await self.consume(self.plan_root_file(root._replace(entry=entry)))

    def plan_root_file(self, root: _Root) -> list[Any]:
        """A root file's events, its target placed already."""
        name = _trim(root.entry["name"])
        self.count(files_found=1)
        if (exists := self.dst_entry_exists(name, root.target)) is not None:
            return [(root.entry, exists)]
        file = ("file", {**root.entry, "name": name}, root.target, True)
        parent = self.parent(root.target)
        if parent and self.parents_required:
            return [("mkdir", name, parent), file]
        return [file]

    async def emit_all(self, errors: list[Exception]) -> None:
        """Emit the errors of roots whose target could not be asked."""
        for error in errors:
            await self._emit_event(({"name": ""}, error))

    def key(self, name: str) -> _Key:
        """A source path's key, by the source's segments: never verified, as a listing gave it."""
        return tuple(self.src.segments(name))

    def dst_entry_exists(
        self, name: str, target: StoragePath
    ) -> FileExistsError | None:
        """The error of a file whose name another took first in its directory, else None.

        Both would go to one place, the second replacing the first, as on a
        storage holding several items of one name, e.g. Google Drive. The
        first goes, and each other is refused, as GNU's `cp` refuses two
        sources onto one destination: it stays where it is, and so does its
        directory.
        """
        names = self.names.setdefault(self.key(name)[:-1], set())
        if name in names:
            return FileExistsError(
                errno.EEXIST,
                "Another file of this name in its directory goes there",
                target,
            )
        names.add(name)
        return None

    # The consumer's: it sends.

    async def consume(self, events: list[Any]) -> list[Any]:
        """Send what a batch of events plans: their outcomes, after the events passed on, e.g. a listing's error."""
        self.out = []
        mkdirs: list[tuple[str, StoragePath]] = []
        files: list[tuple[Entry, StoragePath, bool]] = []
        directories: dict[_Key, tuple[Entry, StoragePath]] = {}
        for event in events:
            if event[0] == "mkdir":
                mkdirs.append(event[1:])
            elif event[0] == "file":
                files.append(event[1:])
            elif event[0] == "dir":
                directories[self.key(event[1]["name"])] = event[1:]
            else:
                self.out.append(event)
        if self.parents_first:
            for source, path in mkdirs:  # In order: each after its parent.
                await self.make(source, path)
        else:
            await asyncio.gather(*(self.make(s, p) for s, p in mkdirs))
        sending = asyncio.ensure_future(self.send_files(files))
        try:
            holding = {self.key(f[0]["name"])[:-1] for f in files}
            await asyncio.gather(
                sending,
                _bottom_up(directories, sending, holding, self.kept, self.settle_dir),
            )
        finally:
            if not sending.done():
                sending.cancel()
                await asyncio.wait([sending])
        out, self.out = self.out, []
        return out

    def parent(self, path: StoragePath) -> str:
        """A destination path's parent: "" at the root."""
        try:
            return self.dst.parent(path)
        except ValueError:
            return ""

    def known_missing(self, path: str) -> bool:
        """Whether nothing is at a destination, as found: it, or a directory above it, was found missing."""
        while path:
            if path in self.missing:
                return True
            path = self.parent(cast("StoragePath", path))
        return False

    async def make(self, source: str, path: StoragePath) -> None:
        """Make a destination directory once, keeping its entry: its error, if any, yielded as ``source``'s.

        One below a directory failing to be made is not tried: its error tells.
        """
        if (
            not path
            or path == self.dst.root
            or path in self.made
            or path in self.unmade
        ):
            return
        parent = self.parent(path)
        if parent in self.unmade:
            self.unmade.add(path)
            return
        made: Entry | None = None

        async def mkdir() -> None:
            nonlocal made
            made = await self.dst.makedirs(
                path,
                exist_ok=True,
                missing=self.known_missing(path),
                parent=self.made.get(parent),
            )

        if await self.write(mkdir, {"name": source}):
            self.made[path] = made
        else:
            self.unmade.add(path)

    async def send_files(self, files: list[tuple[Entry, StoragePath, bool]]) -> None:
        """Send files, but those whose directory failed to be made: the source directory of each that stayed is kept."""
        ready = [
            (e, t, self.made.get(self.parent(t)), self.known_missing(t), alone)
            for e, t, alone in files
            if self.parent(t) not in self.unmade
        ]
        # Unless batching, e.g. between two storages, each goes by the transfer.
        alone = [f for f in ready if f[4] or not self.batching]
        batched = [f for f in ready if not f[4] and self.batching]
        size = self.batch_size
        outcomes = await asyncio.gather(
            *(self.send(batched[i : i + size]) for i in range(0, len(batched), size)),
            *(self.one(*f[:4]) for f in alone),
        )
        gone = {name for names in outcomes for name in names}
        self.kept.update(
            self.key(e["name"])[:-1] for e, _, _ in files if e["name"] not in gone
        )

    async def settle_dir(self, planned: tuple[Entry, StoragePath]) -> bool:
        """Settle a directory once all in it went: whether it went.

        A root going whole goes now, by the storage's one request, with all
        in it. Another, its contents moved, has its source deleted.
        """
        entry, target = planned
        name = _trim(entry["name"])
        if self.key(name) not in self.wholes:
            return await self.remove(entry, name, target)
        parent = self.parent(target)
        if parent in self.unmade:
            return False  # Its parent's error tells.
        primitive = self.src.mv_tree if self.move else self.src.cp_tree
        hint = self.made.get(parent)

        async def send() -> None:
            await primitive(entry, target, parent=hint)

        if not await self.write(send, entry):
            return False
        field = "roots_renamed" if self.move else "roots_copied"
        self.count(**{field: 1})
        self.out.append(("done", name, target))
        return True

    async def one(
        self, entry: Entry, target: StoragePath, hint: Entry | None, missing: bool
    ) -> list[str]:
        """Send a file by the transfer, replacing a file there the destination keeps: its name, if it went."""

        async def send() -> None:
            try:
                await self.transfer(entry, target, missing=missing, parent=hint)
            except FileExistsError as refused:
                if self.move:
                    raise
                await self.replace_file(entry, target, hint, refused)

        if not await self.write(send, entry):
            return []
        self.count(**{"files_moved" if self.move else "files_copied": 1})
        self.out.append(("done", entry["name"], target))
        return [entry["name"]]

    async def replace_file(
        self,
        entry: Entry,
        target: StoragePath,
        hint: Entry | None,
        refused: FileExistsError,
    ) -> None:
        """Copy a file onto one its destination refused to overwrite: deleted first, unless a directory.

        A directory there raises ``refused`` again: ``rm_file`` takes files only.
        """
        found = await self.dst.info(self.dst.entry(target))
        if found["type"] == "directory":
            raise refused
        await self.dst.rm_file({**found, "name": target})
        await self.transfer(entry, target, parent=hint)

    async def send(self, waiting: list[tuple[Any, ...]]) -> list[str]:
        """Send files by the source's ``cp_files`` or ``mv_files``: the names of those gone.

        A copy refused by a destination not overwriting is replaced on its own,
        as `one` replaces it. A batch failing whole emits one error.
        """
        if self.op.cancelling():
            return []
        transfers = [
            FileTransfer(e, t, missing, hint) for e, t, hint, missing, _ in waiting
        ]

        async def transfer() -> tuple[dict[str, Exception | None], list[Exception]]:
            operation = (
                self.src.mv_files(transfers)
                if self.move
                else self.src.cp_files(transfers)
            )
            outcomes: dict[str, Exception | None] = {}
            apart: list[Exception] = []
            async with operation as files:
                async for outcome in files:
                    if not isinstance(outcome, Exception):
                        outcomes[outcome[0]] = None
                    elif (source := getattr(outcome, "source", None)) is None:
                        apart.append(outcome)
                    else:
                        outcomes[source] = outcome
            return outcomes, apart

        try:
            outcomes, apart = await _write(transfer())
        except Exception as error:
            self.out.append((waiting[0][0], error))
            return []
        return await self.settle(waiting, outcomes, apart)

    async def settle(
        self,
        waiting: list[tuple[Any, ...]],
        outcomes: dict[str, Exception | None],
        apart: list[Exception],
    ) -> list[str]:
        """Count and report each outcome of a batch sent: a file absent from it, e.g. cancelled, did not go."""
        gone: list[str] = []
        for entry, target, hint, missing, _ in waiting:
            error = outcomes.get(entry["name"], asyncio.CancelledError())
            if isinstance(error, FileExistsError) and not self.move:
                gone += await self.one(entry, target, hint, missing)
            elif error is None:
                self.count(**{"files_moved" if self.move else "files_copied": 1})
                self.out.append(("done", entry["name"], target))
                gone.append(entry["name"])
            elif not isinstance(error, asyncio.CancelledError):
                self.out.append((entry, error))
        for error in apart:
            self.out.append((waiting[0][0], error))
        return gone

    async def remove(self, entry: Entry, name: str, target: StoragePath) -> bool:
        """Delete a source directory now moved, unless it went with its last file: whether it is gone."""
        if not self.implicit_directories:

            async def rmdir() -> None:
                # Already gone counts as deleted, as _tree_rm counts it.
                with contextlib.suppress(FileNotFoundError):
                    await self.src.rmdir({**entry, "name": name})

            if not await self.write(rmdir, entry):
                return False
        self.count(directories_moved=1)
        self.out.append(("done", name, target))
        return True

    async def write(self, request: Callable[[], Awaitable[None]], entry: Entry) -> bool:
        """Run a request under a permit, to its end even when closed: whether it went.

        None starts once cancelled. An error is reported, and it did not go.
        """

        async def limited() -> bool:
            async with self.limit:
                if self.op.cancelling():
                    return False
                await request()
                return True

        try:
            return await _write(limited())
        except Exception as error:
            self.out.append((entry, error))
            return False

    def count(self, **counts: int) -> None:
        """Count in the state what was done, as it is: yielded or not, e.g. once closed."""
        state = self.op.state
        self.op.state = replace(
            state, **{k: (getattr(state, k) or 0) + v for k, v in counts.items()}
        )

    def outcome(self, event: Any) -> tuple[str, str] | Exception:
        """What the operation yields of an event, counted in its state."""
        state = self.op.state
        if event[0] == "done":
            _, source, target = event
            self.op.state = replace(state, current=source)
            return source, target
        entry, error = event
        current = _trim(entry["name"]) or state.current
        self.op.state = replace(state, current=current, errors=state.errors + 1)
        return cast("Exception", error)
