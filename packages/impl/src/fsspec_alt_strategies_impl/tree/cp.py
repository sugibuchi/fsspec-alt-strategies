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

`_transfers` runs both `_tree_cp` and `_tree_mv`: `common.jobs._CopyJobs`
places each path, as fsspec's ``copy`` does, and `_Sending` copies or moves
it.

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
from ..common.jobs import _CopyJobs, _Roots, _transfer_of
from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim
from ..common.tree import PathTask, TreeTraversal, _write
from .glob import _tree_glob
from .rm import _Call
from .walk import _MAX_RUNNING, _root_task

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Awaitable,
        Callable,
        Coroutine,
        Sequence,
    )

    from ..common.handler import Entry, StorageHandler, StoragePath
    from ..common.jobs import _Job, _Transfer
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
    by ``cp_file``. A batch holds files of one directory, sent once full, as
    the listing finds them: its last, smaller one once all below the
    directory is done.

    The operation yields ``(source, destination)`` for each file copied, or an
    error, and its state is a `CpState`. The call raises ValueError if
    ``maxdepth`` is less than 1. `_tree_mv` does the same, deleting the source as it goes.
    """
    _check_maxdepth(maxdepth)
    given = cp_entry
    cp_entry, same_storage = _transfer_of(src, dst, cp_entry, same_storage)
    return _transfers(
        _Placing(
            handler=src,
            op=None,  # Never run: it only places.
            dst=dst,
            path1=path1,
            path2=path2,
            same_storage=same_storage,
            maxdepth=maxdepth,
            path1_entry=path1_entry,
            batching=same_storage and given is None,
            ils_kwargs=ils_kwargs,
        ),
        cp_entry,
        max_running=max_running,
    )


class _Placing(_CopyJobs):
    """Where a copy's or a move's paths go, as `common.jobs._CopyJobs` places them: never run as jobs.

    `_transfers` calls its ``setup``, ``prepare`` and helpers, e.g. ``target``
    and ``make_directory``: the traversal does the rest.
    """

    def glob(self, pattern: StoragePath | Sequence[StoragePath]) -> Operation[Any, Any]:
        return _tree_glob(
            cast("TreeFileOperationHandler", self.handler),
            pattern,
            maxdepth=self.maxdepth,
            ils_kwargs=self.ils_kwargs,
        )

    # Never run as jobs: the traversal does what these would.

    def state(self) -> Any:
        raise NotImplementedError

    def source(self, roots: _Roots) -> Any:
        raise NotImplementedError

    def fed(self, item: Any) -> list[Any]:
        raise NotImplementedError

    def file_root(self, entry: dict[str, Any]) -> None:
        raise NotImplementedError

    def run(self, job: _Job) -> Awaitable[Any]:
        raise NotImplementedError

    def done(self, job: _Job) -> list[Any]:
        raise NotImplementedError

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        raise NotImplementedError


class _Moving(_Placing):
    """Where a move's paths go: a single path moved onto itself in another case is renamed whole."""

    verb = "move"
    doing = "Moving"
    transfers = "move"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.goes_whole = False  # As _transfers reads it from the profile.

    def recased(self, root: str, destination: StoragePath) -> _Roots:
        """A single path moved onto itself in another case: renamed by one ``mv_tree``.

        The destination is the root itself. Found existing, it would take the
        root into itself: it is taken as missing instead.
        """
        if not self.goes_whole:
            return super().recased(root, destination)
        self.prefix, self.base = root, destination
        self.missing.add(destination)
        self.made.add(self.handler.parent(root))
        # A directory root, as mv_tree takes a file too.
        return _Roots([self.path1_entry or {"name": root}])


def _transfers(
    place: _Placing, transfer: _Transfer, *, max_running: int
) -> Operation[tuple[str, str], Any]:
    """The operation of a copy, or of a move if ``place`` is `_Moving`: ``transfer`` sends each file."""
    move = isinstance(place, _Moving)
    src = cast("TreeFileOperationHandler", place.handler)
    state: CpState | MvState = MvState(directories_moved=0) if move else CpState()

    async def run(op: Operation[tuple[str, str], Any]) -> AsyncGenerator[Any]:
        await place.setup()
        whole = _whole(place, move)
        if isinstance(place, _Moving):
            place.goes_whole = whole  # Read by recased, in prepare.
        if whole:
            op.state = replace(
                op.state, **{"roots_renamed" if move else "roots_copied": 0}
            )
        roots = await place.prepare()
        for error in place.found:
            op.state = replace(op.state, errors=op.state.errors + 1)
            yield error
        if roots is None:
            return

        engine = TreeTraversal(max_running=max_running, split=src.segments)
        op._on_cancel = engine.cancel
        action = _Sending(engine._emit_event, op, place, transfer, move=move)
        tasks = await _tasks(engine, action, roots, whole)

        try:
            async with aclosing(engine.events(tasks)) as events:
                async for event in events:
                    yield action.outcome(event)
        except NotADirectoryError:
            # The one root, a file: it goes alone.
            if op.cancelling() or not isinstance(place.path1, str):
                return
            entry = {"name": _trim(place.path1), "type": "file"}
            for event in await action.file_root(entry, emit=False):
                yield action.outcome(event)

    return Operation(run, state)


def _whole(place: _Placing, move: bool) -> bool:
    """Whether the roots go whole, by one ``cp_tree`` or ``mv_tree``, where they can.

    Only within one storage does the profile tell how the source copies or
    renames. A directory gone whole would take what is deeper than maxdepth
    along. A move's own setup follows: a rename makes no parent, as a write
    may, so its directory is made first unless the profile says it renames
    no file.
    """
    profile = place.profile
    declared = profile.renames_directories if move else profile.copies_directories
    if move:
        place.implicit_directories = place.same_storage and place.implicit_directories
        place.parents_required = place.parents_required or (
            place.same_storage
            and not place.implicit_directories
            and profile.renames_files is not False
        )
    return place.same_storage and bool(declared) and place.maxdepth is None


async def _tasks(
    engine: TreeTraversal, action: _Sending, roots: _Roots, whole: bool
) -> list[PathTask]:
    """A task per root: one going whole, a traversal, or a file's.

    A root goes whole to a target found missing. One whose target exists
    merges into it, so it is traversed. One whose target could not be asked
    yields that error, and stays where it is. Below a pattern's match,
    ``maxdepth`` counts one less.
    """
    place = action.place
    directories = list(roots.directories)
    goes: list[bool | Exception] = [False] * len(directories)
    path1 = place.path1
    if whole and not (isinstance(path1, str) and path1.endswith("/")):
        goes = list(await asyncio.gather(*(action.goes_whole(d) for d in directories)))

    tasks: list[PathTask] = []
    for entry, outcome in zip(directories, goes, strict=True):
        if isinstance(outcome, Exception):
            continue  # Yielded by the first task, as found.
        if outcome:
            tasks.append(_Call(engine, entry, action.whole))
            continue
        depth = place.maxdepth
        if depth is not None and _trim(entry["name"]) in place.matched:
            depth -= 1
        root = {**entry, "type": "directory"}
        tasks.append(
            _root_task(
                engine,
                root,
                action,
                cast("TreeFileOperationHandler", place.handler),
                maxdepth=depth,
                ils_kwargs=place.ils_kwargs,
            )
        )
    tasks.extend(_Call(engine, entry, action.file_root) for entry in roots.files)
    errors = [o for o in goes if isinstance(o, Exception)]
    if errors:
        tasks.append(_Call(engine, {"name": ""}, lambda _: action.emit_all(errors)))
    return tasks


class _Batch:
    """A directory's files waiting to go in one request, and whether all sent so far went."""

    def __init__(self) -> None:
        self.waiting: list[tuple[Entry, StoragePath]] = []
        self.done = True


class _Sending(PathAction):
    """A copy's or a move's action: each file goes as its directory's listing finds it.

    A directory's `map` tells where it goes, refusing a name the destination
    takes not, and makes it first where the storage makes every directory
    before what is in it. A file's task makes its directory once, if needed,
    then sends it, or joins its directory's batch, sent once full. A
    directory's `reduce`, once all below it is done, sends what is left of
    its batch, makes it if empty, and for a move deletes its source, if all
    in it went. A value is whether all went.

    Each request is a `common.tree._write` under a permit of the source's
    ``operation_limit``, none starting once cancelled. Each path gone is
    emitted as ``("done", source, destination)``, each error as ``(entry,
    error)``, as `common.glob`'s tasks emit a listing's.
    """

    def __init__(
        self,
        emit_event: Any,
        op: Operation[Any, Any],
        place: _Placing,
        transfer: _Transfer,
        *,
        move: bool,
    ) -> None:
        super().__init__(emit_event)
        self.op = op
        self.place = place
        self.transfer = transfer
        self.move = move
        self.src = cast("TreeFileOperationHandler", place.handler)
        self.limit = self.src.operation_limit
        self.batch_size = place.transfer_batch_size if place.batching else 1

        # Parents first, each directory is made before what is in it, by one
        # mkdir each: a storage requiring parents makes a missing one by more.
        # Not with maxdepth, as fsspec's copy makes no empty directory then.
        self.parents_first = (
            place.parents_required
            and place.profile.parents_required is True
            and place.maxdepth is None
        )

        # Each destination directory's mkdir, once: whether it was made.
        self.directories: dict[str, asyncio.Future[bool]] = {}
        # The source directories whose names the destination refused.
        self.refused: set[str] = set()
        # The names of the files gone into each source directory: a second of
        # one name, as Google Drive lists, would go where the first went.
        self.names: dict[str, set[str]] = {}
        # Each source directory's batch, until its reduce.
        self.batches: dict[str, _Batch] = {}
        # The source directories listed whole: one at maxdepth is not.
        self.listed_whole: set[str] = set()

    # The traversal's calls.

    async def map(self, entry: Entry, match: bool) -> Any:
        if entry["type"] != "directory":
            return None
        name = _trim(entry["name"])
        target = self.target_of(name, directory=True)
        if isinstance(target, Exception):
            await self._emit_event((entry, target))
            return None
        if target is not None and self.parents_first:
            await self.make(target, name)
        return target

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: list[Any]
    ) -> Any:
        name = _trim(entry["name"])
        if entry["type"] != "directory":
            return await self.file(entry, name)
        if value is None:
            return False  # Refused, with all in it.

        last = self.batches.pop(name, _Batch())
        await self.send(last)
        if name not in self.listed_whole:
            return False  # At maxdepth: found, not listed, and kept.
        went = last.done and all(v is True for v in values)
        if not values:
            # An empty directory is made, whatever the storage. With maxdepth,
            # fsspec's copy leaves out directories, and copies files only: a
            # move then leaves it where it is.
            if self.place.maxdepth is not None:
                return False
            went = await self.make(value, name, always=True) and went
        if self.move and went:
            return await self.remove(entry, name, value)
        return went

    async def listed(self, entry: Entry) -> None:
        self.listed_whole.add(_trim(entry["name"]))

    async def failed(self, entry: Entry) -> Any:
        self.batches.pop(_trim(entry["name"]), None)
        return False

    # The roots.

    async def goes_whole(self, root: Entry) -> bool | Exception:
        """Whether a root can go whole, as its target is missing: an error asking is returned."""
        place = self.place
        target = place.target(root["name"])
        if target in place.missing:
            return True
        if target in place.existing:
            return False
        try:
            async with self.limit:
                await place.dst.info(place.dst.entry(target))
        except FileNotFoundError:
            return True
        except Exception as e:
            return e
        return False

    async def whole(self, root: Entry) -> bool:
        """Copy or move a root with all in it, by the storage's one request, after its parent is made."""
        place = self.place
        name = _trim(root["name"])
        target = place.target(name)
        parent = place.dst.parent(target)
        if not await self.make(parent, name, always=not place.implicit_directories):
            return False
        primitive = self.src.mv_tree if self.move else self.src.cp_tree

        async def send() -> None:
            await primitive(root, target, parent=place.parent_of(target))

        if not await self.write(send, root):
            return False
        field = "roots_renamed" if self.move else "roots_copied"
        self.count(**{field: 1})
        await self._emit_event(("done", name, target))
        return True

    async def file_root(self, entry: Entry, *, emit: bool = True) -> Any:
        """A root that is a file: it goes alone, into its parent made if needed.

        Unless ``emit``, its events are returned, e.g. once the traversal ended.
        """
        name = _trim(entry["name"])
        if not emit:
            events: list[Any] = []
            emit_event, self._emit_event = self._emit_event, _collect(events)
            try:
                await self.file(entry, name, root=True)
            finally:
                self._emit_event = emit_event
            return events
        return await self.file(entry, name, root=True)

    async def emit_all(self, errors: list[Exception]) -> None:
        """Emit the errors of roots whose target could not be asked."""
        for error in errors:
            await self._emit_event(({"name": ""}, error))

    # Each file and directory.

    def target_of(
        self, name: str, *, directory: bool = False
    ) -> StoragePath | Exception | None:
        """Where a source path goes; an error if the destination refuses its name; None below a directory refused."""
        parent = name.rpartition("/")[0]
        if parent in self.refused:
            if directory:
                self.refused.add(name)
            return None  # No error of its own: its directory's tells.
        try:
            return self.place.target(name)
        except ValueError as e:
            if directory:
                self.refused.add(name)
            return e

    async def file(self, entry: Entry, name: str, *, root: bool = False) -> bool:
        """Send a file, or add it to its directory's batch: whether it went, or may still."""
        self.count(files_found=1)
        target = self.target_of(name)
        if target is None:
            return False
        if isinstance(target, Exception):
            await self._emit_event((entry, target))
            return False

        parent = name.rpartition("/")[0]
        names = self.names.setdefault(parent, set())
        if name in names:
            await self._emit_event((entry, _twice(target)))
            return False
        names.add(name)

        directory = self.place.dst.parent(target)
        if not await self.make(directory, parent):
            return False
        # The entry goes with its copy, so that cp_entry need not ask it.
        entry = {**entry, "name": name}
        if self.place.batching and not root:
            batch = self.batches.setdefault(parent, _Batch())
            batch.waiting.append((entry, target))
            if len(batch.waiting) >= self.batch_size:
                await self.send(batch)
            return True
        return await self.one(entry, target)

    async def one(self, entry: Entry, target: StoragePath) -> bool:
        """Send a file by the transfer, replacing a file there the destination keeps."""
        place = self.place

        async def send() -> None:
            try:
                await self.transfer(
                    entry,
                    target,
                    missing=place.known_missing(target),
                    parent=place.parent_of(target),
                )
            except FileExistsError as refused:
                if self.move:
                    raise
                await place.replace_file(self.transfer, entry, target, refused)

        if not await self.write(send, entry):
            return False
        self.count(**{"files_moved" if self.move else "files_copied": 1})
        await self._emit_event(("done", entry["name"], target))
        return True

    async def send(self, batch: _Batch) -> None:
        """Send a directory's files waiting by the source's ``cp_files`` or ``mv_files``, emitting each outcome.

        A copy refused by a destination not overwriting is replaced on its own,
        as `one` replaces it. A batch failing whole emits one error.
        """
        waiting, batch.waiting = batch.waiting, []
        if not waiting:
            return
        if self.op.cancelling():
            batch.done = False
            return
        place = self.place
        transfers = [
            FileTransfer(e, t, place.known_missing(t), place.parent_of(t))
            for e, t in waiting
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
            batch.done = False
            await self._emit_event((waiting[0][0], error))
            return
        await self.settle(batch, waiting, outcomes, apart)

    async def settle(
        self,
        batch: _Batch,
        waiting: list[tuple[Entry, StoragePath]],
        outcomes: dict[str, Exception | None],
        apart: list[Exception],
    ) -> None:
        """Count and emit each outcome of a batch sent: a file absent from it, e.g. cancelled, did not go."""
        for entry, target in waiting:
            error = outcomes.get(entry["name"], asyncio.CancelledError())
            if isinstance(error, FileExistsError) and not self.move:
                if await self.one(entry, target):
                    continue
                batch.done = False
                continue
            if error is None:
                self.count(**{"files_moved" if self.move else "files_copied": 1})
                await self._emit_event(("done", entry["name"], target))
                continue
            batch.done = False
            if not isinstance(error, asyncio.CancelledError):
                await self._emit_event((entry, error))
        for error in apart:
            await self._emit_event((waiting[0][0], error))

    async def make(
        self, directory: StoragePath, source: str, *, always: bool = False
    ) -> bool:
        """Make a destination directory once, if needed, before what goes in it: whether it is there.

        A directory failing to be made emits its error once: what goes in it
        is left out, as each would fail as it did.
        """
        place = self.place
        if not place.needs_mkdir(directory, always=always):
            return True
        made = self.directories.get(directory)
        if made is not None:
            return await made
        made = self.directories[directory] = asyncio.get_running_loop().create_future()
        try:

            async def mkdir() -> None:
                await place.make_directory(directory)

            ok = await self.write(mkdir, {"name": source})
            if ok:
                place.made.add(directory)
        except BaseException:
            made.set_result(False)
            raise
        made.set_result(ok)
        return ok

    async def remove(self, entry: Entry, name: str, target: StoragePath) -> bool:
        """Delete a source directory now moved, unless it went with its last file: whether it is gone."""
        if not self.place.implicit_directories:

            async def rmdir() -> None:
                # Already gone counts as deleted, as _tree_rm counts it.
                with contextlib.suppress(FileNotFoundError):
                    await self.src.rmdir({**entry, "name": name})

            if not await self.write(rmdir, entry):
                return False
        self.count(directories_moved=1)
        await self._emit_event(("done", name, target))
        return True

    async def write(self, request: Callable[[], Awaitable[None]], entry: Entry) -> bool:
        """Run a request under a permit, to its end even when closed: whether it went.

        None starts once cancelled. An error is emitted, and it did not go.
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
            await self._emit_event((entry, error))
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


def _twice(target: StoragePath) -> FileExistsError:
    """A file listed under the name of another in its directory: never sent.

    Both would go to one place, the second replacing the first, as on a
    storage holding several items of one name, e.g. Google Drive. The first
    goes, and each other is refused, as GNU's `cp` refuses two sources onto
    one destination: it stays where it is, and so does its directory.
    """
    return FileExistsError(
        errno.EEXIST, "Another file of this name in its directory goes there", target
    )


def _collect(events: list[Any]) -> Callable[[Any], Coroutine[Any, Any, None]]:
    """An emit_event keeping the events in ``events``."""

    async def emit(event: Any) -> None:
        events.append(event)

    return emit
