# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Recursive deletes that walk the tree and delete while they walk: one for trees, one for keys.

`_tree_rm` and `_flat_rm` are alternatives to fsspec's recursive ``_rm``.
fsspec's lists the whole tree first, then deletes each path. Both delete as
`common.tree`'s traversal lists:

- `_tree_rm`, for a storage whose directories are real, deletes in the
  traversal's map-reduce: each file as its directory's listing finds it, and
  each directory in its reduce, once all below it is gone. The traversal's
  order is the deletion's, and its parallelism too;
- `_flat_rm`, for an object store, whose directories are the prefixes of its
  keys, only lists in the traversal: it emits the keys, which the operation
  deletes in batches of ``rm_batch_size``. No order needs keeping.

This module depends on `glob` and `walk`, and `glob`, `operation`, `paths`,
`strategy` and `tree` of `common`.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import replace
from glob import has_magic
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from fsspec_alt_strategies.states import RmState

from ..common.glob import PathAction, TraverseSubtree, VisitDirectory
from ..common.operation import Operation, _collect_with_errors
from ..common.paths import _check_maxdepth
from ..common.strategy import _check_batch_size
from ..common.tree import PathTask, TreeTraversal, _write
from .glob import _tree_glob
from .walk import _MAX_RUNNING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from ..common.handler import Entry, StorageHandler, StoragePath
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


class _Roots(NamedTuple):
    """Where an operation starts: what it walks, and what it handles alone.

    Each root is its entry, so that a copy or move hands it on without asking
    it again. ``directories`` are walked, or listed, at once. A single path of
    unknown kind is among them, its entry its ``"name"`` alone, as its listing
    tells what it is. ``files`` are known files, e.g. glob matches.
    """

    directories: Sequence[dict[str, Any]]
    files: Sequence[dict[str, Any]] = ()

    @property
    def paths(self) -> list[str]:
        """The paths of all the roots, the directories first."""
        return [e["name"] for e in (*self.directories, *self.files)]


def _roots_of(
    matches: list[dict[str, Any]],
    errors: list[Exception],
    pattern: str | Sequence[str],
) -> _Roots:
    """The roots of a glob's matches, as fsspec's ``_expand_path`` finds them.

    A match inside a directory that matches is left out: walking the directory
    finds it. Matching nothing raises FileNotFoundError, as fsspec does.
    """
    if not matches and not errors:
        raise FileNotFoundError(pattern)

    directories: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []

    # Sorted, a directory comes before all in it.
    for entry in sorted(matches, key=lambda e: e["name"]):
        name = entry["name"].rstrip("/")
        parts = name.split("/")
        if any("/".join(parts[:i]) in directories for i in range(1, len(parts))):
            continue
        if entry["type"] == "directory":
            directories[name] = {**entry, "name": name}
        else:
            files.append({**entry, "name": name})

    return _Roots(list(directories.values()), files)


async def _glob_roots(
    handler: StorageHandler,
    glob: Operation[Any, Any],
    pattern: StoragePath | Sequence[StoragePath],
    maxdepth: int | None,
) -> tuple[_Roots, list[Exception], set[str]]:
    """The roots a glob matches, its errors, and the directories a pattern matched.

    With ``maxdepth``, a directory root a pattern matched, not one named
    literally, is walked one level less deep, as fsspec's glob counted that
    level already.
    """
    matches, errors = await _collect_with_errors(glob)
    roots = _roots_of(matches, errors, pattern)
    matched: set[str] = set()
    if maxdepth is not None:
        named = [pattern] if isinstance(pattern, str) else pattern
        literal = {handler.trim(p) for p in named if not has_magic(p)}
        matched = {d["name"] for d in roots.directories if d["name"] not in literal}
    return roots, errors, matched


async def _rm_roots(
    op: Operation[str, RmState],
    handler: TreeFileOperationHandler,
    path: StoragePath | Sequence[StoragePath],
    maxdepth: int | None,
    ils_kwargs: dict[str, Any] | None,
) -> tuple[_Roots, list[Exception], set[str]]:
    """A deletion's roots, its glob's errors, counted, and the directories a pattern matched.

    A single path is of unknown kind: its listing tells if it is a file.
    """
    if isinstance(path, str) and not has_magic(path):
        name = handler.trim(path)
        return _Roots([{"name": name, "type": "directory"}]), [], set()
    glob = _tree_glob(handler, path, maxdepth=maxdepth, ils_kwargs=ils_kwargs)
    roots, errors, matched = await _glob_roots(handler, glob, path, maxdepth)
    op.state = replace(op.state, errors=op.state.errors + len(errors))
    return roots, errors, matched


def _file_root(handler: TreeFileOperationHandler, path: Any) -> Entry:
    """The one root, found a file."""
    return {"name": handler.trim(str(path)), "type": "file"}


def _tree_rm(
    handler: TreeFileOperationHandler,
    path: StoragePath | Sequence[StoragePath],
    *,
    maxdepth: int | None = None,
    ils_kwargs: dict[str, Any] | None = None,
    max_running: int = _MAX_RUNNING,
) -> Operation[str, RmState]:
    """Delete a directory and everything in it, on a storage whose directories are real.

    The tree is listed a directory per ``ils``, several at once, never by
    ``ils_recursive``: so the deletions run as parallel as the listings. It is
    deleted as it is listed: each file by ``rm_file`` as its directory's listing finds it,
    and each directory by ``rm_dir`` once all below it is gone. So the
    deletions start after the first listing, as many at once as the limit
    lets them, and the deepest directories go first. A directory with
    anything in it that failed is kept, and so is each above it.

    It wins over fsspec's ``_rm`` on storages that list one directory per
    request and have no recursive delete of their own, such as local disks,
    SFTP or Databricks volumes. It also keeps going past a failure, and
    reports each. On an object store it is correct, but sends an ``rm_dir``
    per directory, and deletes no keys in batches: `_flat_rm` wins there.

    With the profile's ``deletes_directories=True``, each root is deleted
    whole by one ``rm_tree``, with no walk, concurrently, under the same limit
    and cancelling. A root whose deletion fails yields the error, and is
    kept. A root deleted whole is yielded once, and counted as a directory:
    what was in it is neither listed, yielded nor counted. ``rm_tree`` is not
    used with ``maxdepth``, as it would delete what is deeper.

    It yields each file and directory once deleted, and the error of each
    directory that failed to list or path that failed to be deleted, as the
    storage raised it: the state's ``current`` is then its path. A file
    found already gone is deleted: it is yielded. A directory found already
    gone is not.

    A glob pattern deletes each path it matches, and all in it, as fsspec's
    ``_rm`` with ``recursive=True`` does. So does a list of paths and
    patterns. They are matched as `_tree_glob` matches them, in one search,
    then the directories matched are traversed at once.

    With ``maxdepth``, it deletes what fsspec's ``_rm`` deletes with it: the
    paths at most ``maxdepth`` levels below ``path``, or below a pattern's
    matches one level less, then ``path`` itself. Directories deeper are not
    listed. A directory at the limit is found but not listed, so it is deleted
    by one ``rm_dir``, as fsspec deletes each path it finds. One with
    anything in it then fails, and the directories above it are kept.

    Once cancelled, it starts no new listing or deletion, and yields the
    outcomes of the deletions already sent. Closed, it waits for those
    deletions without yielding them, as their outcome would otherwise be
    unknown. Iterating it raises FileNotFoundError if ``path`` does not exist,
    or matches nothing, and a root's other failure to list.

    Requests: as `_tree_walk` lists bottom up, per root, then one ``rm_file``
    per file and one ``rm_dir`` per directory, each holding a permit of the
    handler's ``operation_limit``. Patterns, or a list, add the requests of
    their `_tree_glob`. With ``deletes_directories``, a root costs one
    ``rm_tree`` instead. ``rm_dir`` is the handler's ``rmdir``: its
    FileNotFoundError counts as deleted.

    ``path`` is the directory deleted from ``handler``, as its ``ils`` takes
    it. A file is deleted alone. ``maxdepth`` of 1 deletes only what is in
    ``path``, and ``path`` if it is then empty. ``ils_kwargs`` go to each
    ``ils``, a pattern's glob's too. ``max_running`` bounds the directories in
    flight.

    The operation yields each path deleted, or an error, and its state is an
    `RmState`. The call raises ValueError if ``maxdepth`` is less than 1.
    """
    _check_maxdepth(maxdepth)
    single = isinstance(path, str) and not has_magic(path)

    async def run(op: Operation[str, RmState]) -> AsyncGenerator[str | Exception]:
        await handler.open()
        roots, errors, matched = await _rm_roots(
            op, handler, path, maxdepth, ils_kwargs
        )
        for error in errors:
            yield error

        engine = TreeTraversal(max_running=max_running, split=handler.segments)
        op._on_cancel = engine.cancel
        deletion = _TreeDeletion(
            engine._emit_event,
            op,
            handler,
            # rm_tree would delete what is deeper than maxdepth.
            use_rm_tree=maxdepth is None
            and handler.profile.deletes_directories is True,
            missing_raises=single,
        )
        tasks: list[PathTask] = [
            VisitDirectory(
                engine,
                entry,
                deletion,
                handler,
                # Below a pattern's match, one level less: fsspec's glob counted it.
                maxdepth=maxdepth - (entry["name"] in matched)
                if maxdepth is not None
                else None,
                ils_kwargs=ils_kwargs,
            )
            # A file, known one, is reduced unlisted.
            for entry in (*roots.directories, *roots.files)
        ]
        try:
            events = engine.events(tasks)
            async with aclosing(events):
                async for event in events:
                    yield _outcome(op, handler, event)
        except NotADirectoryError:
            # The one root, a file: deleted alone.
            if op.cancelling():
                return
            _found(op, 1)
            outcome = await deletion.delete_file(_file_root(handler, path))
            if outcome:
                yield _outcome(op, handler, outcome)

    return Operation(run, RmState(directories_deleted=0))


def _flat_rm(
    handler: TreeFileOperationHandler,
    path: StoragePath | Sequence[StoragePath],
    *,
    maxdepth: int | None = None,
    ils_kwargs: dict[str, Any] | None = None,
    max_running: int = _MAX_RUNNING,
) -> Operation[str, RmState]:
    """Delete a prefix and every key under it, on an object store, in batches.

    It assumes a storage whose directories are the prefixes of its keys, as
    the profile's ``implicit_directories=True`` declares: a directory goes
    with its last key. So it deletes only keys, in no order. The tree is
    listed by one ``ils_recursive`` per root, which the handler must
    implement, whatever the profile's ``recursive_ls``. The traversal only lists, and emits each key found. The operation deletes
    them as they come, by the handler's ``rm_files``, in batches of its
    ``rm_batch_size``, as it is once open, several batches at once.

    A directory's marker, an entry with ``"marker"``, is deleted by
    ``rm_dir`` once its batch's keys are, unless a key below it stayed. So
    is each root, which may have one whatever its listing says: none with
    ``maxdepth``, as what is deeper stays. Other directories send nothing,
    and are not yielded.

    It wins over fsspec's ``_rm`` on object stores deleting many keys by one
    request, such as S3's DeleteObjects: the deletions start after the first
    listing, not after the last. It also keeps going past a failure, and
    reports each. On a storage whose directories are real it leaves them:
    `_tree_rm` is correct there.

    The operation reads the traversal's keys in rounds: the keys emitted
    while the last round was deleted, up to the larger of 16 and
    ``rm_batch_size``. A key its batch refused yields its error. A batch
    failing whole yields one error.

    Patterns, lists, ``maxdepth``, cancelling, closing, the errors raised and
    those yielded are as `_tree_rm`'s, but a directory at ``maxdepth``'s
    limit, which sends nothing unless it is a marker.

    Requests: as `_tree_walk` lists, per root, then one ``rm_files`` per batch
    of ``rm_batch_size`` keys, by default one ``rm_file`` each, and one
    ``rm_dir`` per marker and root. Patterns, or a list, add the requests of
    their `_tree_glob`. ``rm_dir`` holds a permit of the handler's
    ``operation_limit`` while it runs: ``rm_files`` takes its own.

    The arguments are as `_tree_rm` takes them. The operation yields each
    path deleted, or an error, and its state is an `RmState`. The call raises
    ValueError if ``maxdepth`` is less than 1, and iterating it if the
    handler's ``rm_batch_size`` is, read once it is open.
    """
    _check_maxdepth(maxdepth)

    async def run(op: Operation[str, RmState]) -> AsyncGenerator[str | Exception]:
        await handler.open()
        batch_size = handler.rm_batch_size
        _check_batch_size(batch_size, "rm_batch_size")
        roots, errors, matched = await _rm_roots(
            op, handler, path, maxdepth, ils_kwargs
        )
        for error in errors:
            yield error

        engine = TreeTraversal(
            max_running=max_running,
            # Enough to fill a batch.
            max_events=max(16, batch_size),
            split=handler.segments,
        )
        op._on_cancel = engine.cancel
        # The roots may have a marker whatever their entry says: none with
        # maxdepth, as what is deeper stays, and so do they.
        emptied = [d["name"] for d in roots.directories] if maxdepth is None else []
        deletion = _FlatDeletion(engine._emit_event, op, handler, batch_size, emptied)
        tasks: list[PathTask] = [
            TraverseSubtree(
                engine,
                entry,
                deletion,
                handler,
                # Below a pattern's match, one level less: fsspec's glob counted it.
                maxdepth=maxdepth - (entry["name"] in matched)
                if maxdepth is not None
                else None,
                ils_kwargs=ils_kwargs,
            )
            # A file, known one, is reduced unlisted.
            for entry in (*roots.directories, *roots.files)
        ]
        try:
            outcomes = _consumed(engine, tasks, deletion.consume)
            async with aclosing(outcomes):
                async for event in outcomes:
                    yield _outcome(op, handler, event)
        except NotADirectoryError:
            # The one root, a file: deleted alone.
            if op.cancelling():
                return
            _found(op, 1)
            for event in await deletion.consume([("file", _file_root(handler, path))]):
                yield _outcome(op, handler, event)

    return Operation(run, RmState(directories_deleted=0))


def _found(op: Operation[str, RmState], count: int) -> None:
    """Count files found in the state."""
    op.state = replace(op.state, files_found=op.state.files_found + count)


def _counted(
    op: Operation[str, RmState], *, files: int = 0, directories: int = 0
) -> None:
    """Count paths deleted in the state, as they are: whether yielded or not, e.g. once closed."""
    state = op.state
    op.state = replace(
        state,
        files_deleted=state.files_deleted + files,
        directories_deleted=(state.directories_deleted or 0) + directories,
    )


def _outcome(
    op: Operation[str, RmState], handler: TreeFileOperationHandler, event: Any
) -> str | Exception:
    """What the operation yields of an outcome, ``("deleted", path, is_directory)`` or ``(entry, error)``."""
    state = op.state
    if event[0] == "deleted":
        op.state = replace(state, current=event[1])  # Counted as done: see _counted.
        return cast("str", event[1])
    entry, error = event
    op.state = replace(
        state, current=handler.trim(entry["name"]), errors=state.errors + 1
    )
    return cast("Exception", error)


async def _delete(
    op: Operation[str, RmState],
    handler: TreeFileOperationHandler,
    primitive: Callable[[Entry], Awaitable[Any]],
    entry: Entry,
    *,
    directory: bool,
) -> Any:
    """Delete a path by a primitive, as a `common.tree._write`: its outcome, or None if not sent.

    It holds a permit of the handler's ``operation_limit`` while it runs,
    and checks for cancelling once it has it: so none starts once the
    operation is cancelled. A file found already gone is deleted. A directory
    found already gone, which then was none, e.g. an object store's, is
    ``()``: it is not yielded.
    """
    entry = {**entry, "name": handler.trim(entry["name"])}

    async def send() -> bool:
        async with handler.operation_limit:
            if op.cancelling():
                return False
            try:
                await primitive(entry)
            except FileNotFoundError:
                if directory:
                    raise
            _counted(op, files=not directory, directories=directory)
            return True

    try:
        sent = await _write(send())
    except FileNotFoundError:
        return ()
    except Exception as error:
        return (entry, error)
    return ("deleted", entry["name"], directory) if sent else None


async def _consumed(
    engine: TreeTraversal,
    tasks: list[PathTask],
    consume: Callable[[list[Any]], Awaitable[list[Any]]],
) -> AsyncGenerator[Any]:
    """Run a traversal planning writes, and ``consume`` each batch of its events: the outcomes.

    A batch is what the traversal emitted while the last was consumed, at
    most ``max_events``: so the traversal runs ahead by one batch, no more.
    """
    batches = engine.event_batches(tasks, engine.max_events)
    async with aclosing(batches):
        async for events in batches:
            for outcome in await consume(events):
                yield outcome


_Key = tuple[str, ...]
"""A path by its segments, as the handler's ``segments`` gives them: its parent's is one shorter."""


class _TreeDeletion(PathAction):
    """`_tree_rm`'s map-reduce, which deletes: each file in its reduce, and each directory after all below it.

    Each outcome is emitted as the traversal's event as its request ends. A
    directory's value is whether all below it is gone: False keeps it, and
    each above it.
    """

    def __init__(
        self,
        emit_event: Any,
        op: Operation[str, RmState],
        handler: TreeFileOperationHandler,
        *,
        use_rm_tree: bool,
        missing_raises: bool,
    ) -> None:
        super().__init__(emit_event)
        self.op = op
        self.handler = handler
        # Each directory deleted whole by one rm_tree, unlisted: so only roots are.
        self.use_rm_tree = use_rm_tree
        # The one path given, missing, raises, as the walk would find it.
        self.missing_raises = missing_raises

    def descends(self, entry: Entry, value: Any) -> bool:
        return not self.use_rm_tree

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        if entry["type"] != "directory":
            _found(self.op, 1)
            return await self.emitted(await self.delete_file(entry))
        if self.use_rm_tree:
            return await self.delete_tree(entry)
        gone = True
        async for below in values:
            gone &= below is True
        if not gone:
            return False  # Kept, with what stayed in it.
        rmdir = self.handler.rmdir
        return await self.emitted(
            await _delete(self.op, self.handler, rmdir, entry, directory=True)
        )

    async def failed(self, entry: Entry) -> Any:
        return False  # Kept, with all in it.

    async def delete_file(self, entry: Entry) -> Any:
        """Delete a file by ``rm_file``: its outcome, as `_delete` gives it."""
        rm_file = self.handler.rm_file
        return await _delete(self.op, self.handler, rm_file, entry, directory=False)

    async def delete_tree(self, entry: Entry) -> bool:
        """Delete a directory whole, by one ``rm_tree``: whether it is gone."""
        rm_tree = self.handler.rm_tree
        outcome = await _delete(self.op, self.handler, rm_tree, entry, directory=True)
        if outcome == () and self.missing_raises:
            raise FileNotFoundError(self.handler.trim(entry["name"]))
        return await self.emitted(outcome)

    async def emitted(self, outcome: Any) -> bool:
        """Emit an outcome, if any: whether the path is gone."""
        if outcome:
            await self._emit_event(outcome)
        return outcome is not None and (not outcome or outcome[0] == "deleted")


class _FlatDeletion(PathAction):
    """`_flat_rm`'s map-reduce, which only plans, and `consume`, which deletes.

    A file's `reduce` emits ``("file", entry)``. A directory's emits
    ``("dir", entry)`` where it has a marker, or is a root emptied, once all
    below it is emitted, as it has an object to delete. One with anything
    below that failed to list emits nothing, and its value is False, as is
    each above it. `consume` deletes a round of them: the files by
    ``rm_files``, in batches of ``rm_batch_size``, then the directories,
    unless a file below one stayed. Each path deleted is an outcome
    ``("deleted", path, is_directory)``, and each error ``(entry, error)``.
    """

    def __init__(
        self,
        emit_event: Any,
        op: Operation[str, RmState],
        handler: TreeFileOperationHandler,
        batch_size: int,
        roots: list[str],
    ) -> None:
        super().__init__(emit_event)
        self.op = op
        self.handler = handler
        self.batch_size = batch_size
        self.roots = {self.key(name) for name in roots}
        # The directories holding a file that stayed, each above it too.
        self.kept: set[_Key] = set()

    def key(self, name: str) -> _Key:
        """A path's key, by the handler's segments: never verified, as a listing gave it."""
        return tuple(self.handler.segments(name))

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        if entry["type"] != "directory":
            _found(self.op, 1)
            await self._emit_event(("file", entry))
            return True
        planned = True
        async for below in values:
            planned &= below is True
        if not planned:
            return False  # Kept, with what failed to list in it.
        if entry.get("marker") or self.key(entry["name"]) in self.roots:
            await self._emit_event(("dir", entry))
        return True

    async def failed(self, entry: Entry) -> Any:
        return False  # Kept, with all in it.

    async def consume(self, events: list[Any]) -> list[Any]:
        """Delete what a round of events plans: their outcomes, after the events passed on, e.g. a listing's error."""
        out: list[Any] = []
        files: list[Entry] = []
        directories: list[Entry] = []
        for event in events:
            if event[0] == "file":
                files.append({**event[1], "name": self.handler.trim(event[1]["name"])})
            elif event[0] == "dir":
                directories.append(event[1])
            else:
                out.append(event)
        if files and not self.op.cancelling():
            size = self.batch_size
            batches = await asyncio.gather(
                *(
                    self.delete_batch(files[i : i + size])
                    for i in range(0, len(files), size)
                )
            )
            out.extend(outcome for batch in batches for outcome in batch)
        # Each directory's files were emitted before it, in this round or one before.
        rmdir = self.handler.rmdir
        outcomes = await asyncio.gather(
            *(
                _delete(self.op, self.handler, rmdir, d, directory=True)
                for d in directories
                if self.key(d["name"]) not in self.kept
            )
        )
        out.extend(outcome for outcome in outcomes if outcome)
        return out

    async def delete_batch(self, entries: list[Entry]) -> list[Any]:
        """A batch's outcomes, by the handler's ``rm_files``: an error raised fails it whole.

        Its entries are as the listing gave them, never their names alone:
        two files listed under one name, e.g. on Google Drive, are two. The
        directories above a file that stayed are kept.
        """

        async def delete() -> list[Any]:
            async with self.handler.rm_files(entries) as operation:
                outcomes = [outcome async for outcome in operation]
            # Counted as done, whether yielded or not, e.g. once closed.
            _counted(self.op, files=sum(not isinstance(o, Exception) for o in outcomes))
            return outcomes

        try:
            outcomes = await _write(delete())
        except Exception as error:
            self.keep(entries)
            return [(entries[0], error)]
        deleted = {o for o in outcomes if not isinstance(o, Exception)}
        undone = [e for e in entries if e["name"] not in deleted]
        self.keep(undone)
        undone = undone or entries
        return [
            (undone[0], o) if isinstance(o, Exception) else ("deleted", o, False)
            for o in outcomes
        ]

    def keep(self, entries: list[Entry]) -> None:
        """Keep the directories above files that stayed: their markers are not deleted."""
        for entry in entries:
            key = self.key(entry["name"])
            self.kept.update(key[:i] for i in range(1, len(key)))
