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
import math
from contextlib import aclosing
from dataclasses import replace
from glob import has_magic
from typing import TYPE_CHECKING, Any, cast

from fsspec_alt_strategies.states import RmState

from ..common.glob import PathAction, TraverseSubtree, VisitDirectory, _no_values
from ..common.operation import Operation
from ..common.paths import _check_maxdepth
from ..common.patterns import _Search
from ..common.strategy import _check_batch_size
from ..common.tree import PathTask, TreeTraversal, _write
from .glob import _shared_start, _walks
from .walk import _MAX_RUNNING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from ..common.handler import Entry, StoragePath
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


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
    patterns. They are matched as `_tree_glob` matches them, in the same
    traversal that deletes: each directory is listed once, and the deletions
    start with the first match. A literal path missing in a list matches
    nothing, as fsspec's ``_expand_path`` leaves it out.

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
    handler's ``operation_limit``. Patterns, or a list, list the directories
    on the way to their matches too, as `_tree_glob` does. With
    ``deletes_directories``, a root costs one
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
        engine = TreeTraversal(max_running=max_running, split=handler.segments)
        op._on_cancel = engine.cancel
        deletion = _TreeDeletion(
            engine._emit_event,
            op,
            handler,
            _Depths(handler, path, maxdepth),
            # rm_tree would delete what is deeper than maxdepth.
            use_rm_tree=maxdepth is None
            and handler.profile.deletes_directories is True,
            missing_raises=single,
        )
        tasks = deletion.depths.roots(engine, deletion, _RootDirectory, ils_kwargs)
        events = engine.events(tasks)
        async with aclosing(events):
            async for event in events:
                yield _outcome(op, handler, event)
        deletion.depths.check(op)

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
    ``rm_dir`` per marker and root. A list lists a tree per root, as
    `_tree_glob` does, narrowed to what its matches start with where the
    profile declares ``recursive_ls_with_prefix``. ``rm_dir`` holds a permit of the handler's
    ``operation_limit`` while it runs: ``rm_files`` takes its own.

    The arguments are as `_tree_rm` takes them. The operation yields each
    path deleted, or an error, and its state is an `RmState`. The call raises
    ValueError if ``maxdepth`` is less than 1, and iterating it if the
    handler's ``rm_batch_size`` is, read once it is open.
    """
    _check_maxdepth(maxdepth)

    single = isinstance(path, str) and not has_magic(path)

    async def run(op: Operation[str, RmState]) -> AsyncGenerator[str | Exception]:
        await handler.open()
        batch_size = handler.rm_batch_size
        _check_batch_size(batch_size, "rm_batch_size")
        engine = TreeTraversal(
            max_running=max_running,
            # Enough to fill a batch.
            max_events=max(16, batch_size),
            split=handler.segments,
        )
        op._on_cancel = engine.cancel
        deletion = _FlatDeletion(
            engine._emit_event,
            op,
            handler,
            _Depths(handler, path, maxdepth),
            batch_size,
            missing_raises=single,
        )
        # Narrowed to what every match starts with, where the storage lists so.
        narrowed = handler.profile.recursive_ls_with_prefix is True
        tasks = deletion.depths.roots(
            engine, deletion, _RootSubtree, ils_kwargs, narrowed=narrowed
        )
        outcomes = _consumed(engine, tasks, deletion.consume)
        async with aclosing(outcomes):
            async for event in outcomes:
                yield _outcome(op, handler, event)
        deletion.depths.check(op)

    return Operation(run, RmState(directories_deleted=0))


class _Depths:
    """What a deletion matches, in one traversal: each path given, and all below each match, as deep as ``maxdepth`` lets it.

    Each path is matched as `_tree_glob` matches it, and ``path/**`` with it,
    so that the traversal reaches all below each match. A path's `map` value
    is then how many levels below it are deleted: None for a path only on the
    way to the matches, which is not. Below a match, a path has one level
    less than its directory. A match has ``maxdepth`` where named literally,
    one level less where a pattern matched it, as fsspec's glob counted that
    level already. A match within another counts as the outer one's.
    """

    def __init__(
        self,
        handler: TreeFileOperationHandler,
        path: StoragePath | Sequence[StoragePath],
        maxdepth: int | None,
    ) -> None:
        self.handler = handler
        self.path = path
        self.maxdepth = maxdepth
        self.searches: list[_Search] = []
        self.literals: set[str] = set()
        for given in [path] if isinstance(path, str) else path:
            name = handler.trim(given)
            if not has_magic(given):
                self.literals.add(name)
            self.searches.append(_Search(name, given.endswith("/"), True, maxdepth))
            below = name.rstrip("/") + "/**" if name else "**"
            self.searches.append(_Search(below, False, True, None))
        # The trees' roots, made from the paths: not listed.
        self.tops: set[str] = set()
        # Whether any path matched was found: none raises, as with fsspec's rm.
        self.found = False

    def roots(
        self,
        engine: TreeTraversal,
        action: PathAction,
        task: Callable[..., PathTask],
        ils_kwargs: dict[str, Any] | None,
        *,
        narrowed: bool = False,
    ) -> list[PathTask]:
        """A task per tree the paths are in, as `_tree_glob` walks them."""
        tasks: list[PathTask] = []
        for root, group in _walks(self.searches):
            self.tops.add(root)
            options: dict[str, Any] = {}
            if narrowed:
                options["prefix"] = _shared_start([s.literal for s in group])
            entry = {"name": root, "type": "directory"}
            tasks.append(
                task(
                    engine,
                    entry,
                    action,
                    self.handler,
                    patterns=group,
                    ils_kwargs=ils_kwargs,
                    **options,
                )
            )
        return tasks

    def left(self, entry: Entry, match: bool, parent_value: Any) -> Any:
        """A path's `map` value: the levels below it to delete, or None."""
        if parent_value is not None:
            value = parent_value - 1
        elif not match:
            return None
        elif self.maxdepth is None:
            value = math.inf
        elif entry["name"].rstrip("/") in self.literals:
            value = self.maxdepth
        else:
            value = self.maxdepth - 1
        return value

    def check(self, op: Operation[str, RmState]) -> None:
        """Raise FileNotFoundError if nothing matched, unless an error may have hidden it."""
        if not self.found and not op.state.errors and not op.cancelling():
            raise FileNotFoundError(self.path)


class _TraversalRoot(PathTask):
    """A deletion's root: a file, or missing, as its listing finds it."""

    async def run(self) -> Any:
        task = cast("Any", self)
        try:
            return await super().run()  # type: ignore[safe-super]
        except NotADirectoryError:
            # A file: deleted alone, if a path names it.
            entry = {**task.entry, "type": "file"}
            if not task._matched(entry):
                return None
            value = await task.action.map(entry, True, None)
            return await task.action.reduce(entry, True, value, _no_values())
        except FileNotFoundError:
            if task.action.missing_raises:
                raise  # The one path given, as fsspec's rm raises.
            return None  # Matches nothing.


class _RootDirectory(_TraversalRoot, VisitDirectory):
    """A root of `_tree_rm`, listed a directory per ``ils``."""


class _RootSubtree(_TraversalRoot, TraverseSubtree):
    """A root of `_flat_rm`, listed by one ``ils_recursive``."""


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
        depths: _Depths,
        *,
        use_rm_tree: bool,
        missing_raises: bool,
    ) -> None:
        super().__init__(emit_event)
        self.op = op
        self.handler = handler
        self.depths = depths
        # Each directory deleted whole by one rm_tree, unlisted: so only roots are.
        self.use_rm_tree = use_rm_tree
        # The one path given, missing, raises, as the walk would find it.
        self.missing_raises = missing_raises

    async def map(self, entry: Entry, match: bool, parent_value: Any) -> Any:
        return self.depths.left(entry, match, parent_value)

    async def visit_child(self, entry: Entry, parent_value: Any) -> bool:
        return parent_value is None or parent_value > 0

    def descends(self, entry: Entry, value: Any) -> bool:
        # One at the limit is deleted unlisted, as fsspec's rm deletes it.
        return value != 0 and not (self.use_rm_tree and value is not None)

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        if value is None:
            async for _ in values:
                pass
            return None  # Only on the way to the matches: kept.
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
            self.depths.found = True  # It was there.
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
        depths: _Depths,
        batch_size: int,
        *,
        missing_raises: bool,
    ) -> None:
        super().__init__(emit_event)
        self.op = op
        self.handler = handler
        self.depths = depths
        self.batch_size = batch_size
        self.missing_raises = missing_raises
        # The directories matched first, which may have a marker whatever their
        # entry says: none with maxdepth, as what is deeper stays.
        self.roots: set[_Key] = set()
        # The directories holding a file that stayed, each above it too.
        self.kept: set[_Key] = set()

    def key(self, name: str) -> _Key:
        """A path's key, by the handler's segments: never verified, as a listing gave it."""
        return tuple(self.handler.segments(name))

    async def map(self, entry: Entry, match: bool, parent_value: Any) -> Any:
        value = self.depths.left(entry, match, parent_value)
        if value == math.inf and parent_value is None:
            self.roots.add(self.key(entry["name"]))
        return value

    async def visit_child(self, entry: Entry, parent_value: Any) -> bool:
        return parent_value is None or parent_value > 0

    def descends(self, entry: Entry, value: Any) -> bool:
        return value != 0

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        if value is None:
            async for _ in values:
                pass
            return None  # Only on the way to the matches: kept.
        if entry["type"] != "directory":
            _found(self.op, 1)
            self.depths.found = True
            await self._emit_event(("file", entry))
            return True
        planned = True
        async for below in values:
            planned &= below is True
        if not planned:
            return False  # Kept, with what failed to list in it.
        # An object store's directory exists by its keys, or its marker: one
        # listed, unlike a root, which is the path given.
        name = entry["name"].rstrip("/")
        self.depths.found |= bool(entry.get("marker")) or name not in self.depths.tops
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
