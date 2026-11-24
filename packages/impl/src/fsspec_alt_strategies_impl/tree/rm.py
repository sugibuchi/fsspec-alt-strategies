# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive delete that walks the tree and deletes while it walks.

`_tree_rm` is an alternative to fsspec's recursive ``_rm``. fsspec's lists the
whole tree first, then deletes each path. `_tree_rm` deletes each directory as
soon as all below it is, as a map-reduce on `common.tree`'s traversal, by
`_Deletion`: a directory's reduce deletes its files, then itself.

This module depends on `glob` and `walk`, and `glob`, `jobs`, `operation`,
`paths` and `tree` of `common`.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import replace
from glob import has_magic
from typing import TYPE_CHECKING, Any, NamedTuple

from fsspec_alt_strategies.states import RmState

from ..common.glob import PathAction
from ..common.jobs import _check_batch_size
from ..common.operation import Operation, _collect_with_errors
from ..common.paths import _check_maxdepth, _trim
from ..common.placement import _roots_of
from ..common.tree import PathTask, TreeTraversal, _write
from .glob import _tree_glob
from .walk import _MAX_RUNNING, _root_task

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
    """Delete a directory and everything in it, deleting while the tree is walked.

    The tree is traversed as `_tree_walk` traverses it bottom up. A directory
    is deleted once all below it is: its files, then itself, while other
    directories are still being listed. So the deepest go first. A directory
    with anything in it that failed is kept.

    It yields each file and directory once deleted, and the error of each
    directory that failed to list or path that failed to be deleted, as the
    storage raised it: the state's ``current`` is then its path. A file
    found already gone is deleted, as ``rm_files`` takes it: it is yielded.
    A directory found already gone is not.

    It wins over fsspec's ``_rm`` on storages that list one directory per
    request and have no recursive delete of their own, such as Databricks
    volumes. The deletions then start after the first listings, not after the
    last. It also keeps going past a failure, and reports each.

    It loses on object stores, which delete keys in batches: their own
    ``_rm``, or `_flat_rm`, makes fewer requests there. On a storage deleting a
    directory with everything in it in one request, that request wins: the
    profile's ``deletes_directories=True`` says ``rm_tree`` sends it.

    With ``deletes_directories``, each root is deleted whole by one ``rm_tree``,
    with no walk, concurrently, under the same limit and cancelling. A root
    whose deletion fails yields the error, and is kept. ``path`` missing
    raises, as without it. A root deleted whole is yielded once, and counted
    as a directory: what was in it is neither listed, yielded nor counted.

    A glob pattern deletes each path it matches, and all in it, as fsspec's
    ``_rm`` with ``recursive=True`` does. So does a list of paths and
    patterns. They are matched as `_tree_glob` matches them, in one search,
    then the directories matched are traversed at once.

    Files are deleted by the handler's ``rm_files``, in batches of its
    ``rm_batch_size``, as it is once open: by default 1, each by ``rm_file``,
    as many at once as the limit lets them. The traversal emits each path to
    delete, and the operation reads them in rounds: the paths emitted while
    the last round was deleted, up to the larger of 16 and
    ``rm_batch_size``. A round's files go in batches of files of any
    directory, at once with its directories. Each directory is deleted after
    all below it, and siblings at once. A file its batch refused yields its
    error and keeps its directory. A batch failing whole yields one error,
    and keeps the directories of its files.

    With ``maxdepth``, it deletes what fsspec's ``_rm`` deletes with it: the
    paths at most ``maxdepth`` levels below ``path``, or below a pattern's
    matches one level less, then ``path`` itself. Directories deeper are not
    listed. A directory at the limit is found but not listed, so it is deleted
    by one ``rm_dir``, as fsspec deletes each path it finds. One with
    anything in it then fails, and the directories above it are kept.
    ``rm_tree`` is not used, as it would delete what is deeper.

    Once cancelled, it starts no new listing or deletion, and yields the
    outcomes of the deletions already sent. Closed, it waits for those
    deletions without yielding them, as their outcome would otherwise be
    unknown. Iterating it raises FileNotFoundError if ``path`` does not exist,
    or matches nothing, and a root's other failure to list.

    Requests: as `_tree_walk` lists bottom up, per root. One ``rm_files`` per
    batch of ``rm_batch_size`` files, by default one ``rm_file`` each, and
    one ``rm_dir`` per directory. Patterns, or a list, add the requests of
    their `_tree_glob`. With ``deletes_directories``, a root costs one
    ``rm_tree`` instead. ``rm_dir`` and ``rm_tree`` hold a permit of the
    handler's ``operation_limit`` while they run: ``rm_files`` takes its own.

    The handler's profile's ``implicit_directories=True`` saves the
    ``rm_dir`` of each directory, as a directory goes with its last file. Such
    a directory is not yielded, as one found already gone is not.

    The primitives are named above as the handler's own, but ``rm_dir``, its
    ``rmdir``. A FileNotFoundError from ``rm_dir`` counts as
    deleted, as object stores have no directories to delete.

    ``path`` is the directory deleted from ``handler``, as its ``ils`` takes it. A
    file is deleted alone. A glob pattern, or a list of paths and patterns,
    deletes all they match. ``maxdepth`` of 1 deletes only what is in
    ``path``, and ``path`` if it is then empty. ``ils_kwargs`` go to each
    ``ils``, a pattern's glob's too. ``max_running`` bounds the directories in
    flight.

    The operation yields each path deleted, or an error, and its state is an
    `RmState`. The call raises ValueError if ``maxdepth`` is less than 1, and
    iterating it if the handler's ``rm_batch_size`` is, read once it is open.
    """
    _check_maxdepth(maxdepth)
    single = isinstance(path, str) and not has_magic(path)

    async def run(op: Operation[str, RmState]) -> AsyncGenerator[str | Exception]:
        await handler.open()
        batch_size = handler.rm_batch_size
        _check_batch_size(batch_size, "rm_batch_size")
        profile = handler.profile

        if single:
            # Of unknown kind: its listing tells if it is a file.
            name = _trim(str(path))
            roots = _Globbed([{"name": name, "type": "directory"}], [], [])
        else:
            roots = await _glob_roots(handler, path, maxdepth, ils_kwargs)
            for error in roots.errors:
                op.state = replace(op.state, errors=op.state.errors + 1)
                yield error

        engine = TreeTraversal(
            max_running=max_running,
            max_events=_events_for(batch_size),
            split=handler.segments,
        )
        op._on_cancel = engine.cancel
        deletion = _Deletion(
            engine._emit_event,
            op,
            handler,
            batch_size=batch_size,
            implicit_directories=bool(profile.implicit_directories),
            missing_raises=single,
        )

        whole = maxdepth is None and profile.deletes_directories is True
        tasks = _root_tasks(engine, deletion, roots, whole, maxdepth, ils_kwargs)
        try:
            outcomes = _consumed(engine, tasks, deletion.consume)
            async with aclosing(outcomes):
                async for event in outcomes:
                    yield deletion.outcome(event)
        except NotADirectoryError:
            # The one root, a file: deleted alone.
            if op.cancelling():
                return
            entry = {"name": _trim(str(path)), "type": "file"}
            deletion.files_found(1)
            for event in await deletion.consume([("file", entry)]):
                yield deletion.outcome(event)

    return Operation(run, RmState(directories_deleted=0))


def _root_tasks(
    engine: TreeTraversal,
    deletion: _Deletion,
    roots: _Globbed,
    whole: bool,
    maxdepth: int | None,
    ils_kwargs: dict[str, Any] | None,
) -> list[PathTask]:
    """A task per root directory, and one for the root files, in batches.

    With ``whole``, a directory is deleted by one ``rm_tree``, with no
    traversal. Below a pattern's match, ``maxdepth`` counts one level less:
    0 deletes it alone.
    """
    tasks: list[PathTask] = []
    for entry in roots.directories:
        if whole:
            tasks.append(_Call(engine, entry, deletion.plan_tree))
            continue
        depth = maxdepth
        if depth is not None and entry.get("matched"):
            depth -= 1
        task = _root_task(
            engine,
            entry,
            deletion,
            deletion.handler,
            maxdepth=depth,
            ils_kwargs=ils_kwargs,
        )
        tasks.append(task)
    if roots.files:
        files = list(roots.files)
        deletion.files_found(len(files))
        tasks.append(_Call(engine, files[0], lambda _: deletion.plan_files(files)))
    return tasks


_MAX_EVENTS = 16
"""The events a traversal changing the storage holds unread, at least."""


def _events_for(batch_size: int) -> int:
    """The events a traversal holds unread: enough to fill a batch of its consumer's."""
    return max(_MAX_EVENTS, batch_size)


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


async def _bottom_up(
    directories: dict[_Key, Any],
    files: Awaitable[Any],
    holding: set[_Key],
    kept: set[_Key],
    remove: Callable[[Any], Awaitable[bool]],
) -> None:
    """Remove directories emptied, each after those below it, its siblings at once.

    ``directories`` are by their segments, each emitted after all below it:
    so a subdirectory is either among them, or settled already. One holding
    a file of ``files``, the round's sending, waits for it. One in ``kept``,
    as something in it stayed, is kept, and so is its parent: the parent
    comes after it. So does one ``remove`` fails.
    """
    children: dict[_Key, list[_Key]] = {}
    tops: list[_Key] = []
    for key in directories:
        if key[:-1] in directories:
            children.setdefault(key[:-1], []).append(key)
        else:
            tops.append(key)

    async def go(key: _Key) -> None:
        below = children.get(key, ())
        if below:
            await asyncio.gather(*(go(child) for child in below))
        if key in holding:
            await files
        if key in kept:
            kept.discard(key)
        elif await remove(directories[key]):
            return
        kept.add(key[:-1])

    await asyncio.gather(*(go(key) for key in tops))


class _Globbed(NamedTuple):
    """The roots a glob matched, and its errors, yielded first."""

    directories: Sequence[Entry]
    files: Sequence[Entry]
    errors: list[Exception]


async def _glob_roots(
    handler: TreeFileOperationHandler,
    pattern: StoragePath | Sequence[StoragePath],
    maxdepth: int | None,
    ils_kwargs: dict[str, Any] | None,
) -> _Globbed:
    """The roots a glob matches, as fsspec's ``_expand_path`` finds them, and the glob's errors.

    A directory a pattern matched, not named literally, is marked
    ``"matched"``: ``maxdepth`` counts one level less below it. Matching
    nothing raises FileNotFoundError, as fsspec does.
    """
    glob = _tree_glob(handler, pattern, maxdepth=maxdepth, ils_kwargs=ils_kwargs)
    matches, errors = await _collect_with_errors(glob)
    roots = _roots_of(matches, errors, pattern)
    named = [pattern] if isinstance(pattern, str) else pattern
    literal = {_trim(p) for p in named if not has_magic(p)}
    directories = [
        {**d, "type": "directory", "matched": d["name"] not in literal}
        for d in roots.directories
    ]
    return _Globbed(directories, roots.files, errors)


class _Call(PathTask):
    """A task on a root that is no traversal: e.g. its deletion whole, by one ``rm_tree``."""

    def __init__(
        self,
        engine: TreeTraversal,
        entry: Entry,
        call: Callable[[Entry], Awaitable[Any]],
    ) -> None:
        super().__init__(engine, entry["name"])
        self.entry = entry
        self.call = call

    async def run(self) -> Any:
        return await self.call(self.entry)


class _Deletion(PathAction):
    """rm as a map-reduce planning the deletions, and `consume` sending them.

    The traversal plans: a file's `reduce` emits ``("file", entry)``, and a
    directory's ``("dir", entry)`` once all below it is emitted. One with
    anything below that failed to list emits nothing: its value, as `failed`
    gives it, is False, and so is each above it. A root deleted whole emits
    ``("tree", entry)``.

    `consume` deletes a batch of those events, as `_tree_rm` reads them: the
    files by ``rm_files``, in batches of ``rm_batch_size``, at once with the
    directories, each after all below it. A directory holding anything that
    stayed is kept, and so is each above it, though planned.

    Each deletion is a `common.tree._write`: closed, it runs to its end. Once
    cancelled, none starts. Each path deleted is an outcome
    ``("deleted", path, is_directory)``, and each error ``(entry, error)``,
    as `common.glob`'s tasks emit a listing's.
    """

    def __init__(
        self,
        emit_event: Any,
        op: Operation[str, RmState],
        handler: TreeFileOperationHandler,
        *,
        batch_size: int,
        implicit_directories: bool,
        missing_raises: bool,
    ) -> None:
        super().__init__(emit_event)
        self.op = op
        self.handler = handler
        self.batch_size = batch_size
        self.implicit_directories = implicit_directories
        # The one path given, missing, raises, as the walk would find it.
        self.missing_raises = missing_raises
        # The directories holding something that stayed, until they come.
        self.kept: set[_Key] = set()
        # The outcomes of the batch consumed.
        self.out: list[Any] = []

    # The traversal's calls: they plan.

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        if entry["type"] != "directory":
            self.files_found(1)
            await self._emit_event(("file", entry))
            return True
        planned = True
        async for below in values:
            planned &= below is True
        if not planned:
            return False  # Kept, with what failed to list in it.
        await self._emit_event(("dir", entry))
        return True

    async def failed(self, entry: Entry) -> Any:
        return False  # Kept, with all in it.

    async def plan_tree(self, entry: Entry) -> None:
        """Plan a root's deletion whole, by one ``rm_tree``."""
        await self._emit_event(("tree", entry))

    async def plan_files(self, entries: list[Entry]) -> None:
        """Plan the deletion of root files, counted as found already."""
        for entry in entries:
            await self._emit_event(("file", entry))

    # The consumer's: it deletes.

    async def consume(self, events: list[Any]) -> list[Any]:
        """Delete what a batch of events plans: their outcomes, after the events passed on, e.g. a listing's error."""
        self.out = []
        files: list[Entry] = []
        directories: dict[_Key, Entry] = {}
        trees: list[Entry] = []
        for event in events:
            if event[0] == "file":
                files.append({**event[1], "name": _trim(event[1]["name"])})
            elif event[0] == "dir":
                directories[self.key(event[1]["name"])] = event[1]
            elif event[0] == "tree":
                trees.append(event[1])
            else:
                self.out.append(event)
        sending = asyncio.ensure_future(self.delete_files(files))
        try:
            holding = {self.key(f["name"])[:-1] for f in files}
            await asyncio.gather(
                sending,
                _bottom_up(
                    directories,
                    sending,
                    holding,
                    self.kept,
                    self.delete_directory,
                ),
                *(self.delete_tree(entry) for entry in trees),
            )
        finally:
            if not sending.done():
                sending.cancel()
                await asyncio.wait([sending])
        out, self.out = self.out, []
        return out

    def key(self, name: str) -> _Key:
        """A path's key, by the handler's segments: never verified, as a listing gave it."""
        return tuple(self.handler.segments(name))

    def files_found(self, count: int) -> None:
        state = self.op.state
        self.op.state = replace(state, files_found=state.files_found + count)

    async def delete_files(self, entries: list[Entry]) -> None:
        """Delete files in batches, at once: the directory of each that stayed is kept."""
        deleted: set[str] = set()
        if entries and not self.op.cancelling():
            size = self.batch_size
            batches = await asyncio.gather(
                *(
                    self.delete_batch(entries[i : i + size])
                    for i in range(0, len(entries), size)
                )
            )
            for events in batches:
                for event in events:
                    if event[0] == "deleted":
                        deleted.add(event[1])
                    self.out.append(event)
        self.kept.update(
            self.key(e["name"])[:-1] for e in entries if e["name"] not in deleted
        )

    async def delete_batch(self, entries: list[Entry]) -> list[Any]:
        """A batch's outcomes, by the handler's ``rm_files``: an error raised fails it whole.

        Its entries are as the listing gave them, never their names alone:
        two files listed under one name, e.g. on Google Drive, are two.
        """

        async def delete() -> list[Any]:
            async with self.handler.rm_files(entries) as operation:
                outcomes = [outcome async for outcome in operation]
            # Counted as done, whether yielded or not, e.g. once closed.
            deleted = sum(not isinstance(o, Exception) for o in outcomes)
            self.counted(files=deleted)
            return outcomes

        try:
            outcomes = await _write(delete())
        except Exception as error:
            return [(entries[0], error)]
        deleted = {o for o in outcomes if not isinstance(o, Exception)}
        undone = [e for e in entries if e["name"] not in deleted] or entries
        return [
            (undone[0], o) if isinstance(o, Exception) else ("deleted", o, False)
            for o in outcomes
        ]

    async def delete_directory(self, entry: Entry) -> bool:
        """Delete an empty directory, unless it went with its last file: whether it is gone."""
        if self.implicit_directories:
            return True  # Gone already, as a FileNotFoundError tells.
        name = _trim(entry["name"])
        try:
            sent = await _write(
                self._limited(self.handler.rmdir, {**entry, "name": name})
            )
        except FileNotFoundError:
            return True  # Already gone: an object store's, or deleted by another.
        except Exception as error:
            self.out.append((entry, error))
            return False
        if not sent:
            return False  # Cancelled before it was sent.
        self.out.append(("deleted", name, True))
        return True

    async def delete_tree(self, entry: Entry) -> bool:
        """Delete a root whole, by one ``rm_tree``, as a profile's ``deletes_directories`` lets it."""
        name = _trim(entry["name"])
        try:
            sent = await _write(
                self._limited(self.handler.rm_tree, {**entry, "name": name})
            )
        except FileNotFoundError:
            if self.missing_raises:
                raise  # The path missing, as the walk would find it.
            return True
        except Exception as error:
            self.out.append((entry, error))
            return False
        if not sent:
            return False  # Cancelled before it was sent.
        self.out.append(("deleted", name, True))
        return True

    async def _limited(
        self, primitive: Callable[[Entry], Awaitable[Any]], entry: Entry
    ) -> bool:
        """A primitive's request, holding a permit of the handler's ``operation_limit``: whether it was sent.

        Cancelled while it waits for the permit, it is not: so none starts
        once the operation is cancelled.
        """
        async with self.handler.operation_limit:
            if self.op.cancelling():
                return False
            await primitive(entry)
            # Counted as done, whether yielded or not, e.g. once closed.
            self.counted(directories=1)
            return True

    def counted(self, *, files: int = 0, directories: int = 0) -> None:
        """Count paths deleted in the state."""
        state = self.op.state
        self.op.state = replace(
            state,
            files_deleted=state.files_deleted + files,
            directories_deleted=(state.directories_deleted or 0) + directories,
        )

    def outcome(self, event: Any) -> str | Exception:
        """What the operation yields of an event, counted in its state."""
        state = self.op.state
        if event[0] == "deleted":
            # Counted as it was done: see counted.
            _, name, _ = event
            self.op.state = replace(state, current=name)
            return name
        entry, error = event
        self.op.state = replace(
            state, current=_trim(entry["name"]), errors=state.errors + 1
        )
        return error
