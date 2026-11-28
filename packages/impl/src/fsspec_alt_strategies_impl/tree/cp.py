# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy that walks the tree and copies while it walks, as a scheduler and a pipeline.

`_tree_cp` is an alternative to fsspec's recursive ``copy``. fsspec's lists the
whole tree first, then copies each path. `_tree_cp` copies each file as soon as
its directory's listing finds it, on `common.tree`'s traversal. `_tree_mv`
runs the same, by `_transfers`.

It runs in three parts:

- `_Plan` reads the destination's profile, finds the roots, and places them
  as fsspec's ``copy`` does;
- `_Schedule`, the traversal's map and reduce, emits the operations each path
  needs as events, e.g. ``{"op": "mkdir_dst", ...}`` or ``{"op": "cp_file",
  ...}``, as the task, the profile and the storages decide;
- `_Pipeline` consumes them in batches, through a chain of stages: the
  destination directories made, the files copied or moved, the sources a move
  copied deleted, and the source directories it emptied deleted. Each stage
  runs its operations in batches of its size, and passes the rest on.

This module depends on `glob`, `rm` and `walk`, and `glob`, `handler`,
`operation`, `paths`, `strategy`, `transfer` and `tree` of `common`.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
from contextlib import aclosing
from dataclasses import replace
from functools import partial
from glob import has_magic
from typing import TYPE_CHECKING, Any, cast

from fsspec.utils import common_prefix

from fsspec_alt_strategies.profile import UNKNOWN
from fsspec_alt_strategies.states import CpState, MvState

from ..common.glob import PathAction
from ..common.handler import FileTransfer, StoragePath
from ..common.operation import Operation
from ..common.paths import _check_maxdepth
from ..common.strategy import _check_batch_size
from ..common.transfer import _transfer_of
from ..common.tree import PathTask, TreeTraversal, _write
from .glob import _tree_glob
from .rm import _consumed, _glob_roots, _Key
from .walk import _MAX_RUNNING, _root_task

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from ..common.handler import Entry, StorageHandler
    from ..common.transfer import _Transfer
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


class _Call(PathTask):
    """A task on a root that is no traversal: e.g. its copy whole, by one ``cp_tree``."""

    def __init__(
        self, engine: TreeTraversal, path: str, call: Callable[[], Awaitable[Any]]
    ) -> None:
        super().__init__(engine, path)
        self.call = call

    async def run(self) -> Any:
        return await self.call()


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

    `_plan` places the roots. The traversal's `_Schedule` emits the
    operations each path needs, and `_Pipeline` runs them, stage by stage.
    """
    state: CpState | MvState = MvState(directories_moved=0) if move else CpState()

    def glob(pattern: StoragePath | Sequence[StoragePath]) -> Operation[Any, Any]:
        return _tree_glob(src, pattern, maxdepth=maxdepth, ils_kwargs=ils_kwargs)

    async def run(op: Operation[tuple[str, str], Any]) -> AsyncGenerator[Any]:
        plan = _Plan(src, dst, move=move, same_storage=same_storage, maxdepth=maxdepth)
        await plan.read_profile(batching)
        await plan.place(path1, path2, path1_entry, glob)
        for error in plan.errors:
            op.state = replace(op.state, errors=op.state.errors + 1)
            yield error
        if plan.directories or plan.files:
            # A path ending with "/" goes as its contents, never whole.
            whole = not (isinstance(path1, str) and path1.endswith("/"))
            sent = _send(op, plan, transfer, whole, ils_kwargs, max_running)
            async with aclosing(sent):
                async for outcome in sent:
                    yield outcome

    return Operation(run, state)


async def _send(
    op: Operation[tuple[str, str], Any],
    plan: _Plan,
    transfer: _Transfer,
    whole: bool,
    ils_kwargs: dict[str, Any] | None,
    max_running: int,
) -> AsyncGenerator[Any]:
    """Traverse the roots planned, scheduling their operations, and run them through the pipeline: the outcomes."""
    engine = TreeTraversal(
        max_running=max_running,
        max_events=max(16, plan.largest_batch),  # Enough to fill a batch.
        split=plan.src.segments,
    )
    op._on_cancel = engine.cancel
    pipeline = _Pipeline(op, plan, transfer)
    if plan.whole:
        _count(op, **{"roots_renamed" if plan.move else "roots_copied": 0})
    schedule = _Schedule(engine._emit_event, op, plan)
    tasks, refused = await schedule.root_tasks(engine, plan.whole and whole, ils_kwargs)
    for failure in refused:
        yield pipeline.outcome(failure)
    try:
        outcomes = _consumed(engine, tasks, pipeline.consume)
        async with aclosing(outcomes):
            async for outcome in outcomes:
                yield outcome
    except NotADirectoryError:
        # The one root, a file: it goes alone.
        if op.cancelling() or plan.places or plan.files:
            return
        root = {**plan.directories[0], "type": "file"}
        for outcome in await pipeline.consume(schedule.root_file(root)):
            yield outcome


# The plan: what the profile decides, and where the roots go.


class _Plan:
    """What a copy or move decides before it starts: how the destination takes it, and where its roots go.

    `read_profile` reads the destination's profile. `place` finds the roots,
    as fsspec's ``copy`` finds them, and where each goes: a single path into
    an existing directory or onto its destination, a pattern's matches under
    their common prefix, a list's under their names.

    It is also what is known of the destination, which the pipeline adds to:
    the directories made or found, with their entries, and the paths found
    missing or existing.
    """

    def __init__(
        self,
        src: TreeFileOperationHandler,
        dst: StorageHandler,
        *,
        move: bool,
        same_storage: bool,
        maxdepth: int | None,
    ) -> None:
        self.src = src
        self.dst = dst
        self.move = move
        self.same_storage = same_storage
        self.maxdepth = maxdepth

        # Read from the profile.
        self.profile = UNKNOWN
        self.implicit_directories = False
        self.parents_required = True
        self.parents_first = False
        self.case_insensitive = False
        self.whole = False
        self.batching = False
        self.copies = False

        # The roots: the directories, traversed, and the files, sent alone.
        self.directories: list[Entry] = []
        self.files: list[Entry] = []
        self.matched: set[str] = set()  # Directory roots a pattern matched.
        self.errors: list[Exception] = []
        # Where a path goes: under its root's target, for a list, else under
        # ``base``, in place of ``prefix``.
        self.places: dict[str, StoragePath] = {}
        self.prefix = ""
        self.base = StoragePath("")

        # What is known of the destination.
        self.made: dict[str, Entry | None] = {}
        self.missing: set[str] = set()
        self.existing: set[str] = set()

    async def read_profile(self, batching: bool) -> None:
        """Open both handlers, and read from the destination's profile how the paths go."""
        await self.src.open()
        await self.dst.open()  # A plain filesystem a router copies to: unknown.
        profile = self.profile = self.dst.profile
        within = self.same_storage
        implicit = bool(profile.implicit_directories)
        # Unknown, directories are made first. Within one storage, a file goes
        # by the storage's own copy or move, whose parents the profile tells.
        declared = profile.copy_parents_required if within else profile.parents_required
        self.parents_required = declared is not False and not implicit
        if self.move and within and not implicit and profile.renames_files is not False:
            self.parents_required = True  # A rename makes no parent, as a write may.
        # A move between two storages makes its directories, as it deletes them.
        self.implicit_directories = implicit and (within or not self.move)
        # One directory per request, its parent existing: each made before all
        # below it. Not with maxdepth, as fsspec's copy makes no empty one then.
        self.parents_first = (
            self.parents_required
            and profile.parents_required is True
            and self.maxdepth is None
        )
        self.case_insensitive = within and profile.case_sensitive is False
        # Only within one storage does the profile tell how a directory goes
        # whole. One gone whole would take what is deeper than maxdepth along.
        whole = profile.renames_directories if self.move else profile.copies_directories
        self.whole = within and bool(whole) and self.maxdepth is None
        self.batching = batching
        # A move renaming no file copies its files, then deletes the sources in
        # batches: an object store's move sends one deletion per batch.
        self.copies = self.move and batching and profile.renames_files is False
        for name in self.batch_sizes():
            _check_batch_size(getattr(self.src, name), name)

    def batch_sizes(self) -> list[str]:
        """The names of the source's batch sizes the pipeline reads."""
        if not self.batching:
            return []
        elif self.copies:
            return ["cp_batch_size", "rm_batch_size"]
        else:
            return ["mv_batch_size" if self.move else "cp_batch_size"]

    @property
    def largest_batch(self) -> int:
        """The largest batch a stage sends: the traversal holds as many events."""
        return max([1, *(getattr(self.src, n) for n in self.batch_sizes())])

    # Placing the roots, as fsspec's copy places them.

    async def place(
        self,
        path1: StoragePath | Sequence[StoragePath],
        path2: StoragePath,
        path1_entry: dict[str, Any] | None,
        glob: Callable[[Any], Operation[Any, Any]],
    ) -> None:
        """Find the roots, and where each goes: nothing for a path onto itself."""
        destination = self.dst.verified(path2)  # ValueError if refused.
        if not isinstance(path1, str):
            await self.place_list(path1, destination, glob)
        elif has_magic(path1):
            await self.place_matches(path1, destination, glob)
        elif await self.place_single(path1, path2, destination, path1_entry):
            return  # Renamed onto itself in another case: placed already.
        sources = [e["name"] for e in (*self.directories, *self.files)]
        for source in sources if self.same_storage else ():
            target = self.target(source)
            if self.fold(target) == self.fold(source) and len(sources) == 1:
                self.directories, self.files = [], []  # Onto itself: left.
                return
            folded = self.fold(source)
            if (
                not source
                or self.fold(target) == folded
                or self.fold(target).startswith(folded + "/")
            ):
                verb = "move" if self.move else "copy"
                raise ValueError(f"Cannot {verb} {source!r} into itself: {target!r}")

    async def find(
        self, pattern: Any, glob: Callable[[Any], Operation[Any, Any]]
    ) -> None:
        """The roots a pattern or list matches, as fsspec's ``_expand_path`` finds them."""
        roots, errors, self.matched = await _glob_roots(
            self.src, glob(pattern), pattern, self.maxdepth
        )
        self.errors += errors
        self.directories, self.files = list(roots.directories), list(roots.files)

    async def place_list(
        self, paths: Sequence[StoragePath], destination: StoragePath, glob: Any
    ) -> None:
        """Each root of a list goes into ``destination``, under its name: the first of a name, in path order."""
        await self.find(paths, glob)
        taken: dict[StoragePath, str] = {}
        for source in sorted(e["name"] for e in (*self.directories, *self.files)):
            target = self.dst.join(destination, self.src.name(source))
            if (other := taken.get(target)) is not None:
                error = FileExistsError(target)
                doing = "Moving" if self.move else "Copying"
                error.add_note(
                    f"{doing} {source!r} to {target!r} skipped: {other!r} goes there too."
                )
                self.errors.append(error)
            else:
                taken[target] = source
                self.places[source] = target
        self.directories = [e for e in self.directories if e["name"] in self.places]
        self.files = [e for e in self.files if e["name"] in self.places]

    async def place_matches(
        self, pattern: StoragePath, destination: StoragePath, glob: Any
    ) -> None:
        """A pattern's matches go under ``destination``, below their common prefix: a single file into it."""
        await self.find(pattern, glob)
        sources = [e["name"] for e in (*self.directories, *self.files)]
        if sources:
            self.prefix = common_prefix(sources)
            if not self.directories and len(sources) == 1:
                self.prefix = self.prefix.rpartition("/")[0]
            self.base = destination

    async def place_single(
        self,
        path1: StoragePath,
        path2: StoragePath,
        destination: StoragePath,
        entry: dict[str, Any] | None,
    ) -> bool:
        """A single path goes into ``destination`` if a directory, or ends with "/", else onto it: whether it is renamed onto itself."""
        root = self.src.trim(path1)
        self.prefix = root
        if self.same_storage and self.fold(root) == self.fold(destination):
            # Onto itself: left as it is, unless a move to another case, which
            # only a rename of the root whole does.
            recased = self.move and root != destination
            if recased and not path1.endswith("/") and not path2.endswith("/"):
                if not self.whole:
                    raise ValueError(
                        f"Cannot move {root!r} to {destination!r}, itself in another"
                        " case: the storage renames nothing whole"
                    )
                self.base = destination
                self.missing.add(destination)  # Found, it would be the root itself.
                self.made.setdefault(self.src.parent(root), None)
                self.directories = [entry or {"name": root}]
                return True
            return False

        into = path2.endswith("/")
        if not into:
            try:
                found = await self.dst.info(self.dst.entry(destination))
            except FileNotFoundError:
                self.missing.add(destination)
            else:
                self.existing.add(destination)
                into = found["type"] == "directory"
                if into:
                    self.made[destination] = found
                else:
                    self.made.setdefault(self.parent(destination), None)
        goes_onto = not into or path1.endswith("/")
        self.base = (
            destination
            if goes_onto
            else self.dst.join(destination, self.src.name(root))
        )
        if self.same_storage:
            self.made.setdefault(self.parent(root), None)  # The root's parent exists.
        if entry is not None and entry["type"] != "directory":
            self.files = [{**entry, "name": root}]
        else:
            self.directories = [{"name": root}]  # Of unknown kind: its listing tells.
        return False

    # Paths.

    def target(self, path: str) -> StoragePath:
        """Where a source path below a root goes, each name checked by the destination's ``join``: ValueError if refused.

        A flat storage's marker, a key ending with "/", keeps its "/".
        """
        if self.places:
            root = path
            while root not in self.places and "/" in root:
                root = root.rpartition("/")[0]
            target = self.dst.join(self.places[root], path[len(root) :])
        else:
            target = self.dst.join(self.base, path[len(self.prefix) :])
        return (
            StoragePath(f"{target}/")
            if path.endswith("/") and not target.endswith("/")
            else target
        )

    def fold(self, path: str) -> str:
        """A path as the storage compares it."""
        return path.lower() if self.case_insensitive else path

    def parent(self, path: str) -> str:
        """A destination path's parent: "" at the root."""
        try:
            return self.dst.parent(path)
        except ValueError:
            return ""

    def known_missing(self, path: str) -> bool:
        """Whether a destination, or a directory above it, was found missing: so is all below it."""
        path = path.rstrip("/")
        while path:
            if path in self.missing:
                return True
            path = path.rpartition("/")[0]
        return False

    def exists(self, directory: str) -> bool:
        """Whether a destination directory needs no mkdir: the root, or one made or found."""
        return not directory or directory == self.dst.root or directory in self.made


# The scheduler: the traversal's map and reduce, emitting the operations.


class _Schedule(PathAction):
    """A copy's or a move's map-reduce, which plans: each path's operations, emitted as events.

    `map` places a path under its parent's value, ``(source, destination)``,
    or None for one refused, its error emitted. A file's `reduce` returns its
    entry and destination. A directory's emits its operations, each a dict
    naming its ``"op"``:

    - ``mkdir_dst``, a destination directory, before what goes in it;
    - ``cp_file`` or ``mv_file``, each file in it, and for a move renaming no
      file, ``rm_src_file`` after its copy;
    - ``cp_dir`` or ``mv_dir``, a root going whole;
    - ``rm_src_dir``, for a move, the source directory once all in it went;
    - ``error``, a path refused.

    Which it emits depends on the task, the destination's profile, and
    whether both are one storage, as the `_Plan` read them. A directory's
    value is whether all below it was planned.
    """

    def __init__(self, emit_event: Any, op: Operation[Any, Any], plan: _Plan) -> None:
        super().__init__(emit_event)
        self.op = op
        self.plan = plan
        self.src = plan.src
        self.roots = {
            self.key(e["name"]): plan.target(e["name"]) for e in plan.directories
        }
        # The names gone into each source directory: a second of one name, as
        # Google Drive lists, would go where the first went.
        self.names: dict[_Key, set[str]] = {}

    def key(self, name: str) -> _Key:
        return tuple(self.src.segments(name))

    async def map(self, entry: Entry, match: bool, parent_value: Any) -> Any:
        name = self.src.trim(entry["name"])
        if parent_value is None:
            target = self.roots.get(self.key(name))
            return None if target is None else (name, target)
        try:
            target = self.plan.dst.join(parent_value[1], self.key(name)[-1])
        except ValueError as error:
            await self._emit_event(_error(entry, error))
            return None
        if entry["type"] != "directory" and (taken := self.taken(name, target)):
            await self._emit_event(taken)
            return None
        return name, target

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        if entry["type"] != "directory":
            _count(self.op, files_found=1)
            return False if value is None else ({**entry, "name": value[0]}, value[1])
        if value is None:
            return False  # Refused, with all in it.
        name, target = value
        planned, items, made = await self.plan_items(name, target, values)
        if not items:
            # An empty directory is made. With maxdepth, fsspec's copy makes
            # none, and a move leaves it where it is.
            if self.plan.maxdepth is not None:
                return False
            if not made:
                await self._emit_event(_mkdir(name, target))
        if not planned:
            return False  # Kept, with what failed in it.
        if self.plan.move:
            await self._emit_event(self.rm_dir(entry, name, target))
        return True

    async def plan_items(
        self, name: str, target: StoragePath, values: AsyncGenerator[Any]
    ) -> tuple[bool, int, bool]:
        """Emit the operations of a directory's files, after its mkdir where needed: whether all below was planned, the items, and whether its mkdir was emitted."""
        made = self.plan.parents_first
        if made:
            await self._emit_event(_mkdir(name, target))
        planned, items = True, 0
        async for item in values:
            items += 1
            if not isinstance(item, tuple):
                planned &= item is True  # A subdirectory's value.
                continue
            if not made and self.plan.parents_required:
                await self._emit_event(_mkdir(name, target))
                made = True
            for event in self.file_ops(*item, alone=False):
                await self._emit_event(event)
        return planned, items, made

    async def failed(self, entry: Entry) -> Any:
        return False

    def file_ops(
        self, entry: Entry, target: StoragePath, *, alone: bool
    ) -> list[dict[str, Any]]:
        """A file's operations: its copy or move, and a move's deletion after its copy where it renames no file."""
        if not self.plan.move:
            return [{"op": "cp_file", "entry": entry, "dst": target, "alone": alone}]
        if self.plan.copies and not alone:
            return [
                {
                    "op": "cp_file",
                    "entry": entry,
                    "dst": target,
                    "alone": False,
                    "moving": True,
                },
                {"op": "rm_src_file", "entry": entry, "dst": target},
            ]
        return [{"op": "mv_file", "entry": entry, "dst": target, "alone": alone}]

    def rm_dir(self, entry: Entry, name: str, target: StoragePath) -> dict[str, Any]:
        """A moved directory's deletion: by a request, unless it went with its last file, having no object of its own."""
        request = (
            not self.plan.implicit_directories
            or bool(entry.get("marker"))
            or (self.plan.maxdepth is None and self.key(name) in self.roots)
        )
        return {
            "op": "rm_src_dir",
            "entry": {**entry, "name": name},
            "dst": target,
            "request": request,
        }

    def taken(self, name: str, target: StoragePath) -> dict[str, Any] | None:
        """The error of a file whose name another took first in its directory, as GNU's ``cp`` refuses two sources onto one."""
        names = self.names.setdefault(self.key(name)[:-1], set())
        if name in names:
            error = FileExistsError(
                errno.EEXIST,
                "Another file of this name in its directory goes there",
                target,
            )
            return _error({"name": name}, error)
        names.add(name)
        return None

    # The roots.

    async def root_tasks(
        self, engine: TreeTraversal, whole: bool, ils_kwargs: dict[str, Any] | None
    ) -> tuple[list[PathTask], list[dict[str, Any]]]:
        """A task per root: its traversal, its going whole, or a file's; and the error of each whose target could not be asked."""
        plan = self.plan
        traversed, wholes, errors = list(plan.directories), [], []
        if whole:
            goes = await asyncio.gather(*(self.goes_whole(e) for e in traversed))
            for entry, outcome in zip(list(traversed), goes, strict=True):
                if outcome is not False:
                    traversed.remove(entry)
                    if outcome is True:
                        wholes.append(entry)
                    else:
                        errors.append(_error(entry, outcome))
        tasks: list[PathTask] = [
            _root_task(
                engine,
                {**e, "type": "directory"},
                self,
                self.src,
                # Below a pattern's match, one level less: fsspec's glob counted it.
                maxdepth=plan.maxdepth - (e["name"] in plan.matched)
                if plan.maxdepth is not None
                else None,
                ils_kwargs=ils_kwargs,
            )
            for e in traversed
        ]
        tasks += [
            _Call(engine, e["name"], partial(self.emit_all, self.root_whole(e)))
            for e in wholes
        ]
        tasks += [
            _Call(engine, e["name"], partial(self.emit_all, self.root_file(e)))
            for e in plan.files
        ]
        return tasks, errors

    async def goes_whole(self, root: Entry) -> bool | Exception:
        """Whether a root can go whole, its target missing: an error asking is returned."""
        target = self.plan.target(root["name"])
        if self.plan.known_missing(target):
            return True
        if target in self.plan.existing:
            return False
        try:
            async with self.src.operation_limit:
                await self.plan.dst.info(self.plan.dst.entry(target))
        except FileNotFoundError:
            return True
        except Exception as e:
            return e
        return False

    def root_whole(self, root: Entry) -> list[dict[str, Any]]:
        """A root going whole, by the storage's one request, after its parent."""
        target = self.plan.target(root["name"])
        ops = []
        parent = self.plan.parent(target)
        if parent and (
            not self.plan.implicit_directories or self.plan.parents_required
        ):
            ops.append(_mkdir(root["name"], parent))
        ops.append(
            {
                "op": "mv_dir" if self.plan.move else "cp_dir",
                "entry": root,
                "dst": target,
            }
        )
        return ops

    def root_file(self, root: Entry) -> list[dict[str, Any]]:
        """A root that is a file: it goes alone, into its directory, made if needed."""
        name = self.src.trim(root["name"])
        target = self.plan.target(name)
        _count(self.op, files_found=1)
        if (taken := self.taken(name, target)) is not None:
            return [taken]
        parent = self.plan.parent(target)
        ops = [_mkdir(name, parent)] if parent and self.plan.parents_required else []
        return ops + self.file_ops({**root, "name": name}, target, alone=True)

    async def emit_all(self, ops: list[dict[str, Any]]) -> None:
        for event in ops:
            await self._emit_event(event)


def _mkdir(source: str, path: StoragePath | str) -> dict[str, Any]:
    return {"op": "mkdir_dst", "src": source, "dst": path}


def _error(entry: Entry, error: Exception) -> dict[str, Any]:
    return {"op": "error", "entry": entry, "error": error}


def _done(source: str, target: str) -> dict[str, Any]:
    return {"op": "done", "src": source, "dst": target}


def _count(op: Operation[Any, Any], **counts: int) -> None:
    """Count in the state what was done, as it is: yielded or not, e.g. once closed."""
    state = op.state
    op.state = replace(
        state, **{k: (getattr(state, k) or 0) + v for k, v in counts.items()}
    )


# The pipeline: each stage runs its operations in batches, and passes the rest on.


class _Pipeline:
    """The consumer of a copy's or a move's operations, a chain of stages, each in batches of its size.

    Each batch of events goes through the stages in turn:

    1. `_MakeDirs` makes the destination directories;
    2. `_SendFiles` copies or moves the files, and the roots going whole;
    3. `_RemoveFiles` deletes the sources a move copied;
    4. `_RemoveDirs` deletes the source directories a move emptied.

    Each runs the operations it takes, and passes the rest on, with an
    ``error`` or a ``done`` per operation it ran. What fails keeps its source
    directory, and all above it, in `kept`: a later stage leaves them. The
    last stage's ``done`` and ``error`` are the operation's outcomes.
    """

    def __init__(
        self, op: Operation[Any, Any], plan: _Plan, transfer: _Transfer
    ) -> None:
        self.op = op
        self.plan = plan
        self.src = plan.src
        self.kept: set[_Key] = set()  # Source directories holding what stayed.
        self.unmade: set[str] = (
            set()
        )  # Destination directories failing, with all below.
        self.failed: set[str] = set()  # Sources a stage failed to send.
        self.stages: list[_Stage] = [
            _MakeDirs(self),
            _SendFiles(self, transfer),
            _RemoveFiles(self),
            _RemoveDirs(self),
        ]

    async def consume(self, events: list[dict[str, Any]]) -> list[Any]:
        """Run a batch of operations through the stages: their outcomes."""
        # The traversal's own failures, e.g. a listing's, come as (entry, error).
        events = [_error(*e) if isinstance(e, tuple) else e for e in events]
        for stage in self.stages:
            events = await stage.run(events)
        return [self.outcome(e) for e in events]

    def keep(self, source: str) -> None:
        """Keep a source that stayed: its directory, and so all above it, stays too."""
        self.failed.add(source)
        self.kept.add(tuple(self.src.segments(source))[:-1])

    def outcome(self, event: dict[str, Any]) -> tuple[str, str] | Exception:
        """What the operation yields of an outcome, its state's current its source."""
        state = self.op.state
        if event["op"] == "done":
            self.op.state = replace(state, current=event["src"])
            return event["src"], event["dst"]
        current = self.src.trim(event["entry"]["name"]) or state.current
        self.op.state = replace(state, current=current, errors=state.errors + 1)
        return cast("Exception", event["error"])

    async def write(
        self, request: Callable[[], Awaitable[Any]], entry: Entry, out: list[Any]
    ) -> bool:
        """Run a request under a permit, to its end even when closed: whether it went. None starts once cancelled."""

        async def limited() -> bool:
            async with self.src.operation_limit:
                if self.op.cancelling():
                    return False
                await request()
                return True

        try:
            return await _write(limited())
        except Exception as error:
            out.append(_error(entry, error))
            return False


class _Stage:
    """A stage of the pipeline: the operations it takes, ``ops``, run in batches of ``batch_size``."""

    ops: tuple[str, ...] = ()
    batch_size = 1

    def __init__(self, pipeline: _Pipeline) -> None:
        self.pipeline = pipeline
        self.plan = pipeline.plan
        self.src = pipeline.src
        self.dst = pipeline.plan.dst

    async def run(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run the operations taken, in batches: the rest, with what they gave, in order."""
        mine = [e for e in events if e["op"] in self.ops]
        rest = [e for e in events if e["op"] not in self.ops]
        if not mine:
            return events
        size = self.batch_size
        outs = await self.send([mine[i : i + size] for i in range(0, len(mine), size)])
        return rest + [o for out in outs for o in out]

    async def send(self, batches: list[list[dict[str, Any]]]) -> list[list[Any]]:
        """Send the batches, at once: each one's outcomes."""
        return await asyncio.gather(*(self.batch(b) for b in batches))

    async def batch(self, ops: list[dict[str, Any]]) -> list[Any]:
        raise NotImplementedError


class _MakeDirs(_Stage):
    """Make the destination directories, each once: one failing leaves out what goes in it.

    Where each is made only in its parent, they are made in order. The files
    and roots going into one that failed are passed on no further, their
    sources kept.
    """

    ops = ("mkdir_dst",)

    async def run(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = await super().run(events)
        unmade = self.pipeline.unmade
        kept = []
        for event in out:
            if (
                event["op"] not in ("error", "done", "rm_src_dir")
                and self.plan.parent(event["dst"]) in unmade
            ):
                self.pipeline.keep(self.src.trim(event["entry"]["name"]))
                continue
            kept.append(event)
        return kept

    async def send(self, batches: list[list[dict[str, Any]]]) -> list[list[Any]]:
        if self.plan.parents_first:
            return [await self.batch(b) for b in batches]  # Each after its parent.
        return await super().send(batches)

    async def batch(self, ops: list[dict[str, Any]]) -> list[Any]:
        out: list[Any] = []
        for event in ops:
            await self.make(event["src"], event["dst"], out)
        return out

    async def make(self, source: str, path: str, out: list[Any]) -> None:
        plan, unmade = self.plan, self.pipeline.unmade
        if plan.exists(path) or path in unmade:
            return
        parent = plan.parent(path)
        if parent in unmade:
            unmade.add(path)  # Its parent's error tells.
            return
        made: list[Entry | None] = []

        async def mkdir() -> None:
            made.append(
                await self.dst.makedirs(
                    cast("StoragePath", path),
                    exist_ok=True,
                    missing=plan.known_missing(path),
                    parent=plan.made.get(parent),
                )
            )

        if await self.pipeline.write(mkdir, {"name": source}, out):
            plan.made[path] = made[0]
        else:
            unmade.add(path)


class _SendFiles(_Stage):
    """Copy or move the files, and the roots going whole.

    Batching, within one storage, the files go to the source's ``cp_files``
    or ``mv_files``, in batches of its batch size. Else each goes by the
    transfer. A copy refused by a destination not overwriting replaces the
    file on its own. A root goes whole by the source's ``cp_tree`` or
    ``mv_tree``.
    """

    ops = ("cp_file", "mv_file", "cp_dir", "mv_dir")

    def __init__(self, pipeline: _Pipeline, transfer: _Transfer) -> None:
        super().__init__(pipeline)
        self.transfer = transfer
        name = (
            "mv_batch_size"
            if self.plan.move and not self.plan.copies
            else "cp_batch_size"
        )
        self.batch_size = getattr(self.src, name) if self.plan.batching else 1
        self.files_done = "files_moved" if self.plan.move else "files_copied"

    async def run(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        batched = [
            e
            for e in events
            if e["op"] in ("cp_file", "mv_file")
            and self.plan.batching
            and not e["alone"]
        ]
        ids = {id(e) for e in batched}
        alone = [e for e in events if e["op"] in self.ops and id(e) not in ids]
        rest = [e for e in events if e["op"] not in self.ops]
        size = self.batch_size
        outs = await asyncio.gather(
            *(self.batch(batched[i : i + size]) for i in range(0, len(batched), size)),
            *(self.one(e) for e in alone),
        )
        return rest + [o for out in outs for o in out]

    def hints(self, event: dict[str, Any]) -> tuple[bool, Entry | None]:
        target = event["dst"]
        return self.plan.known_missing(target), self.plan.made.get(
            self.plan.parent(target)
        )

    def went(self, event: dict[str, Any]) -> list[Any]:
        """A file gone: done, unless a move deletes its source next."""
        if event.get("moving"):
            return []
        _count(self.pipeline.op, **{self.files_done: 1})
        return [_done(event["entry"]["name"], event["dst"])]

    async def one(self, event: dict[str, Any]) -> list[Any]:
        entry, target = event["entry"], event["dst"]
        out: list[Any] = []
        missing, parent = self.hints(event)
        if event["op"] in ("cp_dir", "mv_dir"):
            tree = self.src.mv_tree if event["op"] == "mv_dir" else self.src.cp_tree
            if await self.pipeline.write(
                lambda: tree(entry, target, parent=parent), entry, out
            ):
                _count(
                    self.pipeline.op,
                    **{"roots_renamed" if self.plan.move else "roots_copied": 1},
                )
                return [_done(self.src.trim(entry["name"]), target)]
            self.pipeline.keep(entry["name"])
            return out

        async def send() -> None:
            try:
                await self.transfer(entry, target, missing=missing, parent=parent)
            except FileExistsError as refused:
                if self.plan.move:
                    raise
                await self.replace(entry, target, parent, refused)

        if await self.pipeline.write(send, entry, out):
            return self.went(event)
        self.pipeline.keep(entry["name"])
        return out

    async def replace(
        self,
        entry: Entry,
        target: StoragePath,
        parent: Entry | None,
        refused: FileExistsError,
    ) -> None:
        """Copy a file onto one its destination refused to overwrite: deleted first, unless a directory."""
        found = await self.dst.info(self.dst.entry(target))
        if found["type"] == "directory":
            raise refused
        await self.dst.rm_file({**found, "name": target})
        await self.transfer(entry, target, parent=parent)

    async def batch(self, ops: list[dict[str, Any]]) -> list[Any]:
        """Send files by the source's ``cp_files`` or ``mv_files``: an error raised fails the batch whole."""
        if self.pipeline.op.cancelling():
            for e in ops:
                self.pipeline.keep(e["entry"]["name"])
            return []
        transfers = [FileTransfer(e["entry"], e["dst"], *self.hints(e)) for e in ops]
        names = {e["entry"]["name"] for e in ops}
        moving = ops[0]["op"] == "mv_file"

        async def send() -> tuple[dict[str, Exception | None], list[Exception]]:
            operation = (
                self.src.mv_files(transfers) if moving else self.src.cp_files(transfers)
            )
            outcomes: dict[str, Exception | None] = {}
            apart: list[Exception] = []
            async with operation as files:
                async for outcome in files:
                    if not isinstance(outcome, Exception):
                        outcomes[outcome[0]] = None
                    elif (source := operation.state.current or "") in names:
                        outcomes[source] = outcome  # The state names its file.
                    else:
                        apart.append(outcome)
            return outcomes, apart

        try:
            outcomes, apart = await _write(send())
        except Exception as error:
            for e in ops:
                self.pipeline.keep(e["entry"]["name"])
            return [_error(ops[0]["entry"], error)]
        return await self.record(ops, outcomes, apart)

    async def record(
        self,
        ops: list[dict[str, Any]],
        outcomes: dict[str, Exception | None],
        apart: list[Exception],
    ) -> list[Any]:
        """The outcomes of a batch sent: a file absent from them, e.g. cancelled, did not go."""
        out: list[Any] = []
        for event in ops:
            name = event["entry"]["name"]
            failure = outcomes[name] if name in outcomes else asyncio.CancelledError()
            if isinstance(failure, FileExistsError) and not self.plan.move:
                out += await self.one({**event, "alone": True})
            elif failure is None:
                out += self.went(event)
            else:
                self.pipeline.keep(name)
                if not isinstance(failure, asyncio.CancelledError):
                    out.append(_error(event["entry"], failure))
        out += [_error(ops[0]["entry"], e) for e in apart]
        return out


class _RemoveFiles(_Stage):
    """Delete the sources a move copied, by the source's ``rm_files``, in batches of its ``rm_batch_size``.

    A file whose copy failed is kept. Its sources are deleted once copied,
    even once cancelled: the copy is done.
    """

    ops = ("rm_src_file",)

    def __init__(self, pipeline: _Pipeline) -> None:
        super().__init__(pipeline)
        self.batch_size = self.src.rm_batch_size

    async def run(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        failed = self.pipeline.failed
        return await super().run(
            [
                e
                for e in events
                if e["op"] != "rm_src_file" or e["entry"]["name"] not in failed
            ]
        )

    async def batch(self, ops: list[dict[str, Any]]) -> list[Any]:
        entries = [e["entry"] for e in ops]

        async def delete() -> list[Any]:
            async with self.src.rm_files(entries) as operation:
                return [outcome async for outcome in operation]

        try:
            outcomes = await _write(delete())
        except Exception as error:
            for e in entries:
                self.pipeline.keep(e["name"])
            return [_error(entries[0], error)]
        deleted = {o for o in outcomes if not isinstance(o, Exception)}
        undone = [e for e in entries if e["name"] not in deleted] or entries
        out: list[Any] = [
            _error(undone[0], o) for o in outcomes if isinstance(o, Exception)
        ]
        for event in ops:
            name = event["entry"]["name"]
            if name in deleted:
                _count(self.pipeline.op, files_moved=1)
                out.append(_done(name, event["dst"]))
            else:
                self.pipeline.keep(name)
        return out


class _RemoveDirs(_Stage):
    """Delete the source directories a move emptied, each after those below it, those of a depth at once.

    One holding anything that stayed is kept, and so is each above it. One
    with no object of its own went with its last file: it is counted, with no
    request.
    """

    ops = ("rm_src_dir",)

    async def send(self, batches: list[list[dict[str, Any]]]) -> list[list[Any]]:
        ops = [o for b in batches for o in b]
        depth = lambda e: len(self.key(e))  # noqa: E731
        out: list[list[Any]] = []
        for level in sorted({depth(e) for e in ops}, reverse=True):
            out += await asyncio.gather(
                *(self.batch([e]) for e in ops if depth(e) == level)
            )
        return out

    def key(self, event: dict[str, Any]) -> _Key:
        return tuple(self.src.segments(event["entry"]["name"]))

    async def batch(self, ops: list[dict[str, Any]]) -> list[Any]:
        [event] = ops
        entry, key = event["entry"], self.key(event)
        out: list[Any] = []
        kept = self.pipeline.kept
        if key in kept:
            kept.add(key[:-1])
            return out
        if event["request"]:

            async def rmdir() -> None:
                with contextlib.suppress(FileNotFoundError):  # Gone counts as deleted.
                    await self.src.rmdir(entry)

            if not await self.pipeline.write(rmdir, entry, out):
                kept.add(key[:-1])
                return out
        _count(self.pipeline.op, directories_moved=1)
        return [_done(entry["name"], event["dst"])]
