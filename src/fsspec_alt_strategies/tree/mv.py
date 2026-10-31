# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive move that walks the tree and moves while it walks.

`tree_mv` is an alternative to fsspec's recursive ``mv``. fsspec's copies the
whole tree, then deletes it, each stage listing the tree first. `tree_mv` moves
each folder's files as soon as the folder is listed, each by a copy and a
delete, and deletes each source folder once it is empty.

It is for storages without a rename of their own. Where a storage renames a
file or a folder in one request, that request wins.

This module depends on `control`, `profile` and `walk`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass
from glob import has_magic
from typing import TYPE_CHECKING, Any, Literal

from fsspec.asyn import _get_batch_size

from ..control import Operation, State
from ..profile import _resolve
from .walk import _listings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from fsspec.asyn import AsyncFileSystem

    from ..profile import StorageProfile
    from .walk import _Listing

__all__ = ["MvState", "tree_mv"]


@dataclass(frozen=True, kw_only=True)
class MvState(State):
    """The progress of a `tree_mv`.

    Attributes
    ----------
    files_found : int
        The files found so far. It grows as the walk goes on.
    files_moved : int
        The files moved so far.
    folders_moved : int
        The source folders deleted so far, once all in them was moved.
    """

    files_found: int = 0
    files_moved: int = 0
    folders_moved: int = 0


def tree_mv(
    fs: AsyncFileSystem,
    path1: str,
    path2: str,
    *,
    profile: StorageProfile | None = None,
    rm_folder: Callable[[str], Awaitable[Any]] | None = None,
    batch_size: int | None = None,
) -> Operation[tuple[str, str], MvState]:
    """Move a folder and everything in it, moving while the tree is walked.

    It places the paths as fsspec's ``mv`` with ``recursive=True`` does. If
    ``path2`` is an existing folder, or ends with "/", and ``path1`` does not
    end with "/", ``path1`` goes into it: ``path1/x`` becomes
    ``path2/<name of path1>/x``. Otherwise ``path1`` becomes ``path2``:
    ``path1/x`` becomes ``path2/x``. A file moves the same way. Existing files
    are overwritten, as fsspec's ``mv`` does.

    The tree is walked as `tree_walk` walks it. Each folder's files are moved
    as soon as the folder is listed, each by a ``_cp_file`` then an
    ``_rm_file``. A source folder is deleted once everything in it is moved,
    so the deepest go first. A folder is created at the destination before
    files are copied into it, and for each empty folder.

    Unlike fsspec's ``mv``, it keeps going past a failure, and reports each. A
    file that failed to copy stays where it was, and so do the folders above
    it. A file copied but not deleted is then in both places, as its error
    notes.

    It wins over fsspec's ``mv`` on storages that list one folder per request
    and rename nothing in one request. The moves then start after the first
    listing, not after the whole tree is listed twice.

    Once cancelled, it starts no new listing or move, and yields the outcomes
    of the moves already sent. Closed, it waits for those moves without
    yielding them, as their outcome would otherwise be unknown. Iterating it
    raises FileNotFoundError if ``path1`` does not exist, and ValueError if
    ``path2`` is inside it.

    Requests: one ``_info`` for ``path2``, unless it ends with "/". One ``_ls``
    per folder, one ``_cp_file`` and one ``_rm_file`` per file, one
    ``_makedirs`` per destination folder with files or empty, and one
    ``rm_folder`` per folder. A storage refusing to overwrite costs an
    ``_rm_file`` more per file it refused. At most ``batch_size`` listings and
    ``batch_size`` moves run at a time.

    The profile's ``parents_required=False`` saves the ``_makedirs`` of each
    folder with files, as their copies create it.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path1 : str
        The folder or file to move. Patterns are not supported yet.
    path2 : str
        Where to move it, as above.
    profile : StorageProfile, optional
        What the storage does. The filesystem's ``storage_profile`` if None, if
        it has one.
    rm_folder : callable, optional
        An async function deleting an empty folder, as with `tree_rm`.
    batch_size : int, optional
        The most listings, and the most moves, at a time. fsspec's default if
        None.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each file and folder moved, or
        an error. Its state is an `MvState`.

    Raises
    ------
    ValueError
        If ``path1`` is a glob pattern.

    See Also
    --------
    tree_walk : The walk under it.
    tree_rm : The deletion it does as it goes.

    Examples
    --------
    >>> async with tree_mv(fs, "data/2024", "archive/") as op:
    ...     async for moved in op:
    ...         if isinstance(moved, Exception):
    ...             print("failed:", moved)
    >>> op.state.files_moved
    1234
    """
    if has_magic(path1):
        raise ValueError(f"Patterns are not supported yet: {path1!r}")
    resolved = _resolve(fs, profile)

    async def run(
        op: Operation[tuple[str, str], MvState],
    ) -> AsyncGenerator[tuple[str, str] | Exception]:
        move = _Move(fs, resolved, rm_folder, batch_size, op)
        try:
            async with aclosing(move.steps(path1, path2)) as steps:
                async for outcomes in steps:
                    op.state = move.state()
                    for outcome in outcomes:
                        yield outcome
        finally:
            # Closing the steps waits for the moves sent, and counts them.
            op.state = move.state()

    return Operation(run, MvState())


_Kind = Literal["mkdir", "move", "rmdir"]


class _Move:
    """The bookkeeping of one `tree_mv` operation, kept apart from the function."""

    def __init__(
        self,
        fs: AsyncFileSystem,
        profile: StorageProfile,
        rm_folder: Callable[[str], Awaitable[Any]] | None,
        batch_size: int | None,
        op: Operation[tuple[str, str], MvState],
    ) -> None:
        self.fs = fs
        self.parents_required = profile.parents_required is not False
        self.rm_folder = rm_folder or getattr(fs, "_rmdir", None) or fs._rm_file
        self.batch_size = batch_size or fs.batch_size or _get_batch_size(nofiles=True)
        self.op = op
        self.root = ""
        self.base = ""
        self.errors = 0
        self.files_found = 0
        self.files_moved = 0
        self.folders_moved = 0
        self.current: str | None = None
        # As in tree_rm: the paths in each source folder not moved yet. A folder
        # whose count reaches zero is deleted.
        self.remaining: dict[str, int] = {}
        self.parents: dict[str, str] = {}
        # The moves waiting for their destination folder, by that folder.
        self.after_mkdir: dict[str, list[tuple[_Kind, str, str]]] = {}
        # Requests to start, and those running: the kind, source and destination.
        self.waiting: deque[tuple[_Kind, str, str]] = deque()
        self.running: dict[asyncio.Task[Any], tuple[_Kind, str, str]] = {}
        self.next: asyncio.Task[_Listing] | None = None

    def target(self, path: str) -> str:
        """Where a path under the source goes."""
        return self.base + path[len(self.root) :]

    async def steps(
        self, path1: str, path2: str
    ) -> AsyncGenerator[list[tuple[str, str] | Exception]]:
        """Move, yielding what each step moved or failed, as tree_rm's steps do."""
        if not await self.place(path1, path2):
            return
        listings = _listings(
            self.fs, self.root, batch_size=self.batch_size, stopping=self.op.cancelling
        )
        try:
            yield await self.begin(listings)
            while True:
                self.start()
                pending = [*self.running, *([self.next] if self.next else [])]
                if not pending:
                    break
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if (outcome := self.take(task, listings)) is not None:
                        yield outcome
        except GeneratorExit:
            self.waiting.clear()
            while self.running:
                done, _ = await asyncio.wait(self.running)
                for task in done:
                    self.finish(task)
            raise
        finally:
            # Listings are only reads, so cancelling them loses nothing. The
            # moves are cancelled only when this task itself is.
            tasks = [*self.running, *([self.next] if self.next else [])]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await listings.aclose()

    async def place(self, path1: str, path2: str) -> bool:
        """Set where ``path1`` itself goes, as fsspec's ``copy`` places it.

        Returns False if it is there already.
        """
        self.root = self.fs._strip_protocol(path1)
        destination = self.fs._strip_protocol(path2)
        if self.root == destination:
            return False
        if path2.endswith("/"):
            into = True
        else:
            try:
                into = (await self.fs._info(destination))["type"] == "directory"
            except FileNotFoundError:
                into = False
        self.base = destination
        if into and not path1.endswith("/"):
            name = self.root.rsplit("/", 1)[-1]
            self.base = f"{destination}/{name}" if destination else name
        if self.base == self.root:
            return False
        if not self.root or self.base.startswith(self.root + "/"):
            raise ValueError(f"Cannot move {self.root!r} into itself: {self.base!r}")
        return True

    async def begin(self, listings: AsyncGenerator[_Listing]) -> list[Any]:
        """List the source, or queue its own move if a file: the first step."""
        try:
            first = await anext(listings)
        except NotADirectoryError:
            self.files_found = 1
            move: tuple[_Kind, str, str] = ("move", self.root, self.base)
            parent = self.base.rpartition("/")[0]
            if self.parents_required and parent:
                self.after_mkdir[parent] = [move]
                self.waiting.append(("mkdir", "", parent))
            else:
                self.waiting.append(move)
            return []
        self.next = asyncio.ensure_future(anext(listings))
        return self.add(first)

    def take(
        self, task: asyncio.Task[Any], listings: AsyncGenerator[_Listing]
    ) -> list[Any] | None:
        """Handle a finished task, returning its step, or None if the walk ended."""
        if task is not self.next:
            return self.finish(task)
        self.next = None
        try:
            listing = task.result()
        except StopAsyncIteration:
            return None
        self.next = asyncio.ensure_future(anext(listings))
        return self.add(listing)

    def add(self, listing: _Listing) -> list[Any]:
        """Queue the moves of a listed folder's files, and return its error."""
        folder = listing.folder
        self.current = folder
        if listing.error is not None:
            self.errors += 1
            return [listing.error]
        self.remaining[folder] = len(listing.entries)
        moves: list[tuple[_Kind, str, str]] = []
        for entry in listing.entries:
            name = entry["name"].rstrip("/")
            self.parents[name] = folder
            if entry["type"] != "directory":
                self.files_found += 1
                moves.append(("move", name, self.target(name)))
        target = self.target(folder)
        # A folder holding only folders needs no mkdir: theirs create it.
        if not listing.entries or (moves and self.parents_required):
            self.after_mkdir[target] = moves
            self.waiting.append(("mkdir", folder, target))
        else:
            self.waiting.extend(moves)
        return []

    def start(self) -> None:
        if self.op.cancelling():
            self.waiting.clear()
        while self.waiting and len(self.running) < self.batch_size:
            kind, source, destination = self.waiting.popleft()
            if kind == "mkdir":
                coro = self.fs._makedirs(destination, exist_ok=True)
            elif kind == "move":
                coro = self.move_file(source, destination)
            else:
                coro = self.rm_folder(source)
            self.running[asyncio.ensure_future(coro)] = kind, source, destination

    async def move_file(self, source: str, destination: str) -> None:
        try:
            try:
                await self.fs._cp_file(source, destination)
            except FileExistsError:
                # The storage refuses to overwrite, which fsspec's mv does.
                await self.fs._rm_file(destination)
                await self.fs._cp_file(source, destination)
        except Exception as e:
            e.add_note(f"Moving {source!r} to {destination!r} failed.")
            raise
        try:
            await self.fs._rm_file(source)
        except FileNotFoundError:
            pass  # Deleted by another: the file is moved all the same.
        except Exception as e:
            e.add_note(
                f"Copied {source!r} to {destination!r}, but deleting {source!r}"
                " failed: the file is in both places."
            )
            raise

    def finish(self, task: asyncio.Task[Any]) -> list[Any]:
        """Count a finished request, queue what it allows, and return its outcome."""
        kind, source, destination = self.running.pop(task)
        self.current = source or destination
        try:
            task.result()
        except Exception as e:
            if kind == "rmdir" and isinstance(e, FileNotFoundError):
                pass  # Already gone, as tree_rm counts it.
            else:
                if kind == "mkdir":
                    e.add_note(f"Creating the folder {destination!r} failed.")
                elif kind == "rmdir":
                    e.add_note(f"Deleting the folder {source!r} failed.")
                self.errors += 1
                return [e]
        if kind == "mkdir":
            self.waiting.extend(self.after_mkdir.pop(destination))
            if source and self.remaining[source] == 0:
                self.waiting.append(("rmdir", source, destination))
            return []
        if kind == "move":
            self.files_moved += 1
        else:
            self.folders_moved += 1
        parent = self.parents.get(source)
        if parent is not None:
            self.remaining[parent] -= 1
            if self.remaining[parent] == 0:
                self.waiting.append(("rmdir", parent, self.target(parent)))
        return [(source, destination)]

    def state(self) -> MvState:
        return MvState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_moved=self.files_moved,
            folders_moved=self.folders_moved,
        )
