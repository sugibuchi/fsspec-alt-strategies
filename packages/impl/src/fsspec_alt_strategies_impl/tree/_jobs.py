# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The engine under the tree strategies that change a tree, such as _tree_rm.

They walk the tree as _tree_walk does, and start their jobs, e.g. deleting a
file, as each directory's listing arrives. `_TreeJobs` feeds `common`'s
`_Jobs` with the listings of a `_TreeWalk`. A subclass says what a listing
queues, how a job runs, and what its outcome is.

A glob pattern is expanded first, as fsspec's ``_expand_path`` expands it:
its matching directories are then walked at once.

This module depends on `_visit`, `_walker` and `glob`, and `jobs` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ..common.jobs import _Jobs
from ..common.paths import _trim
from ._visit import _Pruned
from ._walker import _TreeWalk
from .glob import _tree_glob

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from ..common.handler import StoragePath
    from ..common.jobs import _Roots
    from ..common.operation import Operation
    from ._walker import _Listing, _Order
    from .handler import TreeFileOperationHandler


class _TreeJobs(_Jobs):
    """The jobs of one operation, run while the tree of ``handler`` is walked.

    ``handler`` is the handler walked, by its ``ils`` and ``info``, each
    request holding a permit of its ``operation_limit``. ``ils_kwargs`` go to
    each ``ils``, a pattern's glob's too. With ``maxdepth``, the
    walk lists no directory deeper than it, as fsspec's ``_expand_path`` finds
    paths. Each subdirectory the walk leaves unlisted goes to `at_limit`, as
    fsspec finds it, but not what is in it.
    """

    handler: TreeFileOperationHandler
    """The handler walked: a tree's, which the tree strategies take."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # The paths in each directory listed that are not done yet, for strategies
        # finishing a directory after all in it, as a deletion does. A directory with
        # a failure in it never reaches zero, and nor do the directories above it.
        self.remaining: dict[str, int] = {}
        self.parents: dict[str, str] = {}

        # The entries of each path counted and not done: two files listed under
        # one name, e.g. on Google Drive, are two, each done on its own.
        self.entries: dict[str, int] = {}

        # The directories done before their parent's listing came, as a
        # recursive listing gives a directory after all below it.
        self.done_early: set[str] = set()

        # The roots walked, whose failure is the operation's.
        self.roots_walked: set[str] = set()

    @property
    def order(self) -> _Order:
        """The order the jobs take listings in: any, as each comes, which lets a subtree be listed whole."""
        return "any"

    @abstractmethod
    def add(self, listing: _Listing) -> list[Any] | None:
        """Queue the jobs for a directory's listing, listed without error.

        It returns what is done at once, if anything, e.g. a directory found gone.
        """

    def glob(self, pattern: StoragePath | Sequence[StoragePath]) -> Operation[Any, Any]:
        return _tree_glob(
            self.handler, pattern, maxdepth=self.maxdepth, ils_kwargs=self.ils_kwargs
        )

    def unlisted(self, directory: str) -> bool:
        """Whether a root directory is at the depth limit already: a match, at 1."""
        return self.maxdepth == 1 and directory in self.matched

    def source(self, roots: _Roots) -> AsyncGenerator[_Listing] | None:
        """The listings of the root directories, if any, walked at once."""
        # Their entries, e.g. as a glob found them, so that each is listed by it.
        directories = [d for d in roots.directories if not self.unlisted(d["name"])]
        if not directories:
            return None
        self.roots_walked = {_trim(d["name"]) for d in directories}
        return _TreeWalk(
            self.handler,
            directories,
            _Pruned(maxdepth=self.maxdepth, matched=self.matched),
            order=self.order,
            ils_kwargs=self.ils_kwargs,
            stopping=self.op.cancelling,
        ).listings()

    async def begin(
        self, roots: _Roots, source: AsyncGenerator[_Listing] | None
    ) -> list[Any]:
        """Queue the files' jobs, and list the directories: the first step.

        A single root of unknown kind is among the directories: if it is a file,
        the walk says so, and its jobs are queued instead.
        """
        for directory in roots.directories:
            if self.unlisted(directory["name"]):
                self.at_limit(directory["name"])
        return await super().begin(roots, source)

    def fed(self, item: _Listing) -> list[Any]:
        self.current = item.directory
        if item.error is not None:
            if item.directory in self.roots_walked:
                if isinstance(item.error, NotADirectoryError):
                    # A file: its listing's entry, so that nothing is asked.
                    self.file_root(
                        getattr(item.error, "entry", None) or {"name": item.directory}
                    )
                    return []
                raise item.error  # The root's failure ends the operation.
            self.errors += 1
            return [item.error]

        if self.maxdepth is not None and item.depth >= self.maxdepth:
            # The walk lists none of these subdirectories: fsspec finds them,
            # but nothing in them.
            for entry in item.entries:
                if entry["type"] == "directory":
                    self.at_limit(entry["name"].rstrip("/"))
        return self.add(item) or []

    # A hook, not abstract: only a deletion acts on a directory it does not list.
    def at_limit(self, directory: str) -> None:
        """Queue the jobs for a directory at the depth limit, found but not listed."""

    def track(self, listing: _Listing) -> None:
        """Count the paths in a directory's listing, for `child_done`.

        The walk lists each subdirectory once, though a storage may list one
        twice. Each file's entry counts, two of one name, e.g. on Google Drive,
        as two. A subdirectory done before this listing came counts as done.
        """
        directory = listing.directory
        self.remaining[directory] = 0
        for entry in listing.entries:
            name = entry["name"].rstrip("/")
            if name in self.done_early:
                self.done_early.discard(name)
                continue
            self.remaining[directory] += 1
            self.parents[name] = directory
            self.entries[name] = self.entries.get(name, 0) + 1

    def child_done(self, path: str) -> str | None:
        """Count a path done, and return its directory if all in it is now done.

        Each entry of a path counts once: a path done more often than listed
        counts no more.
        """
        parent = self.parents.get(path)
        if parent is None:
            # Its parent's listing has not come yet, as a recursive listing
            # gives a directory after all below it.
            self.done_early.add(path)
            return None
        self.entries[path] -= 1
        if not self.entries[path]:
            del self.entries[path], self.parents[path]
        self.remaining[parent] -= 1
        return parent if self.finished(parent) else None

    def finished(self, directory: str) -> bool:
        """Whether all in a directory is done: listed, and each path in it done."""
        return self.remaining.get(directory) == 0
