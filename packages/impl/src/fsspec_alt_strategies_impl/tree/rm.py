# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive delete that walks the tree and deletes while it walks.

`_tree_rm` is an alternative to fsspec's recursive ``_rm``. fsspec's lists the
whole tree first, then deletes each path. `_tree_rm` deletes each directory's files
as soon as the directory is listed, and each directory once it is empty.

This module depends on `_jobs`, and `jobs` and `paths` of `common`.
"""

from __future__ import annotations

from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import RmState

from ..common.jobs import _Job, _keys_of, _Roots
from ..common.paths import _check_maxdepth, _trim
from ._jobs import _TreeJobs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from ..common.handler import StoragePath
    from ..common.operation import Operation
    from .handler import TreeFileOperationHandler
    from .walk import _Listing

__all__: list[str] = []


def _tree_rm(
    handler: TreeFileOperationHandler,
    path: StoragePath | Sequence[StoragePath],
    *,
    maxdepth: int | None = None,
    ils_kwargs: dict[str, Any] | None = None,
) -> Operation[str, RmState]:
    """Delete a directory and everything in it, deleting while the tree is walked.

    The tree is walked as `_tree_walk` walks it. Each directory's files are
    deleted as soon as the directory is listed, while other directories are
    still being listed. A directory is deleted once everything in it is, so the
    deepest go first. A directory with anything in it that failed is kept.

    It yields each file and directory once deleted, and the error of each
    directory that failed to list or path that failed to be deleted. A file
    found already gone is deleted, as ``rm_files`` takes it: it is yielded. A
    directory found already gone is not.

    It wins over fsspec's ``_rm`` on storages that list one directory per
    request and have no recursive delete of their own, such as Databricks
    volumes. The deletions then start after the first listing, not after the
    last. It also keeps going past a failure, and reports each.

    It loses on object stores, which delete keys in batches: their own
    ``_rm``, or `_flat_rm`, makes fewer requests there. On a storage deleting a
    directory with everything in it in one request, that request wins: the
    profile's ``deletes_directories=True`` says ``rm_tree`` sends it.

    With ``deletes_directories``, each root is deleted whole by one ``rm_tree``,
    with no walk, as each file is deleted: concurrently, under the same limit
    and cancelling. A root whose deletion fails yields the error, with its
    cause, and is kept. ``path`` missing raises, as without it. A root deleted
    whole is yielded once, and counted as a directory: what was in it is neither
    listed, yielded nor counted.

    A glob pattern deletes each path it matches, and all in it, as fsspec's
    ``_rm`` with ``recursive=True`` does. So does a list of paths and
    patterns. They are matched as `_tree_glob` matches them, in one search,
    then the directories matched are walked at once.

    Files are deleted by the handler's ``rm_files``, in batches of its
    ``rm_batch_size``, as it is once open: by default 1, each by ``rm_file``,
    as many at once as the limit lets them. A storage deleting several files
    by one request declares more, e.g. Azure's Blob Batch. The files listed
    join a batch, which is sent once full, while the walk goes on, and the
    last, smaller one once the walk ends. A directory is deleted once the
    batches holding its files are. A file its batch refused yields its error
    and keeps its directory. A batch failing whole yields one error, noting
    its files.

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
    or matches nothing.

    Requests: one ``ils_recursive`` per root, which a storage without a
    recursive listing answers with one per directory, or one ``ils`` per
    directory with ``maxdepth``. One ``rm_files`` per batch of
    ``rm_batch_size`` files, by default one ``rm_file`` each, and one ``rm_dir`` per
    directory. Patterns, or a list, add the requests of their `_tree_glob`.
    With ``deletes_directories``, a root costs one ``rm_tree`` instead. Each
    request holds a permit of the handler's ``operation_limit`` while it
    runs.

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
    ``ils``, a pattern's glob's too.

    The operation yields each path deleted, or an error, and its state is an
    `RmState`. The call raises ValueError if ``maxdepth`` is less than 1, and
    iterating it if the handler's ``rm_batch_size`` is, read once it is open.
    """
    _check_maxdepth(maxdepth)
    return _Deletion.operation(
        handler=handler, path=path, maxdepth=maxdepth, ils_kwargs=ils_kwargs
    )


class _Deletion(_TreeJobs):
    """The jobs of one `_tree_rm`: a deletion per batch of files, then per directory."""

    def __init__(
        self, *, path: StoragePath | Sequence[StoragePath], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.path = path
        # Unknown until setup reads the profile.
        self.deletes_directories = False
        self.implicit_directories = False
        self.files_found = 0
        self.files_deleted = 0
        self.directories_deleted = 0

    async def setup(self) -> None:
        await super().setup()
        profile = self.handler.profile
        # A directory deleted whole would take what is deeper than maxdepth.
        self.deletes_directories = (
            self.maxdepth is None and profile.deletes_directories is True
        )
        self.implicit_directories = bool(profile.implicit_directories)

    async def prepare(self) -> _Roots:
        """Find the roots: a glob's matches, or the path, of unknown kind.

        With deletes_directories, each directory root is queued for its rm_tree, as a
        file's deletion, and none is walked: rm_tree takes files too.
        """
        if not isinstance(self.path, str) or has_magic(self.path):
            roots = await self.glob_roots(self.path)
        else:
            # A trailing "/" says a directory, not another name: rm_tree takes the path.
            roots = _Roots([{"name": _trim(self.path)}])

        if not self.deletes_directories:
            return roots
        self.waiting.extend(_Job("tree", d) for d in roots.directories)
        return roots._replace(directories=[])

    def state(self) -> RmState:
        return RmState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_deleted=self.files_deleted,
            directories_deleted=self.directories_deleted,
        )

    def file_root(self, entry: dict[str, Any]) -> None:
        self.files_found += 1
        self.queue_deletion(entry)

    def at_limit(self, directory: str) -> None:
        # Deleted only if empty: one that is not fails, as fsspec's deletion
        # of each path found does. An implicit directory is no path to delete.
        if not self.implicit_directories:
            self.waiting.append(_Job("directory", {"name": directory}))

    def add(self, listing: _Listing) -> None:
        self.track(listing)

        for entry in listing.entries:
            if entry["type"] != "directory":
                self.files_found += 1
                self.queue_deletion({**entry, "name": entry["name"].rstrip("/")})

        # Empty, or all in it deleted before its last part came.
        if listing.last and self.finished(listing.directory):
            self.remove(listing.directory)

    def exhausted(self) -> None:
        # The walk ended: the last, smaller batch goes.
        self.flush_deletions()

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "tree":
            return self.handler.rm_tree(job.source)
        else:
            return self.handler.rmdir(job.source)

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "delete":
            # A file refused keeps its directory, as one rm_file failed would.
            deleted, errors = self.deleted(job)
            self.files_deleted += len(deleted)
            for path in deleted:
                self.after(path)
            return [*deleted, *errors]

        self.directories_deleted += 1
        self.after(job.path)
        return [job.path]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if job.kind == "delete":
            # The batch failed whole: its files, and the directories above, stay.
            error.add_note(f"Deleting {_keys_of(job.paths)} failed.")
            return [error]

        missing = isinstance(error, FileNotFoundError)
        if missing and job.kind == "tree" and job.path == _trim(str(self.path)):
            raise error  # The path missing, as the walk would find it.

        if missing:
            # Already gone: an object store's directory, or deleted by another.
            self.after(job.path)
            return []

        # Yielded among paths, the error must tell its path itself.
        error.add_note(f"Deleting {job.path!r} failed.")
        return [error]

    def after(self, path: str) -> None:
        """Queue the deletion of a path's directory, if it is now empty."""
        if (directory := self.child_done(path)) is not None:
            self.remove(directory)

    def remove(self, directory: str) -> None:
        """Queue the deletion of an empty directory, unless it went with its last file."""
        if self.implicit_directories:
            self.after(directory)  # Gone already, as a FileNotFoundError tells.
        else:
            self.waiting.append(_Job("directory", {"name": directory}))
