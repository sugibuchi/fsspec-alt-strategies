# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy that walks the tree and copies while it walks.

`tree_cp` is an alternative to fsspec's recursive ``copy``. fsspec's lists the
whole tree first, then copies each path. `tree_cp` copies each folder's files
as soon as the folder is listed.

This module depends on `control`, `profile`, `walk` and `_jobs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import has_magic
from typing import TYPE_CHECKING, Any

from ..control import Operation, State
from ..profile import _resolve
from ._jobs import _Job, _Jobs

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from fsspec.asyn import AsyncFileSystem

    from ..profile import StorageProfile
    from .walk import _Listing

__all__ = ["CpState", "tree_cp"]


@dataclass(frozen=True, kw_only=True)
class CpState(State):
    """The progress of a `tree_cp`.

    Attributes
    ----------
    files_found : int
        The files found so far. It grows as the walk goes on.
    files_copied : int
        The files copied so far.
    """

    files_found: int = 0
    files_copied: int = 0


def tree_cp(
    fs: AsyncFileSystem,
    path1: str,
    path2: str,
    *,
    profile: StorageProfile | None = None,
    batch_size: int | None = None,
) -> Operation[tuple[str, str], CpState]:
    """Copy a folder and everything in it, copying while the tree is walked.

    It places the paths as fsspec's ``copy`` with ``recursive=True`` does. If
    ``path2`` is an existing folder, or ends with "/", and ``path1`` does not
    end with "/", ``path1`` goes into it: ``path1/x`` becomes
    ``path2/<name of path1>/x``. Otherwise ``path1`` becomes ``path2``:
    ``path1/x`` becomes ``path2/x``. A file is copied the same way. Existing
    files are overwritten, as fsspec's ``copy`` does.

    The tree is walked as `tree_walk` walks it. Each folder's files are copied
    as soon as the folder is listed, each by a ``_cp_file``. A folder is
    created at the destination before files are copied into it, and for each
    empty folder.

    Unlike fsspec's ``copy``, it keeps going past a failure, and reports each.
    A copy cancelled or failed leaves what was copied so far: it deletes
    nothing.

    It wins over fsspec's ``copy`` on storages that list one folder per
    request. The copies then start after the first listing, not after the
    whole tree is listed.

    Once cancelled, it starts no new listing or copy, and yields the outcomes
    of the copies already sent. Closed, it waits for those copies without
    yielding them. Iterating it raises FileNotFoundError if ``path1`` does not
    exist, and ValueError if ``path2`` is inside it.

    Requests: one ``_info`` for ``path2``, unless it ends with "/". One ``_ls``
    per folder, one ``_cp_file`` per file, and one ``_makedirs`` per
    destination folder with files or empty. A storage refusing to overwrite
    costs an ``_rm_file`` and a ``_cp_file`` more per file it refused. At most
    ``batch_size`` listings and ``batch_size`` copies run at a time.

    The profile's ``parents_required=False`` saves the ``_makedirs`` of each
    folder with files, as their copies create it.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path1 : str
        The folder or file to copy. Patterns are not supported yet.
    path2 : str
        Where to copy it, as above.
    profile : StorageProfile, optional
        What the storage does. The filesystem's ``storage_profile`` if None, if
        it has one.
    batch_size : int, optional
        The most listings, and the most copies, at a time. fsspec's default if
        None.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each file copied, or an error.
        Its state is a `CpState`.

    Raises
    ------
    ValueError
        If ``path1`` is a glob pattern.

    See Also
    --------
    tree_walk : The walk under it.
    tree_mv : The same, deleting the source as it goes.

    Examples
    --------
    >>> async with tree_cp(fs, "data/2024", "backup/") as op:
    ...     async for copied in op:
    ...         if isinstance(copied, Exception):
    ...             print("failed:", copied)
    >>> op.state.files_copied
    1234
    """
    _check_source(path1)
    resolved = _resolve(fs, profile)
    return Operation(
        lambda op: _Copy(fs, path1, path2, resolved, batch_size, op).outcomes(),
        CpState(),
    )


def _check_source(path1: str) -> None:
    if has_magic(path1):
        raise ValueError(f"Patterns are not supported yet: {path1!r}")


class _Copy(_Jobs):
    """The jobs of one `tree_cp`: a folder made where needed, and a copy per file.

    A file's copy waits for its destination folder, when the storage needs it.
    """

    verb = "copy"
    doing = "Copying"

    def __init__(
        self,
        fs: AsyncFileSystem,
        path1: str,
        path2: str,
        profile: StorageProfile,
        batch_size: int | None,
        op: Operation[Any, Any],
    ) -> None:
        super().__init__(fs, batch_size, op)
        self.path1 = path1
        self.path2 = path2
        self.parents_required = profile.parents_required is not False
        self.root = ""
        self.base = ""
        self.files_found = 0
        self.files_done = 0
        # The copies waiting for their destination folder, by that folder.
        self.after_mkdir: dict[str, list[_Job]] = {}

    def state(self) -> Any:
        return CpState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_copied=self.files_done,
        )

    async def prepare(self) -> str | None:
        """Set where ``path1`` itself goes, as fsspec's ``copy`` places it."""
        self.root = self.fs._strip_protocol(self.path1)
        destination = self.fs._strip_protocol(self.path2)
        if self.root == destination:
            return None
        if self.path2.endswith("/"):
            into = True
        else:
            try:
                into = (await self.fs._info(destination))["type"] == "directory"
            except FileNotFoundError:
                into = False
        self.base = destination
        if into and not self.path1.endswith("/"):
            name = self.root.rsplit("/", 1)[-1]
            self.base = f"{destination}/{name}" if destination else name
        if self.base == self.root:
            return None
        if not self.root or self.base.startswith(self.root + "/"):
            raise ValueError(
                f"Cannot {self.verb} {self.root!r} into itself: {self.base!r}"
            )
        return self.root

    def target(self, path: str) -> str:
        """Where a path under the source goes."""
        return self.base + path[len(self.root) :]

    def file_root(self, path: str) -> None:
        self.files_found = 1
        copy = _Job("copy", path, self.base)
        parent = self.base.rpartition("/")[0]
        if self.parents_required and parent:
            self.after_mkdir[parent] = [copy]
            self.waiting.append(_Job("mkdir", "", parent))
        else:
            self.waiting.append(copy)

    def add(self, listing: _Listing) -> None:
        copies = []
        for entry in listing.entries:
            if entry["type"] != "directory":
                name = entry["name"].rstrip("/")
                self.files_found += 1
                copies.append(_Job("copy", name, self.target(name)))
        # A folder holding only folders needs no mkdir: theirs create it.
        if not listing.entries or (copies and self.parents_required):
            target = self.target(listing.folder)
            self.after_mkdir[target] = copies
            self.waiting.append(_Job("mkdir", listing.folder, target))
        else:
            self.waiting.extend(copies)

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "mkdir":
            return self.fs._makedirs(job.destination, exist_ok=True)
        return self.copy_file(job.source, job.destination)

    async def copy_file(self, source: str, destination: str) -> None:
        try:
            try:
                await self.fs._cp_file(source, destination)
            except FileExistsError:
                # The storage refuses to overwrite, which fsspec's copy does.
                await self.fs._rm_file(destination)
                await self.fs._cp_file(source, destination)
        except Exception as e:
            # Yielded among paths, the error must tell its paths itself.
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            self.waiting.extend(self.after_mkdir.pop(job.destination))
            return []
        self.files_done += 1
        return [(job.source, job.destination)]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        # A file's error is noted where it happened, in copy_file.
        if job.kind == "mkdir":
            error.add_note(f"Creating the folder {job.destination!r} failed.")
        return [error]
