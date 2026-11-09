# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive move of the keys of a prefix, each moved as soon as it is listed.

`_flat_mv` is an alternative to fsspec's recursive ``mv`` for flat storages.
fsspec's copies every key, then deletes them all, each stage listing the
prefix first. `_flat_mv` moves each key as soon as the listing gives it: by
one ``_mv_file``, or by a copy, deleting the copied keys in batches while the
listing goes on.

This module depends on `_jobs`, `cp`, and `filesystem`, `jobs`, `operation`
and `paths` of `common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import MvState

from ..common.filesystem import _profile_of
from ..common.jobs import _Job
from ..common.operation import Operation
from ..common.paths import _check_maxdepth
from ._jobs import _check_batch_size, _keys_of
from .cp import _Copy, _CopyOrMove

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from ..common.jobs import _Transfer

__all__: list[str] = []


def _flat_mv(
    src_fs: Any,
    dst_fs: Any,
    path1: str | Sequence[str],
    path2: str,
    *,
    cp_file: _Transfer | None = None,
    same_storage: bool | None = None,
    maxdepth: int | None = None,
) -> Operation[tuple[str, str], MvState]:
    """Move the keys under a prefix, each as soon as it is listed.

    It places the paths as fsspec's ``mv`` with ``recursive=True`` does, as
    `_flat_cp` places them. Existing files are overwritten.

    It moves each key in one of two ways, by the destination's profile:

    - by the source's ``_mv_file``: each key is moved by one ``_mv_file`` as
      soon as the listing gives it, and yielded once moved. It wins on
      storages renaming a key in one request, which copies no data.
    - by copies: each key is copied by one ``cp_file`` as soon as the listing
      gives it. The keys copied are deleted in batches of the source's
      ``rm_batch_size``, each by one of its ``_rm_files``, and the last,
      smaller batch once all is copied. A key is moved, and yielded, once its
      batch is deleted. It wins on object stores, which rename nothing, but
      delete many keys in one request.

    It moves by ``_mv_file`` only within one storage whose profile's
    ``renames_files`` is True. Unknown, it copies and deletes in batches,
    which is right on any storage: about one request per key either way, but
    a rename copies no data.

    The destination is described by ``dst_fs`` and its profile, as with
    `_flat_cp`. Between two storages, ``same_storage`` is False, and
    ``cp_file`` copies to the other storage. ``_rm_files`` always deletes the
    source's.

    A folder's marker, a key ending with "/", is moved too, but not yielded. In
    batches, it is deleted without being copied, as `_flat_cp` copies none: its
    folder is made at the destination instead, as `_flat_cp` makes it.

    With ``maxdepth``, it moves the keys `_flat_cp` copies with it, and no
    marker. fsspec's ``mv`` copies as its ``copy`` does with ``maxdepth``, but
    then deletes the whole source prefix, deeper keys too. This keeps them.

    Unlike fsspec's ``mv``, it keeps going past a failure, and reports each. A
    key that failed to move, or to copy, stays where it was. If a batch fails
    to be deleted, its keys are then in both places, as its error notes.

    It wins over fsspec's ``mv`` on large prefixes of flat storages: the moves
    start after the first page of the listing, and the prefix is listed once,
    not twice.

    Once cancelled, it takes nothing more from the listing, and starts no new
    move or copy. In batches, it still deletes the keys already copied, so
    that none is in both places. It yields the outcomes of what was sent.
    Closed, it waits for the requests in flight without yielding them: keys
    copied but not yet deleted are then in both places. Iterating it raises
    FileNotFoundError if ``path1`` does not exist, or matches nothing, and
    ValueError if ``path2`` is inside it.

    Requests: as `_flat_cp`, with one ``_mv_file`` per key instead of its
    ``cp_file``, or with one ``_rm_files`` more per batch.

    Parameters
    ----------
    src_fs : FlatFileSystem
        The filesystem moved from.
    dst_fs : AsyncFileSystem
        The filesystem moved to: ``src_fs`` within one storage.
    path1 : str or sequence of str
        The folder or file to move, a glob pattern, or a list of them, as
        the source's ``_iexpand`` takes paths.
    path2 : str
        Where to move it, as with `_flat_cp`.
    cp_file : callable, optional
        Copies a file to the destination, when moving by copies, called as
        ``cp_file(source, destination)``: the source's ``_cp_file`` by
        default, which copies within one storage.
    same_storage : bool, optional
        Whether the destination is the source's storage: whether
        ``dst_fs`` is ``src_fs`` by default.
    maxdepth : int, optional
        The deepest level to move: 1 moves only the keys right in ``path1``.
        No limit by default.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each key moved, or an error. Its
        state is an `MvState`.

    Raises
    ------
    ValueError
        If the source's ``rm_batch_size``, or ``maxdepth``, is less than 1.

    See Also
    --------
    _flat_cp : The same, keeping the sources.
    _tree_mv : The same, for storages listing one folder per request.
    """
    _check_maxdepth(maxdepth)
    if same_storage is None:
        same_storage = src_fs is dst_fs
    options = {
        "fs": src_fs,
        "dst_fs": dst_fs,
        "path1": path1,
        "path2": path2,
        "same_storage": same_storage,
        "maxdepth": maxdepth,
    }

    # A rename wins only where the storage is known to rename.
    if same_storage and _profile_of(dst_fs).renames_files is True:
        return Operation(
            lambda op: _Rename(op=op, **options).outcomes(),
            MvState(),
        )

    _check_batch_size(src_fs.rm_batch_size)
    cp_file = src_fs._cp_file if cp_file is None else cp_file
    return Operation(
        lambda op: _Move(cp_file=cp_file, op=op, **options).outcomes(),
        MvState(),
    )


class _Rename(_CopyOrMove):
    """The jobs of one `_flat_mv` by ``_mv_file``: one move per key.

    The source's ``_mv_file`` moves each key where a copy would put it. A
    folder's marker is moved too, but not yielded.
    """

    verb = "move"
    doing = "Moving"

    async def transfer_file(self, source: str, destination: str) -> None:
        try:
            await self.fs._mv_file(source, destination)
        except Exception as e:
            # Yielded among paths, the error must tell its paths itself.
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise

    def state(self) -> MvState:
        return MvState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_moved=self.files_done,
        )

    def add(self, entry: dict[str, Any]) -> None:
        name = entry["name"]
        if name.endswith("/"):
            # With maxdepth, folders stay, as fsspec's copy leaves them out.
            if self.maxdepth is None:
                self.waiting.append(_Job("marker", name, self.target(name)))
            return
        super().add(entry)

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "marker":
            return []
        return super().done(job)


class _Move(_Copy):
    """The jobs of one `_flat_mv` by copies: a copy per key, then a deletion per batch."""

    verb = "move"
    doing = "Moving"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.batch_size: int = self.fs.rm_batch_size

        # The keys to delete. Each key moving, from its listing to its
        # deletion, has where it is copied, None for a marker: listed again,
        # it is skipped.
        self.batch: list[str] = []
        self.copied: dict[str, str | None] = {}

        # The copies queued or running, and whether the listing has ended: the
        # last batch goes once both say all is copied.
        self.copying = 0
        self.listed = False

    def state(self) -> MvState:
        return MvState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_moved=self.files_done,
        )

    def add(self, entry: dict[str, Any]) -> None:
        name = entry["name"]
        if name in self.copied:
            return
        if name.endswith("/"):
            if self.maxdepth is not None:
                return  # Folders stay, as fsspec's copy leaves them out.
            # A folder's marker: deleted, its folder made as _flat_cp makes it.
            self.copied[name] = None
            self.delete_later(name)
            super().add(entry)
            return

        self.copied[name] = self.target(name)
        self.copying += 1
        super().add(entry)

    def exhausted(self) -> None:
        self.listed = True
        self.send_if_ready()

    def stopping(self) -> None:
        # The copies not started are dropped, but the keys copied are deleted,
        # so that none is in both places.
        deletions = [j for j in self.waiting if j.kind == "delete"]
        self.waiting.clear()
        self.waiting.extend(deletions)
        self.copying = sum(j.kind == "copy" for j in self.running.values())
        self.send_if_ready()

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "delete":
            return self.fs._rm_files(list(job.paths))
        return super().run(job)

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            return super().done(job)
        if job.kind == "copy":
            self.copying -= 1
            self.delete_later(job.source)
            return []

        moved = []
        for path in job.paths:
            if (destination := self.copied.pop(path)) is not None:
                self.files_done += 1
                moved.append((path, destination))
        return moved

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if job.kind == "mkdir":
            return super().failed(job, error)
        if job.kind == "copy":
            self.copying -= 1
            self.copied.pop(job.source)
            self.send_if_ready()
            return super().failed(job, error)

        for path in job.paths:
            self.copied.pop(path)

        error.add_note(
            f"Deleting {_keys_of(job.paths)} failed, once copied: they are in both"
            " places."
        )
        return [error]

    def delete_later(self, source: str) -> None:
        """Add a key to the batch to delete, and send the batch if it is ready."""
        self.batch.append(source)
        self.send_if_ready()

    def send_if_ready(self) -> None:
        """Send the batch if full, or if nothing more can join it."""
        if not self.batch:
            return

        done = (self.listed or self.op.cancelling()) and self.copying == 0
        if len(self.batch) >= self.batch_size or done:
            paths, self.batch = tuple(self.batch), []
            self.waiting.append(_Job("delete", paths[-1], paths=paths))
