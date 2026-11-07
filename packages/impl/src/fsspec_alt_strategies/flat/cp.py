# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy that copies the keys of a prefix as it lists them.

`flat_cp` is an alternative to fsspec's recursive ``copy`` for flat storages.
fsspec's lists every key first, then copies them. `flat_cp` copies each key as
soon as the listing gives it.

This module depends on `_jobs`, and `jobs`, `monitors` and `placement` of
`common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies_common.states import CpState

from ..common.jobs import _Job
from ..common.monitors import Operation
from ..common.placement import _Placing
from ._jobs import _expand, _FlatJobs

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Sequence

    from ..common.jobs import _Roots
    from ..common.operations import _Ils, _Info, _Request

__all__ = ["CpState", "flat_cp"]


def flat_cp(
    path1: str | Sequence[str],
    path2: str,
    *,
    ils: _Ils,
    info: _Info,
    cp_file: _Request,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[tuple[str, str], CpState]:
    """Copy the keys under a prefix, copying as the listing goes on.

    It places the paths as fsspec's ``copy`` with ``recursive=True`` does, as
    `tree_cp` places them: into ``path2`` if it is an existing folder or ends
    with "/", and ``path1`` does not end with "/", otherwise onto it. A glob
    pattern copies each path it matches, relative to the matches' common
    prefix. A list copies each path it names or matches into ``path2``, under
    its name, as ``cp -r`` does. Existing files are overwritten.

    The keys are listed as `flat_find` lists them, and each is copied by one
    ``cp_file`` as soon as the listing gives it. A folder's marker, a key
    ending with "/", is not copied, as fsspec's ``copy`` copies no folder:
    flat storages need none for the files in it.

    Unlike fsspec's ``copy``, it keeps going past a failure, and reports each.
    A copy cancelled or failed leaves what was copied so far: it deletes
    nothing.

    It wins over fsspec's ``copy`` on large prefixes of flat storages: the
    copies start after the first page of the listing, not after the last. It
    loses on storages with a hierarchical namespace, whose folders a listing
    by prefix does not see: `tree_cp` suits those.

    Once cancelled, it takes nothing more from the listing, and starts no new
    copy, but yields the outcomes of those sent. Closed, it waits for those
    without yielding them. Iterating it raises FileNotFoundError if ``path1``
    does not exist, or matches nothing, and ValueError if ``path2`` is inside
    it.

    Requests: one ``info`` for ``path2``, unless it ends with "/" or ``path1``
    is a pattern or a list. The pages of one ``ils`` per folder copied, one
    after another, and one ``cp_file`` per key, as ``concurrency_limit``
    allows. Patterns, or a list, add the requests of their `flat_glob`.

    Parameters
    ----------
    path1 : str or sequence of str
        The folder or file to copy, a glob pattern, or a list of them, as
        ``ils`` takes paths.
    path2 : str
        Where to copy it, as above.
    ils : callable
        Lists every key starting with a prefix, called as ``ils(prefix)``: an
        async iterator of entries as fsspec's ``_ls`` gives them, in any order.
    info : callable
        Describes a path, called as ``info(path)``: e.g. the filesystem's
        ``_info``.
    cp_file : callable
        Copies a file, called as ``cp_file(source, destination)``: e.g. the
        filesystem's ``_cp_file``.
    concurrency_limit : asyncio.Semaphore
        Held by each ``cp_file`` and ``info`` while it runs. The listing holds
        none. Share one to limit several operations, or files, together.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each key copied, or an error.
        Its state is a `CpState`.

    See Also
    --------
    tree_cp : The same, for storages listing one folder per request.
    flat_mv : The same, deleting the sources as it goes.

    Examples
    --------
    >>> cp = flat_cp(
    ...     "data/2024",
    ...     "backup/",
    ...     ils=my_ils,
    ...     info=fs._info,
    ...     cp_file=fs._cp_file,
    ...     concurrency_limit=asyncio.Semaphore(32),
    ... )
    >>> async for copied in cp:
    ...     if isinstance(copied, Exception):
    ...         print("failed:", copied)
    """
    return Operation(
        lambda op: _Copy(
            path1,
            path2,
            ils=ils,
            info=info,
            cp_file=cp_file,
            concurrency_limit=concurrency_limit,
            op=op,
        ).outcomes(),
        CpState(),
    )


class _Copy(_Placing, _FlatJobs):
    """The jobs of one `flat_cp`: a copy per key."""

    def __init__(
        self,
        path1: str | Sequence[str],
        path2: str,
        *,
        ils: _Ils,
        info: _Info,
        cp_file: _Request,
        concurrency_limit: asyncio.Semaphore,
        op: Operation[Any, Any],
    ) -> None:
        super().__init__(ils, info, concurrency_limit, op)
        self.path1 = path1
        self.path2 = path2
        self.cp_file = cp_file
        self._placing()
        self.files_found = 0
        self.files_done = 0

    def state(self) -> Any:
        return CpState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_copied=self.files_done,
        )

    async def expand(self, pattern: str | Sequence[str]) -> _Roots:
        return await _expand(self.ils, self.info, pattern, self.concurrency_limit)

    def add(self, entry: dict[str, Any]) -> None:
        name = entry["name"]
        if name.endswith("/"):
            return  # A folder's marker: no folder is copied.
        self.files_found += 1
        self.waiting.append(_Job("copy", name, self.target(name)))

    def run(self, job: _Job) -> Awaitable[Any]:
        return self.copy_file(job.source, job.destination)

    async def copy_file(self, source: str, destination: str) -> None:
        try:
            await self.cp_file(source, destination)
        except Exception as e:
            # Yielded among paths, the error must tell its paths itself.
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise

    def done(self, job: _Job) -> list[Any]:
        self.files_done += 1
        return [(job.source, job.destination)]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        # Noted where it happened, in copy_file.
        return [error]
