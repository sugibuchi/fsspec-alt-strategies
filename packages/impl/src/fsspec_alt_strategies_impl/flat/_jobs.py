# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The engine under the flat strategies that change a storage, such as _flat_rm.

They list the keys under each root, and start their jobs, e.g. a batch
deletion, as the keys arrive: `_FlatJobs` feeds `common`'s `_Jobs` with the
keys. A subclass says what a key queues, how a job runs, and what its outcome
is. The roots are listed one after another.

A glob pattern is expanded first, with `_flat_glob`, as fsspec's
``_expand_path`` expands it: the keys under its matching directories are then
listed.

This module depends on `_keys`, `find`, `glob`, `handler`, and `handler` and `jobs` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ..common.jobs import _Jobs
from ._keys import _close, _listing
from .find import _within
from .glob import _flat_glob

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from ..common.jobs import _Roots
    from ..common.operation import Operation
    from .handler import FlatFileOperationHandler


class _FlatJobs(_Jobs):
    """The jobs of one operation, run while the keys under its roots are listed.

    ``handler`` is the handler listed, by its ``ils_recursive`` and ``info``.
    ``ils_kwargs`` go to each ``ils_recursive``, a pattern's glob's too. With ``maxdepth``, the keys deeper than it are listed, as a prefix lists
    every key under it, but left out, as fsspec's ``_expand_path`` finds paths.

    A batch deletion runs by `_Jobs`'s `delete`, by the handler's ``rm_files``.
    """

    handler: FlatFileOperationHandler
    """The handler listed: a flat one's, which the flat strategies take."""

    def glob(self, pattern: str | Sequence[str]) -> Operation[Any, Any]:
        return _flat_glob(
            self.handler, pattern, maxdepth=self.maxdepth, ils_kwargs=self.ils_kwargs
        )

    @abstractmethod
    def add(self, entry: dict[str, Any]) -> None:
        """Queue the jobs for a key: a file, or a directory's marker ending with "/"."""

    def file_root(self, entry: dict[str, Any]) -> None:
        self.add(entry)

    def source(self, roots: _Roots) -> AsyncGenerator[dict[str, Any]] | None:
        """The keys under the root directories, if any, listed one after another."""
        if not roots.directories:
            return None
        else:
            return self._keys(roots.directories)

    def fed(self, item: dict[str, Any]) -> list[Any]:
        self.current = item["name"]
        try:
            self.add(item)
        except ValueError as e:
            # A copy's or move's key whose name the destination refuses, e.g. "..".
            return [self.refuse_name(item["name"], e)]  # type: ignore[attr-defined]
        return []

    async def _keys(
        self, roots: Sequence[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any]]:
        """The keys under each root directory, as its recursive ``ils`` lists them, by its entry.

        A directory with nothing under it is asked for with ``info``: a file is
        yielded as a key, and a missing one raises FileNotFoundError, ending the
        operation, as a missing root does.
        """
        for root in roots:
            directory = root["name"]
            depth = self.maxdepth
            if depth is not None and directory in self.matched:
                depth -= 1
                if depth < 1:
                    continue  # At the limit already: nothing in it is taken.

            under = False
            prefix = f"{directory}/" if directory else ""
            entries = _listing(self.handler, prefix, root, self.ils_kwargs)
            try:
                async for entry in entries:
                    under = True
                    if _within(entry["name"].rstrip("/")[len(prefix) :], depth):
                        yield entry
                    if self.op.cancelling():
                        return
            finally:
                await _close(entries)

            if under or not directory:
                continue

            async with self.concurrency_limit:
                entry = await self.handler.info(root)
            if entry["type"] != "directory":
                yield entry
