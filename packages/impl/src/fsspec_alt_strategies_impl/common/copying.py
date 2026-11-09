# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What the jobs of a recursive copy or move do, whatever the family.

`_Copying` is the base of `_tree_cp`'s and `_flat_cp`'s jobs, and of the
moves'. It places each path as fsspec's ``copy`` does, describes the
destination by its filesystem and profile, and counts the files. A family's
copy joins it with the family's jobs, e.g. ``_CopyOrMove(_Copying,
_TreeJobs)``: the family finds what to copy, `_Copying` where it goes. The
two define no method in common, and their constructors take keywords and pass
the rest on, so either order works.

This module depends on `filesystem`, `jobs` and `paths`.
"""

from __future__ import annotations

from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec.utils import common_prefix

from fsspec_alt_strategies.states import CpState

from .filesystem import _profile_of
from .jobs import _Jobs, _Roots
from .paths import _trim

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import TypeAlias

    _Transfer: TypeAlias = Callable[[str, str], Awaitable[None]]
    """How one file goes from a source path to a destination path.

    It is the source's ``_cp_file`` or ``_mv_file`` within one storage, or a
    router's copy or move of a file between two.
    """


class _Copying(_Jobs):
    """The jobs of a copy or a move: where each path goes, onto ``destination``.

    A path under ``prefix`` goes to the same place under ``base``. For a list,
    a path under a root of ``places`` goes there instead. `prepare` sets them,
    before `target` is used.

    The destination is described by ``destination``, the filesystem copied
    to, and its profile, unknown if it has none. Within one storage, it is the
    source, and ``same_storage`` is True: a path copied onto itself, or under
    itself, is refused. Between two, a path may go to the same path, or under
    it, as that is another storage's.
    """

    verb = "copy"
    doing = "Copying"

    def __init__(
        self,
        *,
        path1: str | Sequence[str],
        path2: str,
        destination: Any,
        same_storage: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.path1 = path1
        self.path2 = path2
        self.same_storage = same_storage

        # The profile is the destination's, as what a copy saves depends on
        # where it writes. Within one storage, it is the source's too.
        self.destination = destination
        self.profile = _profile_of(destination)

        # Unknown, folders are made first. Implicit folders need none.
        self.implicit_folders = bool(self.profile.implicit_folders)
        self.parents_required = (
            self.profile.parents_required is not False and not self.implicit_folders
        )

        self.prefix = ""
        self.base = ""

        # For a list: where each root goes, by the root.
        self.places: dict[str, str] = {}

        # The destinations found missing, which need no asking again, e.g.
        # by a move renaming a root onto one.
        self.missing: set[str] = set()

        self.files_found = 0
        self.files_done = 0

    def state(self) -> Any:
        return CpState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_copied=self.files_done,
        )

    async def prepare(self) -> _Roots | None:
        """Set where the paths go, as fsspec's ``copy`` places them."""
        destination = _trim(self.path2)

        if not isinstance(self.path1, str):
            roots = await self._place(destination)
            sources = [*roots.folders, *roots.files]
        elif has_magic(self.path1):
            roots = await self.glob_roots(self.path1)
            sources = [*roots.folders, *roots.files]
            if not sources:
                return roots  # Only errors, in found, to yield.
            self.prefix = common_prefix(sources)
            if sources == list(roots.files) and len(sources) == 1:
                # A single file goes into the destination.
                self.prefix = self.prefix.rpartition("/")[0]
            self.base = destination
        else:
            root = _trim(self.path1)
            if root == destination and self.same_storage:
                return None
            roots = _Roots([root])
            sources = [root]
            self.prefix = root
            self.base = await self._base(root, destination)

        for source in sources if self.same_storage else ():
            target = self.target(source)
            if target == source and len(sources) == 1:
                return None
            if not source or target == source or target.startswith(source + "/"):
                raise ValueError(
                    f"Cannot {self.verb} {source!r} into itself: {target!r}"
                )

        return roots

    async def _place(self, destination: str) -> _Roots:
        """Place each root of a list into ``destination``, under its name.

        Two roots of the same name would go to the same place. As GNU's cp
        does, the first, in path order, goes, and each other is an error. That
        error is the root's own: what is in it is not copied either.
        """
        roots = await self.glob_roots(self.path1)
        folders = set(roots.folders)
        placed: dict[str, str] = {}

        for source in sorted([*roots.folders, *roots.files]):
            name = source.rpartition("/")[2]
            target = f"{destination}/{name}" if destination else name
            if (other := placed.get(target)) is not None:
                error = FileExistsError(target)
                error.add_note(
                    f"{self.doing} {source!r} to {target!r} skipped: {other!r} goes"
                    " there too."
                )
                self.found.append(error)
                continue

            placed[target] = source
            self.places[source] = target

        return _Roots(
            [s for s in self.places if s in folders],
            [s for s in self.places if s not in folders],
        )

    async def _base(self, root: str, destination: str) -> str:
        """Where ``root``, a single path, goes: into ``destination``, or onto it."""
        if self.path2.endswith("/"):
            into = True
        else:
            try:
                entry = await self.destination._info(destination)
                into = entry["type"] == "directory"
            except FileNotFoundError:
                self.missing.add(destination)
                into = False

        if not into or self.path1.endswith("/"):
            return destination

        name = root.rsplit("/", 1)[-1]
        return f"{destination}/{name}" if destination else name

    def target(self, path: str) -> str:
        """Where a path goes."""
        if self.places:
            # The root it is under, found from the path up.
            root = path
            while root not in self.places and "/" in root:
                root = root.rpartition("/")[0]
            return self.places[root] + path[len(root) :]

        if self.prefix:
            return self.base + path[len(self.prefix) :]
        return f"{self.base}/{path}" if self.base else path

    async def copy_file(
        self, cp_file: _Transfer, source: str, destination: str
    ) -> None:
        """Copy a file by ``cp_file``, replacing a file there the destination keeps.

        A destination refusing to overwrite, as fsspec's copy does not, costs
        an ``_info``, an ``_rm_file`` and a copy more. A folder there is no file
        to replace: ``_rm_file`` takes files only. The job holds its permit
        already. An error raised notes its paths, as it is yielded among paths.
        """
        try:
            try:
                await cp_file(source, destination)
            except FileExistsError:
                entry = await self.destination._info(destination)
                if entry["type"] == "directory":
                    raise

                await self.destination._rm_file(destination)
                await cp_file(source, destination)
        except Exception as e:
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise
