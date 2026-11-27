# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Where a copy's or a move's paths go, as fsspec's ``copy`` places them, whatever the family.

`_Placement` places the roots of a copy or move, and each path below them,
onto the destination: by its ``join``, folded on a storage whose names are
not case-sensitive. It keeps what it learns of the destination on the way:
the paths found missing or existing, the directories made, and their
entries, which the primitives take as hints.

`_place_roots` gives a tree copy its roots alone, each a `_Root` with what is
known of its target. The traversal places what is below them.

`_glob_roots` finds the roots a glob pattern matches, as fsspec's
``_expand_path`` finds them, for any operation changing a storage.

This module depends on `handler`, `operation` and `paths`.
"""

from __future__ import annotations

from glob import has_magic
from typing import TYPE_CHECKING, Any, NamedTuple

from fsspec.utils import common_prefix

from fsspec_alt_strategies.profile import UNKNOWN

from .handler import StoragePath
from .operation import _collect_with_errors

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fsspec_alt_strategies.profile import StorageProfile

    from .handler import Entry, FileOperationHandler, StorageHandler
    from .operation import Operation

__all__: list[str] = []


class _Roots(NamedTuple):
    """Where an operation starts: what it walks, and what it handles alone.

    Each root is its entry, so that a copy or move hands it on without asking
    it again. ``directories`` are walked, or listed, at once. A single path of
    unknown kind is among them, its entry its ``"name"`` alone, as its listing
    tells what it is. ``files`` are known files, e.g. glob matches.
    """

    directories: Sequence[dict[str, Any]]
    files: Sequence[dict[str, Any]] = ()

    @property
    def paths(self) -> list[str]:
        """The paths of all the roots, the directories first."""
        return [e["name"] for e in (*self.directories, *self.files)]


def _roots_of(
    matches: list[dict[str, Any]],
    errors: list[Exception],
    pattern: str | Sequence[str],
) -> _Roots:
    """The roots of a glob's matches, as fsspec's ``_expand_path`` finds them.

    A match inside a directory that matches is left out: walking the directory
    finds it. Matching nothing raises FileNotFoundError, as fsspec does.
    """
    if not matches and not errors:
        raise FileNotFoundError(pattern)

    directories: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []

    # Sorted, a directory comes before all in it.
    for entry in sorted(matches, key=lambda e: e["name"]):
        name = entry["name"].rstrip("/")
        parts = name.split("/")
        if any("/".join(parts[:i]) in directories for i in range(1, len(parts))):
            continue
        if entry["type"] == "directory":
            directories[name] = {**entry, "name": name}
        else:
            files.append({**entry, "name": name})

    return _Roots(list(directories.values()), files)


async def _glob_roots(
    handler: StorageHandler,
    glob: Operation[Any, Any],
    pattern: StoragePath | Sequence[StoragePath],
    maxdepth: int | None,
) -> tuple[_Roots, list[Exception], set[str]]:
    """The roots a glob matches, its errors, and the directories a pattern matched.

    With ``maxdepth``, a directory root a pattern matched, not one named
    literally, is walked one level less deep, as fsspec's glob counted that
    level already.
    """
    matches, errors = await _collect_with_errors(glob)
    roots = _roots_of(matches, errors, pattern)
    matched: set[str] = set()
    if maxdepth is not None:
        named = [pattern] if isinstance(pattern, str) else pattern
        literal = {handler.trim(p) for p in named if not has_magic(p)}
        matched = {d["name"] for d in roots.directories if d["name"] not in literal}
    return roots, errors, matched


class _Placement:
    """Where a copy's or a move's paths go, onto ``dst``.

    A path under ``prefix`` goes to the same place under ``base``. For a list,
    a path under a root of ``places`` goes there instead. `prepare` sets them,
    before `target` is used.

    ``handler`` is the source's. ``dst`` is the handler copied to, described
    by its profile, unknown if it has none. Within one storage, it is the
    source, and ``same_storage`` is True: a path copied onto itself, or under
    itself, is refused. Between two, a path may go to the same path, or under
    it, as that is another storage's. The paths are compared folded on a
    storage whose names are not case-sensitive, as its profile says.

    ``path1_entry`` is the entry of ``path1``, a single path, if the caller
    found it already, e.g. to tell a file. A file's is then a root known to be
    a file, not listed to find what it is.

    ``glob`` is the strategy's glob of a pattern, within ``maxdepth``, which
    finds the roots of patterns and lists. With ``batching``, `setup` reads
    the source's ``cp_batch_size``, or ``mv_batch_size`` where ``transfers``
    is "move", into ``transfer_batch_size``.

    The errors of finding the roots go to ``found``, which the operation
    yields first.
    """

    verb = "copy"
    doing = "Copying"
    transfers = "copy"
    """Which of the handler's strategies a batch of files goes to: "copy" for ``cp_files``, "move" for ``mv_files``."""

    def __init__(
        self,
        *,
        handler: FileOperationHandler,
        dst: StorageHandler,
        path1: StoragePath | Sequence[StoragePath],
        path2: StoragePath,
        same_storage: bool,
        glob: Callable[[StoragePath | Sequence[StoragePath]], Operation[Any, Any]],
        maxdepth: int | None = None,
        path1_entry: dict[str, Any] | None = None,
        batching: bool = False,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.handler: FileOperationHandler = handler
        self.maxdepth = maxdepth
        self.ils_kwargs: dict[str, Any] = ils_kwargs or {}
        self.glob = glob

        self.path1 = path1
        self.path2 = path2
        self.same_storage = same_storage
        self.path1_entry = path1_entry
        self.batching = batching
        self.transfer_batch_size = 1

        # The profile is the destination's, as what a copy saves depends on
        # where it writes. Within one storage, it is the source's too. Unknown
        # until setup awaits it.
        self.dst: StorageHandler = dst
        self.profile = UNKNOWN
        self.implicit_directories = False
        self.parents_required = True
        self.case_insensitive = False

        # Set by prepare: the destination's path, verified, where the prefix goes.
        self.prefix = ""
        self.base = StoragePath("")

        # For a list: where each root goes, by the root.
        self.places: dict[str, StoragePath] = {}

        # The directory roots a pattern matched, walked one level less deep.
        self.matched: set[str] = set()

        # The errors found in finding the roots, e.g. by a glob: yielded first.
        self.found: list[Exception] = []

        # The destinations found missing, and those found existing, which need
        # no asking again, e.g. by a move renaming a root onto one.
        self.missing: set[str] = set()
        self.existing: set[str] = set()

        # The destination directories made, tried, or known to exist: their
        # copies need not wait for a mkdir.
        self.made: set[str] = set()

        # The entries of the destination directories found or made, by path:
        # each is the parent hint of what is made inside it.
        self.directories: dict[str, Entry] = {}

    async def setup(self) -> None:
        """Open both handlers, and read the destination's profile, and the source's batch size where batching."""
        await self.handler.open()
        # The destination's profile, loaded once as its handler opens. A plain
        # fsspec filesystem a router copies to has an unknown one.
        await self.dst.open()
        self.profile = self.dst.profile
        # Unknown, directories are made first. Implicit directories need none.
        # Within one storage, a file goes by the storage's own copy or move,
        # whose parents the profile tells apart from a write's.
        self.implicit_directories = bool(self.profile.implicit_directories)
        declared = (
            self.profile.copy_parents_required
            if self.same_storage
            else self.profile.parents_required
        )
        self.parents_required = declared is not False and not self.implicit_directories
        # Only within one storage are a source and its destination compared.
        self.case_insensitive = (
            self.same_storage and self.profile.case_sensitive is False
        )
        if self.batching:
            # Within one storage, the source's handler is the destination's.
            name = "cp_batch_size" if self.transfers == "copy" else "mv_batch_size"
            self.transfer_batch_size = getattr(self.handler, name)
            if self.transfer_batch_size < 1:
                raise ValueError(
                    f"{name} must be at least 1: {self.transfer_batch_size}"
                )

    def folded(self, path: str) -> str:
        """A path as the storage compares it: folded unless its names are case-sensitive."""
        return path.lower() if self.case_insensitive else path

    async def glob_roots(self, pattern: StoragePath | Sequence[StoragePath]) -> _Roots:
        """The roots matching glob patterns, noting those a pattern matched. The glob's errors go to ``found``."""
        roots, errors, matched = await _glob_roots(
            self.handler, self.glob(pattern), pattern, self.maxdepth
        )
        self.found.extend(errors)
        self.matched = matched
        return roots

    async def prepare(self) -> _Roots | None:
        """Set where the paths go, as fsspec's ``copy`` places them: the roots, or None if there is nothing to do."""
        # The caller's destination, as the handler verifies it: ValueError if refused.
        destination = self.dst.verified(self.path2)

        if not isinstance(self.path1, str):
            roots = await self._place(destination)
            sources = roots.paths
        elif has_magic(self.path1):
            roots = await self.glob_roots(self.path1)
            sources = roots.paths
            if not sources:
                return roots  # Only errors, in found, to yield.
            self.prefix = common_prefix(sources)
            if not roots.directories and len(sources) == 1:
                # A single file goes into the destination.
                self.prefix = self.prefix.rpartition("/")[0]
            self.base = destination
        else:
            root = self.handler.trim(self.path1)
            if self.same_storage and self.folded(root) == self.folded(destination):
                # Onto itself: left as it is, unless a move to another case.
                recased = root != destination and self.verb == "move"
                if not recased or self.path1.endswith("/") or self.path2.endswith("/"):
                    return None
                return self.recased(root, destination)
            roots = await self._single(root, destination)
            sources = [root]

        for source in sources if self.same_storage else ():
            target, folded = self.folded(self.target(source)), self.folded(source)
            if target == folded and len(sources) == 1:
                return None
            if not source or target == folded or target.startswith(folded + "/"):
                raise ValueError(
                    f"Cannot {self.verb} {source!r} into itself: {self.target(source)!r}"
                )

        return roots

    def recased(self, root: str, destination: StoragePath) -> _Roots:
        """The root of a single path moved onto itself in another case, to be renamed.

        On a storage whose names are not case-sensitive, only a rename of the
        root whole changes its case: a move doing so places it, as a root
        going to a destination found missing. Here, it raises ValueError.
        """
        raise ValueError(
            f"Cannot {self.verb} {root!r} to {destination!r}, itself in another"
            " case: the storage renames nothing whole"
        )

    async def _place(self, destination: StoragePath) -> _Roots:
        """Place each root of a list into ``destination``, under its name.

        Two roots of the same name would go to the same place. As GNU's cp
        does, the first, in path order, goes, and each other is an error. That
        error is the root's own: what is in it is not copied either.
        """
        roots = await self.glob_roots(self.path1)
        directories = {d["name"]: d for d in roots.directories}
        files = {f["name"]: f for f in roots.files}
        placed: dict[StoragePath, str] = {}

        for source in sorted(roots.paths):
            name = source.rpartition("/")[2]
            target = self.dst.join(destination, name)
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
            [directories[s] for s in self.places if s in directories],
            [files[s] for s in self.places if s in files],
        )

    async def _single(self, root: str, destination: StoragePath) -> _Roots:
        """Place a single path, and return it as a root: a file if its entry says so."""
        self.prefix = root
        self.base = await self._base(root, destination)
        if self.same_storage:
            # The root's parent exists: e.g. a file renamed in its directory needs no mkdir.
            self.made.add(root.rpartition("/")[0])

        entry = self.path1_entry
        if entry is not None and entry["type"] != "directory":
            return _Roots([], [{**entry, "name": root}])
        else:
            return _Roots([{"name": root}])

    async def _base(self, root: str, destination: StoragePath) -> StoragePath:
        """Where ``root``, a single path, goes: into ``destination``, or onto it.

        What its ``_info`` finds is kept: a destination missing or existing,
        and the directory it shows to exist, which needs no mkdir.
        """
        if self.path2.endswith("/"):
            into = True
        else:
            try:
                entry = await self.dst.info(self.dst.entry(destination))
            except FileNotFoundError:
                self.missing.add(destination)
                into = False
            else:
                self.existing.add(destination)
                into = entry["type"] == "directory"
                self.made.add(destination if into else destination.rpartition("/")[0])
                if into:
                    self.directories[destination] = entry

        if not into or (isinstance(self.path1, str) and self.path1.endswith("/")):
            return destination

        name = root.rsplit("/", 1)[-1]
        return self.dst.join(destination, name)

    def target(self, path: str) -> StoragePath:
        """Where a path goes, each name below its root checked by the destination's ``join``.

        A name the destination refuses, e.g. "..", raises ValueError: a path
        listed on the source never leaves the destination. A flat storage's
        directory marker, a key ending with "/", keeps its "/".
        """
        if self.places:
            # The root it is under, found from the path up.
            root = path
            while root not in self.places and "/" in root:
                root = root.rpartition("/")[0]
            target = self.dst.join(self.places[root], path[len(root) :])
        elif self.prefix:
            target = self.dst.join(self.base, path[len(self.prefix) :])
        else:
            target = self.dst.join(self.base, path)

        if path.endswith("/") and not target.endswith("/"):
            # A flat storage's marker is a key: its directory's path and a "/".
            return StoragePath(f"{target}/")
        else:
            return target

    def known_missing(self, target: str) -> bool:
        """Whether nothing is at a destination, as the operation knows: it, or a directory above it, was found missing.

        All below a destination found missing is missing too, unless another
        client puts something there meanwhile: the primitives making it are
        told so, by ``missing=True``, and a storage whose creation never
        conflicts asks nothing first.
        """
        path = target.rstrip("/")
        while path:
            if path in self.missing:
                return True
            path = path.rpartition("/")[0]
        return False

    def needs_mkdir(self, directory: StoragePath, *, always: bool = False) -> bool:
        """Whether a destination directory is made before what goes in it.

        Never the root, nor one made, tried or known to exist. Else only where
        the storage requires parents, unless ``always``, e.g. for an empty
        directory, which nothing else makes.
        """
        if directory == self.dst.root or not directory or directory in self.made:
            return False
        return always or self.parents_required

    async def replace_file(
        self,
        cp_entry: Any,
        entry: dict[str, Any],
        destination: StoragePath,
        refused: FileExistsError,
    ) -> None:
        """Copy a file onto one its destination refused to overwrite: deleted first, unless a directory.

        A directory there raises ``refused`` again: ``rm_file`` takes files only.
        """
        found = await self.dst.info(self.dst.entry(destination))
        if found["type"] == "directory":
            raise refused

        await self.dst.rm_file({**found, "name": destination})
        await cp_entry(entry, destination, parent=self.parent_of(destination))

    def parent_of(self, path: StoragePath) -> Entry | None:
        """The entry of a destination's parent, as found or made, as the primitives take it: None if unknown."""
        try:
            return self.directories.get(self.dst.parent(path))
        except ValueError:
            return None  # A path at the root: no parent of the destination's.


class _Moving(_Placement):
    """Where a move's paths go: a single path moved onto itself in another case is renamed whole."""

    verb = "move"
    doing = "Moving"
    transfers = "move"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.goes_whole = False  # As _place_roots reads it from the profile.

    def recased(self, root: str, destination: StoragePath) -> _Roots:
        """A single path moved onto itself in another case: renamed by one ``mv_tree``.

        The destination is the root itself. Found existing, it would take the
        root into itself: it is taken as missing instead.
        """
        if not self.goes_whole:
            return super().recased(root, destination)
        self.prefix, self.base = root, destination
        self.missing.add(destination)
        self.made.add(self.handler.parent(root))
        # A directory root, as mv_tree takes a file too.
        return _Roots([self.path1_entry or {"name": root}])


class _Root(NamedTuple):
    """A root of a copy or move, placed: where it goes, and what is known of the destination there.

    ``missing`` is whether nothing is at ``target``, or above it, as found,
    and ``existing`` whether something is. ``parent`` is the entry of the
    target's parent, if found, and ``parent_exists`` whether it needs no
    mkdir: made, found, or the root. ``matched`` is whether a pattern matched
    it, not named literally: ``maxdepth`` counts one level less below it.
    """

    entry: dict[str, Any]
    target: StoragePath
    missing: bool
    existing: bool
    parent: Entry | None
    parent_exists: bool
    matched: bool
    made: bool
    directory: Entry | None


class _Placed(NamedTuple):
    """The roots of a copy or move, placed, and how the destination takes them.

    ``errors`` are those of finding the roots, yielded first. ``whole`` is
    whether a directory root goes whole, by one ``cp_tree`` or ``mv_tree``,
    where its target is missing. ``batch_size`` is the source's
    ``cp_batch_size`` or ``mv_batch_size`` where batching, else 1.
    """

    directories: list[_Root]
    files: list[_Root]
    errors: list[Exception]
    profile: StorageProfile
    implicit_directories: bool
    parents_required: bool
    batching: bool
    batch_size: int
    whole: bool


async def _place_roots(*, move: bool, **kwargs: Any) -> _Placed:
    """Place the roots of a copy, or of a move, as fsspec's ``copy`` places them.

    ``kwargs`` are `_Placement`'s. It opens both handlers, and reads the
    destination's profile. A path onto itself, left as it is, has no root.
    """
    place = _Moving(**kwargs) if move else _Placement(**kwargs)
    await place.setup()
    whole = _whole(place, move)
    if isinstance(place, _Moving):
        place.goes_whole = whole  # Read by recased, in prepare.
    roots = await place.prepare() or _Roots([])

    def placed(entry: dict[str, Any]) -> _Root:
        target = place.target(entry["name"])
        try:
            parent = place.dst.parent(target)
        except ValueError:
            parent = None  # At the root: no parent of the destination's.
        return _Root(
            entry=entry,
            target=target,
            missing=place.known_missing(target),
            existing=target in place.existing,
            parent=place.parent_of(target),
            parent_exists=parent is None or not place.needs_mkdir(parent, always=True),
            matched=place.handler.trim(entry["name"]) in place.matched,
            made=not place.needs_mkdir(target, always=True),
            directory=place.directories.get(target),
        )

    return _Placed(
        directories=[placed(d) for d in roots.directories],
        files=[placed(f) for f in roots.files],
        errors=place.found,
        profile=place.profile,
        implicit_directories=place.implicit_directories,
        parents_required=place.parents_required,
        batching=place.batching,
        batch_size=place.transfer_batch_size if place.batching else 1,
        whole=whole,
    )


def _whole(place: _Placement, move: bool) -> bool:
    """Whether the roots go whole, by one ``cp_tree`` or ``mv_tree``, where they can.

    Only within one storage does the profile tell how the source copies or
    renames. A directory gone whole would take what is deeper than maxdepth
    along. A move's own setup follows: a rename makes no parent, as a write
    may, so its directory is made first unless the profile says it renames
    no file.
    """
    profile = place.profile
    declared = profile.renames_directories if move else profile.copies_directories
    if move:
        place.implicit_directories = place.same_storage and place.implicit_directories
        place.parents_required = place.parents_required or (
            place.same_storage
            and not place.implicit_directories
            and profile.renames_files is not False
        )
    return place.same_storage and bool(declared) and place.maxdepth is None
