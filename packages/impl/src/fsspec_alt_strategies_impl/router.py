# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A filesystem routing each path to the storage of its handler.

`RoutingFileSystem` is the filesystem of a namespace spanning several
storages, e.g. a cloud's volumes or containers, each with a handler of its
own. A `RoutingHandler`, which a developer implements, says where each path
goes. The router makes the filesystem of each storage it routes to, on its
handler, and passes each call on to it, so that the storage's strategies
run. A copy or move between two storages runs the source handler's
strategy, each file by the routing handler's ``cp_across`` or
``mv_across``, which stream the data by `transfer._cross_cp_file` unless
overridden.

This module depends on `filesystem`, `handler`, `operation`, `paths` and
`transfer` of `common`. It imports no family: each storage's handler runs
its own strategies.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from abc import abstractmethod
from contextlib import aclosing
from glob import has_magic
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from fsspec.asyn import AsyncFileSystem, sync
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.spec import AbstractFileSystem

from fsspec_alt_strategies.monitor import Call, Monitored, routing

from .common.filesystem import (
    BaseFileSystem,
    _as_given,
    _copy_errors,
    _raise,
    _range_bounds,
)
from .common.handler import (
    FileOperationHandler,
    StorageHandler,
    _limit,
    _PathHandler,
)
from .common.operation import _collect_with_monitors
from .common.paths import _trim
from .common.transfer import _cross_cp_file

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Awaitable,
        Callable,
        Hashable,
        Sequence,
    )

    from fsspec.callbacks import Callback

    from .common.handler import Entry, StoragePath
    from .common.jobs import _Transfer
    from .common.operation import Operation

__all__ = ["RoutingFileSystem", "RoutingHandler"]

_Group = tuple[AsyncFileSystem, list[str], list[str]]
"""An inner filesystem, the inner paths given to it, and the router's paths of them."""

_Made = TypeVar("_Made", bound="FileOperationHandler | RoutingHandler")
"""The class of a handler a routing handler makes, e.g. one per volume."""

_Through = tuple[tuple["RoutingHandler", Any], ...]
"""The routers a path goes through, outermost first, each with the handler it routed to."""


class RoutingHandler(_PathHandler):
    """Where each path of a namespace spanning several storages goes, for a `RoutingFileSystem`.

    A developer implements, beside `verify_path`, which checks the router's
    own paths:

    - `route`, the handler of a path's storage and its path there, or None
      for a path above every storage, e.g. a catalog holding volumes;
    - `outer`, the router's path of a path of one of the handlers it routes
      to, so that the names answered, e.g. by ``ls`` and ``find``, are the
      router's;
    - `roots`, if any path is above the storages, a listing there.

    It has no primitives: each storage's handler has its own. The handler
    `route` gives may be another routing handler, which routes the path it
    is given again. Each handler checks its own paths: a path through two
    routers is checked three times, by each router and by the storage's
    handler.

    `inner` makes the handlers it routes to, once each, so that `aclose` and
    `close` close them. A copy or move between two storages goes a file at
    a time by `cp_across` or `mv_across`, ``transfer_concurrency`` at a time.
    Through nested routers, the outermost router's are those used.

    Parameters
    ----------
    transfer_concurrency : int or asyncio.Semaphore
        The files copied between two storages at a time. A semaphore is used
        as it is, shared with what else holds it.

    Raises
    ------
    ValueError
        If ``transfer_concurrency`` is less than 1.

    See Also
    --------
    RoutingFileSystem : The filesystem on it.

    Examples
    --------
    >>> class Lake(RoutingHandler):
    ...     protocol = "lake"
    ...
    ...     def verify_path(self, path):
    ...         if self.strip_protocol(path).startswith("/"):
    ...             raise ValueError(f"Not a path of the lake: {path!r}")
    ...
    ...     def route(self, path):
    ...         volume, _, inner = path.partition("/")
    ...         if not volume:
    ...             return None  # The root, above every volume.
    ...         return self.inner(volume, lambda: VolumeHandler(volume)), inner
    ...
    ...     def outer(self, handler, path):
    ...         return f"{handler.volume}/{path}".rstrip("/")
    ...
    ...     async def roots(self, entry):
    ...         return [
    ...             {"name": v, "type": "directory", "size": 0}
    ...             for v in await self.client.list_volumes()
    ...         ]
    >>> fs = RoutingFileSystem(Lake(transfer_concurrency=8))
    """

    def __init__(self, *, transfer_concurrency: int | asyncio.Semaphore) -> None:
        self.transfer_concurrency = transfer_concurrency
        self.transfer_limit = _limit(transfer_concurrency, "transfer_concurrency")
        self._inner: dict[Hashable, Any] = {}

    # What a developer implements.

    @abstractmethod
    def route(
        self, path: StoragePath
    ) -> tuple[FileOperationHandler | RoutingHandler, str] | None:
        """The handler of a path's storage, and its path there, or None above them all.

        ``path`` is the router's, verified, without a trailing "/". It may be a
        glob pattern: where its literal part routes, the whole pattern must
        route to the same handler. The handler given checks the path given
        back, by its ``verify_path``.

        It raises FileNotFoundError for a path no storage has, e.g. a missing
        volume. It is sync, as fsspec's sync methods route too: it makes the
        handler, by `inner`, without a request.
        """

    @abstractmethod
    def outer(self, handler: FileOperationHandler | RoutingHandler, path: str) -> str:
        """The router's path of ``path``, a path of ``handler``, one `route` gave."""

    # What a developer may override.

    async def roots(self, entry: Entry) -> list[Entry]:
        """List a directory above the storages, its entries named by the router's paths.

        It raises FileNotFoundError for a directory that does not exist. Here,
        there is none: `route` never answers None, e.g. for URLs each naming
        its storage. A handler with paths above its storages overrides it.
        """
        raise FileNotFoundError(entry["name"])

    async def root_info(self, entry: Entry) -> Entry:
        """Describe a path above the storages, as ``info`` does.

        Here, the root is a bare directory entry, and any other path the entry
        of its name in `roots` of its parent. A handler knowing more, e.g. the
        root's own entry, or one request describing a path, overrides it. It
        raises FileNotFoundError for a path that does not exist.
        """
        path = entry["name"]
        if path == self.root:
            return {"name": path, "size": 0, "type": "directory"}

        for listed in await self.roots(self.entry(self.parent(path))):
            if listed["name"].rstrip("/") == path:
                return listed

        raise FileNotFoundError(path)

    async def cp_across(
        self,
        src: StorageHandler,
        entry: Entry,
        dst: StorageHandler,
        path: StoragePath,
    ) -> None:
        """Copy a file from one storage to another, the paths their handlers'.

        ``entry`` is the source's, as its listing or ``info`` gave it, its
        ``"name"`` the source path. Its size, e.g., may choose how the data
        goes. It may be stale. ``path`` is the destination's, verified.

        Here, the data is streamed through the caller, by
        `transfer._cross_cp_file`: a block at a time, leaving no part of a
        file, and checking the size written. A handler overrides it with the
        cloud's copy on the server, e.g. a copy from a URL, and falls back on
        this one where that cannot be done, e.g. between two clouds. It runs
        holding a permit of ``transfer_limit``.
        """
        await _cross_cp_file(src, entry, dst, path)

    async def mv_across(
        self,
        src: StorageHandler,
        entry: Entry,
        dst: StorageHandler,
        path: StoragePath,
    ) -> None:
        """Move a file from one storage to another, as `cp_across` copies one.

        Here, it is `cp_across`, then the source's ``rm_file``: a source whose
        copy failed is kept. It runs holding a permit of ``transfer_limit``.
        """
        await self.cp_across(src, entry, dst, path)
        await src.rm_file(entry)

    # The handlers it makes, and its lifecycle.

    def inner(self, key: Hashable, make: Callable[[], _Made]) -> _Made:
        """The handler of ``key``, made by ``make`` the first time, e.g. one per volume.

        `route` makes the handlers it routes to by it, so that `aclose` and
        `close` close them. ``make`` sends no request: a handler opens as
        its storage is first used.
        """
        made = self._inner.get(key)
        if made is None:
            made = self._inner[key] = make()
        return made

    async def open(self) -> None:
        """Nothing: each handler made opens as its storage is first used."""

    async def aclose(self) -> None:
        """Close the handlers made by `inner`, by their `aclose`, and forget them: a later route makes them again."""
        made, self._inner = self._inner, {}
        for handler in made.values():
            await handler.aclose()

    def close(self) -> None:
        """Close the handlers made by `inner` at once, by their sync `close`, and forget them.

        The filesystem's sync ``close()`` calls it. A later route makes them
        again.
        """
        made, self._inner = self._inner, {}
        for handler in made.values():
            handler.close()


class _Storage(NamedTuple):
    """A storage a router routed to: its handler, its filesystem, and the routers it goes through."""

    handler: FileOperationHandler
    fs: BaseFileSystem
    through: _Through


class RoutingFileSystem(AsyncFileSystem, Monitored):
    """An async filesystem passing each call to the filesystem of the path's storage.

    It suits a namespace spanning several storages, each with a handler of
    its own, e.g. a cloud's volumes, or an account's containers each with its
    own client. Its `RoutingHandler` says where each path goes. The router
    makes the filesystem of each storage it routes to, a `BaseFileSystem` on
    the storage's handler, and passes each fsspec call on to it.

    The primitives, e.g. ``_info``, ``_cat_file`` and ``_open``, call the
    storage's filesystem's own. The recursive methods, e.g. ``_find``,
    ``_rm`` and ``_copy``, call its own too, so that its strategies run.
    Several paths, e.g. of one ``rm``, are grouped by storage, each group sent
    at once. A failure raised by a storage gets a note naming the router's
    path, and its inner one.

    Above the storages, ``ls`` answers from the handler's ``roots``, and
    ``info`` from its ``root_info``. ``find``, ``du`` and ``walk`` are
    fsspec's own walk, a directory after another, handing each storage's
    directory to its filesystem. A glob whose literal prefix routes goes whole
    to that storage. Another lists the prefix's roots, as fsspec's ``glob``
    does. A change above the storages, e.g. ``rm`` of a catalog, raises
    PermissionError.

    A copy or move between two paths of one storage is the storage's own.
    Between two storages, it is the source handler's ``cp`` or ``mv``,
    walking or listing the source as its strategy does, with ``maxdepth``
    too. Each file goes by the handler's
    ``cp_across`` or ``mv_across``, which stream the data unless the handler
    overrides them, e.g. with a copy on the server. A move between two
    storages deletes each source file once its copy succeeds, and the source
    directories left empty.

    Each path is checked by the routing handler, and its path in a storage by
    the storage's handler. A routing handler may route to another, whose
    paths are checked too, and whose names are mapped back through both.

    One transaction is the router's: ``open`` passes ``autocommit`` on, and
    fsspec adds each file written to the router's transaction. Its commit, or
    discard, is file by file: it is not atomic across storages. In it,
    ``pipe_file`` and ``put_file`` write through ``open``, so that their files
    join it. ``open_async`` takes no part in it, as in fsspec, whose
    ``open_async`` knows no transaction: its files are the storage's own,
    written as they close. Neither does a copy between two storages.

    The router has no profile and no limit of its own: each storage's are
    its handler's, and the files copied across storages at once are bounded
    by the routing handler's ``transfer_limit``.

    A `Monitor` sees the calls of the storages' filesystems, their
    ``Call.via`` naming the routers they came through. A copy across storages
    is a call of the router itself.

    It takes a routing handler. By convention, a class below it takes the
    handler's storage options, not a handler: its ``__init__`` makes its
    handler from them, so that fsspec makes it from a URL or a registry. It
    names its ``protocol``, ``root_marker``, ``_strip_protocol`` and
    ``_get_kwargs_from_urls`` as its handler class's ``protocol``, ``root``,
    ``strip_protocol`` and ``options_from_url``.

    Parameters
    ----------
    handler : RoutingHandler
        Where each path goes.
    asynchronous, loop, use_listings_cache, listings_expiry_time, max_paths
        fsspec's own options, as `BaseFileSystem` takes them.

    Raises
    ------
    TypeError
        If given an option it does not name, e.g. one misspelled.

    See Also
    --------
    RoutingHandler : Where each path goes.
    BaseFileSystem : The filesystem of each storage.

    Examples
    --------
    >>> class LakeFileSystem(RoutingFileSystem):
    ...     protocol = Lake.protocol
    ...     root_marker = Lake.root
    ...     _strip_protocol = Lake.strip_protocol
    ...     _get_kwargs_from_urls = Lake.options_from_url
    ...
    ...     def __init__(self, *, transfer_concurrency, **kwargs):
    ...         super().__init__(Lake(transfer_concurrency=transfer_concurrency), **kwargs)
    >>> fs = LakeFileSystem(transfer_concurrency=8)
    >>> fs = RoutingFileSystem(Lake(transfer_concurrency=8), asynchronous=True)
    """

    handler: RoutingHandler
    """The routing handler, which says where each path goes."""

    def __init__(
        self,
        handler: RoutingHandler,
        *,
        asynchronous: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        use_listings_cache: bool = True,
        listings_expiry_time: float | None = None,
        max_paths: int | None = None,
    ) -> None:
        # fsspec's own options, named, as `BaseFileSystem` names them.
        super().__init__(
            asynchronous=asynchronous,
            loop=loop,
            use_listings_cache=use_listings_cache,
            listings_expiry_time=listings_expiry_time,
            max_paths=max_paths,
        )

        self.handler = handler
        # The storages routed to, by their handler's id, each kept with it.
        self._storages: dict[int, _Storage] = {}

    def unstrip_protocol(self, name: str) -> str:
        """A path of the router as a URL: its handler's."""
        return self.handler.unstrip_protocol(name)

    def close(self) -> None:
        """Release the handlers the routing handler made, and forget their filesystems.

        A call after makes them again. Their clients' sessions, which closing
        takes a request, are closed by the routing handler's `aclose`.
        """
        self._storages = {}
        self.handler.close()

    # Routing.

    def _resolve(
        self, path: str
    ) -> tuple[_Through, RoutingHandler, FileOperationHandler | None, str]:
        """Route a router's path through each router, to a storage's handler or above them.

        It gives the routers gone through, the last router, the storage's
        handler, or None above the storages of the last router, and the path
        there. Each router checks its path, and the storage's handler its own.
        A trailing "/" of the path is kept on a non-empty inner path, as it
        means a directory to a copy or a glob: a route never sees it.
        """
        through: list[tuple[RoutingHandler, Any]] = []
        router = self.handler
        while True:
            plain = path.rstrip("/") or path
            target = router.route(router.verified(plain))
            if target is None:
                return tuple(through), router, None, plain

            handler, inner = target
            if path != plain and inner and not inner.endswith("/"):
                inner = f"{inner}/"
            through.append((router, handler))
            if isinstance(handler, RoutingHandler):
                router, path = handler, inner
                continue
            if not isinstance(handler, FileOperationHandler):
                raise TypeError(
                    f"{router!r} routed {plain!r} to {handler!r}: a storage's handler"
                    " with strategies is needed, a FileOperationHandler. Wrap a"
                    " filesystem, e.g. in FsspecTreeHandler(fs)."
                )

            handler.verify_path(inner.rstrip("/") or handler.root)
            return tuple(through), router, handler, inner

    def _route(self, path: str) -> tuple[BaseFileSystem, str] | None:
        """The filesystem of a path's storage, and its path there, or None above them all."""
        through, _, handler, inner = self._resolve(path)
        if handler is None:
            return None
        return self._storage(handler, through).fs, inner

    def _storage(self, handler: FileOperationHandler, through: _Through) -> _Storage:
        """The storage of a handler, its filesystem made once."""
        storage = self._storages.get(id(handler))
        if storage is not None and storage.handler is handler:
            return storage

        # Never cached: the router keeps it, and fsspec would key it by the
        # handler's repr. fsspec's metaclass takes skip_instance_cache, before
        # __init__ sees the options.
        fs = BaseFileSystem(  # type: ignore[call-arg]
            handler,
            asynchronous=self.asynchronous,
            loop=self.loop,
            skip_instance_cache=True,
        )
        storage = self._storages[id(handler)] = _Storage(handler, fs, through)
        return storage

    def _outer(self, fs: AsyncFileSystem, path: str) -> str:
        """The router's path of ``path``, a path of a storage's filesystem."""
        handler = fs._handler
        return self._lifted(self._storages[id(handler)].through, path)

    @staticmethod
    def _lifted(through: _Through, path: str) -> str:
        """A path of the last handler gone through, mapped back to the outermost router's."""
        for router, handler in reversed(through):
            path = router.outer(handler, path)
        return path

    async def _roots(self, path: str) -> list[dict[str, Any]]:
        """List a directory above the storages, by the routing handler's ``roots``."""
        through, router, _, inner = self._resolve(path)
        listed = await router.roots(router.entry(inner))
        return [{**e, "name": self._lifted(through, e["name"])} for e in listed]

    async def _root_info(self, path: str) -> dict[str, Any]:
        """Describe a path above the storages, by the routing handler's ``root_info``."""
        through, router, _, inner = self._resolve(path)
        entry = await router.root_info(router.entry(inner))
        return {**entry, "name": self._lifted(through, entry["name"])}

    def _target(self, path: str) -> tuple[AsyncFileSystem, str] | None:
        """`_route` of a path given to a method, its protocol stripped."""
        return self._route(self.handler.verified(path, strip_only_protocol=True))

    def _required(self, path: str) -> tuple[AsyncFileSystem, str]:
        """``_target`` of a path that must be in a storage, e.g. of a file."""
        target = self._target(path)
        if target is None:
            raise PermissionError(
                f"{path!r} is above the storages of {self!r}: it can only be listed."
            )

        return target

    def _pattern_target(self, path: str) -> tuple[AsyncFileSystem, str] | None:
        """``_target`` of a path or glob pattern, if all it matches is in one storage."""
        if not has_magic(path):
            return self._target(path)

        literal = path[: min(path.find(c) % (len(path) + 1) for c in "*?[")]
        base = self._target(literal.rpartition("/")[0])
        if base is None:
            return None

        whole = self._target(path)
        return whole if whole is not None and whole[0] is base[0] else None

    async def _storage_roots(self, path: str) -> list[str]:
        """The roots of the storages below a path above them, found by `_roots`, level by level."""
        roots: list[str] = []
        frontier = [path.rstrip("/")]
        while frontier:
            prefix = frontier.pop()
            for entry in await self._roots(prefix):
                name = entry["name"].rstrip("/")
                if self._target(name) is not None:
                    roots.append(name)
                elif entry.get("type") == "directory":
                    frontier.append(name)
        return sorted(roots)

    async def _spread(self, pattern: str, maxdepth: int | None) -> list[str] | None:
        """A pattern spanning storages as one pattern per storage it can match, or None to glob as fsspec does.

        The segments above the storages are expanded one at a time: a literal
        one is taken as it is, and one with wildcards matched against the
        names `_roots` lists there. Once a path is in a storage, the rest of the
        pattern is that storage's to glob, so a storage matching no segment
        above it is never listed: e.g. ``data*/x.txt`` lists the buckets, then
        ``x.txt`` in each bucket named ``data...``.

        It answers None where fsspec's glob counts what is above the storages:
        a ``**`` there, a ``maxdepth`` with a ``**``, or a pattern ending above
        the storages, matching their roots themselves.
        """
        segments = pattern.rstrip("/").split("/")
        trailing = "/" if pattern.endswith("/") else ""
        if maxdepth is not None and "**" in segments:
            return None

        spread: list[str] = []
        frontier = [("", 0)]
        while frontier:
            prefix, index = frontier.pop()
            if prefix and self._target(prefix) is not None:
                rest = "/".join(segments[index:])
                spread.append(
                    f"{prefix}/{rest}{trailing}" if rest else prefix + trailing
                )
                continue
            if index >= len(segments) - 1:
                return None  # The pattern ends above the storages.
            segment = segments[index]
            if segment == "**":
                return None
            if not has_magic(segment):
                frontier.append(
                    (f"{prefix}/{segment}" if prefix else segment, index + 1)
                )
                continue
            for entry in await self._roots(prefix):
                name = entry["name"].rstrip("/")
                if fnmatch.fnmatchcase(name.rpartition("/")[2], segment):
                    frontier.append((name, index + 1))
        return sorted(spread)

    async def _grouped(self, path: str | Sequence[str]) -> list[_Group]:
        """The paths of a call changing the storages, grouped by inner filesystem.

        A pattern matching in several storages is expanded by ``_glob`` first.
        A path above the storages raises PermissionError.
        """
        groups: dict[int, _Group] = {}

        def add(target: tuple[AsyncFileSystem, str], outer: str) -> None:
            fs, inner = target
            group = groups.setdefault(id(fs), (fs, [], []))
            group[1].append(inner)
            group[2].append(outer)

        for p in [path] if isinstance(path, str) else path:
            target = self._pattern_target(
                self.handler.verified(p, strip_only_protocol=True)
            )
            if target is not None:
                add(target, p)
            elif has_magic(p):
                for match in await self._glob(p):
                    add(self._required(match), match)
            else:
                self._required(p)  # Raises, above the storages.

        return list(groups.values())

    def _note(
        self, error: BaseException, fs: AsyncFileSystem, inner: Any, outer: Any
    ) -> None:
        """Tell the router's path in an inner filesystem's error."""
        if isinstance(error, OSError) and error.filename == inner:
            error.filename = outer
        error.add_note(f"{outer!r} was routed to {fs!r} as {inner!r}.")

    async def _on(
        self, fs: AsyncFileSystem, inner: Any, outer: Any, call: Awaitable[Any]
    ) -> Any:
        """Await an inner filesystem's call, as made through this router."""
        with routing(self):
            try:
                return await call
            except Exception as e:
                self._note(e, fs, inner, outer)
                raise

    async def _forward(self, path: str, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call the inner filesystem's ``method`` on the inner path of ``path``."""
        fs, inner = self._required(path)
        return await self._on(
            fs, inner, path, getattr(fs, method)(inner, *args, **kwargs)
        )

    def _entry(self, fs: AsyncFileSystem, entry: dict[str, Any]) -> dict[str, Any]:
        """An inner filesystem's entry, named by the router's path."""
        return {**entry, "name": self._outer(fs, entry["name"])}

    async def _each(
        self,
        groups: list[_Group],
        run: Callable[[_Group], Awaitable[Any]],
        doing: str,
        errors: Sequence[Exception] = (),
    ) -> None:
        """Run a call per group at once, raising their errors when all are done.

        No group is stopped by another's failure: a deletion or copy under
        way finishes, as its outcome would otherwise be unknown. ``errors``
        are those found before, raised with the groups'.
        """
        outcomes = await asyncio.gather(*map(run, groups), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(
                outcome, Exception
            ):
                raise outcome  # A cancellation, never swallowed.

        _raise([*errors, *(o for o in outcomes if isinstance(o, Exception))], doing)

    def _placed(
        self, paths: Sequence[str], destination: str, doing: str
    ) -> tuple[list[str], list[Exception]]:
        """The roots of a list copied or moved into ``destination``, each name once.

        Each root goes to the destination under its name, whatever its
        storage, as a family's copy of a list places it. Two roots of one name
        would go to the same place: as there, the first in path order goes,
        and each other is a FileExistsError. Roots in different storages are
        checked here, as no inner filesystem sees them both. A pattern is
        left to the filesystem expanding it.
        """
        placed: dict[str, str] = {}
        errors: list[Exception] = []
        skipped: set[str] = set()
        base = _trim(self.handler.verified(destination, strip_only_protocol=True))

        for path in sorted(
            paths,
            key=lambda p: _trim(self.handler.verified(p, strip_only_protocol=True)),
        ):
            if has_magic(path):
                continue

            name = _trim(
                self.handler.verified(path, strip_only_protocol=True)
            ).rpartition("/")[2]
            target = f"{base}/{name}" if base else name
            if (other := placed.get(target)) is not None:
                error = FileExistsError(target)
                error.add_note(
                    f"{doing} {path!r} to {target!r} skipped: {other!r} goes there too."
                )
                errors.append(error)
                skipped.add(path)
                continue

            placed[target] = path

        return [p for p in paths if p not in skipped], errors

    @staticmethod
    def _one(paths: list[str]) -> str | list[str]:
        """A group's paths as given to a method: one alone, as fsspec's semantics differ."""
        return paths[0] if len(paths) == 1 else paths

    # The primitives, each its storage's own.

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        target = self._target(path)
        if target is not None:
            fs, inner = target
            return self._entry(fs, await self._on(fs, inner, path, fs._info(inner)))
        else:
            return await self._root_info(self._strip_protocol(path))

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        target = self._target(path)
        if target is None:
            entries = await self._roots(self._strip_protocol(path))
        else:
            fs, inner = target
            listed = await self._on(
                fs, inner, path, fs._ls(inner, detail=True, **kwargs)
            )
            entries = [self._entry(fs, e) for e in listed]

        return entries if detail else [e["name"] for e in entries]

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        return await self._forward(path, "_cat_file", start=start, end=end, **kwargs)

    async def _cat_ranges(
        self,
        paths: list[str],
        starts: Any,
        ends: Any,
        max_gap: int | None = None,
        batch_size: int | None = None,
        on_error: str = "return",
        **kwargs: Any,
    ) -> list[Any]:
        """Read byte ranges of files, as fsspec's ``cat_ranges`` does, each storage's by its own.

        fsspec's calls ``_cat_file`` once per range. This gives each inner
        filesystem its ranges in one ``_cat_ranges``, all at once, so that a
        filesystem merging nearby ranges, as the families' do, merges them. A
        path no storage has, or above them, fails in place of its range. The
        answers come in the order asked, as fsspec's do.
        """
        starts, ends = _range_bounds(paths, starts, ends)
        out: list[Any] = [None] * len(paths)
        # Each inner filesystem with the ranges it reads: their indices, and inner paths.
        groups: dict[int, tuple[AsyncFileSystem, list[int], list[str]]] = {}
        for i, path in enumerate(paths):
            try:
                fs, inner = self._required(path)
            except Exception as e:
                out[i] = e
                continue
            groups.setdefault(id(fs), (fs, [], []))[1].append(i)
            groups[id(fs)][2].append(inner)

        async def read(
            fs: AsyncFileSystem, indices: list[int], inners: list[str]
        ) -> None:
            options = {} if max_gap is None else {"max_gap": max_gap}
            with routing(self):
                results = await fs._cat_ranges(
                    inners,
                    [starts[i] for i in indices],
                    [ends[i] for i in indices],
                    batch_size=batch_size,
                    on_error="return",
                    **options,
                    **kwargs,
                )
            for i, inner, result in zip(indices, inners, results, strict=True):
                if isinstance(result, BaseException):
                    self._note(result, fs, inner, paths[i])
                out[i] = result

        await asyncio.gather(*(read(*group) for group in groups.values()))
        if on_error != "return":
            for result in out:
                if isinstance(result, BaseException):
                    raise result
        return out

    async def _pipe_file(self, path: str, value: bytes, **kwargs: Any) -> None:
        if self._intrans:
            pipe_file = AbstractFileSystem.pipe_file
            await self._in_transaction(path, pipe_file, path, value, **kwargs)
        else:
            await self._forward(path, "_pipe_file", value, **kwargs)

    async def _get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        await self._forward(rpath, "_get_file", lpath, **kwargs)

    async def _put_file(self, lpath: str, rpath: str, **kwargs: Any) -> None:
        if self._intrans:
            await self._in_transaction(rpath, _put_file, lpath, rpath, **kwargs)
        else:
            fs, inner = self._required(rpath)
            await self._on(fs, inner, rpath, fs._put_file(lpath, inner, **kwargs))

    async def _in_transaction(
        self, path: str, write: Callable[..., None], *args: Any, **kwargs: Any
    ) -> None:
        """Write ``path`` by fsspec's sync ``write``, through ``open``, so that it joins the transaction.

        An inner filesystem's own write goes through ``open_async``, whose
        files know no transaction. ``write`` takes the router first, as
        fsspec's sync methods do. The sync file runs in a thread, as its
        methods may wait on this loop.
        """
        fs, inner = self._required(path)
        await self._on(fs, inner, path, asyncio.to_thread(write, self, *args, **kwargs))

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        await self._forward(path, "_rm_file", **kwargs)

    async def _rmdir(self, path: str) -> None:
        await self._forward(path, "_rmdir")

    async def _mkdir(
        self, path: str, create_parents: bool = True, **kwargs: Any
    ) -> None:
        await self._forward(path, "_mkdir", create_parents=create_parents, **kwargs)

    async def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        if self._target(path) is None and await self._exists(path):
            if exist_ok:
                return
            raise FileExistsError(path)

        await self._forward(path, "_makedirs", exist_ok=exist_ok)

    async def _exists(self, path: str, **kwargs: Any) -> bool:
        try:
            target = self._target(path)
        except FileNotFoundError:
            return False

        if target is None:
            return await super()._exists(path, **kwargs)

        fs, inner = target
        return await self._on(fs, inner, path, fs._exists(inner, **kwargs))

    async def _isdir(self, path: str) -> bool:
        try:
            target = self._target(path)
        except FileNotFoundError:
            return False

        if target is None:
            return await super()._isdir(path)

        fs, inner = target
        return await self._on(fs, inner, path, fs._isdir(inner))

    async def _isfile(self, path: str) -> bool:
        try:
            target = self._target(path)
        except FileNotFoundError:
            return False

        if target is None:
            return False  # Above the storages, all are directories.

        fs, inner = target
        return await self._on(fs, inner, path, fs._isfile(inner))

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        fs1, inner1 = self._required(path1)
        fs2, inner2 = self._required(path2)

        if fs1 is fs2:
            await self._on(fs1, inner1, path1, fs1._cp_file(inner1, inner2, **kwargs))
            return

        # The source's entry, for the copy across, tells a directory too.
        entry = await self._on(fs1, inner1, path1, fs1._info(inner1))
        if entry["type"] == "directory":
            # fsspec's copy gives directories too, which each filesystem makes.
            await self._on(fs2, inner2, path2, fs2._makedirs(inner2, exist_ok=True))
        else:
            copy_file = self._across(self._handler_of(fs1), self._handler_of(fs2))
            entry = {**entry, "name": inner1}
            await self._on(fs1, inner1, path1, copy_file(entry, inner2))

    async def _mv_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        fs1, inner1 = self._required(path1)
        fs2, inner2 = self._required(path2)

        if fs1 is fs2:
            await self._on(fs1, inner1, path1, fs1._mv_file(inner1, inner2, **kwargs))
        else:
            entry = await self._on(fs1, inner1, path1, fs1._info(inner1))
            mv_file = self._across(
                self._handler_of(fs1), self._handler_of(fs2), move=True
            )
            entry = {**entry, "name": inner1}
            await self._on(fs1, inner1, path1, mv_file(entry, inner2))

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return await self._forward(path, "open_async", mode, **kwargs)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        # autocommit is passed on: fsspec's open adds the file to the router's
        # transaction, which commits it with the others.
        fs, inner = self._required(path)
        return fs._open(
            inner,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_options=cache_options,
            **kwargs,
        )

    def created(self, path: str) -> Any:
        fs, inner = self._required(path)
        return fs.created(inner)

    def modified(self, path: str) -> Any:
        fs, inner = self._required(path)
        return fs.modified(inner)

    def sign(self, path: str, expiration: int = 100, **kwargs: Any) -> str:
        fs, inner = self._required(path)
        return fs.sign(inner, expiration=expiration, **kwargs)

    def invalidate_cache(self, path: str | None = None) -> None:
        super().invalidate_cache(path)

        if path is None:
            for storage in self._storages.values():
                storage.fs.invalidate_cache()
            return

        target = self._target(path)
        if target is not None:
            target[0].invalidate_cache(target[1])

    # The recursive methods, each its storage's own, so that its strategies run.

    async def _walk(
        self, path: str, maxdepth: int | None = None, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, Any, Any]]:
        target = self._target(path)
        if target is None:
            # fsspec's walk, listing from _roots, then each storage's own below.
            async for item in super()._walk(path, maxdepth=maxdepth, **kwargs):
                yield item
            return

        fs, inner = target
        detail = kwargs.get("detail", False)
        walk = fs._walk(inner, maxdepth=maxdepth, **kwargs)

        async with aclosing(walk):
            while True:
                # A step at a time, as the generator runs in its caller's context.
                with routing(self):
                    try:
                        root, dirs, files = await anext(walk)
                    except StopAsyncIteration:
                        return
                    except Exception as e:
                        self._note(e, fs, inner, path)
                        raise

                if detail:
                    dirs = {k: self._entry(fs, v) for k, v in dirs.items()}
                    files = {k: self._entry(fs, v) for k, v in files.items()}

                yield self._outer(fs, root), dirs, files

    async def _find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        **kwargs: Any,
    ) -> Any:
        target = self._target(path)
        detail = kwargs.pop("detail", False)
        if target is None:
            return await super()._find(
                path, maxdepth=maxdepth, withdirs=withdirs, detail=detail, **kwargs
            )

        fs, inner = target
        found = await self._on(
            fs,
            inner,
            path,
            fs._find(
                inner, maxdepth=maxdepth, withdirs=withdirs, detail=True, **kwargs
            ),
        )

        return self._found(fs, found, detail)

    def _found(self, fs: AsyncFileSystem, found: dict[str, Any], detail: bool) -> Any:
        """An inner ``find`` or ``glob``'s answer with details, as the router's."""
        entries = {self._outer(fs, n): self._entry(fs, e) for n, e in found.items()}
        names = sorted(entries)
        return {n: entries[n] for n in names} if detail else names

    async def _du(
        self, path: str, total: bool = True, maxdepth: int | None = None, **kwargs: Any
    ) -> Any:
        target = self._target(path)
        if target is None:
            if maxdepth is not None:
                # fsspec's du, as maxdepth counts the levels above the storages.
                return await super()._du(path, total=total, maxdepth=maxdepth, **kwargs)
            # Each storage's own du, which sends no request per file.
            measured = [
                await self._du(root, total=total, **kwargs)
                for root in await self._storage_roots(path)
            ]
            if total:
                return sum(measured)
            return {name: size for sizes in measured for name, size in sizes.items()}

        fs, inner = target
        sizes = await self._on(
            fs, inner, path, fs._du(inner, total=total, maxdepth=maxdepth, **kwargs)
        )

        return sizes if total else {self._outer(fs, n): s for n, s in sizes.items()}

    async def _glob(self, path: str, maxdepth: int | None = None, **kwargs: Any) -> Any:
        verified = self.handler.verified(path, strip_only_protocol=True)
        target = self._pattern_target(verified)
        detail = kwargs.pop("detail", False)
        if target is None:
            spread = await self._spread(verified, maxdepth)
            if spread is None:
                # fsspec's glob, finding from the literal prefix over _roots.
                return await super()._glob(
                    path, maxdepth=maxdepth, detail=detail, **kwargs
                )
            # Each storage's own glob of what is below it: none other is listed.
            entries: dict[str, Any] = {}
            for pattern in spread:
                entries.update(await self._glob(pattern, detail=True, **kwargs))
            names = sorted(entries)
            return {n: entries[n] for n in names} if detail else names

        fs, inner = target
        found = await self._on(
            fs, inner, path, fs._glob(inner, maxdepth=maxdepth, detail=True, **kwargs)
        )

        return self._found(fs, found, detail)

    async def _expand_path(
        self,
        path: str | list[str],
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        target = (
            self._pattern_target(self.handler.verified(path, strip_only_protocol=True))
            if isinstance(path, str)
            else None
        )

        if target is None:
            return await super()._expand_path(
                path, recursive=recursive, maxdepth=maxdepth, **kwargs
            )

        fs, inner = target
        expanded = await self._on(
            fs,
            inner,
            path,
            fs._expand_path(inner, recursive=recursive, maxdepth=maxdepth, **kwargs),
        )

        return sorted(self._outer(fs, p) for p in expanded)

    async def _rm(
        self,
        path: str | list[str],
        recursive: bool = False,
        batch_size: int | None = None,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None:
        if maxdepth is not None:
            kwargs["maxdepth"] = maxdepth

        async def rm(group: _Group) -> None:
            fs, inner, outer = group
            await self._on(
                fs,
                self._one(inner),
                self._one(outer),
                fs._rm(
                    self._one(inner),
                    recursive=recursive,
                    batch_size=batch_size,
                    **kwargs,
                ),
            )

        await self._each(await self._grouped(path), rm, "Deleting")

    async def _copy(
        self,
        path1: str | list[str],
        path2: str | list[str],
        recursive: bool = False,
        on_error: str | None = None,
        maxdepth: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Copy, as fsspec's ``copy`` does: within a storage, or between two.

        Sources in several storages are copied per storage, at once. A list
        stays a list in each storage, so that each root goes into the
        destination under its name, as `_placed` checks across storages.
        """
        if not isinstance(path2, str):
            # Pairs of paths: fsspec's, a _cp_file per pair.
            await super()._copy(
                path1, path2, recursive, on_error, maxdepth, batch_size, **kwargs
            )
            return

        fs2, inner2 = self._required(path2)
        callback = kwargs.get("callback")
        listed = not isinstance(path1, str)
        clashes: list[Exception] = []
        if listed:
            path1, clashes = self._placed(path1, path2, "Copying")

        async def copy(group: _Group) -> None:
            fs1, inner1, outer1 = group
            sources = inner1 if listed else self._one(inner1)
            outers = outer1 if listed else self._one(outer1)

            if fs1 is fs2:
                await self._on(
                    fs1,
                    sources,
                    outers,
                    fs1._copy(
                        sources,
                        inner2,
                        recursive,
                        on_error,
                        maxdepth,
                        batch_size,
                        **kwargs,
                    ),
                )
            elif recursive:
                op = await self._cross_fs_cp(fs1, sources, fs2, inner2, maxdepth)
                errors = await self._transferred(
                    op, fs1, sources, fs2, inner2, outers, path2, callback
                )
                _raise(_copy_errors(errors, on_error), "Copying")
            else:
                await super(RoutingFileSystem, self)._copy(
                    outers,
                    path2,
                    recursive,
                    on_error,
                    maxdepth,
                    batch_size,
                    **kwargs,
                )

        await self._each(await self._grouped(path1), copy, "Copying", clashes)

    async def _mv(
        self,
        path1: str | list[str],
        path2: str,
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Move, as fsspec's ``mv`` does: within a storage, or between two.

        Between two storages, each source file is deleted once copied, and the
        source directories left empty once all are. A source whose copy failed
        is kept.
        """
        if path1 == path2:
            return

        fs2, inner2 = self._required(path2)
        listed = not isinstance(path1, str)
        clashes: list[Exception] = []
        if listed:
            path1, clashes = self._placed(path1, path2, "Moving")

        async def mv(group: _Group) -> None:
            fs1, inner1, outer1 = group
            sources = inner1 if listed else self._one(inner1)
            outer = outer1 if listed else self._one(outer1)

            if fs1 is fs2:
                move = fs1._mv(
                    sources, inner2, recursive=recursive, maxdepth=maxdepth, **kwargs
                )
                await self._on(fs1, sources, outer, move)
            elif recursive:
                op = await self._cross_fs_mv(fs1, sources, fs2, inner2, maxdepth)
                errors = await self._transferred(
                    op,
                    fs1,
                    sources,
                    fs2,
                    inner2,
                    outer,
                    path2,
                    kwargs.get("callback"),
                    delete=True,
                )
                _raise(errors, "Moving")
            else:
                await super(RoutingFileSystem, self)._copy(
                    outer, path2, recursive, "raise", maxdepth, **kwargs
                )
                await self._rm(outer, recursive=recursive)

        await self._each(await self._grouped(path1), mv, "Moving", clashes)

    async def _cross_fs_cp(
        self,
        fs1: AsyncFileSystem,
        path1: str | list[str],
        fs2: AsyncFileSystem,
        path2: str,
        maxdepth: int | None = None,
    ) -> Operation[tuple[str, str], Any]:
        """The recursive copy from ``fs1`` to ``fs2``, by the source's handler's ``cp``.

        Each file goes by the routing handler's ``cp_across``. The operation is
        returned not yet started.
        """
        src, dst = self._handler_of(fs1), self._handler_of(fs2)
        return src.cp(
            dst,
            _as_given(src, path1),
            dst.verified(path2, strip_only_protocol=True),
            cp_entry=self._across(src, dst),
            same_storage=False,
            maxdepth=maxdepth,
        )

    async def _cross_fs_mv(
        self,
        fs1: AsyncFileSystem,
        path1: str | list[str],
        fs2: AsyncFileSystem,
        path2: str,
        maxdepth: int | None = None,
    ) -> Operation[tuple[str, str], Any]:
        """The recursive move from ``fs1`` to ``fs2``, by the source's handler's ``mv``.

        As `_cross_fs_cp`, each file by the routing handler's ``mv_across``,
        or, where the move copies, e.g. a flat one deleting the keys copied in
        batches, by its ``cp_across``: two storages never rename.
        """
        src, dst = self._handler_of(fs1), self._handler_of(fs2)
        return src.mv(
            dst,
            _as_given(src, path1),
            dst.verified(path2, strip_only_protocol=True),
            cp_entry=self._across(src, dst),
            mv_entry=self._across(src, dst, move=True),
            same_storage=False,
            maxdepth=maxdepth,
        )

    @staticmethod
    def _handler_of(fs: AsyncFileSystem) -> FileOperationHandler:
        """The handler of a storage's filesystem, one the router made."""
        return fs._handler

    def _across(
        self, src: StorageHandler, dst: StorageHandler, *, move: bool = False
    ) -> _Transfer:
        """A file's copy, or move, from ``src`` to ``dst``, under ``transfer_limit``."""
        hook = self.handler.mv_across if move else self.handler.cp_across

        # The hints go nowhere: a copy between storages writes, and the hooks
        # take none.
        async def request(
            entry: dict[str, Any],
            path2: str,
            *,
            missing: bool = False,
            parent: dict[str, Any] | None = None,
        ) -> None:
            with routing(self):
                async with self.handler.transfer_limit:
                    await hook(src, entry, dst, dst.verified(path2))

        return request

    async def _transferred(
        self,
        op: Operation[tuple[str, str], Any],
        fs1: AsyncFileSystem,
        path1: str | list[str],
        fs2: AsyncFileSystem,
        path2: str,
        outer1: str | list[str],
        outer2: str,
        callback: Any,
        delete: bool = False,
    ) -> list[Exception]:
        """Run a copy or move between two storages, returning its paths' errors.

        The call is the router's own, watched by the monitors as such.
        """
        call = Call(self, "mv" if delete else "copy", outer1, outer2)
        _, errors = await _collect_with_monitors(op, call, callback)
        for error in errors:
            error.add_note(
                f"Through {self!r}, from {fs1!r} as {path1!r} to {fs2!r} as {path2!r}."
            )

        return errors

    # fsspec makes sync methods only from the async ones it knows: _mv and
    # _rmdir are not.

    def mv(self, path1: str | list[str], path2: str, **kwargs: Any) -> None:
        return sync(self.loop, self._mv, path1, path2, **kwargs)

    def rmdir(self, path: str) -> None:
        return sync(self.loop, self._rmdir, path)


def _put_file(
    fs: AbstractFileSystem,
    lpath: str,
    rpath: str,
    mode: str = "overwrite",
    callback: Callback = DEFAULT_CALLBACK,
    **kwargs: Any,
) -> None:
    """fsspec's ``put_file`` through ``open``, without its ``makedirs``.

    A storage needs no directory for the files put in it, and fsspec's sync
    ``makedirs`` cannot run from a thread of an async router.
    """
    if os.path.isdir(lpath):
        return

    with open(lpath, "rb") as local:
        callback.set_size(os.fstat(local.fileno()).st_size)
        with fs.open(rpath, "xb" if mode == "create" else "wb", **kwargs) as f:
            while chunk := local.read(fs.blocksize):
                f.write(chunk)
                callback.relative_update(len(chunk))
