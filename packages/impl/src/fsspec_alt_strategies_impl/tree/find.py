# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive find that lists several folders at a time.

`_tree_find` is an alternative to fsspec's ``_find``, with the same paths. fsspec's
walks a tree one folder at a time, unless the filesystem lists whole trees.
`_tree_find` lists several folders at a time, as `_tree_walk` does.

This module depends on `walk`, and `operation` and `paths` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import FindState

from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim
from .walk import _chunks, _Listing

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

__all__: list[str] = []


def _tree_find(
    fs: Any,
    path: str,
    *,
    maxdepth: int | None = None,
    withdirs: bool = False,
    ils_kwargs: dict[str, Any] | None = None,
) -> Operation[dict[str, Any], FindState]:
    """Find the files in a tree, listing several folders at a time.

    It finds the paths fsspec's ``_find`` finds. A missing ``path`` finds
    nothing, and a file finds itself. A folder that fails to list yields its
    error, and the rest of the tree is still searched.

    It yields each path as its folder's listing arrives, so the paths are not
    in order. fsspec's ``_find`` sorts them, by name. A storage listing a
    folder twice, e.g. as a prefix and as its marker object, yields it twice.

    It wins over fsspec's ``_find`` on storages that list one folder per request,
    with many folders: their listings overlap. It loses on object stores, which
    list a whole tree in one paginated request.

    Once cancelled, it lists no new folder, and yields the paths of the
    listings in flight.

    Requests: one ``_ils`` per folder, as the filesystem's ``operation_limit`` allows. One
    ``info`` more for ``path`` itself, if a file, or a folder with
    ``withdirs``.

    Parameters
    ----------
    fs : TreeFileSystem
        The filesystem searched, by its ``_ils`` and ``_info``.
    path : str
        The folder to search, as ``_ils`` takes it.
    maxdepth : int, optional
        The deepest level to search: 1 finds only the files in ``path``. No
        limit by default.
    withdirs : bool
        Whether to find folders too, ``path`` included.
    ils_kwargs : dict, optional
        Passed to each ``_ils``, as fsspec's ``find`` passes its own on.

    Returns
    -------
    Operation
        Yielding the ``_ils`` entry of each path found, or the error of a folder
        that failed to list. Its state is a `FindState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    _tree_walk : The walk under it.
    """
    _check_maxdepth(maxdepth)

    async def run(
        op: Operation[dict[str, Any], FindState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        root = _trim(path)
        steps = _find_listings(
            fs,
            root,
            maxdepth=maxdepth,
            ils_kwargs=ils_kwargs,
            stopping=op.cancelling,
            file_root=True,
            folder_root=withdirs and bool(root),
        )
        state = op.state

        async with aclosing(steps):
            async for listing in steps:
                found = [
                    e for e in listing.entries if withdirs or e["type"] != "directory"
                ]
                op.state = state = FindState(
                    errors=state.errors + (listing.error is not None),
                    current=listing.folder,
                    # Depth 0 is the root's own entry: no folder listed.
                    folders_listed=state.folders_listed
                    + (listing.depth > 0 and listing.last),
                    found=state.found + len(found),
                )

                if listing.error is not None:
                    yield listing.error

                for entry in found:
                    yield entry

    return Operation(run, FindState(folders_listed=0))


async def _find_listings(
    fs: Any,
    path: str,
    *,
    maxdepth: int | None = None,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    ils_kwargs: dict[str, Any] | None = None,
    stopping: Callable[[], object] | None = None,
    file_root: bool = False,
    folder_root: bool = False,
) -> AsyncGenerator[_Listing]:
    """Walk as the strategies that only read do, such as find, du and glob.

    It yields each folder's listing in parts, as ``_chunks`` does, but treats ``path``
    itself as fsspec's ``_find`` and ``_glob`` do:

    - missing, it yields nothing;
    - a file, it yields the file's ``info`` entry if ``file_root``, else
      nothing;
    - a folder, it yields the folder's own ``info`` entry after its listing's
      last part, if ``folder_root`` and the walk is not stopping.

    The root's own entry comes as a `_Listing` of depth 0, which is no listing.
    So the strategies filter and count it as any entry, but count no folder
    listed for it.
    """
    root = _trim(path)
    listings = _chunks(
        fs,
        root,
        maxdepth=maxdepth,
        descend=descend,
        ils_kwargs=ils_kwargs,
        stopping=stopping,
    )

    async with aclosing(listings):
        try:
            root_listing = await anext(listings)
        except StopAsyncIteration:
            return  # Stopping before the root was listed.
        except (FileNotFoundError, NotADirectoryError) as e:
            # Only the root's own listing raises them: any later one is yielded.
            if file_root and isinstance(e, NotADirectoryError):
                async with fs.operation_limit:
                    entry = await fs._info(root)
                yield _Listing(root, 0, [entry])
            return

        yield root_listing
        if root_listing.last:
            async for listing in _root_entry(fs, root, folder_root, stopping):
                yield listing

        async for listing in listings:
            yield listing
            if listing.folder == root and listing.last:
                async for entry in _root_entry(fs, root, folder_root, stopping):
                    yield entry


async def _root_entry(
    fs: Any,
    root: str,
    folder_root: bool,
    stopping: Callable[[], object] | None,
) -> AsyncGenerator[_Listing]:
    """The root folder's own ``info`` entry, as a listing of depth 0, if asked."""
    if folder_root and not (stopping and stopping()):
        async with fs.operation_limit:
            entry = await fs._info(root)
        yield _Listing(root, 0, [entry])
