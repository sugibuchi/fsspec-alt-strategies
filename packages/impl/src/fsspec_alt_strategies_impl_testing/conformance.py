# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Checks that a handler keeps what its profile declares, run against its storage.

The library never detects what a storage does: the developer declares it, and a
wrong declaration gives wrong answers. These checks catch one before it does,
in a handler's tests, on a dummy or a live storage.

`check_recursive_order` checks the order of a tree handler's recursive ``ils``
against the profile's ``recursive_order``. Unknown, the order is taken as
parents first, right on any storage: "depth_first" is the declaration a wrong
listing would break, a directory then taken as done too early. A handler
keeping its family's default, a directory per request by ``super().ils``,
needs no check: the walk reads such a subtree a directory at a time itself.

This module depends on the strategies' `handler` and `operation` of `common`,
and `handler` of `tree`.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies_impl.common.operation import _collect_with_errors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fsspec_alt_strategies_impl.tree.handler import TreeFileOperationHandler

__all__ = ["check_recursive_order", "recursive_order_errors"]

_TREE = ("a/b/c/1", "a/b/2", "a/b-x/d/3", "a/4", "a.txt")
"""Files whose directories interleave in name order, "a/b", "a/b-x", then "a/b/c" below "a/b", and two levels deep.

Level by level, a listing leaves "a/b-x" for "a/b", then comes back to "a/b-x/d": not depth first.
"""


async def check_recursive_order(handler: TreeFileOperationHandler, root: str) -> None:
    """Check that a tree handler's recursive ``ils`` keeps the order its profile declares, on its storage.

    It writes a few files in a new directory below ``root``, chosen so that
    name order and depth-first order differ, lists the directory by the
    handler's own recursive ``ils``, then deletes it. The listing must give
    each file once, and keep the order of the profile's ``recursive_order``,
    as `recursive_order_errors` checks it.

    Parameters
    ----------
    handler : TreeFileOperationHandler
        The handler checked, opened by the check.
    root : str
        A directory the check may write in, verified by the handler.

    Raises
    ------
    AssertionError
        If the listing leaves out a file, gives one twice, or breaks the order
        declared, each error said.
    """
    await handler.open()
    base = handler.join(handler.verified(root), f"order-check-{secrets.token_hex(4)}")
    try:
        for name in _TREE:
            path = handler.join(base, name)
            await handler.makedirs(handler.parent(path), exist_ok=True)
            await handler.pipe_file(path, b"x")

        listing = handler.ils(handler.entry(base), recursive=True)
        entries = [entry async for entry in listing]
        order = handler.profile.recursive_order
        errors = recursive_order_errors(entries, base, order)
        files = sorted(
            e["name"].rstrip("/") for e in entries if e["type"] != "directory"
        )
        expected = sorted(f"{base}/{name}" for name in _TREE)
        if files != expected:
            errors.insert(0, f"The files listed are {files}, not {expected}.")
        if errors:
            raise AssertionError(
                f"{handler!r} lists {base!r} out of the order its profile declares,"
                f" {order!r}:\n- " + "\n- ".join(errors)
            )
    finally:
        await _collect_with_errors(handler.rm(base))


def recursive_order_errors(
    entries: Sequence[dict[str, Any]], root: str, order: str | None
) -> list[str]:
    """The ways a recursive listing of ``root`` breaks ``order``: none if it keeps it.

    - "depth_first", which covers name order: once the listing has passed
      what is below a directory, nothing more below it comes. A directory's
      own entry may come apart from what is below it, as "a/b-x" comes
      between "a/b" and "a/b/c" in name order.
    - "parents_first", or None, unknown, taken so: a directory's entry, where
      listed, comes before what is below it. A directory left out, as object
      stores list only files, breaks nothing.

    Each error names the entry breaking the order.
    """
    names = [e["name"].rstrip("/") for e in entries]
    prefix = f"{root.rstrip('/')}/" if root.rstrip("/") else ""
    directories = {
        e["name"].rstrip("/") for e in entries if e.get("type") == "directory"
    }
    errors: list[str] = []
    if order == "depth_first":
        # The directories something below which has come, and those the listing then passed.
        entered: set[str] = set()
        passed: set[str] = set()
        for name in names:
            above = _above(name, prefix)
            errors.extend(
                f"{name!r} comes after the listing passed {directory!r}, below which it is."
                for directory in sorted(passed & set(above))
            )
            passed |= entered - set(above)
            entered = (entered & set(above)) | set(above)
    else:
        seen: set[str] = set()
        for name in names:
            parent = name.rpartition("/")[0]
            if parent in directories and parent not in seen:
                errors.append(f"{name!r} comes before its directory {parent!r}.")
            seen.add(name)
    return errors


def _above(name: str, prefix: str) -> list[str]:
    """The directories below the root that a path is in, the shallowest first."""
    parts = name[len(prefix) :].split("/")[:-1]
    return [prefix + "/".join(parts[: i + 1]) for i in range(len(parts))]
