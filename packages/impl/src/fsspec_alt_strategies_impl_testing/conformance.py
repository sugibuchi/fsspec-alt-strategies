# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Checks that a handler keeps what its profile declares, run against its storage.

The library never detects what a storage does: the developer declares it, and a
wrong declaration gives wrong answers. These checks catch one before it does,
in a handler's tests, on a dummy or a live storage.

`check_recursive_order` checks that a handler's ``ils_recursive``, which its
profile's ``recursive_ls`` declares, lists a tree as it must: depth first in
preorder, every directory once, before all below it. A wrong listing would
have a directory taken as done too early. A handler with no
``ils_recursive`` needs no check: the strategies list a directory at a time.

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
    """Check that a handler's ``ils_recursive`` lists a tree as it must, on its storage.

    It writes a few files in a new directory below ``root``, chosen so that
    name order and depth-first order differ, lists the directory by the
    handler's ``ils_recursive``, then deletes it. The listing must give each
    file once, and keep the order `recursive_order_errors` checks.

    Parameters
    ----------
    handler : TreeFileOperationHandler
        The handler checked, opened by the check.
    root : str
        A directory the check may write in, verified by the handler.

    Raises
    ------
    AssertionError
        If the listing leaves out a file, gives one twice, or breaks the
        order, each error said.
    """
    await handler.open()
    base = handler.join(handler.verified(root), f"order-check-{secrets.token_hex(4)}")
    try:
        for name in _TREE:
            path = handler.join(base, name)
            await handler.makedirs(handler.parent(path), exist_ok=True)
            await handler.pipe_file(path, b"x")

        entries = [entry async for entry in handler.ils_recursive(handler.entry(base))]
        errors = recursive_order_errors(entries, base)
        files = sorted(e["name"] for e in entries if e["type"] != "directory")
        expected = sorted(f"{base}/{name}" for name in _TREE)
        if files != expected:
            errors.insert(0, f"The files listed are {files}, not {expected}.")
        if errors:
            raise AssertionError(
                f"{handler!r} lists {base!r} out of the order of ils_recursive:\n- "
                + "\n- ".join(errors)
            )
    finally:
        await _collect_with_errors(handler.rm(base))


def recursive_order_errors(entries: Sequence[dict[str, Any]], root: str) -> list[str]:
    """The ways a listing of all below ``root`` breaks the order of ``ils_recursive``: none if it keeps it.

    - Each entry's directory below ``root`` comes before it, once: those
      the storage keeps only as prefixes too.
    - Depth first: once the listing has passed what is below a directory,
      nothing more below it comes.
    - No name ends with "/": a marker is its directory's entry.

    Each error names the entry breaking the order.
    """
    prefix = f"{root.rstrip('/')}/" if root.rstrip("/") else ""
    errors: list[str] = []
    opened: list[str] = []  # The directories holding the last entry.
    seen: set[str] = set()
    for entry in entries:
        name = entry["name"]
        if name.endswith("/"):
            errors.append(f"{name!r} is a marker, not its directory's entry.")
            name = name.rstrip("/")
        if name in seen:
            errors.append(f"{name!r} comes twice.")
        while opened and not name.startswith(f"{opened[-1]}/"):
            opened.pop()
        parent = name.rpartition("/")[0]
        if parent != root.rstrip("/") and (not opened or opened[-1] != parent):
            if parent in seen:
                errors.append(
                    f"{name!r} comes after the listing passed {parent!r}, below which it is."
                )
            else:
                errors.append(f"{name!r} comes before its directory {parent!r}.")
        seen.add(name)
        if entry.get("type") == "directory" and name.startswith(prefix):
            opened.append(name)
    return errors
