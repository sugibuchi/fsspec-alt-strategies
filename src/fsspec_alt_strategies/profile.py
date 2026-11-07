# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What a storage does that fsspec's API leaves open, as the developer declares it.

Storages answer the same request differently. Some create a file's missing
parent folders, others raise FileNotFoundError. A strategy works on any of
them, but sends fewer requests when it knows. `StorageProfile` tells it.

A strategy taking a ``profile`` uses `StorageProfile()`, which knows nothing,
if given None. A filesystem wrapping strategies may hold one for all its calls.

The library never detects a storage's behaviour: the developer declares it. A
field left None is unknown, and a strategy then takes the way that is right on
any storage. So a profile only saves requests, and never changes the outcome,
unless it is wrong.

A fact gets a field only once a strategy saves requests with it. Facts that
cost nothing to handle as they come, such as a write refused because the file
exists, have none.

This module depends on nothing else in the package.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["StorageProfile"]


@dataclass(frozen=True, kw_only=True)
class StorageProfile:
    """What a storage does that fsspec's API leaves open.

    Each strategy's docstring says the fields it reads, and the requests each
    saves.

    Parameters
    ----------
    parents_required : bool, optional
        Whether writing a file into a missing folder fails, rather than
        creating its parents. Unknown by default, taken as True: folders are
        created before files are written into them.

    Examples
    --------
    >>> profile = StorageProfile(parents_required=False)
    >>> op = tree_cp(
    ...     "data/2024",
    ...     "archive/2024",
    ...     ls=fs._ls,
    ...     info=fs._info,
    ...     cp_file=fs._cp_file,
    ...     makedirs=fs._makedirs,
    ...     rm_file=fs._rm_file,
    ...     profile=profile,
    ... )
    """

    parents_required: bool | None = None
