# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Strategies that list a flat namespace by key prefix, as object stores do.

They suit storages without real directories, which list every key under a prefix
in one paginated request. They take that listing from the handler's
``ils_recursive``, an async iterator of the entries in order of their names.
Each module holds one strategy, e.g. ``fsspec_alt_strategies_impl.flat.find``.
`handler.FlatFileOperationHandler` runs them all, as a filesystem's recursive
methods. This package imports none of them.
"""
