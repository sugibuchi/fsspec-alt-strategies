# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Strategies that walk a directory tree, listing several directories at a time.

They suit storages with a hierarchical namespace, which list by directory. They
list a whole tree by the handler's ``ils_recursive``, and a tree they prune a
directory per request. Each module holds one strategy, e.g.
``fsspec_alt_strategies_impl.tree.find``. `handler.TreeFileOperationHandler`
runs them all, as a filesystem's recursive methods. This package imports none
of them.
"""
