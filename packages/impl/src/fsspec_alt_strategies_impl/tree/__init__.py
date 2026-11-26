# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Strategies that traverse any storage as a tree, listing several directories at a time.

They list a directory per request, or a subtree nothing prunes by the
handler's ``ils_recursive``, where the profile declares ``recursive_ls``: so
they suit storages with a hierarchical namespace and object stores alike. Each module holds one strategy, e.g.
``fsspec_alt_strategies_impl.tree.find``. `handler.TreeFileOperationHandler`
runs them all, as a filesystem's recursive methods. This package imports none
of them.
"""
