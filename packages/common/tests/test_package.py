"""Tests of the package itself: what it re-exports."""

from __future__ import annotations

import fsspec_alt_strategies_common
from fsspec_alt_strategies_common import control, profile, states


def test_version() -> None:
    assert fsspec_alt_strategies_common.__version__


def test_the_interface_is_reexported() -> None:
    for module in (control, profile, states):
        for name in module.__all__:
            assert getattr(fsspec_alt_strategies_common, name) is getattr(module, name)
            assert name in fsspec_alt_strategies_common.__all__
