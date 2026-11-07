"""Tests of the package itself."""

from __future__ import annotations

import fsspec_alt_strategies
from fsspec_alt_strategies import control, profile


def test_version() -> None:
    assert fsspec_alt_strategies.__version__


def test_the_shared_interface_is_reexported() -> None:
    for module in (control, profile):
        for name in module.__all__:
            assert getattr(fsspec_alt_strategies, name) is getattr(module, name)
            assert name in fsspec_alt_strategies.__all__
