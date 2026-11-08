"""Fixtures shared by the tests."""

from __future__ import annotations

import pytest

from fsspec_alt_strategies_impl_testing import SimulatedFileSystem


@pytest.fixture
def flat_fs() -> SimulatedFileSystem:
    """An empty async filesystem modelling an object store."""
    return SimulatedFileSystem(asynchronous=True)


@pytest.fixture
def hierarchical_fs() -> SimulatedFileSystem:
    """An empty async filesystem with a hierarchical namespace."""
    return SimulatedFileSystem(hierarchical=True, asynchronous=True)


@pytest.fixture(params=["flat", "hierarchical"])
def any_fs(request: pytest.FixtureRequest) -> SimulatedFileSystem:
    """Each of the two filesystems above, in turn."""
    return SimulatedFileSystem(
        hierarchical=request.param == "hierarchical", asynchronous=True
    )
