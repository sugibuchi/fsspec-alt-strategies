"""Tests of the package itself."""

from __future__ import annotations

from importlib.metadata import requires

from packaging.requirements import Requirement

import fsspec_alt_strategies_impl


def test_version() -> None:
    assert fsspec_alt_strategies_impl.__version__


def test_the_interface_of_one_minor_version() -> None:
    # The strategies use what their minor version of the interface adds: an
    # older one installed beside them would fail on import. release-impl.yaml
    # checks the bound is the release's own minor version.
    (interface,) = [
        r
        for r in map(Requirement, requires("fsspec-alt-strategies-impl") or [])
        if r.name == "fsspec-alt-strategies"
    ]
    bounds = {s.operator: s.version for s in interface.specifier}
    assert set(bounds) == {">=", "<"}
    major, minor = map(int, bounds[">="].split("."))
    assert bounds["<"] == f"{major}.{minor + 1}"
