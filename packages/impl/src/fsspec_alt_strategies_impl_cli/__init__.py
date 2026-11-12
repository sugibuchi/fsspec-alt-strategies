# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The command line of the strategies: copying them into another package.

Users copy the strategies into their own package, e.g. as ``_contrib/strategy``.
``fsspec-alt-strategies-impl copy`` copies the package as it is installed, so
the copy carries the ``_version.py`` its build wrote, and tells the version it
was copied at. A copy of the source tree would have none.

The copy also carries the package's ``LICENSE``, from its installed metadata.
The Apache License asks that whoever passes the code on gives a copy of it, and
a package under another license would otherwise ship the copy without it.

It is a package of its own, apart from the strategies, so that copies leave it
out. It needs click, which the ``cli`` extra installs:

    uv run --with "fsspec-alt-strategies-impl[cli]" fsspec-alt-strategies-impl copy src/mypkg/_contrib/strategy

This module depends on click, and on `fsspec_alt_strategies_impl`, the package
it copies.
"""

from __future__ import annotations

import shutil
from importlib.metadata import distribution
from pathlib import Path

import click

import fsspec_alt_strategies_impl

_DISTRIBUTION = "fsspec-alt-strategies-impl"
_VERSION_FILE = "_version.py"
_LICENSE_FILE = "LICENSE"


def _license() -> Path:
    """The license file the installed distribution records, in its ``licenses`` folder."""
    dist = distribution(_DISTRIBUTION)
    for file in dist.files or ():
        if file.name == _LICENSE_FILE and file.parent.name == "licenses":
            return Path(str(dist.locate_file(file)))
    raise click.ClickException(
        f"{_DISTRIBUTION} records no {_LICENSE_FILE}: reinstall it"
    )


@click.group()
@click.version_option(fsspec_alt_strategies_impl.__version__, prog_name=_DISTRIBUTION)
def main() -> None:
    """Tools for fsspec-alt-strategies-impl."""


@main.command()
@click.argument("target", type=click.Path(file_okay=False, path_type=Path))
def copy(target: Path) -> None:
    """Copy the installed strategies into TARGET, replacing a previous copy.

    TARGET is a directory of your package, e.g. src/mypkg/_contrib/strategy. It
    must be missing, empty, or a previous copy: one holding _version.py.
    The copy carries the license, as LICENSE: keep it with the code.
    """
    source = Path(fsspec_alt_strategies_impl.__file__).parent
    if not (source / _VERSION_FILE).is_file():
        raise click.ClickException(
            f"{source} has no {_VERSION_FILE}: install the package by building it, e.g. from PyPI"
        )
    license_file = _license()

    if target.exists() and any(target.iterdir()):
        if not (target / _VERSION_FILE).is_file():
            raise click.ClickException(
                f"{target} is neither empty nor a previous copy, with {_VERSION_FILE}: not replaced"
            )
        shutil.rmtree(target)

    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    shutil.copyfile(license_file, target / _LICENSE_FILE)
    click.echo(
        f"Copied {_DISTRIBUTION} {fsspec_alt_strategies_impl.__version__} to {target}"
    )
