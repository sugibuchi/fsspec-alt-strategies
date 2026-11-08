# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The command line of the strategies: copying them into another package.

Users copy the strategies into their own package, e.g. as ``_contrib/strategy``.
``fsspec-alt-strategies-impl copy`` copies the package as it is installed, so
the copy carries the ``_version.py`` its build wrote, and tells the version it
was copied at. A copy of the source tree would have none.

It is a package of its own, apart from the strategies, so that copies leave it
out. It needs click, which the ``cli`` extra installs:

    uv run --with "fsspec-alt-strategies-impl[cli]" fsspec-alt-strategies-impl copy src/mypkg/_contrib/strategy
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click

import fsspec_alt_strategies_impl

_VERSION_FILE = "_version.py"


@click.group()
@click.version_option(
    fsspec_alt_strategies_impl.__version__, prog_name="fsspec-alt-strategies-impl"
)
def main() -> None:
    """Tools for fsspec-alt-strategies-impl."""


@main.command()
@click.argument("target", type=click.Path(file_okay=False, path_type=Path))
def copy(target: Path) -> None:
    """Copy the installed strategies into TARGET, replacing a previous copy.

    TARGET is a folder of your package, e.g. src/mypkg/_contrib/strategy. It
    must be missing, empty, or a previous copy: one holding _version.py.
    """
    source = Path(fsspec_alt_strategies_impl.__file__).parent
    if not (source / _VERSION_FILE).is_file():
        raise click.ClickException(
            f"{source} has no {_VERSION_FILE}: install the package by building it, e.g. from PyPI"
        )

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
    click.echo(
        f"Copied fsspec-alt-strategies-impl {fsspec_alt_strategies_impl.__version__} to {target}"
    )
