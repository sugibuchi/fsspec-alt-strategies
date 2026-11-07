# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Python goes through uv (never call `python`/`pip` directly).

```bash
uv sync                                   # the package, editable, with dev dependencies
uv run pytest                             # the tests
uv run pytest tests/test_package.py -k version  # one file / matching tests
uv run pytest --cov                       # with coverage of src/fsspec_alt_strategies
uvx ruff check .                          # lint (ruff is not a dev dependency; config in pyproject.toml)
uvx ruff format .                         # format
```

CI, in `.github/workflows/ci.yaml`, runs ruff and pytest on pull requests and pushes to `main`. In the workflows:

- Pin every action to a full commit SHA, with its version in a comment, e.g. `@<sha>  # v7.0.1`. Dependabot updates
  them. Also pin uv's `version`.
- Check out with `persist-credentials: false`: no job pushes.
- Check changes with `uvx zizmor --offline .github/workflows` and `uvx --from actionlint-py actionlint`.

## What this project is

A third-party library of alternative, filesystem-independent strategies for fsspec. README.md lists the planned
features. DEVELOPMENT.md's Design section has the choices that shape the code: read it before adding a strategy.

## The traps

- Never use names suggesting ownership of fsspec's fundamentals, such as "base" or "core" packages of fsspec.
- Each strategy is a stateless function taking a filesystem. Filesystem classes only wrap those functions. File
  objects, in `files/`, and `control.Operation` are the exceptions: see DEVELOPMENT.md's "File objects".
- A strategy calls only the filesystem's primitives, such as `_ls`, `_info` and `_rm_file`. It never calls another
  high-level method, such as `_find` or `_glob`: another strategy may have replaced it. A strategy needing another
  takes it as a parameter.
- Every strategy follows DEVELOPMENT.md's "The strategy API":
  - it is a plain function, `strategy(fs, ..., concurrency_limit=...)`, the limit an `asyncio.Semaphore`,, returning an `Operation` that works as it is iterated;
  - the operation yields what it did mixed with exception objects, one per failed path, each noting its path;
  - it raises only errors ending the whole operation, such as a missing root;
  - its stopping mirrors `asyncio.Task`: `cancel()` starts nothing new but yields what is in flight, and `aclose()`
    stops at once, waiting for writes in flight. There are no callbacks and no result classes.
- The tree strategies, in `tree/`, walk with `tree.walk._listings`, which is private: never expose its `_Listing`.
  Two layers, by what a strategy sends: one that only reads builds on `tree.walk._scan`, which handles the root; one
  that changes the storage builds on `tree._jobs._Jobs`. Never write another loop or root handling.
- What a storage does that fsspec leaves open is declared, never detected, in `profile.StorageProfile`. A strategy
  takes `profile=`, falling back on `fs.storage_profile`, then on unknown. Unknown must be correct on any storage; a
  field only saves requests. Add a field only when a strategy saves requests with it.
- A strategy's name tells its family, e.g. `tree_rm` and `tree_glob`, never a bare `rm` or `glob`. Code using fsspec
  has those names already, as methods and modules.
- Never swallow `CancelledError`.
- The package's `__init__.py` re-exports nothing but `__version__`, and `tree/__init__.py` nothing at all. Users import
  from each module, so a copy can leave out modules it does not use.
- Strategies use only fsspec's API that every implementation has. Never special-case an implementation's class.
- The library never chooses a strategy for the developer. Each strategy's docstring says what it assumes of the
  storage, and when it wins or loses.
- Async first: write the async version; a sync one only when asked.
- Users copy the modules into their own package. So the package imports itself only with relative imports, and depends
  on nothing but fsspec and the standard library.
- `SimulatedFileSystem` lives in its own top-level package, `fsspec_alt_strategies_testing`, so that copies of the
  strategies leave it out. Never move it, or anything for tests only, into `fsspec_alt_strategies`.
- Each file in `src/` starts with the SPDX license header, as copies must keep it. `tests/test_copyable.py` checks
  this and the import rules.
- `_version.py` is written by hatch-vcs and git-ignored: never edit or commit it.
- Test folders have no `__init__.py`. Never import one test file from another.
- Test strategies on `SimulatedFileSystem`, with the fixtures of `tests/conftest.py`: `flat_fs`, `hierarchical_fs` and
  `any_fs`, which runs a test on both. Assert the requests made, from `fs.stats.calls`.
- A strategy replacing an fsspec method answers as it does. Test it against fsspec's own, e.g.
  `AsyncFileSystem._glob(fs, ...)`: the subclass's method may be overridden.

## Conventions

### Code

- Comments and docstrings are written in full sentences explaining *why*. Keep that density and tone.
- `from __future__ import annotations` everywhere, with type-only imports under `TYPE_CHECKING`: ruff's `TC` rules.
- Public API docstrings are in numpy style: `Parameters`, `Returns`, `Raises`, `See Also`, `Examples`. Internal
  docstrings stay prose.
- The public API is what the package's `__all__` exports. Keep machinery out of it.

### Writing docs

Write docs so that people can review them. That covers this file, DEVELOPMENT.md and the README.

- One idea per sentence: about 30 words, never more than 50.
- At most one parenthesis per sentence, never nested.
- Three or more items make a list.
- Describe what is, not what was: git keeps the history.
- The README uses absolute URLs for links to files, as PyPI cannot resolve relative paths.
