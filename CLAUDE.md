# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Python goes through uv (never call `python`/`pip` directly).

```bash
uv sync                                   # both packages, editable, with dev dependencies
uv run pytest                             # the tests of both, from the workspace root
uv run pytest packages/impl/tests/test_package.py -k version  # one file / matching tests
uv run pytest --cov                       # with coverage of both packages
uv build --all-packages                   # both distributions
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
features. It is a uv workspace of two packages, each built and published on its own:

- `packages/common`, `fsspec_alt_strategies`: the interface the strategies share with their callers, in
  `monitor.py`, `profile.py` and `states.py`. It is installed, never copied.
- `packages/impl`, `fsspec_alt_strategies_impl`: the strategies, copied into the projects using them, and
  `fsspec_alt_strategies_impl_testing`, the simulated filesystem.

The workspace root holds the tooling: ruff, pytest and coverage settings, CI, and the docs. DEVELOPMENT.md's Design section has the choices that shape the code: read it before adding a strategy.

## The traps

- Never use names suggesting ownership of fsspec's fundamentals, such as "base" or "core" packages of fsspec.
- Each strategy is a stateless function taking the storage's operations as functions, never a filesystem: e.g.
  `ils=fs._ils`. Filesystem classes only wrap those functions. File objects, in `files/`, and `common.operation.Operation` are the
  exceptions: see DEVELOPMENT.md's "File objects".
- A strategy takes only primitive operations, such as `ils`, `info` and `rm_file`, called as fsspec calls its own,
  e.g. `info(path)`. A listing is an async iterator of entries, `ils(path)`, as fsspec's `_ls` gives them. It never takes a high-level one, such as `_find` or `_glob`: another strategy may have
  replaced it. A strategy needing another takes it as a parameter. Paths come without a protocol, as the operations
  take them.
- Every strategy follows DEVELOPMENT.md's "The strategy API":
  - it is a plain function, `strategy(paths, *, ils=..., ..., concurrency_limit=...)`, the operations keyword-only and
    required, the limit an `asyncio.Semaphore`, returning an `Operation` that works as it is iterated;
  - the operation yields what it did mixed with exception objects, one per failed path, each noting its path;
  - it raises only errors ending the whole operation, such as a missing root;
  - its stopping mirrors `asyncio.Task`: `cancel()` starts nothing new but yields what is in flight, and `aclose()`
    stops at once, waiting for writes in flight. There are no callbacks and no result classes.
- The tree strategies, in `tree/`, walk with `tree.walk._chunks`, which is private: never expose its `_Listing`. It streams each folder in parts: never hold a folder's listing whole, but in `tree_walk`, which answers as fsspec's `walk` does.
  Two layers, by what a strategy sends: one that only reads builds on `tree.find._find_listings`, which handles the root as `_find` does; one
  that changes the storage builds on `tree._jobs._TreeJobs`. Never write another loop or root handling.
- The strategies changing a storage, in any family, run on `common.jobs._Jobs`, fed by the family's source: listings
  for `tree`, keys for `flat`. A copy or move places its paths with `common.placement._Placing`.
- A family of strategies, such as `tree/` or `flat/`, never imports another. What two share goes in `common/`, a
  module per concern, e.g. `common/patterns.py` for glob patterns. What builds on both families sits at the package's
  root: `wrapper.py`, the wrappers, and `router.py`, the router, which imports a family's strategies only when
  copying or moving from a storage of it, as its profile's `fs_type` says.
- The flat strategies, in `flat/`, list with `iexpand(prefix)`, an async iterator of entries in order of their names, never pages: see
  DEVELOPMENT.md's "The flat strategies". Those that only read derive folders from keys with `flat._keys`.
  `flat.find._find_entries` handles the root as fsspec's `_find` does, and `flat_walk` as `tree_walk` does.
- What a storage does that fsspec leaves open is declared, never detected, in `profile.StorageProfile`. A strategy
  takes `profile=`, required: `StorageProfile()` is unknown. Unknown must be correct on any storage; a
  field only saves requests or data copied. Add a field only when a strategy saves either with it.
- A strategy's name tells its family, e.g. `tree_rm` and `tree_glob`, never a bare `rm` or `glob`. Code using fsspec
  has those names already, as methods and modules.
- Never swallow `CancelledError`.
- `fsspec_alt_strategies` holds only what callers use, and what every copy must share: one `OperationHandler`, one
  set of monitors, one `StorageProfile`. The machinery stays in the copied package, e.g. `common/operation.py`. The
  interface depends on nothing but fsspec and the standard library, and never imports the strategies.
- `fsspec_alt_strategies/__init__.py` re-exports its public API. In `fsspec_alt_strategies_impl`, the package's
  `__init__.py` re-exports `monitored` only, and no other `__init__.py` re-exports anything: users import from each
  module, so a copy can leave out modules it does not use.
- Strategies use only fsspec's API that every implementation has. Never special-case an implementation's class.
- The library never chooses a strategy for the developer. Each strategy's docstring says what it assumes of the
  storage, and when it wins or loses.
- Async first: write the async version; a sync one only when asked.
- Users copy the modules into their own package. So the package imports itself only with relative imports, and depends
  on nothing but fsspec, `fsspec_alt_strategies` and the standard library.
- `SimulatedFileSystem` lives in its own top-level package, `fsspec_alt_strategies_impl_testing`, so that copies of the
  strategies leave it out. Never move it, or anything for tests only, into `fsspec_alt_strategies_impl`.
- Each file in `packages/*/src/` starts with the SPDX license header, as copies must keep it.
  `packages/impl/tests/test_copyable.py` checks this and the import rules for the copied package.
- `_version.py` is written by hatch-vcs and git-ignored: never edit or commit it.
- Test folders have no `__init__.py`. Never import one test file from another. The interface's tests never import
  the strategies.
- Test strategies on `SimulatedFileSystem`, with the fixtures of `packages/impl/tests/conftest.py`: `flat_fs`, `hierarchical_fs` and
  `any_fs`, which runs a test on both. Assert the requests made, from `fs.stats.calls`.
- A strategy replacing an fsspec method answers as it does. Test it against fsspec's own, e.g.
  `AsyncFileSystem._glob(fs, ...)`: the subclass's method may be overridden.

## Conventions

### Code

- Comments and docstrings are written in full sentences explaining *why*. Keep that density and tone.
- `from __future__ import annotations` everywhere, with type-only imports under `TYPE_CHECKING`: ruff's `TC` rules.
- Public API docstrings are in numpy style: `Parameters`, `Returns`, `Raises`, `See Also`, `Examples`. Internal
  docstrings stay prose.
- The public API is what each module's `__all__` exports. Keep machinery out of it.

### Writing docs

Write docs so that people can review them. That covers this file, DEVELOPMENT.md, USAGE.md and the README.

- One idea per sentence: about 30 words, never more than 50.
- At most one parenthesis per sentence, never nested.
- Three or more items make a list.
- Describe what is, not what was: git keeps the history.
- The README uses absolute URLs for links to files, as PyPI cannot resolve relative paths. It is both packages' readme on PyPI, read by each package's `hatch_build.py`: they have none of their own.
