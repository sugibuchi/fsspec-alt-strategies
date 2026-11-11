# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Python goes through uv: never call `python` or `pip` directly.

```bash
uv sync                                   # both packages, editable, with dev dependencies
uv run pytest                             # the tests of both, from the workspace root
uv run pytest packages/impl/tests/test_package.py -k version  # one file / matching tests
uv run pytest --cov                       # with coverage of both packages
uv build --all-packages                   # both distributions
uvx ruff check .                          # lint (ruff is not a dev dependency; config in pyproject.toml)
uvx ruff format .                         # format
```

CI runs ruff, zizmor, actionlint and pytest on pull requests and pushes to `main`. In the workflows:

- Pin every action to a full commit SHA, with its version in a comment, e.g. `@<sha>  # v7.0.1`. Dependabot updates
  them. Also pin uv's `version`.
- Check out with `persist-credentials: false`: no job pushes.
- Check changes with `uvx zizmor --offline .github/workflows` and `uvx --from actionlint-py actionlint`.

## What this project is

A third-party library of alternative, filesystem-independent strategies for fsspec. It is a uv workspace of two
packages, each built and published on its own:

- `packages/common`, `fsspec_alt_strategies`: the interface the strategies share with their callers. It is installed,
  never copied.
- `packages/impl`, `fsspec_alt_strategies_impl`: the strategies, copied into the projects using them. Two packages
  beside it are never copied:
  - `fsspec_alt_strategies_impl_testing`, the simulated filesystem;
  - `fsspec_alt_strategies_impl_cli`, the click command line copying the strategies, with the `cli` extra.

The workspace root holds the tooling and the docs. DEVELOPMENT.md's Design section has the choices that shape the
code: read it before adding a strategy.

## The traps

### Strategies

- The abstract classes, `TreeFileSystem` and `FlatFileSystem`, and the wrappers are the API. Their recursive methods,
  such as `_rm` and `_copy`, start a private strategy, e.g. `_tree_rm`, and answer as fsspec's method does.
- A strategy takes the filesystem it works on, never its methods as functions: `_tree_rm(fs, path, maxdepth=...)`. A
  copy or move takes `src_fs` and `dst_fs`, the same within one storage. It reads the primitives, `operation_limit`
  and `storage_profile` from them.
- The only function a strategy takes is a copy's or a move's `cp_entry` or `mv_entry`, a `common.jobs._Transfer`,
  taking the source's entry. The router passes its copy of a file between two storages so. Between two storages it is
  required.
- A copy or move hands each file's entry, as its listing found it, to `_cp_file_entry` or `_mv_file_entry`. Never drop it
  for the path alone: a storage may need its size.
- A strategy calls only a storage's primitives, such as `_ils`, `_info` and `_rm_file`, never a high-level method such
  as `_find` or `_glob`: another strategy may have replaced it.
- Strategies use only fsspec's API that every implementation has. Never special-case an implementation's class.
- Every strategy follows DEVELOPMENT.md's "Operations":
  - paths come without a protocol;
  - it returns a private `Operation`, an `OperationHandler` working as it is iterated. No public method returns one:
    callers watch operations with a `Monitor`, which stops one through its `OperationHandler`;
  - the operation yields what it did, mixed with an exception per failed path, each noting its path;
  - the call raises argument errors, and the iteration only errors ending the whole operation, such as a missing root;
  - its stopping mirrors `asyncio.Task`: `cancel()` starts nothing new but yields what is in flight, and `aclose()`
    stops at once, waiting for writes in flight. There are no callbacks and no result classes.
- A strategy's name tells its family, e.g. `_tree_rm`, never a bare `_rm`: that is the method's name.
- A strategy replacing an fsspec method answers as it does. Where it departs on purpose, its docstring says why.
- What a storage does that fsspec leaves open is declared in `profile.StorageProfile`, never detected. A strategy reads
  it with `await fs._get_storage_profile()`, in its jobs' `setup`, so `StorageProfile()` must be correct on any
  storage. A copy or move reads the destination's. Add a field only when a strategy saves requests or data copied with
  it.
- A filesystem has one profile, which never changes. A storage whose behaviour differs by path, e.g. per bucket or
  container, is several filesystems under a router. Never add a strategy hook taking a path to tell them apart.
- The library never chooses a strategy for the developer. Each docstring says what the strategy assumes of the
  storage, and when it wins or loses.
- Async first: write the async version, and a sync one only when asked.
- Never swallow `CancelledError`.

### Families

- A family, such as `tree/` or `flat/`, never imports another. What two share goes in `common/`, a module per
  concern. What builds on both sits at the package's root: `wrapper.py` and `router.py`. The router imports a
  family only when copying or moving from a storage of it, as its profile's `fs_type` says.
- A family's modules export nothing public but its filesystem class, in `filesystem.py`.
- The tree strategies walk with `tree.walk._chunks`, which is private: never expose its `_Listing`. It streams each
  directory in parts: never hold a directory's listing whole, except in `_tree_walk`, which answers as fsspec's `walk`.
- Those that only read build on `tree.find._find_listings`, which handles the root as `_find` does. Those changing
  the storage build on `tree._jobs._TreeJobs`. Never write another loop or root handling.
- The flat strategies list with `flat._keys._listing(fs, prefix)`, a recursive `_ils` of the prefix's directory, an
  async iterator of entries in name order, never pages: see DEVELOPMENT.md's "The flat strategies". Those that only read derive directories with `flat._keys`. Those changing the
  storage build on `flat._jobs._FlatJobs`.
- The strategies changing a storage, in any family, run on `common.jobs._Jobs`, and start with `_Jobs.operation`. A
  copy or move also builds on `common.jobs._CopyJobs`, which places its paths: `_CopyOrMove(_CopyJobs, _TreeJobs)`.
  Their constructors take keywords only, and pass the rest on.

### Packages

- Never use names suggesting ownership of fsspec's fundamentals, such as "base" or "core" packages of fsspec.
- `fsspec_alt_strategies` holds only what callers use and every copy must share: one `OperationHandler`, one set of
  monitors, one `StorageProfile`. It depends on fsspec and the standard library alone, and never imports the
  strategies. Its `__init__.py` re-exports its public API.
- The copied package imports itself only relatively, and depends only on fsspec, `fsspec_alt_strategies` and the
  standard library. Its `__init__.py` re-exports `monitored` only, and no other `__init__.py` re-exports anything.
- Never move anything for tests only into `fsspec_alt_strategies_impl`: it goes in
  `fsspec_alt_strategies_impl_testing`. Only the CLI package may use click.
- Each file in `packages/*/src/` starts with the SPDX license header, and a module docstring ending with the modules
  it depends on. `test_copyable.py` checks the header and the import rules.
- `_version.py` is written by hatch-vcs and git-ignored: never edit or commit it.

### Tests

- Test directories have no `__init__.py`. Never import one test file from another. The interface's tests never import
  the strategies.
- Test strategies on `SimulatedFileSystem`, with the fixtures `flat_fs`, `hierarchical_fs` and `any_fs`, which runs a
  test on both. Start a strategy on a `TreeView` or `FlatView` of it, which replaces a primitive and sets the profile
  and limit. Assert the requests made, from `fs.stats.calls`.
- Test a strategy replacing an fsspec method against fsspec's own, e.g. `AsyncFileSystem._glob(fs, ...)`.

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
- Each doc has its reader:
  - the README tells users what the library offers, without technical details;
  - USAGE.md tells them how to install and use each feature;
  - DEVELOPMENT.md tells contributors the layout and the design.
- After each release, rewrite its GitHub release notes as DEVELOPMENT.md's "Release notes" says, without being
  asked.
- The README is both packages' readme on PyPI, read by each package's `hatch_build.py`. It links files by absolute
  URL, as PyPI cannot resolve relative paths.
