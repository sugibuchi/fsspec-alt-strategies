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
uv run --with mypy==2.4.1 mypy            # types of both packages' sources (config in pyproject.toml)
```

CI runs ruff, mypy, zizmor, actionlint and pytest on pull requests and pushes to `main`. In the workflows:

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
code: read it before adding a strategy or a handler.

## The traps

### Handlers and filesystems

- A handler is how a storage is accessed. A filesystem enforces the contract between the client code and the storage:
  the public API, paths verified before the storage is accessed, monitoring, and the client side's cache updated and
  dropped. Put each concern on its side: never a contract's part in a handler, nor a storage's access in the
  filesystem.
- A storage is a handler, which a developer writes: its primitives, paths, profile, limits and clients. It subclasses
  its family's, `TreeFileOperationHandler` or `FlatFileOperationHandler`, which declares the family's primitives and
  runs its strategies. The strategies take handlers, never filesystems.
- A `FileOperationHandler`, in `common/handler.py`, is a `StorageHandler` with the `Strategy` mixin's methods, `walk`,
  `find`, `glob`, `du`, `rm`, `rm_files`, `cp` and `mv`, in `common/strategy.py`. Each takes its strategy's arguments
  but the handler, which is `self`, and returns an operation not started. A family's handler implements them by its
  strategies, e.g. `rm` by `_tree_rm`. A storage's handler overrides a method to choose another strategy.
- `rm_files(entries)` deletes files a caller found: by default `common.rm_files._rm_each_file`, each by `rm_file`. A
  storage deleting many by one request overrides it with `_rm_by_request`, and declares `rm_batch_size`, 1 by
  default. The recursive deletions and moves send it their files in batches, by `_Jobs.send_batch`.
- `BaseFileSystem` implements fsspec's API on a `FileOperationHandler`, a required argument. Nobody overrides its methods. Its recursive methods, such as `_rm` and `_copy`, run the handler's strategy, e.g. its `rm`, and answer as
  fsspec's method does.
- A filesystem keeps its handler private, as `self._handler`, so that no caller gets an operation from it. It copies
  nothing from its handler: no profile, no limit, no `rm_batch_size`. It reads them from `self._handler` where it uses
  them.
- The filesystems' and the router's `__init__` name fsspec's options, `asynchronous`, `loop` and the `dircache`'s, so
  that any other raises TypeError. Never take `**kwargs` there, nor fsspec's `batch_size`.
- A storage's own class, by convention, writes its `__init__`, its signature naming each of its handler's options and
  passing fsspec's on, with no helper splitting them. Its `**` is fsspec's, or else options another library names,
  e.g. a token credential's, the handler passing on those its constructor names. It makes its handler from them. It names
  `protocol`, `root_marker`, `_strip_protocol` and `_get_kwargs_from_urls` as its handler class's `protocol`, `root`,
  `strip_protocol` and `options_from_url`, and overrides nothing else. There is no
  metaclass, no `__init_subclass__` and no decorator reflecting on a class.
- `FileSystemWrapper` alone takes its handler's paths at runtime, and alone is not cached: it serves a wrapped
  filesystem. The router makes its storages' `BaseFileSystem`s uncached.
- A primitive takes an entry for an item that exists, and a `StoragePath` for one to be made. `open_file` and
  `open_async_file` take an entry whatever the mode, holding only the name of a file to write. `open_file` gives
  Python's sync API, and `open_async_file` fsspec's async one: the strategies use only the latter. Async adapters
  and transactions are each handler's to choose. Build paths only by the
  handler's `verified`, `entry`, `join` and `parent`, which verify them. Never concatenate strings.
- A filesystem passes its caller's paths to `verified` or `entry` as given, never stripped first: `verify_path` sees
  a URL whole, e.g. abfs's account, and `verified` strips it.
- A name a listing gave is never verified again. Pass a listing's entries on whole, never wrapped again from their
  names: a storage may need their size or id.

### Strategies

- A strategy takes the handler it works on, never its methods as functions: `_tree_rm(handler, path, maxdepth=...)`.
  A copy or move takes `src` and `dst`, handlers, the same within one storage.
- The only functions a strategy takes are a copy's `cp_entry`, and a move's `cp_entry` and `mv_entry`, each a
  `common.jobs._Transfer` taking the source's entry. Between two storages, `cp_entry` defaults to
  `common.transfer._across`, which streams the data and leaves no part of a file. A move's `mv_entry` defaults to
  `cp_entry`, then the source's `rm_file`. The router passes its routing handler's `cp_across` and `mv_across` so.
- A copy or move tells `makedirs`, `cp_file`, `mv_file` and its `cp_entry` or `mv_entry` `missing=True` where
  `_CopyJobs.known_missing` holds: the destination, or a directory above it, was found missing. A write's entry holds
  its final `"size"` where the caller knows it. Every override of those primitives takes `missing`, used or not.
- A strategy calls only a handler's primitives, such as `ils`, `info` and `rm_file`, never a filesystem's method: the
  filesystem is the caller's.
- Strategies use only fsspec's API that every implementation has. Never special-case an implementation's class.
- Every strategy follows DEVELOPMENT.md's "Operations":
  - paths come without a protocol, verified;
  - it returns a private `Operation`, an `OperationHandler` working as it is iterated. Only a
    `FileOperationHandler`'s strategy methods return one, and the filesystem keeps its handler private: callers watch
    operations with a `Monitor`, which stops one through its `OperationHandler`;
  - the operation yields what it did, mixed with an exception per failed path, each noting its path;
  - the call raises argument errors, and the iteration only errors ending the whole operation, such as a missing root;
  - its stopping mirrors `asyncio.Task`: `cancel()` starts nothing new but yields what is in flight, and `aclose()`
    stops at once, waiting for writes in flight. There are no callbacks and no result classes.
- A strategy's name tells its family, e.g. `_tree_rm`, never a bare `_rm`: that is the method's name.
- A strategy replacing an fsspec method answers as it does. Where it departs on purpose, its docstring says why.
- What a storage does that fsspec leaves open is declared in `profile.StorageProfile`, never detected. A strategy
  needing it awaits its handler's `open()`, then reads `handler.profile`, as its operation starts, e.g. in its jobs'
  `setup`. The default, `UNKNOWN`, is `StorageProfile()`: a strategy must be correct with it on any storage. A copy or
  move reads the destination's. Add a field only when a strategy saves requests or data copied with it.
- A storage has one profile, which never changes. The handler's `load_profile()` and `open()` take no path, so that
  the API enforces it. A storage whose behaviour differs by path, e.g. per bucket or container, is several handlers
  under a router. Never add a hook taking a path to tell them apart.
- The callers hold the limits' permits, never the primitives: the strategies hold `operation_limit` around each
  primitive. A job running another strategy's operation, e.g. a batch's `rm_files`, holds none: the operation's
  requests take theirs. Data holds `io_limit` where its requests are sent: the files per part, and a handler's own `cat_file`,
  `pipe_file`, `get_file` or `put_file` per request. Their defaults, through `open_async_file`, hold none.
- The filesystem calls only what `StorageHandler` declares. Never find a handler's method by `getattr`: declare it
  on `StorageHandler`, with a default that works.
- The library never chooses a strategy for the developer: the developer chooses a family's handler, and overrides a
  strategy method to choose another. Each docstring says what the strategy assumes of the storage, and when it wins
  or loses.
- fsspec's `dircache` is for fsspec's own methods and callers only. `BaseFileSystem._ls` keeps a listing once it has
  listed it whole. The strategies list by the handler, which never reads or fills it, and `_info` never reads it.
  Every method changing the storage calls `self._drop_listings()` in a `finally`: a new one does too.
- Async first: write the async version, and a sync one only when asked.
- Never swallow `CancelledError`.

### Families

- A family, such as `tree/` or `flat/`, never imports another. What two share goes in `common/`, a module per
  concern. What builds on both sits at the package's root: `wrapper/`, `monitor.py` and `router.py`. The router
  imports no family: each storage's handler runs its own strategies. A handler's class tells its family: a
  `TreeFileOperationHandler` or a `FlatFileOperationHandler`, never an attribute naming it.
- A wrapper of one library, such as `wrapper/s3fs.py`, is a handler, and never imports the library: it calls the
  wrapped filesystem's methods. Write one only where the library saves requests or data beyond fsspec's API.
- A family's modules export nothing public but its handler class, in `handler.py`.
- A tree handler's `ils_recursive` defaults to `tree.listing._tree_ils`, a directory per request by `ils`. A storage
  listing a tree in one request overrides it, and falls back on `_tree_ils` itself where its storage refuses.
- The tree strategies walk with `tree.walk._chunks`, which is private: never expose its `_Listing`. It streams each
  directory in parts: never hold a directory's listing whole, except in `_tree_walk`, which answers as fsspec's `walk`.
- Those that only read build on `tree.find._find_listings`, which handles the root as `_find` does. Those changing
  the storage build on `tree._jobs._TreeJobs`. Never write another loop or root handling.
- The flat strategies list with `flat._keys._listing(handler, prefix)`, the handler's `ils_recursive` of the prefix's
  directory, an async iterator of entries in name order, never pages: see DEVELOPMENT.md's "The flat strategies".
  Those that only read derive directories with `flat._keys`. Those changing the storage build on
  `flat._jobs._FlatJobs`.
- The strategies changing a storage, in any family, run on `common.jobs._Jobs`, and start with `_Jobs.operation`. A
  copy or move also builds on `common.jobs._CopyJobs`, which places its paths by the destination's `join`:
  `_CopyOrMove(_CopyJobs, _TreeJobs)`. Their constructors take keywords only, and pass the rest on.
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
  test on both. Start a strategy on a `TreeView` or `FlatView` of it, a handler which replaces a primitive and sets
  the profile and limit. Assert the requests made, from `fs.stats.calls`.
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
