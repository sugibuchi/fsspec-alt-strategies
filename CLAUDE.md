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
  - `fsspec_alt_strategies_impl_testing`, the simulated filesystem and the conformance checks;
  - `fsspec_alt_strategies_impl_cli`, the click command line copying the strategies, with the `cli` extra.

The workspace root holds the tooling and the docs. DEVELOPMENT.md's Design section has the choices that shape the
code: read it before adding a strategy or a handler.

## The traps

### Handlers and filesystems

- A handler is how a storage is accessed. A filesystem enforces the contract between the client code and the storage:
  - the public API;
  - paths verified before the storage is accessed;
  - monitoring;
  - the client side's cache, updated and dropped.

  Put each concern on its side: never a contract's part in a handler, nor a storage's access in the filesystem.
- A storage is a handler, which a developer writes: its primitives, paths, profile, limits and clients. It subclasses
  its family's, `TreeFileOperationHandler` or `FlatFileOperationHandler`, which declares the family's primitives and
  runs its strategies.
- A `FileOperationHandler`, in `common/handler.py`, is a `StorageHandler` with the `Strategy` mixin's methods, in
  `common/strategy.py`: `walk`, `find`, `glob`, `du`, `rm`, `rm_files`, `cp`, `cp_files`, `mv` and `mv_files`. Each
  takes its strategy's arguments but the handler, which is `self`, and returns an operation not started. A family's
  handler implements them by its strategies, e.g. `rm` by `_tree_rm`. The batch strategies, `rm_files`, `cp_files`
  and `mv_files`, keep the mixin's defaults, which no family changes.
- How the strategies run in bulk is the `Strategy` mixin's, never `StorageHandler`'s: `rm_batch_size`,
  `cp_batch_size` and `mv_batch_size` are declared there, each 1 by default. A primitive acts on one item.
- `rm_files(entries)` deletes files a caller found: by default `common.rm_files._rm_each_file`, each by `rm_file`. A
  storage deleting many by one request overrides it with `rm_by_request`, and declares `rm_batch_size`. The
  recursive deletions and moves send it their files in batches, by `_Jobs.queue_deletion` and `flush_deletions`.
  `_Jobs.setup` reads the size once the handler is open.
- `rm_by_request`, `cp_by_request` and `mv_by_request`, in `common/`, and `SyncClientSupport.run_in_thread` are public:
  they are what a storage's handler calls. The strategies stay private.
- `cp_files(transfers)` and `mv_files(transfers)` copy and move `FileTransfer`s within one storage, by default
  `common.cp_files._cp_each_file` and `_mv_each_file`. A storage overrides them with `cp_by_request` or
  `mv_by_request`, and declares `cp_batch_size` or `mv_batch_size`. The recursive copies and moves within one storage
  send them their files by `_CopyJobs.release`, unless the caller gives its own `cp_entry` or `mv_entry`. Each error
  they yield has its source's path as its `source`.
- `BaseFileSystem`, in `common/filesystem.py`, implements fsspec's API on a `FileOperationHandler`, a required
  argument. Nobody overrides its fsspec methods. Its recursive methods, such as `_rm` and `_copy`, run the handler's
  strategy, e.g. its `rm`, and answer as fsspec's method does.
- A `BaseFileSystem` keeps its handler private, as `self._handler`, so that no caller gets an operation from it. It
  copies nothing from its handler: no profile, no limit, no `rm_batch_size`. It reads them from `self._handler` where
  it uses them. The router's `RoutingHandler` has no strategies, so `RoutingFileSystem` keeps it as `handler`.
- The filesystems' and the router's `__init__` name fsspec's options, `asynchronous`, `loop` and the `dircache`'s, so
  that any other raises TypeError. Never take `**kwargs` there, nor fsspec's `batch_size`.
- A storage's own class, by convention, writes its `__init__`. Its signature names each of its handler's options and
  passes fsspec's on, with no helper splitting them. Its `**` is fsspec's, or else options another library names,
  e.g. a token credential's: the handler passes on those its constructor names. It makes its handler from them.
- A storage's own class names `protocol`, `root_marker`, `_strip_protocol` and `_get_kwargs_from_urls` as its handler
  class's `protocol`, `root`, `strip_protocol` and `options_from_url`, and overrides nothing else. There is no
  metaclass, no `__init_subclass__` and no decorator reflecting on a class.
- `FileSystemWrapper` alone takes its handler's paths at runtime, and alone sets `cachable = False`: it serves a
  wrapped filesystem. The router makes its storages' `BaseFileSystem`s uncached, by `skip_instance_cache=True`.
- A primitive takes an entry for an item that exists, and a `StoragePath` for one to be made. `open_file` and
  `open_async_file` take an entry whatever the mode. A file to write has an entry holding its name, its final `"size"`
  where the caller knows it, and its `"parent"`'s entry where the caller holds it.
- `open_file` gives Python's sync API, and `open_async_file` fsspec's async one: the strategies use only the latter.
  Async adapters and transactions are each handler's to choose.
- Build paths only by the handler's `verified`, `entry`, `join` and `parent`, which verify them. Never concatenate
  strings.
- A filesystem passes its caller's paths to `verified` or `entry` as given, never stripped first: `verify_path` sees
  a URL whole, e.g. abfs's account, and `verified` strips it.
- A name a listing gave is never verified again. Pass a listing's entries on whole, never wrapped again from their
  names: a storage may need their size or id.
- The filesystem calls only what `StorageHandler` declares. Never find a handler's method by `getattr`: declare it
  on `StorageHandler`, with a default that works.
- fsspec's `dircache` is for fsspec's own methods and callers only. `BaseFileSystem._ls` keeps a listing once it has
  listed it whole. The strategies list by the handler, which never reads or fills it, and `_info` never reads it.
  Every method changing the storage calls `self._drop_listings()` in a `finally`: a new one does too.

### Strategies

- A strategy takes the handler it works on, never a filesystem, nor its methods as functions:
  `_tree_rm(handler, path, maxdepth=...)`. A copy or move takes `src` and `dst`, handlers, the same within one storage.
- A strategy calls only a handler's primitives, such as `ils`, `info` and `rm_file`, never a filesystem's method: the
  filesystem is the caller's.
- The only functions a strategy takes are:
  - the options fsspec's method takes, such as `walk`'s `on_error`, and the walk's `descend`, which prunes it;
  - a copy's `cp_entry`, and a move's `cp_entry` and `mv_entry`, each a `common.jobs._Transfer` taking the source's
    entry;
  - the `request` of `rm_by_request`, `cp_by_request` and `mv_by_request`, the storage's request on a batch.
- Between two storages, `cp_entry` defaults to `common.transfer._across`, which streams the data and leaves no part of
  a file. A move's `mv_entry` defaults to `cp_entry`, then the source's `rm_file`. The router passes its routing
  handler's `cp_across` and `mv_across` so.
- A copy or move tells `makedirs`, `cp_file`, `mv_file` and its `cp_entry` or `mv_entry` two hints:
  - `missing=True` where `_CopyJobs.known_missing` holds: the destination, or a directory above it, was found missing;
  - `parent=`, the destination's parent's entry, as `makedirs` returned it or an `info` found it, kept in
    `_CopyJobs.directories`. `cp_tree` and `mv_tree` take it too.

  `makedirs` returns the entry of the directory made, or None. Every override of those primitives takes `missing` and
  `parent`, used or not.
- Strategies use only fsspec's API that every implementation has. Never special-case an implementation's class.
- Every strategy follows DEVELOPMENT.md's "Operations":
  - paths come without a protocol, verified, typed `StoragePath`: a pattern or a list too, as `filesystem._as_given`
    gives them;
  - it returns a private `Operation`, an `OperationHandler` working as it is iterated. Callers never get one: they
    watch operations with a `Monitor`, which stops one through its `OperationHandler`;
  - the operation yields what it did, mixed with an exception per failed path, each noting its path;
  - the call raises argument errors, and the iteration only errors ending the whole operation, such as a missing root;
  - its stopping mirrors `asyncio.Task`: `cancel()` starts nothing new but yields what is in flight, and `aclose()`
    stops at once, waiting for writes in flight. There are no callbacks and no result classes.
- A strategy's name tells its family, e.g. `_tree_rm`, never a bare `_rm`: that is the method's name.
- A strategy replacing an fsspec method answers as it does. Where it departs on purpose, its docstring says why.
- The library never chooses a strategy for the developer. The developer chooses a family's handler, and overrides a
  strategy method to choose another. Each docstring says what the strategy assumes of the storage, and when it wins
  or loses.
- What a storage does that fsspec leaves open is declared in `profile.StorageProfile`, never detected. A strategy
  needing it awaits its handler's `open()`, then reads `handler.profile`, as its operation starts, e.g. in its jobs'
  `setup`. A copy or move reads the destination's.
- The default profile, `UNKNOWN`, is `StorageProfile()`: a strategy must be correct with it on any storage. Add a
  field only when a strategy saves requests or data copied with it.
- A storage has one profile, which never changes. The handler's `load_profile()` and `open()` take no path, so that
  the API enforces it. A storage whose behaviour differs by path, e.g. per bucket or container, is several handlers
  under a router. Never add a hook taking a path to tell them apart.
- The callers hold the limits' permits, never the primitives: the strategies hold `operation_limit` around each
  primitive. A job running another strategy's operation, e.g. a batch's `rm_files`, holds none: the operation's
  requests take theirs.
- Data holds `io_limit` where its requests are sent: the files per part, and a handler's own `cat_file`, `pipe_file`,
  `get_file` or `put_file` per request. Their defaults, through `open_async_file`, hold none.
- Async first: write the async version, and a sync one only when asked.
- Never swallow `CancelledError`.

### Families

- A family, such as `tree/` or `flat/`, never imports another. What two share goes in `common/`, a module per
  concern. What builds on both, `wrapper/` and `monitor.py`, sits at the package's root.
- The router, `RoutingFileSystem` on a `RoutingHandler`, sits at the root too, in `router.py`. It imports no family:
  each storage's handler runs its own strategies.
- A handler's class tells its family: a `TreeFileOperationHandler` or a `FlatFileOperationHandler`, never an
  attribute naming it.
- A family's modules export nothing public but its handler class, in `handler.py`.
- A wrapper of one library, such as `wrapper/s3fs.py`, is a handler, and never imports the library: it calls the
  wrapped filesystem's methods. Write one only where the library saves requests or data beyond fsspec's API.
- A wrapper of one library declares its profile, limits and batch sizes as class attributes. A `known_implementations`
  entry gives them only to the generic `FsspecTreeHandler` and `FsspecFlatHandler`.
- The strategies changing a storage, in any family, run on `common.jobs._Jobs`, and start with `_Jobs.operation`. A
  copy or move also builds on `common.jobs._CopyJobs`, which places its paths by the destination's `join`, e.g.
  `_CopyOrMove(_CopyJobs, _TreeJobs)`. Their constructors take keywords only, and pass the rest on.

#### The tree family

- A tree handler's `ils_recursive` defaults to its `ils_by_directory`, `tree.listing._tree_ils`: a directory per
  request by `ils`. A storage listing a tree in one request overrides it, and falls back on `ils_by_directory` where
  its storage refuses. Outside `tree/`, call the method, never `_tree_ils`.
- The profile's `recursive_order` says the order of a storage's own listing: the walk can never tell it.
  - Unknown is taken as "parents_first", right on any storage.
  - "depth_first", declared, covers name order too. It lets the walk close each directory as the listing passes it.
  - `_tree_ils`'s listing is a `_DepthFirst`, which the walk reads a directory at a time itself, whatever the profile
    says.
- The tree strategies walk with `tree._walker._TreeWalk`, which is private: never expose its `_Part`. It reads each
  directory in parts: never hold a directory's listing whole, except in `_tree_walk`, which answers as fsspec's `walk`.
- The walk asks for each part, and nothing pushes one: no queue between the readers and it, and no task listing ahead.
  Its directories to read wait on a stack, depth first. A reader holds one permit while it is read, taken by the walk
  before it starts and given back once it ends or pauses. Keep it so: a queue or a task per directory would list
  beyond the walk's control, and past a cancel.
- The walk takes a `tree._visit._Visit`: which directories the operation needs, e.g. within `maxdepth`. What it found
  but did not list, it yields as events: a `_Skipped` subdirectory, a `_FileRoot`, and, when asked, a
  `_SubtreeListed`. A consumer acts on those, and never checks `maxdepth` or catches a file root again.
- How the walk lists what the visit needs is the handler's, by the tree handler's `lists_subtree(entry, pruned=...)`:
  a subtree needed whole by one `ils_recursive` unless it says no, a pruned one only if it says yes. The walk then
  keeps what the visit would list, by path: only a visit whose `by_path` holds allows it. Never decide it in a
  visit, nor by the storage in a strategy.
- Each reader drops a subdirectory its directory listed before, though a storage may list one twice. Beside them, only
  two places keep names alone:
  - a copy's file names per directory, `tree.cp._CopyOrMove.names`, to refuse a second file of one name;
  - a recursive listing's, `tree._walker._OpenDirectory.seen`, to resume a break, as the directories it was in are read
    again. Depth first, it keeps those of the directories the listing is in. Parents first, it keeps all until the
    listing ends.
- Those that only read build on `tree.find._find_listings`, which handles the root as `_find` does. Those changing
  the storage build on `tree._jobs._TreeJobs`. Never write another loop or root handling.

#### The flat family

- The flat strategies list with `flat._keys._listing(handler, prefix)`, the handler's `ils_recursive` of the prefix's
  directory. It is an async iterator of entries in name order, never pages: see DEVELOPMENT.md's "The flat
  strategies". It holds no permit of `operation_limit`.
- Those that only read derive directories with `flat._keys`. Those changing the storage build on
  `flat._jobs._FlatJobs`.

### Packages

- Never use names suggesting ownership of fsspec's fundamentals, such as "base" or "core" packages of fsspec.
- `fsspec_alt_strategies` holds only what callers use and every copy must share, e.g. one `OperationHandler`, one set
  of monitors, one `StorageProfile`. It depends on fsspec and the standard library alone, and never imports the
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
