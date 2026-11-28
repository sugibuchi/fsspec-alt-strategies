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
  `TreeFileOperationHandler`, which declares the optional primitives, such as `rm_tree`, and runs the strategies. One
  class serves storages with a hierarchical namespace and object stores alike: the profile tells them apart.
- A `FileOperationHandler`, in `common/handler.py`, is a `StorageHandler` with the `Strategy` mixin's methods, in
  `common/strategy.py`: `walk`, `find`, `glob`, `du`, `rm`, `rm_files`, `cp`, `cp_files`, `mv` and `mv_files`. Each
  takes its strategy's arguments but the handler, which is `self`, and returns an operation not started.
  `TreeFileOperationHandler` implements them by the tree strategies, e.g. `rm` by `_tree_rm`. The batch strategies,
  `rm_files`, `cp_files` and `mv_files`, keep the mixin's defaults.
- How the strategies run in bulk is the `Strategy` mixin's, never `StorageHandler`'s: `rm_batch_size`,
  `cp_batch_size` and `mv_batch_size` are declared there, each 1 by default. A primitive acts on one item.
- `rm_files(entries)` deletes files a caller found: by default `common.rm_files._rm_each_file`, each by `rm_file`. A
  storage deleting many by one request overrides it with `rm_by_request`, and declares `rm_batch_size`.
  `_flat_rm` sends it its keys in batches, and so do the moves copying within a storage that declares
  `renames_files=False`. They read the size once the handler is open.
- `rm_by_request`, `cp_by_request` and `mv_by_request`, in `common/`, and `SyncClientSupport.run_in_thread` are public:
  they are what a storage's handler calls. The strategies stay private.
- `cp_files(transfers)` and `mv_files(transfers)` copy and move `FileTransfer`s within one storage, by default
  `common.cp_files._cp_each_file` and `_mv_each_file`. A storage overrides them with `cp_by_request` or
  `mv_by_request`, and declares `cp_batch_size` or `mv_batch_size`. The recursive copies and moves within one storage
  send them their files in batches, as the traversal plans them, unless the caller gives its own `cp_entry` or
  `mv_entry`. Each error is yielded as the storage raised it, the state's `current` naming its source.
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
  - the options fsspec's method takes, such as `walk`'s `on_error`;
  - a copy's `cp_entry`, and a move's `cp_entry` and `mv_entry`, each a `common.transfer._Transfer` taking the source's
    entry;
  - the `request` of `rm_by_request`, `cp_by_request` and `mv_by_request`, the storage's request on a batch.
- Between two storages, `cp_entry` defaults to `common.transfer._across`, which streams the data and leaves no part of
  a file. A move's `mv_entry` defaults to `cp_entry`, then the source's `rm_file`. The router passes its routing
  handler's `cp_across` and `mv_across` so.
- A copy or move tells `makedirs`, `cp_file`, `mv_file` and its `cp_entry` or `mv_entry` two hints:
  - `missing=True` where the destination, or a directory above it, was found missing;
  - `parent=`, the destination's parent's entry, as `makedirs` returned it or an `info` found it. `cp_tree` and
    `mv_tree` take it too.

  The copy's `tree.cp._Plan` keeps them, in its `made` and `missing`, which the pipeline adds to.

  `makedirs` returns the entry of the directory made, or None. Every override of those primitives takes `missing` and
  `parent`, used or not.
- Strategies use only fsspec's API that every implementation has. Never special-case an implementation's class.
- Every strategy follows DEVELOPMENT.md's "Operations":
  - paths come without a protocol, verified, typed `StoragePath`: a pattern or a list too, as `filesystem._as_given`
    gives them;
  - it returns a private `Operation`, an `OperationHandler` working as it is iterated. Callers never get one: they
    watch operations with a `Monitor`, which stops one through its `OperationHandler`;
  - the operation yields what it did, mixed with an exception per failed path, as the storage raised it, with no
    note: a walk's state names the directory whose error it yields, as its `current`;
  - the call raises argument errors, and the iteration only errors ending the whole operation, such as a missing root;
  - its stopping mirrors `asyncio.Task`: `cancel()` starts nothing new but yields what is in flight, and `aclose()`
    stops at once, waiting for writes in flight. There are no callbacks and no result classes.
- A strategy's name tells its kind, e.g. `_tree_rm`, never a bare `_rm`: that is the method's name.
- A strategy replacing an fsspec method answers as it does. Where it departs on purpose, its docstring says why.
- The library never chooses a strategy for the developer. The developer declares the handler's profile, and overrides
  a strategy method to choose another. Each docstring says what the strategy assumes of the storage, and when it wins
  or loses.
- What a storage does that fsspec leaves open is declared in `profile.StorageProfile`, never detected. A strategy
  needing it awaits its handler's `open()`, then reads `handler.profile`, as its operation starts. A copy or move
  reads the destination's.
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

### Layout

- The strategies are in `tree/`. What they share with the handlers, the filesystems and the router goes in
  `common/`, a module per concern. What builds on them, `wrapper/` and `monitor.py`, sits at the package's root.
- The router, `RoutingFileSystem` on a `RoutingHandler`, sits at the root too, in `router.py`. It imports no strategy:
  each storage's handler runs its own.
- `tree/` exports nothing public but its handler class, in `handler.py`.
- A wrapper of one library, such as `wrapper/s3fs.py`, is a handler, and never imports the library: it calls the
  wrapped filesystem's methods. Write one only where the library saves requests or data beyond fsspec's API.
- A wrapper of one library declares its profile, limits and batch sizes as class attributes. A `known_implementations`
  entry gives them only to the generic `FsspecTreeHandler` and `FsspecFlatHandler`. The latter wraps an object store:
  its `ils_recursive` lists by `find`.
- A copy or move places its roots as fsspec's `copy` does, by `tree.cp._Plan`, and each path below
  them by the destination's `join`, under its directory's map value.

### Listings

- `ils(entry)` lists a directory's items. `ils_recursive(entry, prefix=False)` lists all below it, by the storage's
  own listing: it raises NotImplementedError by default. Neither takes `**kwargs`.
- A handler may add keyword options of its own to either, each optional, for a feature of its storage: its docstring
  names them. The strategies pass a caller's `ils_kwargs` on as given, so that an option the handler does not take
  raises TypeError.
- `ils_recursive` lists a tree, whatever the storage keeps:
  - each directory once, before all below it, all below it together: depth first, in preorder;
  - every directory, those kept only as the prefixes of keys too;
  - a directory's marker, a key ending with "/", as the directory's entry with `"marker": True`, never as an item.

  A storage listing keys in name order meets it by `common.listing.implied_directories`. A file lists itself, and a
  missing directory raises FileNotFoundError.
- With `prefix=True`, the entry's name is a string all listed start with, e.g. "data/2024-". Nothing missing raises.
- The profile declares what `ils_recursive` does: the strategies can never tell it.
  - `recursive_ls`: it is implemented. A subtree nothing prunes is then streamed by one `ils_recursive`. False, the
    default, each directory is listed on its own.
  - `recursive_ls_with_prefix`: it takes `prefix`. A glob then lists a tree by one listing, narrowed to its pattern's
    literal prefix. It needs `recursive_ls`.
- A handler whose storage refuses its own listing, or where it costs more, declares no `recursive_ls`: the strategies
  then list each directory on their own. The filesystem's `_ils` lists one directory, by `ils`.
- On a storage with `implicit_directories`, a directory has no object of its own but its marker. `rm` and `mv` delete
  a directory by `rmdir` only where its entry says `"marker"`, and a root emptied, which may have one.

### The traversal

- The strategies run on `common.tree.TreeTraversal`, with `common.glob`'s tasks: `VisitDirectory`, a directory per
  listing, and `TraverseSubtree`, a subtree by one `ils_recursive`. `tree.walk._root_task` chooses between them by
  the profile. Never write another walk.
- What a strategy does on each path is a `common.glob.PathAction`: `map` before what is below a path, `descends`
  to leave a directory unlisted, `visit_child` to prune each item then listed, and `reduce` reading the values below
  it as they end. A directory an action takes whole, e.g. by `rm_tree`, is a `descends` of False, never a task apart. A task holds a permit of
  `operation_limit` around each page of a listing, never across it.
- A write goes through `common.tree._write`, under a permit checked against the operation's `cancelling()`. So none
  starts once cancelled, and a close waits for those in flight.
- A root's failure to list is raised by the traversal. Each strategy decides what it means, as its fsspec method does.
- A recursive listing that breaks off fails its whole subtree. Nothing is listed again: a storage resumes its own
  listing, e.g. a page by its continuation token.
- Those that only read build on `tree.find._read`, which handles the root as `_find` does.
- rm has two strategies, in `tree/rm.py`. A handler chooses one by overriding `rm`:
  - `_tree_rm`, for real directories, deletes in its map-reduce: each file as it is listed, and each directory in its
    `reduce`, after all below it. It lists by `VisitDirectory` alone, a directory per `ils`: the traversal gives the
    order and the parallelism;
  - `_flat_rm`, for an object store, only lists in its map-reduce, by `TraverseSubtree` alone, one
    `ils_recursive` per root, and emits the keys. Its operation deletes them in
    batches of `rm_batch_size`, in no order, and then the markers.

  `FsspecFlatHandler` and `FlatView` choose `_flat_rm`. Never make `_tree_rm` batch, nor `_flat_rm` keep an order.
- `cp` and `mv` each have a `PathAction` of their own, which only plans. Their operation consumes its events in
  batches, by `tree.rm._consumed`. The traversal holds at least a batch of events unread.
- cp and mv schedule, then run, in `tree/cp.py`:
  - `_Plan` reads the profile and places the roots, before the traversal starts;
  - `_Schedule`, the `PathAction`, emits operations as dicts, e.g. `{"op": "mkdir_dst", ...}` or `{"op": "cp_file",
    ...}`. Which ones depends on the task, the profile and whether both are one storage;
  - `_Pipeline` runs each batch of them through four stages, each with a batch size: `_MakeDirs`, `_SendFiles`,
    `_RemoveFiles` and `_RemoveDirs`. Each stage runs its operations, and passes the rest on.

  A decision about what to do goes in `_Schedule`, and how to send it in a stage.
- A move within a storage declaring `renames_files=False` copies its files in batches, then deletes their sources in
  batches of `rm_batch_size`. Elsewhere, each file moves by `mv_file`, or by its batch strategy.
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
