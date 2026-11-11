# Development

## Layout

The repository is a uv workspace of two packages, each built and published on its own. The root holds the tooling:
ruff, pytest and coverage settings in `pyproject.toml`, CI, and the docs.

| Path                                                    | Contents                                                                   |
|---------------------------------------------------------|----------------------------------------------------------------------------|
| `packages/common/src/fsspec_alt_strategies/`            | The interface, installed once per project                                  |
| `.../monitor.py`                                        | Operations' control, monitors, and the routers a call came through         |
| `.../profile.py`                                        | `StorageProfile`: what a storage does that fsspec leaves open              |
| `.../states.py`                                         | The states the strategies report, e.g. `FindState`                         |
| `.../wrapper.py`                                        | `known_implementations`: how each known filesystem class is wrapped        |
| `packages/impl/src/fsspec_alt_strategies_impl/`         | The strategies, copied into projects                                       |
| `.../tree/`, `.../flat/`                                | The two families: each one's abstract class and strategies                 |
| `.../common/`                                           | What the families share: the base class, the jobs, operations and paths    |
| `.../files/`                                            | File objects, their async adapters, and transactions                       |
| `.../wrapper/`, `.../monitor.py`                        | The generic wrappers, those of s3fs, gcsfs and adlfs, and `monitored()`    |
| `.../router.py`                                         | `AsyncRoutingFileSystem`, spanning several storages                        |
| `packages/impl/src/fsspec_alt_strategies_impl_testing/` | `SimulatedFileSystem` and its views, apart so that copies leave them out   |
| `packages/impl/src/fsspec_alt_strategies_impl_cli/`     | The command line copying the strategies, with the `cli` extra              |
| `packages/*/tests/`                                     | Each package's tests                                                       |
| `packages/*/hatch_build.py`                             | Gives each package the root README.md as its PyPI readme                   |
| `.github/workflows/`                                    | CI, CodeQL, and one release workflow per package                           |

The interface depends on fsspec alone, the strategies on fsspec and the interface. Both take their version from the
git tag, through hatch-vcs.

## Setup and tests

All Python commands go through [uv](https://docs.astral.sh/uv/):

```bash
uv sync                          # both packages, editable, with the dev dependencies
uv run pre-commit install        # ruff and file checks before each commit
uv run pytest --cov              # the tests of both packages, with coverage
uv build --all-packages          # both distributions
uvx ruff check . ; uvx ruff format .
```

CI runs ruff, zizmor and actionlint, and pytest on every supported Python version. When writing tests:

- Run strategies on `SimulatedFileSystem`, with the fixtures `flat_fs`, `hierarchical_fs`, or `any_fs` for both.
  `add_file` and `add_dir` set up files without requests.
- Start a strategy on a `TreeView` or a `FlatView` of the simulator, e.g. `_tree_rm(TreeView(fs), "data")`. A view
  gives the strategy the primitives it calls, and lets a test replace one, e.g. with one that fails. It also sets the
  storage's `storage_profile` and `operation_limit`.
- Assert the requests made, e.g. `fs.stats.calls == {"list": 1, "delete": 3}`: they are what a strategy is for.
- Test a strategy replacing an fsspec method against fsspec's own, e.g. `AsyncFileSystem._glob(fs, ...)`.
- `parents_required` and `overwrite` make the simulator fail as some storages do, to test profiles.
- Async tests need no marker. Test directories have no `__init__.py`, and the interface's tests never import the strategies.

## Design

The choices below shape the whole project. Keep to them, or change them on purpose.

### Principles

The library aims at large cloud storages, such as data lakes holding millions of files across many directories.

- **Streaming first.** `_ils` is the primitive every strategy lists with: an async generator yielding paths as the
  storage is scanned, where fsspec lists every path first with `expand_path`. A strategy acts on each part of a listing
  as it arrives, and never holds a whole directory or tree. Only `_walk` holds a directory whole, as fsspec's `walk`
  answers a directory at a time.
- **Hierarchical namespaces, used.** Where a storage lists by directory, the tree strategies scan directories
  recursively, several at once, e.g. to prune a tree by a pattern. A storage's own operations on a whole tree plug in
  as optional primitives, such as `_rm_tree` and `_mv_tree`.
- **One implementation, many storages.** A storage's behaviour is declared in its `StorageProfile`, never detected, and
  the strategies pick their way from it. fsspec's implementations override its methods instead, each for its own
  storage.
- **Adaptive file objects.** They aim at good performance from random reads of short ranges to bulk sequential reads,
  with nothing to tune per workload. The file classes resize their blocks as a file is read or written, and send their
  requests in the background, in parallel.
- **Observable and cancellable.** A `Monitor` sees each step of an operation, and can stop it through the operation's
  `OperationHandler`.

### The classes

```text
AsyncFileSystem                    fsspec's
└── _BaseFileSystem                common/filesystem.py
    ├── TreeFileSystem             tree/filesystem.py
    │   ├── AsyncTreeWrapper       wrapper/tree.py
    │   │   └── AdlfsWrapper       wrapper/adlfs.py
    │   └── SyncTreeWrapper
    └── FlatFileSystem             flat/filesystem.py
        ├── AsyncFlatWrapper       wrapper/flat.py
        │   ├── S3fsWrapper        wrapper/s3fs.py
        │   └── GcsfsWrapper       wrapper/gcsfs.py
        └── SyncFlatWrapper
AsyncRoutingFileSystem             router.py, holding filesystems of the above
```

`_BaseFileSystem` holds what both families share:

- the two limits: `operation_limit` for the strategies' requests, `io_limit` for reads and writes of data;
- the profile: `storage_profile`, `UNKNOWN` unless set, and `async _load_storage_profile(path)` overridden to find
  it with a request;
- the data methods, `_cat_file`, `_pipe_file`, `_get_file` and `_put_file`, through `open_async`'s files.

Each family declares the primitives its subclasses implement, and overrides fsspec's recursive methods:

| Class            | Its own primitives                           | Overrides                                               |
|------------------|----------------------------------------------|---------------------------------------------------------|
| `TreeFileSystem` | `_ils`, and optional `_rm_tree`, `_mv_tree`  | `_walk`, `_find`, `_du`, `_glob`, `_rm`, `_copy`, `_mv` |
| `FlatFileSystem` | `_ils`, `_rm_files`, `rm_batch_size`         | The same                                                |

Each family also makes the sync methods fsspec does not, `mv` and `rmdir`. Both declare abstract `_info`, `_rm_file`,
`_rmdir`, `_makedirs`, `open_async` and `_open`. fsspec's default `_rmdir` and `_makedirs` do nothing, so a subclass
forgetting them would fail silently. A `FlatFileSystem`'s handle buckets.

Both also declare abstract the two methods copying and moving a file:

- `_cp_file_entry(entry, path2)`, a copy within the storage;
- `_mv_file_entry(entry, path2)`, the storage's rename, or a copy then a delete where it has none.

Each takes the source's entry, as the listing gave it, so that a copy needing the size, e.g. to go in parts, need not
ask it again. The entry may be stale, so a copy must still be right if the file changed. fsspec's `_cp_file` and
`_mv_file` ask `_info`, then call them.

A method raises its strategy's failures: one as it is, several in an ExceptionGroup.

### Choosing

A developer chooses, per method:

- a family, by subclassing its class or wrapping a filesystem with it;
- fsspec's own algorithm instead, by putting it back, e.g. `_rm = AsyncFileSystem._rm`;
- a storage's own operation, by overriding the method;
- a storage's recursive delete, by implementing `_rm_tree`, which `deletes_directories=True` declares;
- a storage's rename of a directory, by implementing `_mv_tree`, which `renames_directories=True` declares.

`monitored(fs)` picks the wrapper of a filesystem from `known_implementations`, a table the developer may replace. An
entry's `wrapper` names a class by a string, as the interface never imports the strategies. A bare name is one of the
built-in wrappers in `monitor.py`'s `_BUILT_IN`, and a full name is imported.

### Operations

Each recursive method starts a private strategy of its family, e.g. `tree/rm.py`'s `_tree_rm`:

- It takes the filesystem it works on: `_tree_rm(fs, path, maxdepth=...)`. A copy or move takes two, `src_fs` and
  `dst_fs`, which are the same within one storage.
- It reads what it needs from the filesystem: the primitives, `operation_limit` and `storage_profile`.
- It takes no function but one: a copy's or a move's `cp_entry` or `mv_entry`, a `_Transfer`, taking the source's
  entry. The router passes its copy of a file between two storages so.
- Paths come without a protocol.
- Argument errors are raised by the call. Errors ending the whole operation, such as a missing root, are raised by
  the iteration.

It returns an `Operation`, private, which works as it is iterated, a few requests ahead of the caller:

- It yields what it did, mixed with an exception per failed path. Each exception notes its path.
- Its `state` is a frozen dataclass of counts so far, from `fsspec_alt_strategies.states`.
- Its stopping follows `asyncio.Task`. `cancel()` starts nothing new, but yields what is in flight. `aclose()`, or
  leaving the loop, stops at once, waiting only for writes in flight.
- It never swallows `CancelledError`.
- It yields in the order the storage answers. A caller wanting fsspec's order sorts.

The class runs it through `common/operation.py`, which tells the monitors:

- A `Monitor` overrides `started`, `yielded` and `ended`, awaited where the operation runs.
- `with monitoring(monitor):` watches every call within the block, sync calls included.
- A monitor stops a call with `op.cancel()`. The call then raises `CallCancelled`.
- fsspec's `callback=` is updated per path, as fsspec's own methods do.

A strategy replacing an fsspec method answers as it does. Where it departs on purpose, its docstring says why:

- It keeps going past a failure, and reports each, where fsspec stops at the first.
- A copy or move of a list keeps each directory's tree, under its name, as `cp -r` does. fsspec's flattens the trees.
- With `maxdepth`, a copy or move places the files as without it. fsspec's places them by the files found, which
  moves them when all are in one subdirectory.
- With `maxdepth`, a move keeps the files deeper than the limit, and the directories holding them. fsspec's deletes the
  whole source.

### Jobs

The strategies changing a storage run on `common/jobs.py`:

- `_Jobs` finds what to change and does it at once, e.g. deleting each file as its directory is listed. Each request
  holds a permit of the filesystem's `operation_limit`, through a `_Queue`.
- `_TreeJobs` and `_FlatJobs` say where the items come from, in each family's `_jobs.py`.
- `_CopyJobs` places each path of a copy or move, as fsspec's `copy` does, whatever the family.

A family's copy joins the two, e.g. `_CopyOrMove(_CopyJobs, _TreeJobs)`. Their constructors take keywords only, and
pass on what they do not take, so either order works.

A strategy's jobs say what an item queues, how a job runs, and what its outcome is. `_Jobs.operation(...)` makes the
jobs and their `Operation` at once.

### Routers

`AsyncRoutingFileSystem`, in `router.py`, passes each call to the filesystem of the path's storage. A subclass
implements two methods, and a third when some paths are above every storage:

- `_route(path)` gives the inner filesystem and its path, or None above every storage;
- `_outer(fs, path)` gives the router's path of an inner one;
- `_roots(path)` lists a directory above the storages. Without such paths, e.g. for URLs naming their storage, the
  default raises FileNotFoundError.

`_route` makes each inner filesystem once, with `_backend`, in the router's mode and on its loop: a sync router's
`open` and `sign` call their sync methods. `backends_loop` is the loop they run on, where a subclass closes what
they share, e.g. a credential. The class is generic in its inner filesystems' class, e.g.
`AsyncRoutingFileSystem[S3FileSystem]`, which its hooks then take.

Within one storage, a call is that storage's own. Between two, a recursive copy or move runs the source family's
strategy on the two inner filesystems, chosen by the source profile's `fs_type`, with `maxdepth` too. Without one, it
is fsspec's copy, a file at a time. A plain fsspec filesystem counts as one of unknown profile.

Each file goes through `_cross_fs_cp_entry` or `_cross_fs_mv_entry`, with the source's entry. By default these stream
the data through both storages' `open_async`. A subclass can use a server-side copy instead, or the size to choose how
the data goes.

The router imports a family only when copying from it. Its inner filesystems must be async, and `close()` closes them.
Its transaction commits file by file, so it is not atomic across storages.

### The tree strategies

The tree strategies list with `_ils(path)`, a directory's entries in any order. With `recursive=True`, it lists all
below a directory, at any depth, in one of two orders, which tell the walk when a directory is done:

- Depth first: a directory's entry, then all below it, before any other entry. Siblings may come in any order.
- Sorted by the whole name, as object stores list keys. A directory's entry may come apart from what is below it, as
  "a/b-x" comes between "a/b" and "a/b/c".

A directory may be left out, as object stores list only files: the walk tells it from the paths below it. A storage
with no recursive listing answers with `_tree_ils` of `tree/listing.py`. That lists depth first, a directory per
request, several ahead at a time. The generic tree wrappers list so, but `AdlfsWrapper` lists by Azure's own recursive
listing.

The strategies walk with a private engine, `tree.walk._chunks`. It streams each directory in parts of at most 1000
entries. A huge directory is never held whole, except by `_tree_walk`, which answers as fsspec's `walk` does.

- A tree that nothing prunes is listed by one recursive `_ils` per root. Its order tells when each directory is done.
  Where the storage has a recursive listing, that is one paginated request, which usually beats a directory per
  request.
- A tree pruned, by `maxdepth` or a glob's pattern, is listed a directory per request, many at once. Below a glob's
  `**`, nothing is pruned, so it is listed recursively again. Pruning wins on a large tree, and loses on a small
  subtree that one recursive listing would cover. The library cannot tell which without guessing the tree's size, so
  it always prunes.
- A failure ends a recursive listing, as there is no way past it. The root's error is yielded, noting the directory that
  failed. Listed a directory per request, the rest of the tree goes on.
- `_tree_walk` top down lists a directory per request: a recursive listing ends a directory after all below it, so the
  whole tree would be held.

They come in two layers:

| Strategies                             | They send                         | They build on    |
|----------------------------------------|-----------------------------------|------------------|
| `_tree_find`, `_tree_du`, `_tree_glob` | Listings, and one `_info` at most | `_find_listings` |
| `_tree_rm`, `_tree_cp`, `_tree_mv`     | Changes, as the listings come     | `_TreeJobs`      |

- The changes run along with the walk still in flight.
- Patterns are expanded by the family's glob first, as copies place paths from the common prefix of all matches.

### The flat strategies

The flat strategies list with `_ils(path, recursive=True, prefix=...)`: every key below `path` starting with
`path/prefix`, at any depth, in name order, as object stores list them. Without `recursive`, `_ils` lists one directory,
for fsspec's own methods. The two are one request with or without a delimiter, as on S3, GCS and Azure.

- The strategies list by a raw prefix, split at its last "/" by `flat._keys._listing`. So `_flat_glob` lists only keys
  starting as its pattern does, e.g. `data` with the prefix `2024-` for `data/2024-*`.
- Name order tells when a directory is done, so `_flat_walk` streams its directories. Each key is listed once.
- Directories are derived from the keys, by `flat/_keys.py`. A key ending with "/" marks a directory.
- `_flat_rm` deletes in batches of `rm_batch_size`, with `_rm_files`. A batch may span buckets. `_rm_files` returns
  the keys it could not delete, each with its error, so that the others count as deleted.
- `_flat_mv` renames with `_mv_file_entry` when the profile says the storage renames. Otherwise it copies, then
  deletes in batches.
- A listing error ends the operation: there is no other listing to go on with.

### Storage profiles

A `StorageProfile` declares what a storage does that fsspec leaves open. A strategy is right on any storage, and a
profile only saves requests:

| Field                  | Saves                                                              |
|------------------------|--------------------------------------------------------------------|
| `fs_type`              | The router's per-file fallback, by naming the family               |
| `parents_required`     | False: the `_makedirs` of each destination directory written to    |
| `implicit_directories` | True: those, and the deletion of each directory                    |
| `renames_files`        | True: the data copied by `_flat_mv`                                |
| `renames_directories`  | True: every request in a directory moved whole by one `_mv_tree`   |
| `deletes_directories`  | True: every request in a directory deleted whole by one `_rm_tree` |

- A filesystem has one profile, which never changes. It is `storage_profile`, or what `_load_storage_profile(path)`
  finds with a request, asking the storage at that path, e.g. the container a credential is limited to.
- A strategy awaits it with `_get_storage_profile(paths)`, in its jobs' `setup()`, with the paths it is given. Once
  loaded, `storage_profile` is returned. Until then, each path is tried in turn, under a lock, and the first profile
  loaded is set as `storage_profile`. If all fail, it raises. `storage_profile` is `UNKNOWN` unless a
  subclass sets what it knows.
- Paths of one filesystem whose storages differ go to a router, each storage with a filesystem of its own.
- A deletion reads the profile of the filesystem it deletes from.
- A copy or move reads the destination's. `renames_files` and `renames_directories` count only within one storage.
- A field left None is unknown, and the strategy takes the way right on any storage.
- A wrong field gives wrong results, as the library never checks it against the storage.
- A field is added only when a strategy saves requests or data with it.

### Copied, not only installed

We recommend copying the strategies into your own package, e.g. as `_contrib/strategy`, so that behaviour never
changes with a release. The interface is installed instead, so that every copy shares one `CallCancelled`, one
`StorageProfile` and one set of monitors. To keep the code copyable:

- the package imports itself only relatively, and depends only on fsspec, the interface and the standard library;
- each source file keeps its SPDX license header;
- no `__init__.py` but the package's own re-exports anything, so a copy can leave modules out.

`packages/impl/tests/test_copyable.py` checks these rules. The command line, `fsspec-alt-strategies-impl copy`,
copies the package as installed, with the `_version.py` its build wrote. It adds the `LICENSE` file the distribution
records, as the Apache License asks that the code goes with it. It lives in its own package,
`fsspec_alt_strategies_impl_cli`, so that copies leave it and click out.

### File objects

`files/` holds abstract file classes, by what the storage allows:

- `AdaptiveReadableFile` and `AdaptiveWritableFile` read byte ranges and write in parts. They adapt their block sizes
  to how the file is used, sending as few requests as they can.
- Each request holds a `concurrency_limit`, whose scope the caller chooses: a file, a filesystem or the process.
- A file read in many requests reads one version, and raises `ESTALE` if the file changes.
- `AsyncFileAdapter` and `SyncFileAdapter` give files the API of those `open_async` returns.
- A `TransactionFile` holds writes back until fsspec's transaction commits, by a temporary path, a local buffer, or the
  adaptive file's own last request.

They follow Python's file API, so errors are raised.

### Logging

The library tells its callers by what it returns and raises, not by logging. A file object logs a warning only for a
failure no caller can be told of, such as a discard by the garbage collector. A caller wanting a log of operations
uses `LoggingMonitor`.

## Releases

A release is a version tag pushed to the repository, e.g. `git push origin v0.1.0`:

1. `release-interface.yaml` builds `fsspec-alt-strategies`, creates the GitHub release, and publishes to PyPI.
2. `release-impl.yaml` then does the same for `fsspec-alt-strategies-impl`. It can be rerun by hand for a tag.

Before the first release, set up on PyPI a trusted publisher for each project, with its workflow and the `pypi`
environment. The repository also needs the `pypi` environment, and the secret `CODECOV_TOKEN`.

### Release notes

The workflows create the GitHub release with notes generated from the pull requests. Once both have succeeded,
rewrite the notes for users, e.g. with `gh release edit v0.2.0 --notes-file notes.md`. They hold, in order:

1. An opening line, only when the release as a whole needs one, e.g. the first release.
2. The sections that apply, in this order:
   - `## New`: each feature as a bold title, a colon, then what it does for the user;
   - `## Changes`: what behaves differently;
   - `## Fixes`: what was wrong, and is now right.
3. A `Documentation:` list linking the sections of the README or USAGE.md that describe the new features.
4. `## Install`, with the commands to upgrade, and what else to do after upgrading, if anything.
5. The generated sections, kept as they are: `## What's Changed`, `## New Contributors` and `**Full Changelog**`.

The notes describe what users see, not how it is built. They follow the rules for docs of CLAUDE.md, and link files
by absolute URL.
