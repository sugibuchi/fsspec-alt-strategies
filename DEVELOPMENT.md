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
| `.../tree/`, `.../flat/`                                | The two families: walk, find, du, glob, rm, cp, mv, and the abstract class |
| `.../common/`                                           | Private machinery the families share                                       |
| `.../files/`                                            | File objects, their async adapters, and transactions                       |
| `.../wrapper.py`, `.../monitor.py`                      | Wrappers of existing filesystems, and `monitored()`                        |
| `.../router.py`                                         | `AsyncRoutingFileSystem`, spanning several storages                        |
| `packages/impl/src/fsspec_alt_strategies_impl_testing/` | `SimulatedFileSystem`, apart so that copies leave it out                   |
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

CI runs ruff and pytest on every supported Python version. When writing tests:

- Run strategies on `SimulatedFileSystem`, with the fixtures `flat_fs`, `hierarchical_fs`, or `any_fs` for both.
  `add_file` and `add_dir` set up files without requests.
- Assert the requests made, e.g. `fs.stats.calls == {"list": 1, "delete": 3}`: they are what a strategy is for.
- Test a strategy replacing an fsspec method against fsspec's own, e.g. `AsyncFileSystem._glob(fs, ...)`.
- `parents_required` and `overwrite` make the simulator fail as some storages do, to test profiles.
- Async tests need no marker. Test folders have no `__init__.py`, and the interface's tests never import the strategies.

## Design

The choices below shape the whole project. Keep to them, or change them on purpose.

### Proposed: class first

This section is a proposal under review. Once accepted and ported, it replaces "Principles" and "The strategy API"
below.

The strategies are functions today, so that they could apply to any filesystem's methods. Two things now do that
better:

- the wrappers apply a family to an existing filesystem, by passing its primitives through;
- a router holds the filesystems of both storages, so a copy between them can call each directly.

So the abstract classes become the API, and the strategies become their methods. Passing operations as functions,
e.g. `ils=fs._ils`, goes away.

#### The classes

```text
AsyncFileSystem                    fsspec's
└── _BaseFileSystem                common/filesystem.py
    ├── TreeFileSystem             tree/filesystem.py
    │   ├── AsyncTreeWrapper       wrapper.py
    │   └── SyncTreeWrapper
    └── FlatFileSystem             flat/filesystem.py
        ├── AsyncFlatWrapper
        └── SyncFlatWrapper
AsyncRoutingFileSystem             router.py, holding filesystems of the above
```

`_BaseFileSystem` holds what both families share:

- the abstract primitives both need: `_info`, `_rm_file`, `_cp_file`, `_makedirs`, `open_async` and `_open`;
- `_mv_file`, fsspec's copy and delete unless a subclass overrides it with the storage's rename;
- `storage_profile`, a property a subclass may compute per instance, e.g. from the account it connects to;
- the two limits, `operation_limit` and `io_limit`;
- the running of operations: `Operation`, the monitors, and the job engine of `common/jobs.py`;
- the data methods through `open_async`, the error grouping, and the sync methods fsspec does not make.

Each family adds its listing primitive, and its versions of fsspec's recursive methods:

| Class            | Adds the primitives          | Overrides                                                |
|------------------|------------------------------|----------------------------------------------------------|
| `TreeFileSystem` | `_ils`, `_rmdir`, `_rm_tree` | `_walk`, `_find`, `_du`, `_glob`, `_rm`, `_copy`, `_mv` |
| `FlatFileSystem` | `_iexpand`, `_rm_files`      | The same                                                 |

`_rm_tree` stays optional. A wrapper is a family's class whose primitives are the wrapped filesystem's, as now.

#### Operations

Each recursive method runs an operation object, made for the call:

- It is a private class of its family's module, e.g. `tree/rm.py`'s `_TreeRm`.
- It takes the filesystems it works on, not their methods: `_TreeRm(fs, path, ...)`.
- A copy or a move takes two: `_TreeCopy(source, destination, ...)`. Within one storage, both are the same.
- It computes where each path goes in its own methods, as fsspec's `copy` does. `_Placement` goes away.
- Its outcomes still stream through `Operation`, which the monitors see, and which `cancel()` and `aclose()` stop.

Private helpers stay functions where several operations share them: `tree.walk._chunks`, `tree.find._find_listings`,
`common.jobs._Queue`, the patterns of `common/patterns.py`, and the paths of `common/paths.py`.

#### Choosing

A developer still chooses, per method:

- a family, by subclassing its class or wrapping with it;
- fsspec's own algorithm instead, by putting it back, e.g. `_rm = AsyncFileSystem._rm`;
- a storage's native operation, by overriding the method.

The library still never chooses for them.

#### The router

A copy or move between two storages is the router's method. It takes the two inner filesystems from `_route`, and
runs the source family's copy operation on them. The per-file hooks, `_cross_fs_cp_file` and `_cross_fs_mv_file`,
stay the router's.

#### What changes for callers

- The strategy functions, e.g. `tree_rm`, leave the public API. The classes and wrappers are the API.
- `Operation` stays private: no method returns one. A caller watches and stops an operation through an
  `OperationHandler`, such as a `Monitor`.
- `_walk` is the family's walk, listing with `_ils` or `_iexpand`, as `tree_walk` and `flat_walk` do now.
- The other recursive methods, e.g. `_glob`, `_copy` and `_mv`, keep streaming each folder in parts. Only `_walk`
  holds a folder whole, as fsspec's `walk` answers a folder at a time.
- This breaks 0.1.0's API: it is released as 0.2.0.

The operation classes of both families share a base in `common/`. It holds what a copy or move does whatever the
family, such as placing each path as fsspec's `copy` does.

#### Porting

1. Port the tree family, keeping its tests, which assert the requests made through `SimulatedFileSystem`.
2. Port the flat family the same way.
3. Port the router's copy and move between storages.
4. Rewrite the docs and CLAUDE.md's rules on strategies, then remove this section.

### Principles

- **A third party.** The library offers alternatives to fsspec's algorithms, and never patches fsspec.
- **Strategies are functions.** Each is stateless, and takes the storage's operations as functions, e.g. `ils=fs._ils`.
  It works with any filesystem, without subclassing. File objects and `Operation` are the only classes with state.
- **Developers choose.** The library never picks a strategy. Each docstring says what it assumes of the storage, and
  when it wins or loses.
- **Primitives only.** A strategy takes primitive operations, such as `ils`, `info` and `rm_file`, never another
  strategy's method. So choosing one strategy never changes another.
- **Async first.** Strategies are async. Sync filesystems get them through the sync wrappers.
- **Observable and cancellable.** Long operations are iterated, so the caller sees each step and can stop it.

### The strategy API

Every strategy has the same shape, so that it is learnt once:

```python
rm = tree_rm(
    "data/2024",
    ils=fs._ils,
    info=fs._info,
    rm_file=fs._rm_file,
    rm_folder=fs._rmdir,
    profile=fs.storage_profile,
    concurrency_limit=asyncio.Semaphore(32),
)
async with rm as op:
    async for path in op:
        if isinstance(path, Exception):
            print("failed:", path)
print(op.state.files_deleted)
```

- Paths come first, without a protocol. The operations and options are keyword-only.
- The operations are required: a strategy guesses nothing. Only a native operation saving requests is optional, e.g.
  `tree_rm`'s `rm_tree`.
- A listing is an async iterator of entries: `ils(path)` for one folder in `tree/`, `iexpand(prefix)` for every key
  under a prefix in `flat/`.
- `concurrency_limit` is an `asyncio.Semaphore`, required. Sharing one limits several operations together.
- Argument errors are raised by the call. Errors ending the whole operation, such as a missing root, are raised by
  the iteration.

The `Operation` returned works as it is iterated, a few requests ahead of the caller:

- It yields what it did, mixed with an exception per failed path. Each exception notes its path.
- Its `state` is a frozen dataclass of counts so far.
- Its stopping follows `asyncio.Task`. `cancel()` starts nothing new, but yields what is in flight. `aclose()`, or
  leaving the loop, stops at once, waiting only for writes in flight.
- It never swallows `CancelledError`.
- It yields in the order the storage answers. A caller wanting fsspec's order sorts.

A strategy replacing an fsspec method answers as it does. Where it departs on purpose, its docstring says why:

- It keeps going past a failure, and reports each, where fsspec stops at the first.
- A copy or move of a list keeps each folder's tree, under its name, as `cp -r` does. fsspec's flattens the trees.
- With `maxdepth`, a copy or move places the files as without it. fsspec's places them by the files found, which
  moves them when all are in one subfolder.
- With `maxdepth`, a move keeps the files deeper than the limit, and the folders holding them. fsspec's deletes the
  whole source.

### Filesystem classes, wrappers and monitors

A strategy is attached in one of three ways:

- **By hand:** override fsspec's method, e.g. `_rm`, to call the strategy. fsspec builds the sync method from it.
- **By subclassing** `TreeFileSystem` or `FlatFileSystem`. The subclass implements the primitives, such as `_ils`,
  `_info`, `_rm_file` and `open_async`, which are abstract. It takes two limits: `operation_concurrency` for the
  strategies, `io_concurrency` for data.
- **By wrapping** an existing filesystem with a tree or flat wrapper, async or sync. `monitored(fs)` picks the wrapper
  from `known_implementations`.

The classes raise a strategy's failures: one as it is, several in an ExceptionGroup. They run each operation through
`common/operation.py`, which tells the monitors:

- A `Monitor` overrides `started`, `yielded` and `ended`, awaited where the operation runs.
- `with monitoring(monitor):` watches every call within the block, sync calls included.
- A monitor stops a call with `op.cancel()`. The call then raises `CallCancelled`.
- fsspec's `callback=` is updated per path, as fsspec's own methods do.

### Routers

`AsyncRoutingFileSystem`, in `router.py`, passes each call to the filesystem of the path's storage. A subclass
implements three methods:

- `_route(path)` gives the inner filesystem and its path, or None above every storage;
- `_outer(fs, path)` gives the router's path of an inner one;
- `_roots(path)` lists a folder above the storages.

Within one storage, a call is that storage's own. Between two, a recursive copy or move runs the source family's
strategy, chosen by the profile's `fs_type`. Without one, or with `maxdepth`, it is fsspec's copy, a file at a time.
Each file goes through `_cross_fs_cp_file` or `_cross_fs_mv_file`. By
default these stream the data through both storages' `open_async`; a subclass can use a server-side copy instead.

The router imports a family only when copying from it. Its inner filesystems must be async, and `close()` closes them.
Its transaction commits file by file, so it is not atomic across storages.

### The tree strategies

The tree strategies walk with a private engine, `tree.walk._chunks`. It lists many folders at once, and streams each
folder in parts of at most 1000 entries. A huge folder is never held whole, except by `tree_walk`, which answers as
fsspec's `walk` does.

They come in two layers:

| Strategies                          | They send                         | They build on    |
|-------------------------------------|-----------------------------------|------------------|
| `tree_find`, `tree_du`, `tree_glob` | Listings, and one `info` at most  | `_find_listings` |
| `tree_rm`, `tree_cp`, `tree_mv`     | Changes, as the listings come     | `_TreeJobs`      |

- The changes run on `common.jobs._Jobs`, along with the walk still in flight.
- A copy or move places its paths with `common.placement._Placement`, as fsspec's `copy` does.
- A copy or move within one storage is a special case of one between two: `tree_local_cp` passes one storage's
  operations twice, with `same_storage=True`.
- Patterns are expanded with `tree_glob` first, as copies place paths from the common prefix of all matches.

### The flat strategies

The flat strategies list with `iexpand(prefix)`: every key under a prefix, at any depth, in name order, as object
stores list them.

- The prefix is raw text, so `flat_glob` lists only keys starting as its pattern does.
- Name order tells when a folder is done, so `flat_walk` streams its folders. Each key is listed once.
- Folders are derived from the keys. A key ending with "/" marks a folder.
- `flat_rm` deletes in batches, with `rm_files`. `flat_mv` moves by `mv_file`, or copies and deletes in batches.
- A listing error ends the operation: there is no other listing to go on with.

### Storage profiles

A `StorageProfile` declares what a storage does that fsspec leaves open. A strategy is right on any storage, and a
profile only saves requests:

| Field              | Saves                                                         |
|--------------------|---------------------------------------------------------------|
| `fs_type`          | The router's per-file fallback, by naming the family          |
| `parents_required` | False: the `makedirs` of each destination folder              |
| `implicit_folders` | True: those, and the deletion of each folder                  |
| `renames_files`    | True: the data copied by `flat_mv`                            |
| `renames_folders`  | True: every request in a folder moved whole by one `mv_file`  |

- A field left None is unknown, and the strategy takes the way right on any storage.
- A wrong field gives wrong results. The library never detects a storage's behaviour: the developer declares it.
- A field is added only when a strategy saves requests or data with it.

### Copied, not only installed

We recommend copying the strategies into your own package, e.g. as `_contrib/strategy`, so that behaviour never
changes with a release. The interface is installed instead, so that every copy shares one `CallCancelled`, one
`StorageProfile` and one set of monitors. To keep the code copyable:

- the package imports itself only relatively, and depends only on fsspec, the interface and the standard library;
- each source file keeps its SPDX license header;
- no `__init__.py` but the package's own re-exports anything, so a copy can leave modules out.

`packages/impl/tests/test_copyable.py` checks these rules. The command line, `fsspec-alt-strategies-impl copy`,
copies the package as installed, with the `_version.py` its build wrote. It lives in its own package,
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
