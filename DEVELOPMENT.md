# Development

## Layout

The repository is a uv workspace of two packages, each built and published on its own. The root holds the tooling:
ruff, pytest and coverage settings in `pyproject.toml`, CI, and the docs.

| Path                                                    | Contents                                                                   |
|---------------------------------------------------------|----------------------------------------------------------------------------|
| `packages/common/src/fsspec_alt_strategies/`            | The interface, installed once per project                                  |
| `.../monitor.py`                                        | Operations' control, monitors, and the routers a call came through          |
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
strategy, chosen by the profile's `fs_type`. Each file goes through `_cross_fs_cp_file` or `_cross_fs_mv_file`. By
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
- A copy or move places its paths with `common.placement._Placing`, as fsspec's `copy` does.
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

`packages/impl/tests/test_copyable.py` checks these rules.

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
