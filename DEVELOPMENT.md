# Development

## Layout

The repository is a uv workspace of two packages, each built and published on its own. The root holds the tooling:
ruff, mypy, pytest and coverage settings in `pyproject.toml`, CI, and the docs.

| Path                                                    | Contents                                                                    |
|---------------------------------------------------------|-----------------------------------------------------------------------------|
| `packages/common/src/fsspec_alt_strategies/`            | The interface, installed once per project                                   |
| `.../monitor.py`                                        | Operations' control, monitors, and the routers a call came through          |
| `.../profile.py`                                        | `StorageProfile`: what a storage does that fsspec leaves open               |
| `.../states.py`                                         | The states the strategies report, e.g. `FindState`                          |
| `.../wrapper.py`                                        | `known_implementations`: how each known filesystem class is wrapped         |
| `packages/impl/src/fsspec_alt_strategies_impl/`         | The strategies, copied into projects                                        |
| `.../tree/`, `.../flat/`                                | The two families: each one's handler class, filesystem class and strategies |
| `.../common/`                                           | What the families share, e.g. the base handler, the jobs and the transfer   |
| `.../files/`                                            | File objects, their async adapters, and transactions                        |
| `.../wrapper/`, `.../monitor.py`                        | The handlers wrapping fsspec filesystems, their filesystems, `monitored()`   |
| `.../router.py`                                         | `RoutingHandler` and `RoutingFileSystem`, spanning several storages         |
| `packages/impl/src/fsspec_alt_strategies_impl_testing/` | `SimulatedFileSystem` and its views, apart so that copies leave them out    |
| `packages/impl/src/fsspec_alt_strategies_impl_cli/`     | The command line copying the strategies, with the `cli` extra               |
| `packages/*/tests/`                                     | Each package's tests                                                        |
| `packages/*/hatch_build.py`                             | Gives each package the root README.md as its PyPI readme                    |
| `.github/workflows/`                                    | CI, CodeQL, and one release workflow per package                            |

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
uv run --with mypy==2.4.1 mypy   # types of both packages' sources
```

CI runs ruff, mypy, zizmor and actionlint, and pytest on every supported Python version. When writing tests:

- Run strategies on `SimulatedFileSystem`, with the fixtures `flat_fs`, `hierarchical_fs`, or `any_fs` for both.
  `add_file` and `add_dir` set up files without requests.
- Start a strategy on a `TreeView` or a `FlatView` of the simulator, e.g. `_tree_rm(TreeView(fs), "data")`. A view is
  a handler: it gives the strategy the primitives it calls, and lets a test replace one, e.g. with one that fails.
  It also sets the storage's profile and `operation_limit`, and opens files of the simulator.
- Assert the requests made, e.g. `fs.stats.calls == {"list": 1, "delete": 3}`: they are what a strategy is for.
- Test a strategy replacing an fsspec method against fsspec's own, e.g. `AsyncFileSystem._glob(fs, ...)`.
- `parents_required`, `copy_parents_required` and `overwrite` make the simulator fail as some storages do, to test
  profiles.
- Async tests need no marker. Test directories have no `__init__.py`, and the interface's tests never import the
  strategies.

## Design

The choices below shape the whole project. Keep to them, or change them on purpose.

### Principles

The library aims at large cloud storages, such as data lakes holding millions of files across many directories.

- **Streaming first.** A handler's listing is the primitive every strategy lists with: an async iterator yielding
  entries as the storage is scanned, where fsspec lists every path first with `expand_path`. A strategy acts on each
  part of a listing as it arrives, and never holds a whole directory or tree. Only `_walk` holds a directory whole, as
  fsspec's `walk` answers a directory at a time.
- **Hierarchical namespaces, used.** Where a storage lists by directory, the tree strategies scan directories
  recursively, several at once, e.g. to prune a tree by a pattern. A storage's own operations on a whole tree plug in
  as optional primitives, such as `rm_tree`, `mv_tree` and `cp_tree`.
- **One implementation, many storages.** A storage's behaviour is declared in its `StorageProfile`, never detected, and
  the strategies pick their way from it. fsspec's implementations override its methods instead, each for its own
  storage.
- **Adaptive file objects.** They aim at good performance from random reads of short ranges to bulk sequential reads,
  with nothing to tune per workload. The file classes resize their blocks as a file is read or written, and send their
  requests in the background, in parallel.
- **Observable and cancellable.** A `Monitor` sees each step of an operation, and can stop it through the operation's
  `OperationHandler`.
- **AI oriented.** We assume AI assistants will largely develop and maintain the fsspec implementations built on the
  library. Its docs and docstrings are material for them: each says what an implementation must do and why, so that
  an assistant has enough context to write one, and to keep it right as the library changes.

### Handlers and filesystems

A storage is implemented in two parts:

- a **storage handler**, which a developer writes: the storage's primitives, its paths, its profile, its limits, and
  its clients, opened and closed;
- a **filesystem**, `TreeFileSystem` or `FlatFileSystem`, which implements fsspec's whole API on one handler. Nobody
  overrides its methods.

The strategies take handlers, never filesystems. So a convention the library keeps, e.g. a path verified or a listing
dropped after a change, is kept in one place.

```text
_PathHandler                       common/handler.py: paths
├── StorageHandler                 common/handler.py: lifecycle, profile, limits, primitives
│   ├── TreeHandler                tree/handler.py
│   │   └── FsspecTreeHandler      wrapper/tree.py
│   │       └── AdlfsHandler       wrapper/adlfs.py
│   └── FlatHandler                flat/handler.py
│       └── FsspecFlatHandler      wrapper/flat.py
│           ├── S3fsHandler        wrapper/s3fs.py
│           └── GcsfsHandler       wrapper/gcsfs.py
└── RoutingHandler                 router.py: no primitives

AsyncFileSystem                    fsspec's
├── _BaseFileSystem                common/filesystem.py
│   ├── TreeFileSystem             tree/filesystem.py
│   │   └── TreeFileSystemWrapper  wrapper/tree.py
│   └── FlatFileSystem             flat/filesystem.py
│       └── FlatFileSystemWrapper  wrapper/flat.py
└── RoutingFileSystem              router.py
```

#### The handler

`StorageHandler` holds what both families share:

- **Its lifecycle.** It is made without a request. `await open()` makes its clients on the running loop, and loads its
  profile with `load_profile()`, once. `await aclose()` closes them, and a later call opens it again. Its sync
  `close()` is the filesystem's: it releases at once what it can without awaiting, e.g. threads. A handler may refuse
  calls after it, as a closed filesystem does, so `aclose()` never calls it.
- **Its profile,** `profile`, as the next sections say.
- **Its limits,** `operation_limit` for the strategies' requests and `io_limit` for reads and writes of data. The
  callers hold the permits, never the primitives: the strategies hold one of `operation_limit` around each primitive.
  Data is the exception: the adaptive files hold one of `io_limit` per part, and a handler's own `cat_file`,
  `pipe_file`, `get_file` or `put_file` one per request it sends.
- **Its paths:** its `protocol`, `strip_protocol` and `unstrip_protocol`, its `root`, and the path methods below.
- **The primitives of single paths:** `ils`, `info`, `rm_file`, `rmdir`, `makedirs`, `cp_file` and `mv_file`, and its
  files: `open_file`, with Python's sync API, for fsspec's sync `open`, and `open_async_file`, whose methods are
  awaited, for `open_async` and the strategies. Both raise NotImplementedError by default. A handler gives its files
  either API, e.g. through `AsyncFileAdapter`, and supports fsspec's transactions in `open_file`, as it chooses.

Each family's handler class adds what its strategies call:

| Class         | Adds                                                                                          |
|---------------|-----------------------------------------------------------------------------------------------|
| `TreeHandler` | Optional `ils_recursive` with its `recursive_order`, and `rm_tree`, `mv_tree`, `cp_tree`      |
| `FlatHandler` | Required `ils_recursive(entry, prefix=...)`, and `rm_files` of at most `rm_batch_size` keys   |

A handler's class tells its family, as a router reads it. The filesystem calls only what `StorageHandler` declares,
never finding a method by `getattr`. Beside the primitives, it declares methods with defaults a handler may override:

- `cat_file`, `pipe_file`, `get_file` and `put_file` move data whole, by default through `open_async_file`. A
  handler sending a request of its own instead, e.g. one download, holds a permit of `io_limit` around it, as the
  files do. The defaults hold none: the files take their own.
- `invalidate_cache` drops what the handler keeps of listings, by default nothing, and `wrapped` is the filesystem a
  wrapper wraps, which monitors name, by default None.

`StorageHandler.sign(entry, expiration)`, `created(entry)` and `modified(entry)` answer fsspec's `sign`, `created`
and `modified`. They are async, as a storage may need a request, e.g. Azure's user delegation key to sign with. A
handler without them keeps the defaults, which raise NotImplementedError, as fsspec's do.

#### Paths and entries

A path a strategy uses is a `StoragePath`, a `NewType` of `str`: no protocol, no trailing "/". Only the handler makes
one, so each is verified:

- `verify_path(path)`, abstract, raises ValueError for a path the storage refuses, or the handler does not serve,
  e.g. another account's. It sends no request. It sees the path as the caller gave it, a URL or not, so that it
  checks what a URL names beyond the path, e.g. abfs's account. A handler checking names strips the path first.
- `verified(path)` is a caller's path once verified, then stripped by `strip_protocol`, and `entry(path)` its entry,
  `{"name": path}`. The filesystem passes its caller's paths to them as they come: `verified` is where a path loses
  its protocol. A pattern alone is stripped by the filesystem, as no handler verifies one.
- `join(directory, *paths)` joins paths as `os.path.join` does, verifying the path built once. `_split` splits each
  path into segments by the class's `delimiter`, "/" by default, and `_verify_segment` refuses a segment that is empty,
  ".", "..", or holds the delimiter. A storage overrides either. Unlike `os.path.join`, a path starting with the
  delimiter never starts again from the root.
- `parent(path)` is verified too, and `name(path)` is the last name, both by the delimiter.

So a copy places each file by joining names below its destination, never by string operations. A key holding ".."
can never leave the destination, whatever the source allows.

An item that exists is named by its entry, a dict as fsspec's `ls` with `detail` gives it:

- Only `"name"` is sure. The other keys, e.g. `"size"`, or a storage's own, e.g. an `"id"`, are there when the storage
  told them.
- A key missing is unknown, never absent: a missing `"size"` is not 0. A key may be stale, listed before a change.
- The primitives take an entry for an item that exists, and a path for one to be made. A strategy passes the entries
  its listing gave, whole, so that a storage addressing items by id, or needing a size, asks nothing again. A name a
  listing gave is never verified again.
- `open_file` and `open_async_file` take an entry whatever the mode, so that one method opens a file to read or to
  write. A file to write may not exist yet: its entry then holds its name alone, as `entry` makes it from a verified
  path.

#### The filesystem

`_BaseFileSystem` holds what both families' filesystems share:

- It takes a handler, as a required argument. It exposes neither its profile nor its limits: they are the handler's.
- It strips the protocol of each path a caller gives, verifies it by the handler, and wraps it by `handler.entry`
  where it names an item.
- Its data methods, `_cat_file`, `_pipe_file`, `_get_file` and `_put_file`, call the handler's extras where it has
  them, else go through its files.
- `_copy` and `_mv` start the copy and move its family's class names, `_cp_strategy` and `_mv_strategy`.
- It answers and raises as fsspec's methods do, and tells the monitors of each call.
- It has the sync methods fsspec does not make, `mv` and `rmdir`, and fsspec's `dircache`, as the next section says.

A storage's own class, by convention, is a small subclass of a family's:

```python
class MyFileSystem(TreeFileSystem):
    protocol = MyHandler.protocol
    root_marker = MyHandler.root
    _strip_protocol = MyHandler.strip_protocol

    def __init__(self, account, *, io_concurrency=8, **kwargs):
        super().__init__(MyHandler(account, io_concurrency=io_concurrency), **kwargs)
```

It takes storage options, not a handler, so that fsspec makes it from a URL or a registry, caches it and pickles it.
It names its paths as its handler class's, as fsspec reads them of the class before a filesystem is made. It
overrides nothing else.

Its `__init__` is what its users call, so its signature names each option, repeating its handler's:

- No helper splits the options for it: a signature tells its users and their tools what it takes.
- It passes fsspec's options on as keywords. `_BaseFileSystem` and `RoutingFileSystem` name them, so any other option
  raises TypeError. fsspec's own constructors take any keyword, silently ignoring a misspelled one.
- Options whose names another library decides, e.g. those of an Azure token credential, are the only catch-all. The
  class then names fsspec's options too, so that its `**` holds those alone, and the handler passes on those that
  the credential's constructor names.
- fsspec's `batch_size` is not taken: the handler's `io_concurrency` bounds the paths handled at a time.

`TreeFileSystemWrapper` and `FlatFileSystemWrapper` serve a wrapped fsspec filesystem's paths: as they are made, they
take the protocol, root and stripping of the handler they are given. They are never cached, as fsspec would key them
by their handler's repr, which two wrappers of one filesystem share.

A method raises its strategy's failures: one as it is, several in an ExceptionGroup.

#### fsspec's listings cache

fsspec lets a filesystem keep its listings in `dircache`, for its own methods and callers. Here:

- `_ls` keeps a directory's listing once it has listed it whole, and answers a later `_ls` from it, unless
  `refresh=True`. `use_listings_cache=False` keeps none, as in fsspec.
- The strategies never read or fill it: they list by the handler. `_info` never reads it either. So a listing kept
  never changes what an operation does, and the strategies need not handle a stale one.
- Each method changing the storage calls `_drop_listings()` in a `finally`, dropping every listing kept once it ends.
  A new one does too.
- Opening a file to write drops them too, as the file is opened: the file knows no filesystem, so its closing cannot.
  A listing taken while it is written may then miss it once it is closed.
- `invalidate_cache(path)` drops the listings of `path`, of what is below it, and of its parents. It passes on to a
  wrapper's handler, whose wrapped filesystem manages its own cache.

### Choosing

A developer chooses, per storage:

- a family, by the handler class a handler subclasses;
- a storage's recursive listing, by implementing `ils_recursive` and declaring its `recursive_order`;
- a storage's recursive delete, by implementing `rm_tree`, which `deletes_directories=True` declares;
- a storage's rename of a directory, by implementing `mv_tree`, which `renames_directories=True` declares;
- a storage's copy of a directory, by implementing `cp_tree`, which `copies_directories=True` declares.

`monitored(fs)` picks the handler of a filesystem from `known_implementations`, a table the developer may replace. An
entry's `handler` names a class by a string, as the interface never imports the strategies. A bare name is one of the
built-in handlers in `monitor.py`'s `_BUILT_IN`, and a full name is imported.

### Operations

Each recursive method starts a private strategy of its family, e.g. `tree/rm.py`'s `_tree_rm`:

- It takes the handler it works on: `_tree_rm(handler, path, maxdepth=...)`. A copy or move takes two, `src_fs` and
  `dst_fs`, which are the same handler within one storage.
- It reads what it needs from the handler: the primitives, `operation_limit` and `profile`, once open.
- It takes no function but one: a copy's or a move's `cp_entry` or `mv_entry`, a `_Transfer`, taking the source's
  entry. Between two storages, it defaults to `common/transfer.py`'s `_across`.
- Paths come without a protocol, verified.
- Argument errors are raised by the call. Errors ending the whole operation, such as a missing root, are raised by
  the iteration.

It returns an `Operation`, private, which works as it is iterated, a few requests ahead of the caller:

- It yields what it did, mixed with an exception per failed path. Each exception notes its path.
- Its `state` is a frozen dataclass of counts so far, from `fsspec_alt_strategies.states`.
- Its stopping follows `asyncio.Task`. `cancel()` starts nothing new, but yields what is in flight. `aclose()`, or
  leaving the loop, stops at once, waiting only for writes in flight.
- It never swallows `CancelledError`.
- It yields in the order the storage answers. A caller wanting fsspec's order sorts.

The filesystem runs it through `common/operation.py`, which tells the monitors:

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
  holds a permit of the handler's `operation_limit`, through a `_Queue`.
- `_TreeJobs` and `_FlatJobs` say where the items come from, in each family's `_jobs.py`.
- `_CopyJobs` places each path of a copy or move, as fsspec's `copy` does, whatever the family. It builds each target
  by the destination's `join`.

A family's copy joins the two, e.g. `_CopyOrMove(_CopyJobs, _TreeJobs)`. Their constructors take keywords only, and
pass on what they do not take, so either order works.

A strategy's jobs say what an item queues, how a job runs, and what its outcome is. `_Jobs.operation(...)` makes the
jobs and their `Operation` at once.

### Copies between storages

A copy or move between two handlers sends each file by `common/transfer.py`, unless its caller gives another way:

- It reads and writes by the source's and the destination's `open_async_file`, a block of 8 MiB at a time.
  So each handler keeps the primitives of its own storage only.
- It leaves no part of a file. Where the destination's profile declares `atomic_writes`, it writes in place.
  Otherwise it writes to a temporary name beside the destination, then moves it there by `mv_file`, deleting it on a
  failure.
- It checks the bytes written against the source's size, the reader's or else the entry's. A size that differs raises
  OSError inside the writer, which then completes nothing.
- A move copies, then deletes the source by its `rm_file`. A source whose copy failed is kept.

### Routers

`RoutingFileSystem`, in `router.py`, passes each call to the filesystem of the path's storage. Where each path goes is
its `RoutingHandler`'s, which a developer implements:

- `verify_path(path)` checks the router's own paths;
- `route(path)` gives the handler of the path's storage and its path there, or None above every storage;
- `outer(handler, path)` gives the router's path of a path of one of its handlers;
- `roots(entry)` lists a directory above the storages. Without such paths, e.g. for URLs naming their storage, the
  default raises FileNotFoundError.

`route` makes each handler once, by `inner(key, make)`, without a request. The routing handler's `aclose()` closes
them, and its sync `close()` releases them at once, as the filesystem's `close()` does.

For each storage routed to, the router makes a `TreeFileSystem` or `FlatFileSystem`, as its handler's class is a
`TreeHandler` or a `FlatHandler`, uncached. Each path is checked twice: by the routing handler, then its path in the storage by the storage's handler.

`route` may give another routing handler, which routes the inner path again. Each level checks its own path, and the
names answered are mapped back through each level's `outer`. The roots of a nested router are listed by it.

Within one storage, a call is that storage's own. Between two, a recursive copy or move runs the source family's
strategy, with `maxdepth` too. Each file goes by the routing handler's `cp_across` or `mv_across`, under its
`transfer_limit`. By default these stream the data, as the previous section says. A routing handler can copy on the
server instead, e.g. from a URL. Through nested routers, the outermost router's are used.

The router imports a family's filesystem only when routing to a storage of it, and its strategies only when copying or
moving from one. Its transaction commits file by file, so it is not atomic across storages.

### The tree strategies

The tree strategies list a directory with the handler's `ils`, its entries in any order. A handler listing a whole
tree in one request implements `ils_recursive`, and declares its order in `recursive_order`, which tells the walk when
a directory is done:

- "depth first": a directory's entry, then all below it, before any other entry. Siblings may come in any order.
- "name": sorted by the whole name, as object stores list keys. A directory's entry may come apart from what is below
  it, as "a/b-x" comes between "a/b" and "a/b/c".

A directory may be left out, as object stores list only files: the walk tells it from the paths below it. A handler
with no `recursive_order` has no recursive listing: `tree/listing.py`'s `_tree_ils` lists depth first, a directory per
request, several ahead at a time. The generic tree handler lists so, but `AdlfsHandler` lists by Azure's own recursive
listing.

A storage may refuse a recursive listing in a way only asking tells, e.g. a WebDAV server refusing `Depth: infinity`.
Its `ils_recursive` then raises NotImplementedError before its first entry, and the strategies list by `_tree_ils`
instead.

The strategies walk with a private engine, `tree.walk._chunks`. It streams each directory in parts of at most 1000
entries. A huge directory is never held whole, except by `_tree_walk`, which answers as fsspec's `walk` does.

- A tree that nothing prunes is listed by one recursive listing per root. Its order tells when each directory is
  done. Where the storage has a recursive listing, that is one paginated request, which usually beats a directory per
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
| `_tree_find`, `_tree_du`, `_tree_glob` | Listings, and one `info` at most  | `_find_listings` |
| `_tree_rm`, `_tree_cp`, `_tree_mv`     | Changes, as the listings come     | `_TreeJobs`      |

- The changes run along with the walk still in flight.
- Patterns are expanded by the family's glob first, as copies place paths from the common prefix of all matches.

### The flat strategies

The flat strategies list with the handler's `ils_recursive(entry, prefix=...)`: every key below the directory starting
with the prefix, at any depth, in name order, as object stores list them. `ils` lists one directory, for fsspec's own
methods. The two are one request with or without a delimiter, as on S3, GCS and Azure.

- The strategies list by a raw prefix, split at its last "/" by `flat._keys._listing`. So `_flat_glob` lists only keys
  starting as its pattern does, e.g. `data` with the prefix `2024-` for `data/2024-*`.
- Name order tells when a directory is done, so `_flat_walk` streams its directories. Each key is listed once.
- Directories are derived from the keys, by `flat/_keys.py`. A key ending with "/" marks a directory.
- `_flat_rm` deletes in batches of the handler's `rm_batch_size`, with `rm_files`. A batch may span buckets.
  `rm_files` returns the keys it could not delete, each with its error, so that the others count as deleted.
- `_flat_mv` renames with `mv_file` when the profile says the storage renames. Otherwise it copies, then deletes in
  batches.
- A listing error ends the operation: there is no other listing to go on with.

### Storage profiles

A `StorageProfile` declares what a storage does that fsspec leaves open. A strategy is right on any storage, and a
profile only saves requests or data:

| Field                   | Saves                                                                  |
|-------------------------|------------------------------------------------------------------------|
| `fs_type`               | Names the family `monitored()` wraps a known filesystem as             |
| `parents_required`      | False: the `makedirs` of each destination directory written to         |
| `copy_parents_required` | False: the `makedirs` of each directory the storage copies a file to   |
| `implicit_directories`  | True: those, and the deletion of each directory                        |
| `renames_files`         | True: the data copied by `_flat_mv`                                    |
| `renames_directories`   | True: every request in a directory moved whole by one `mv_tree`        |
| `deletes_directories`   | True: every request in a directory deleted whole by one `rm_tree`      |
| `copies_directories`    | True: every request in a directory copied whole by one `cp_tree`       |
| `case_sensitive`        | False: a path moved onto itself in another case, by one `mv_tree`      |
| `atomic_writes`         | True: the temporary file and the move of a copy between storages       |

- A storage has one profile, which never changes. It is the handler's `profile`, `UNKNOWN` unless the handler sets
  what it knows. Or it is what the handler's `load_profile()` finds with a request.
- A strategy awaits the handler's `open()` as its operation starts, e.g. in its jobs' `setup()`, then reads
  `profile`. `open()` loads it once, under a lock. A load that fails raises, and the next call tries again.
- Neither takes a path, so that the profile cannot differ by path. A storage whose behaviour does, e.g. GCS's buckets
  with or without a hierarchical namespace, is several handlers under a router: one per bucket.
- The filesystem exposes no profile: it is the handler's alone.
- A deletion reads the profile of the handler it deletes from.
- A copy or move reads the destination's. `copy_parents_required`, `renames_files`, `renames_directories`,
  `copies_directories` and `case_sensitive` count only within one storage, and `atomic_writes` only between two.
- Between two storages, a file copied is written: `parents_required` counts. Within one, it goes by the storage's own
  copy, which may make no parent where a write does, as on Microsoft Graph: `copy_parents_required` counts.
- A field left None is unknown, and the strategy takes the way right on any storage.
- `case_sensitive` is the exception: no way is right on both kinds of names. Unknown, paths are compared as fsspec
  compares them, as strings. On a storage whose names are not case-sensitive, a path copied or moved onto itself in
  another case then goes into itself, as with fsspec's own.
- A wrong field gives wrong results, as the library never checks it against the storage.
- A field is added only when a strategy saves requests or data with it.

### Copied, not only installed

We recommend copying the strategies into your own package, e.g. as `_contrib/strategy`, so that behaviour never
changes with a release. The interface is installed instead, so that every copy shares one `CallCancelled`, one
`StorageProfile` and one set of monitors. To keep the code copyable:

- the package imports itself only relatively, and depends only on fsspec, the interface and the standard library;
- each source file keeps its SPDX license header;
- no `__init__.py` but the package's own re-exports anything, so a copy can leave modules out.

`packages/impl/tests/test_copyable.py` checks the first two. The command line, `fsspec-alt-strategies-impl copy`,
copies the package as installed, with the `_version.py` its build wrote. It adds the `LICENSE` file the distribution
records, as the Apache License asks that the code goes with it. It lives in its own package,
`fsspec_alt_strategies_impl_cli`, so that copies leave it and click out.

### File objects

`files/` holds abstract file classes, by what the storage allows:

- `AdaptiveReadableFile` and `AdaptiveWritableFile` read byte ranges and write in parts. They adapt their block sizes
  to how the file is used, sending as few requests as they can. A handler's `open_file` returns them as they are,
  and its `open_async_file` through `AsyncFileAdapter`.
- Each request holds a `concurrency_limit`, whose scope the caller chooses: a file, a handler or the process.
- A file read in many requests reads one version, and raises `ESTALE` if the file changes.
- An `AdaptiveWritableFile` left by an exception completes nothing: a multipart upload is aborted.
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

The strategies require the interface of their own minor version, e.g. `fsspec-alt-strategies>=0.4,<0.5`, as they use
what it adds. A release of a new minor version sets that bound in `packages/impl/pyproject.toml` in its pull request.
`release-impl.yaml` refuses to publish a wheel whose bound is another version's.

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
