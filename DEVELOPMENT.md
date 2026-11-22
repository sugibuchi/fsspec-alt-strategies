# Development

## Layout

The repository is a uv workspace of two packages, each built and published on its own. The root holds the tooling:
ruff, mypy, pytest and coverage settings in `pyproject.toml`, CI, and the docs.

| Path                                                    | Contents                                                 |
|---------------------------------------------------------|----------------------------------------------------------|
| `packages/common/src/fsspec_alt_strategies/`            | The interface, installed once per project                |
| `.../monitor.py`                                        | Operations' control, monitors, the routers of a call     |
| `.../profile.py`                                        | `StorageProfile`: what fsspec leaves open of a storage   |
| `.../states.py`                                         | The states the strategies report, e.g. `FindState`       |
| `.../wrapper.py`                                        | `known_implementations`: how known ones are wrapped      |
| `packages/impl/src/fsspec_alt_strategies_impl/`         | The strategies, copied into projects                     |
| `.../tree/`, `.../flat/`                                | Each family's handler class and strategies               |
| `.../common/`                                           | What the families share, e.g. `BaseFileSystem`, the jobs |
| `.../files/`                                            | File objects, their async adapters, and transactions     |
| `.../wrapper/`, `.../monitor.py`                        | Handlers wrapping fsspec filesystems, `monitored()`      |
| `.../router.py`                                         | `RoutingHandler`, `RoutingFileSystem`: many storages     |
| `packages/impl/src/fsspec_alt_strategies_impl_testing/` | `SimulatedFileSystem`, views and checks, never copied    |
| `packages/impl/src/fsspec_alt_strategies_impl_cli/`     | The command line copying the strategies: `cli` extra     |
| `packages/*/tests/`                                     | Each package's tests                                     |
| `packages/*/hatch_build.py`                             | Gives each package the root README.md as its PyPI readme |
| `.github/workflows/`                                    | CI, CodeQL, and one release workflow per package         |

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
  directory's listing as it arrives, and never holds a whole tree. A directory's listing comes whole, as fsspec's
  `walk` and `os.walk` give it.
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

A storage is implemented in two parts, each with one concern:

- a **storage handler** is how the storage is accessed. A developer writes it: the storage's primitives, its paths,
  its profile, its limits, and its clients, opened and closed. It subclasses its family's, `TreeFileOperationHandler`
  or `FlatFileOperationHandler`, which runs the family's strategies;
- a **filesystem**, `BaseFileSystem`, enforces the contract between the client code and the storage. It implements
  fsspec's whole public API on one handler, running the handler's strategies as its recursive methods. Nobody
  overrides its methods.

The filesystem's side of the contract is what the client relies on, whatever the storage:

- the public API, answering and raising as fsspec's methods do;
- each path verified by the handler before the storage is accessed;
- each call told to the monitors;
- the client side's cache, fsspec's `dircache`, updated and dropped as the storage changes.

So nothing a handler does can skip a part of the contract. The strategies take handlers, never filesystems, and a
handler knows nothing of the client. A convention the library keeps, e.g. a path verified or a listing dropped after
a change, is kept in one place: the filesystem.

```text
_PathHandler                           common/handler.py: paths
├── StorageHandler                     common/handler.py: lifecycle, profile, limits, primitives
│   └── FileOperationHandler           common/handler.py: + Strategy, of common/strategy.py
│       ├── TreeFileOperationHandler   tree/handler.py
│       │   └── FsspecTreeHandler      wrapper/tree.py
│       │       └── AdlfsHandler       wrapper/adlfs.py
│       └── FlatFileOperationHandler   flat/handler.py
│           └── FsspecFlatHandler      wrapper/flat.py
│               ├── S3fsHandler        wrapper/s3fs.py
│               └── GcsfsHandler       wrapper/gcsfs.py
└── RoutingHandler                     router.py: no primitives

AsyncFileSystem                        fsspec's
├── BaseFileSystem                     common/filesystem.py: also a storage's, under a router
│   └── FileSystemWrapper              wrapper/filesystem.py
└── RoutingFileSystem                  router.py
```

#### The handler

`StorageHandler` holds what both families share:

- **Its lifecycle.** It is made without a request. `await open()` loads its profile with `load_profile()`, once.
  `await aclose()` releases what `open()` made, and a later call opens it again. Its sync `close()` is the
  filesystem's: it releases at once what it can without awaiting, e.g. threads. A handler may refuse calls after it,
  as a closed filesystem does, so `aclose()` never calls it.
- **Its clients,** by a mixin of `common/clients.py`. `SyncClientSupport` runs a sync client's calls in threads of
  the handler's own. `AsyncClientSupport` makes an async client's clients by `create_clients`, in the handler's own
  fields, once `ensure_clients` asks for them on the running loop. It notes the loop, and closes them on it by
  `close_clients`: awaited there, or sent there as a task.
  - A closing the sync `close` sent is awaited before new clients are made, so it never sees them.
  - A loop ended leaves its clients dropped, as nothing can close them.
  - A closing's failure is dropped unlogged: a storage's message may carry credentials.
- **Its profile,** `profile`, as "Storage profiles" says.
- **Its limits,** `operation_limit` for the strategies' requests and `io_limit` for reads and writes of data. The
  callers hold the permits, never the primitives: the strategies hold one of `operation_limit` around each primitive.
  Data is the exception: the adaptive files hold one of `io_limit` per part, and a handler's own `cat_file`,
  `pipe_file`, `get_file` or `put_file` one per request it sends.
- **Its paths:** its `protocol`, `strip_protocol` and `unstrip_protocol`, its `root`, and the path methods below.
- **The primitives of single paths:** `ils`, `info`, `rm_file`, `rmdir`, `makedirs`, `cp_file` and `mv_file`.
- **Its files:** `open_file`, with Python's sync API, for fsspec's sync `open`, and `open_async_file`, whose methods
  are awaited, for `open_async` and the strategies. Both raise NotImplementedError by default. A handler gives its
  files either API, e.g. through `AsyncFileAdapter`, and supports fsspec's transactions in `open_file`, as it chooses.

Beside the primitives, `StorageHandler` declares methods with defaults a handler may override:

- `cat_file`, `pipe_file`, `get_file` and `put_file` move data whole, by default through `open_async_file`. A handler
  may send a request of its own instead, e.g. one ranged download. The defaults hold no permit: the files take their
  own.
- `sign(entry, expiration)`, `created(entry)` and `modified(entry)` answer fsspec's `sign`, `created` and `modified`.
  They are async, as a storage may need a request, e.g. Azure's user delegation key to sign with. Their defaults raise
  NotImplementedError, as fsspec's do.
- `invalidate_cache` drops what the handler keeps of listings, by default nothing.

The filesystem calls only what `StorageHandler` declares, never finding a method by `getattr`.

`FileOperationHandler` adds the `Strategy` mixin's methods, one per strategy: `walk`, `find`, `glob`, `du`, `rm`,
`rm_files`, `cp`, `cp_files`, `mv` and `mv_files`. Each takes its strategy's arguments but the handler, which is
`self`. It returns an operation, not started, which the filesystem runs. A copy or move takes the destination's
handler, `dst`. The mixin also declares how its batch strategies run in bulk: `rm_batch_size`, `cp_batch_size` and
`mv_batch_size`. That is no concern of `StorageHandler`, whose primitives each act on one item.

Each family's handler class implements the strategy methods by its strategies, but the batch ones, whose defaults are
the mixin's. It adds the primitives its strategies call:

| Class                      | Adds                                                                                          |
|----------------------------|-----------------------------------------------------------------------------------------------|
| `TreeFileOperationHandler` | `ils(entry, recursive=True)`, by default a directory per request, and optional `rm_tree`, `mv_tree`, `cp_tree` |
| `FlatFileOperationHandler` | Required `ils(entry, recursive=True, prefix=...)`                                             |

`ils` is a handler's one listing primitive: a directory's items, or with `recursive` all below it.

A handler's class tells its family. A storage's handler overrides a strategy method to choose another strategy, e.g.
its own.

`rm_files(entries)` is the strategy deleting files a caller found. The recursive deletions and moves of both families
send it their files in batches of the handler's `rm_batch_size`, 1 by default. Its default, `rm_files._rm_each_file`,
deletes each by `rm_file`, as many at once as `operation_limit` allows. A storage deleting several files by one
request overrides it with `rm_files.rm_by_request` and that request. It declares its batch, e.g. 1000 on S3 or 256
for Azure's Blob Batch.

`cp_files(transfers)` and `mv_files(transfers)` are the strategies copying and moving files within one storage. Each
`FileTransfer` is a source's entry, where it goes, whether that is known missing, and its parent's entry. Their
defaults, `cp_files._cp_each_file` and `_mv_each_file`, send each by `cp_file` or `mv_file`. A storage changing many
files by one request, e.g. the Hugging Face Hub's commit, overrides them with `cp_by_request` or `mv_by_request`. It
declares `cp_batch_size` or `mv_batch_size`.

- The recursive copies and moves within one storage send their files to them, unless the caller gives its own
  `cp_entry` or `mv_entry`. Between two storages, each file streams on its own.
- A file joins its batch once its destination's directory is ready, whatever the directory. A batch goes once full,
  or once the walk has ended with no directory left to make.
- Each file's outcome then goes to the jobs' `done` or `failed`, as a file's own job would. A copy a destination
  refused to overwrite is replaced on its own, as without batches.

A batch's job holds no permit of `operation_limit`: its operation's requests take theirs, and a permit held around
them would wait for itself once all were so held.

#### Paths and entries

A path a strategy uses is a `StoragePath`, a `NewType` of `str`: no protocol, no trailing "/". Only the handler makes
one, so each is verified:

- `verify_path(path)`, abstract, raises ValueError for a path the storage refuses, or the handler does not serve,
  e.g. another account's. It sends no request. It sees the path as the caller gave it, a URL or not, so that it
  checks what a URL names beyond the path, e.g. abfs's account. A handler checking names strips the path first.
- It sees glob patterns too, given to `glob`, or to a copy or deletion. A handler whose names refuse `*`, `?` or `[`
  checks a pattern's fixed prefix instead, by `paths.fixed_prefix`: the directory the glob starts from. The names
  it matches below are listed, never verified again.
- `verified(path)` is a caller's path once verified, then stripped by `strip_protocol`, and `entry(path)` its entry,
  `{"name": path}`. The filesystem passes its caller's paths to them as they come: `verified` is where a path loses
  its protocol. With `strip_only_protocol=True`, a trailing "/" stays, for the strategies taking a path as given.
- `join(directory, *paths)` joins paths as `os.path.join` does, verifying the path built once. `_split` splits each
  path into segments by the class's `delimiter`, "/" by default, and `_verify_segment` refuses a segment that is
  empty, ".", "..", or holds the delimiter. A storage overrides either. Unlike `os.path.join`, a path starting with
  the delimiter never starts again from the root.
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
  write. A file to write may not exist yet: its entry then holds its name, as `entry` makes it from a verified path.
  It also holds the file's final `"size"` where the caller knows it, as `pipe_file`'s and `put_file`'s defaults and a
  copy between storages put it there. A storage needing the size up front, e.g. for an upload session, then need not
  hold the file.
- `makedirs`, `cp_file` and `mv_file` take `missing`: True where the caller knows nothing is at the path. A copy or
  move says so below a destination it found missing, as nothing is there unless another client puts it. A storage
  whose creation never conflicts, e.g. Google Drive, which makes a second item of the name, skips its lookup. Any
  other may ignore it.
- `makedirs`, `cp_file`, `mv_file`, `cp_tree` and `mv_tree` take `parent` too: the entry of the path's parent, where
  the caller holds it, else None. A write's entry holds it as `"parent"`. A storage addressing items by id, e.g. Box
  or Google Drive, then need not resolve the parent's path.
- `makedirs` returns the entry of the directory it made, where the storage tells it. A copy or move keeps it, and the
  entry of a destination directory its `info` found, as the parent of what goes inside.

#### The filesystem

`BaseFileSystem` is the one filesystem of both families:

- It takes a handler, as a required argument, and keeps it private, as `_handler`: its strategy methods return
  operations, which callers never get. It exposes neither its profile nor its limits: they are the handler's.
- It passes each path a caller gives to the handler's `verified`, or to `entry` where the path names an item, as
  given. They verify it before they strip its protocol.
- Its data methods, `_cat_file`, `_pipe_file`, `_get_file` and `_put_file`, call the handler's, whose defaults go
  through its files.
- `_walk`, `_find`, `_du`, `_glob`, and `_rm`, `_copy` and `_mv` when recursive, run the handler's `walk`, `find`,
  `du`, `glob`, `rm`, `cp` and `mv`. A copy or move of a single file runs the handler's `cp` or `mv` too, so that a
  move renames where the storage can.
- The paths of a copy, a move, a deletion and a glob are the handler's `verified` with `strip_only_protocol=True`,
  which keeps a trailing "/".
- It answers and raises as fsspec's methods do, and tells the monitors of each call. A method raises its strategy's
  failures: one as it is, several in an ExceptionGroup.
- It has the sync methods fsspec does not make, `mv` and `rmdir`, and fsspec's `dircache`, as the next section says.

A monitor's `Call` names the filesystem whose method was called, by its `_unwrapped()`. That is the filesystem
itself, or, for `FileSystemWrapper`, the filesystem it wraps, as its user made it. Under a router, a call names the
filesystem the router made on the storage's handler, and the routers in `Call.via`.

A storage's own class, by convention, is a small subclass of it:

```python
class MyFileSystem(BaseFileSystem):
    protocol = MyHandler.protocol
    root_marker = MyHandler.root
    _strip_protocol = MyHandler.strip_protocol
    _get_kwargs_from_urls = MyHandler.options_from_url

    def __init__(self, account, *, io_concurrency=8, **kwargs):
        super().__init__(MyHandler(account, io_concurrency=io_concurrency), **kwargs)
```

It takes storage options, not a handler, so that fsspec makes it from a URL or a registry, caches it and pickles it.
It names its paths as its handler class's, as fsspec reads them of the class before a filesystem is made. So does
`_get_kwargs_from_urls`, the handler class's `options_from_url`: the storage options a URL names beyond its path,
e.g. an account or a host. fsspec asks for them before a filesystem, or a handler, is made. The class overrides
nothing else.

Its `__init__` is what its users call, so its signature names each option, repeating its handler's:

- No helper splits the options for it: a signature tells its users and their tools what it takes.
- It passes fsspec's options on as keywords. `BaseFileSystem` and `RoutingFileSystem` name them, so any other option
  raises TypeError. fsspec's own constructors take any keyword, silently ignoring a misspelled one.
- Options whose names another library decides, e.g. those of an Azure token credential, are the only catch-all. The
  class then names fsspec's options too, so that its `**` holds those alone, and the handler passes on those that
  the credential's constructor names.
- fsspec's `batch_size` is not taken: the handler's `io_concurrency` bounds the paths handled at a time.

`FileSystemWrapper` serves a wrapped fsspec filesystem's paths: as it is made, it takes the protocol, root and
stripping of the handler it is given. It is never cached, as fsspec would key it by its handler's repr, which two
wrappers of one filesystem share.

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
- `invalidate_cache(path)` drops the listings of `path`, of what is below it, and of its parents. It passes on to the
  handler, e.g. a wrapper's, whose wrapped filesystem manages its own cache.

### Choosing

A developer chooses, per storage:

- a family, by the handler class a handler subclasses, whose strategy methods run the family's strategies;
- another strategy for one method, by overriding it, e.g. `glob` with one of the storage's own;
- a storage's recursive listing, by answering `recursive` in its `ils`, or a directory per request, by `super().ils`;
- a storage's recursive delete, by implementing `rm_tree`, which `deletes_directories=True` declares;
- a storage's rename of a directory, by implementing `mv_tree`, which `renames_directories=True` declares;
- a storage's copy of a directory, by implementing `cp_tree`, which `copies_directories=True` declares.

`monitored(fs)` picks the handler of a filesystem from `known_implementations`, a table the developer may replace. An
entry's `handler` names a `FileOperationHandler` by a string, as the interface never imports the strategies. A bare
name is one of the built-in handlers in `monitor.py`'s `_BUILT_IN`, e.g. the generic `FsspecTreeHandler` and
`FsspecFlatHandler`, and a full name is imported. The handler's class is what chooses the strategies. The profile
names no family, as a family's adaptation to its storage is its handler's.

Only a generic handler takes its profile, limits and batch sizes from its entry, as it knows nothing of the storage.
A dedicated handler, e.g. `S3fsHandler`, knows its storage, so it declares them as class attributes, beside the code
relying on them. `monitored()` makes it with the filesystem alone, and refuses an entry giving it options.

### Operations

Each recursive method runs its handler's strategy method, a private strategy of its family, e.g. `tree/rm.py`'s
`_tree_rm`:

- It takes the handler it works on: `_tree_rm(handler, path, maxdepth=...)`. A copy or move takes two, `src` and
  `dst`, which are the same handler within one storage.
- It reads what it needs from the handler: the primitives, `operation_limit` and `profile`, once open.
- The only functions it takes are a copy's `cp_entry`, and a move's `cp_entry` and `mv_entry`. Each is a `_Transfer`,
  taking the source's entry. Between two storages, `cp_entry` defaults to `common/transfer.py`'s `_across`. A move's
  `mv_entry` defaults to `cp_entry`, then the source's `rm_file`.
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
- With `maxdepth`, a move keeps the files deeper than the limit, and the directories holding them. fsspec's deletes
  the whole source.

### Jobs

Two modules serve the strategies changing a storage:

- `common/placement.py`'s `_Placement` places each path of a copy or move, as fsspec's `copy` does, whatever the
  family. It builds each target by the destination's `join`, and keeps the hints it learns on the way.
- `common/jobs.py`'s `_Jobs` runs the flat strategies changing a storage: it finds what to change and does it at
  once. Each request holds a permit of the handler's `operation_limit`, through a `_Queue`. `_FlatJobs` says where
  the items come from.

A tree copy or move uses `_Placement` alone: the traversal does the rest. A flat one joins both, as
`_CopyOrMove(_CopyJobs, _FlatJobs)`, where `_CopyJobs` is `_Placement` and `_Jobs`.

### Copies between storages

A copy or move between two handlers sends each file by `common/transfer.py`, unless its caller gives another way:

- It reads and writes by the source's and the destination's `open_async_file`, a block of the source's
  `transfer_size` at a time, 8 MiB by default. So each handler keeps the primitives of its own storage only.
- It leaves no part of a file. Where the destination's profile declares `atomic_writes`, it writes in place.
  Otherwise it writes to a temporary name beside the destination, then moves it there by `mv_file`, deleting it on a
  failure.
- It checks the bytes written against the source's size, the reader's or else the entry's. A size that differs raises
  OSError inside the writer, which then completes nothing.
- A move copies, then deletes the source by its `rm_file`. A source whose copy failed is kept.

### Routers

`RoutingFileSystem`, in `router.py`, passes each call to the filesystem of the path's storage. Where each path goes is
its `RoutingHandler`'s. A developer implements the first three of its methods, and overrides the last two where paths
lie above the storages:

- `verify_path(path)` checks the router's own paths;
- `route(path)` gives the handler of the path's storage and its path there, or None above every storage;
- `outer(handler, path)` gives the router's path of a path of one of its handlers;
- `roots(entry)` lists a directory above the storages. Its default raises FileNotFoundError, for a router with no such
  paths, e.g. one whose URLs name their storage;
- `root_info(entry)` describes a path above the storages, by default by its parent's `roots`.

`route` makes each handler once, by `inner(key, make)`, without a request. The routing handler's `aclose()` closes
them, and its sync `close()` releases them at once, as the filesystem's `close()` does.

For each storage routed to, the router makes a `BaseFileSystem` on its handler, uncached. The handler must be a
`FileOperationHandler`, which runs its own strategies. Each path is checked twice: by the routing handler, then its
path in the storage by the storage's handler.

`route` may give another routing handler, which routes the inner path again. Each level checks its own path, and the
names answered are mapped back through each level's `outer`. The roots of a nested router are listed by it.

Above the storages, the router lists by `roots`, and hands each storage what is below it:

- A glob pattern spanning storages, e.g. one bucket per storage and `data*/x.txt`, is expanded a segment at a time
  over `roots` above them. Each storage it reaches then globs the rest by its own strategy, and a storage the pattern
  cannot match is never listed. A `**` above the storages, or a pattern ending there, is fsspec's glob.
- A `du` above the storages sums each storage's own, which sends no request per file. With `maxdepth`, it is
  fsspec's, which counts the levels above the storages.

Within one storage, a call is that storage's own. Between two, a recursive copy or move runs the source handler's
`cp` or `mv`, with `maxdepth` too. Each file goes by the routing handler's `cp_across` or `mv_across`, under its
`transfer_limit`. By default these stream the data, as the previous section says. A routing handler can copy on the
server instead, e.g. from a URL. Through nested routers, the outermost router's are used.

The router imports no family: each storage's handler runs its own strategies. Its transaction commits file by file,
so it is not atomic across storages.

### The tree strategies

The tree strategies list a directory with the handler's `ils`, its entries in any order, and a tree with its
`ils(entry, recursive=True)`. A recursive listing may leave a directory out, as object stores list only files: the walk
tells it from the paths below it.

By default, a tree handler's recursive `ils` runs `tree/handler.py`'s `_tree_ils`: depth first, a directory per
request, one at a time. The strategies never ask it, as they list a directory per request themselves, many at once:
it serves other callers, e.g. a filesystem's recursive listing. A handler listing a directory alone reaches it by
`super().ils(entry, recursive=True)`. The generic tree
handler lists so, but `AdlfsHandler` lists by Azure's own recursive listing. A storage may refuse a recursive listing
in a way only asking tells, e.g. a WebDAV server refusing `Depth: infinity`. Its handler's `ils` then falls back on
`super().ils` itself.

The profile's `recursive_ls_depth_first_preorder` declares the order of a storage's own recursive listing:

- True: each directory's descendants come together. That covers depth first, siblings in any order, and the byte
  order of names, as object stores list keys. A subtree nothing prunes is then streamed by one recursive listing.
- False, the default: nothing is promised, e.g. breadth first, as Dropbox's recursive listing documents no more. Each
  directory is then listed on its own, a failure failing only itself.

The strategies cannot tell one order from another. By the time an entry comes under a directory taken as done, that
directory may have been reduced, and a move may have deleted it. So the order is declared, never detected.

#### The traversal

The tree strategies run on `common/tree.py`'s `TreeTraversal`, with `common/glob.py`'s tasks:

- `VisitDirectory` lists a directory by one `ils`, and runs a task per item as the listing streams.
- `TraverseSubtree` reads a subtree from one recursive `ils`, as it streams, with no task per item. In byte order, a
  subdirectory listed apart from what is below it is held until its first path comes.
- `tree/walk.py`'s `_root_task` takes `TraverseSubtree` where the profile declares the order and nothing prunes the
  root, else `VisitDirectory`, which streams each subtree it can.

What a strategy does on each path is a `PathAction`: `map` before what is below the path, `reduce` after, with the
values of the items in it. So a strategy is a map-reduce:

| Strategies                             | Their action                                             |
|----------------------------------------|----------------------------------------------------------|
| `_tree_walk` top down                  | `_WalkTopDown`: each directory, which the caller prunes  |
| `_tree_walk` bottom up                 | `_WalkBottomUp`: each directory once all below it is     |
| `_tree_find`, `_tree_glob`, `_tree_du` | `_Found` and `_Du`, by `tree/find.py`'s `_read`          |
| `_tree_rm`                             | `_Deletion`: a directory deleted once all in it is       |
| `_tree_cp`, `_tree_mv`                 | `_Sending`: directories made first, each file as found   |

How the traversal goes:

- **Permits.** A task holds a permit of `operation_limit` around each page of its listing, never across it. A listing
  streaming into tasks would otherwise wait for room among them while they wait for permits.
- **Order.** The task taken next is the pending one with the first path, so the traversal goes quasi depth first.
- **Writes.** Each goes through `common/tree.py`'s `_write`. A close waits for those in flight, and none starts once
  the operation is cancelled.
- **Batches.** A file joins its directory's batch, sent once full: a batch never spans two directories.
- **Stopping.** Once cancelled, no task starts. Those running finish, their pages bound by the permits.
- **Failures.** A root's failure to list is raised, and each strategy answers as its fsspec method does. A directory's
  failure is emitted with its entry, as the storage raised it. A recursive listing broken off fails its whole subtree:
  nothing is listed again, as a storage resumes its own listing where it can.
- **Patterns.** They are expanded by the family's glob first, as copies place paths from the common prefix of all
  matches.

### The flat strategies

The flat strategies list with the handler's `ils(entry, recursive=True, prefix=...)`: every key below the directory
starting with the prefix, at any depth, in name order, as object stores list them. Without `recursive`, `ils` lists one
directory, for fsspec's own methods. The two are one request with or without a delimiter, as on S3, GCS and Azure.

- The strategies list by a raw prefix, split at its last "/" by `flat._keys._listing`. So `_flat_glob` lists only keys
  starting as its pattern does, e.g. `data` with the prefix `2024-` for `data/2024-*`.
- Name order tells when a directory is done, so `_flat_walk` streams its directories. Each key is listed once.
- Directories are derived from the keys, by `flat/_keys.py`. A key ending with "/" marks a directory.
- `_flat_rm` deletes in batches of the handler's `rm_batch_size`, by its `rm_files`. A batch may span buckets.
  `rm_files` yields each key deleted, and the error of each it could not delete, so that the others count as deleted.
- `_flat_mv` renames with `mv_file` within one storage whose profile says it renames files. Otherwise it copies, then
  deletes in batches.
- A listing error ends the operation: there is no other listing to go on with.

### Storage profiles

A `StorageProfile` declares what a storage does that fsspec leaves open. A strategy is right on any storage, and a
profile only saves requests or data:

| Field                   | Saves                                                                  |
|-------------------------|------------------------------------------------------------------------|
| `parents_required`      | False: the `makedirs` of each destination directory written to         |
| `copy_parents_required` | False: the `makedirs` of each directory the storage copies a file to   |
| `implicit_directories`  | True: those, and the deletion of each directory                        |
| `renames_files`         | True: the data copied by `_flat_mv`                                    |
| `renames_directories`   | True: every request in a directory moved whole by one `mv_tree`        |
| `deletes_directories`   | True: every request in a directory deleted whole by one `rm_tree`      |
| `copies_directories`    | True: every request in a directory copied whole by one `cp_tree`       |
| `case_sensitive`        | False: a path moved onto itself in another case, by one `mv_tree`      |
| `atomic_writes`         | True: the temporary file and the move of a copy between storages       |
| `recursive_ls_depth_first_preorder` | True: the names of a tree held until its listing ends      |

- A storage has one profile, which never changes. It is the handler's `profile`: `UNKNOWN` unless the handler sets
  what it knows, or what its `load_profile()` finds with a request.
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
- `recursive_ls_depth_first_preorder` tells of a tree handler's own recursive `ils` only, as "The tree strategies"
  says. It is False by default, never None: parents first is the order right on any storage.
- Any other field left None is unknown, and the strategy takes the way right on any storage.
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
