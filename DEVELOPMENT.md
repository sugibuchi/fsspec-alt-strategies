# Development

## Layout

The repository is a uv workspace of two packages, each built and published on its own:

- `packages/common`, the distribution `fsspec-alt-strategies-common`, is the interface the strategies share with
  their callers. It is installed.
- `packages/impl`, the distribution `fsspec-alt-strategies`, is the strategies, meant to be copied into a project,
  and the simulated filesystem their tests use.

The root holds the tooling: the ruff, pytest and coverage settings in `pyproject.toml`, CI, and the docs.

| Path                            | Contents                                                       |
|---------------------------------|----------------------------------------------------------------|
| `packages/common/src/fsspec_alt_strategies_common/` | The interface, installed: one for all the copies in a project. |
| `.../monitor.py`                | `State` and `OperationHandler`: what callers see of an operation, and its control. `Monitored`, `Call`, `CallCancelled`, `Monitor`, `monitoring` and `current_monitors`: watching the operations behind the filesystems' fsspec methods. `routing` and `current_routers`: the routers a call came through. |
| `.../profile.py`                | `StorageProfile`, what a storage does that fsspec's API leaves open. |
| `.../states.py`                 | The states the strategies report, e.g. `FindState` of `tree_find` and `flat_find`. |
| `.../wrapper.py`                | `KnownImplementation` and `known_implementations`: how each known filesystem class is wrapped, to be monitored. |
| `packages/common/tests/`        | The interface's tests, which never import the strategies.      |
| `packages/impl/src/fsspec_alt_strategies/` | The strategies, copied.                              |
| `.../monitor.py`                | `monitored(fs)`: the filesystem if `Monitored`, else wrapped as its known implementation says. |
| `.../routing.py`                | `AsyncRoutingFileSystem`, the base of a router: a filesystem passing each path to its storage's filesystem. |
| `.../streams.py`                | `streaming_copy`, a `copy_file` between two storages, a block at a time through their files. |
| `.../tree/`                     | The strategies walking a folder tree, a folder per listing.    |
| `.../tree/walk.py`              | `tree_walk`, the concurrent folder walk under the tree strategies.  |
| `.../tree/find.py`              | `tree_find`, a find listing several folders at a time.              |
| `.../tree/du.py`                | `tree_du`, a tree's size from its listings alone.                   |
| `.../tree/glob.py`              | `tree_glob`, a glob listing only the folders it can match in.       |
| `.../tree/rm.py`                | `tree_rm`, a recursive delete that deletes while it walks.          |
| `.../tree/cp.py`                | `tree_cp`, a recursive copy that copies while it walks.        |
| `.../tree/transfer.py`          | `tree_transfer`, `tree_cp` between two storages, each file by a `copy_file` the caller gives. |
| `.../tree/mv.py`                | `tree_mv`, a recursive move: a `mv_file` per file, or per folder where it moves folders. |
| `.../tree/filesystem.py`        | `TreeFileSystem`, an abstract async filesystem whose recursive methods are the tree strategies. |
| `.../tree/wrapper.py`           | `AsyncTreeWrapper` and `SyncTreeWrapper`: an existing filesystem, as a `TreeFileSystem`. |
| `.../tree/_jobs.py`             | The private engine's tree side: it feeds `common`'s with the walk's listings. |
| `.../flat/`                     | The strategies listing a flat namespace by key prefix, as object stores do. |
| `.../flat/walk.py`              | `flat_walk`, a walk of one listing of a prefix, its folders derived from the keys. |
| `.../flat/find.py`              | `flat_find`, a find streaming one listing of a prefix.          |
| `.../flat/du.py`                | `flat_du`, the size of the files under a prefix, from the listing alone. |
| `.../flat/glob.py`              | `flat_glob`, a glob listing only the keys starting as its pattern does. |
| `.../flat/rm.py`                | `flat_rm`, a recursive delete in batches, sent as the listing goes on. |
| `.../flat/cp.py`                | `flat_cp`, a recursive copy that copies each key as it is listed. |
| `.../flat/transfer.py`          | `flat_transfer`, `flat_cp` between two storages, each key by a `copy_file` the caller gives. |
| `.../flat/mv.py`                | `flat_mv`, a recursive move: a `mv_file` per key, or a copy per key and a deletion per batch. |
| `.../flat/filesystem.py`        | `FlatFileSystem`, an abstract async filesystem whose recursive methods are the flat strategies. |
| `.../flat/wrapper.py`           | `AsyncFlatWrapper` and `SyncFlatWrapper`: an existing filesystem, as a `FlatFileSystem`. |
| `.../flat/_keys.py`             | The private listing under them: a prefix's keys, and the folders they imply. |
| `.../flat/_jobs.py`             | The private engine's flat side: it feeds `common`'s with the keys of each root. |
| `.../common/`                   | Private machinery the families share, so that none imports another. |
| `.../common/filesystem.py`      | What the filesystem classes share: the two limits, and data moved through file objects. |
| `.../common/wrapper.py`         | What the wrappers share: the calls to the filesystem wrapped, awaited or in threads. |
| `.../common/operation.py`       | `Operation`, which a strategy returns, implementing `OperationHandler`, and how one is run to its end, as the monitors and fsspec's `callback=` watch it. |
| `.../common/jobs.py`            | The private engine running a strategy's jobs, e.g. deletions, while what to change is found, and the queue of requests under a concurrency limit, which the walk uses too. |
| `.../common/placement.py`       | Where a copy or move puts each path, as fsspec's `copy` does.   |
| `.../common/types.py`           | The types of the storage's operations, such as `ls` and `info`. |
| `.../common/paths.py`           | Paths and depths, as the strategies take them.                  |
| `.../common/patterns.py`        | Glob patterns, matched as fsspec's `_glob` matches them, by both globs. |
| `.../files/`                    | Base classes for a filesystem's file objects, by what its storage allows. |
| `.../files/adaptive.py`         | `AdaptiveReadableFile` and `AdaptiveWritableFile`, for range reads and writes in parts. |
| `.../files/fsspec_transaction.py` | Files written in a transaction, by strategy: a temporary path, a local buffer, or the storage's own. |
| `.../files/fsspec_async.py`     | `AsyncFileAdapter` and `SyncFileAdapter`, giving file objects the API of those `open_async` returns. |
| `.../files/_async_file.py`      | The private machinery under them: the loop, the concurrency limit, the requests. |
| `packages/impl/src/fsspec_alt_strategies_testing/` | `SimulatedFileSystem`, for tests and benchmarks: a package apart, so that copies leave it out. |
| `packages/impl/tests/`          | The strategies' tests, in the package's layout, with the shared fixtures in `conftest.py`. |
| `.github/workflows/ci.yaml`     | Lint and tests, on pull requests and pushes to `main`.         |

The interface depends on fsspec alone, and the strategies on fsspec and the interface. Both take their version from
the git tag, through hatch-vcs.

## Setup

All Python commands go through [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # both packages, editable, with the dev dependencies
uv run pre-commit install        # ruff and file checks before each commit
```

## Tests

```bash
uv run pytest                    # the tests of both packages, from the root
uv run pytest --cov              # with coverage of both packages
uv build --all-packages          # both distributions
uvx ruff check .                 # lint
uvx ruff format .                # format
```

CI runs ruff and pytest on every supported Python version.

- Tests run with pytest-asyncio in `auto` mode: an `async def` test needs no marker.
- Tests run on `SimulatedFileSystem`, from the public `fsspec_alt_strategies_testing`. It counts the requests a real
  storage would get, in `stats`, and can delay each by a latency and by its bytes over a bandwidth.
- Its two modes are an object store, where folders are only prefixes, and a hierarchical namespace, with real
  folders. Test a strategy on both, with the fixture `any_fs`.
- Its options `parents_required` and `overwrite` make writes fail as some storages do. Test a strategy's profile with
  them.
- The fixtures `flat_fs`, `hierarchical_fs` and `any_fs` are empty async instances. Set up files with `add_file` and
  `add_dir`, which make no request.
- Assert the requests a strategy makes, e.g. `fs.stats.calls == {"list": 1, "delete": 3}`. They are what the
  strategy is for.
- A strategy replacing an fsspec method answers as that method does. Test it against fsspec's own, called as
  e.g. `AsyncFileSystem._glob(fs, ...)` on the same filesystem.
- Test folders have no `__init__.py`. Share code through `conftest.py`. The interface's tests never import the
  strategies, as the interface never does.

## Design

The choices below shape the whole project. Keep to them, or change them on purpose.

### A third party to fsspec

The library is not part of fsspec. It offers alternatives, and never patches fsspec or its filesystems.

### Strategies are functions

Each algorithm is a stateless function. It takes the storage's operations as functions, e.g. `ls=fs._ls`, and the
operation's arguments. It keeps no state between calls.

- Users can call a function with any fsspec filesystem's operations, without subclassing.
- Each family's abstract filesystem wraps the functions as its methods, for implementations that want them by default.
- A function only uses fsspec's API that every implementation has, e.g. `_ls` and `_info` on async filesystems. So it
  works on any implementation.
- File objects and `Operation` are the exceptions, as they keep state between calls.

### Developers choose

Each fsspec feature may have several strategies. Developers of a filesystem pick those that suit their storage. The
library never picks for them.

- A tree walk, for example, wins on storages listing one folder per call. On object stores, one flat listing wins.
- So each strategy's docstring says what it assumes of the storage, when it wins or loses, and the calls it makes.
- A strategy takes only the storage's primitive operations, such as `ls`, `info` and `rm_file`, as functions. So
  picking one strategy never changes the behaviour of another.
- A strategy needing another, or a native operation of the storage, takes it as a parameter too.

### The strategy API

Every strategy follows the same shape, so that developers learn it once.

#### Calling

A strategy is a plain function returning an `Operation`. The operation's paths come first, then, keyword-only, the
storage's operations it uses and the options:

```python
limit = asyncio.Semaphore(32)  # Shared by all that uses the storage, files included.
rm = tree_rm(
    "data/2024",
    ls=fs._ls,
    info=fs._info,
    rm_file=fs._rm_file,
    rm_folder=fs._rmdir,
    concurrency_limit=limit,
)
async with rm as op:
    async for path in op:
        if isinstance(path, Exception):
            print("failed:", path)
print(op.state.files_deleted)
```

- The storage's operations are functions, not a filesystem. Any coroutine function called as fsspec calls its own
  serves, e.g. `ls(path, detail=True)`, so one may come from a storage's SDK rather than a filesystem. It also leaves
  room for another kind of operation later, e.g. a listing yielding pages.
- They are required: a strategy guesses nothing, e.g. which request deletes a folder. Only a native operation that
  saves requests is optional, e.g. `tree_rm`'s `rm_tree`: without it, the strategy does the work itself.
- Paths come without a protocol, as the operations take them. The method wrapping a strategy strips it first.
- An operation deleting files, such as `rm_file` or `rm_files`, is given only paths the strategy found to be files,
  by a listing or an `info`. So its implementation need not check the path again, e.g. to refuse a folder.

- `concurrency_limit` is an `asyncio.Semaphore`, of which each request holds a permit while it runs. Shared, it limits
  several operations, and the file objects, together: the caller chooses its scope.
- It is required, with no default. The right limit depends on the storage, high enough to hide each request's latency
  and low enough not to be throttled, so the caller sets it. fsspec's `batch_size` does not fit: it is how many
  coroutines fsspec gathers at once, sized from the limit of open files.
- Errors in the arguments, such as a bad `maxdepth`, are raised by the call.
- Errors ending the whole operation are raised by the iteration, as fsspec raises them. An example is a missing root,
  which raises `FileNotFoundError`.

#### Operations

An `Operation`, in `common/operation.py`, does its work as it is iterated, a few requests ahead of the caller. Its
control is the interface's `OperationHandler`, which a `Monitor` is given, so that any copy's operations are
stopped the same way.

- It yields what it did, e.g. each path deleted, mixed with the error of each path that failed, as an exception object.
  Errors on single paths are never raised: the caller decides what to do with each.
- An error yielded carries its path in a note, e.g. "Deleting 'a/b' failed.", as nothing else in the stream does.
- Its `state` is a frozen dataclass of counts so far, such as the files deleted, and the path last worked on. Each
  strategy has its own state class.
- It does nothing until iterated, and a slow loop body slows it. This is backpressure: it never runs far ahead.

Its stopping follows `asyncio.Task`'s, so that it needs no learning:

- `cancel()` asks it to stop. It starts no new request, but the loop goes on through the outcomes of those already sent,
  then ends. It returns False if the operation is already done.
- `cancelling()` counts the stops asked. `cancelled()` tells whether it ended after one, and `done()` whether it ended.
- Cancelled before its first iteration, it sends nothing at all.
- Leaving the loop early, or `aclose()`, stops it at once without yielding what is in flight. Read requests in flight
  are then cancelled. Requests changing the storage, such as deletions, are waited for, as their outcome would
  otherwise be unknown.
- Use it as an `async with` block, so that leaving the block closes it.

Cancelling the task iterating it, e.g. with `asyncio.timeout`, raises `CancelledError` as usual. The operation then
cancels its requests and never swallows the error.

A `CancelledError` raised by one of the operation's own requests, e.g. from an SDK, is no error of a path. It ends the
whole operation, as any cancellation does, rather than being yielded.

Strategies yield in the order the storage answers. A caller wanting fsspec's order sorts, as fsspec's `_find` and
`_glob` do.

#### Attaching

Developers attach a strategy by overriding the fsspec method themselves, in a few lines:

```python
class MyFileSystem(AsyncFileSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # One limit for the filesystem's operations and files. It belongs to the
        # loop it is first used on, as fsspec's own requests do.
        self.limit = asyncio.Semaphore(32)

    async def _rm(self, path, recursive=False, **kwargs):
        if recursive:
            op = tree_rm(
                self._strip_protocol(path),
                ls=self._ls,
                info=self._info,
                rm_file=self._rm_file,
                rm_folder=self._rmdir,
                concurrency_limit=self.limit,
            )
            errors = [e async for e in op if isinstance(e, Exception)]
            if errors:
                raise ExceptionGroup("rm failed", errors)
            return
        return await super()._rm(path, recursive=recursive, **kwargs)
```

- The override can choose a strategy per instance, e.g. from the storage's settings.
- fsspec makes the sync method, `rm`, from the async one, `_rm`. So sync callers get the strategy too.
- The library adds no mixins or descriptors: there is no machinery to learn.

Or they subclass a family's abstract filesystem, `TreeFileSystem` in `tree/filesystem.py` or `FlatFileSystem` in
`flat/filesystem.py`, which overrides all of fsspec's recursive methods with the family's strategies:

- The subclass implements the storage's primitives, e.g. `_ls`, `_info` and `_rm_file`, and `open_async` and `_open`.
  Each family's class declares them all as abstract methods, so a subclass missing one fails when instantiated.
- It takes two required limits: `operation_concurrency`, for the strategies' requests, and `io_concurrency`, for the
  reads and writes of data. `open_async` gives the files it opens `io_limit`, as their concurrency limit.
- A storage whose SDK blocks backs the primitives with `_blocking`, e.g. `return await self._blocking(client.list,
  path)`. It runs the call on the filesystem's own threads, as many as its two limits allow at once.
  - They are made when first needed. `close()` shuts them down without waiting, as does collecting the filesystem.
  - A cancelled call raises `CancelledError` at once, but its thread runs to the end, as a thread cannot be stopped.
- `_cat_file`, `_pipe_file`, `_get_file` and `_put_file` go through `open_async`'s files, so data moves as the file
  classes move it.
- The failures of a strategy are raised: one as it is, several in an ExceptionGroup. A folder that fails to list is
  yielded empty by a walk, as fsspec's `walk` yields it, and `on_error` handles every folder's failure, not only the
  root's as fsspec's does.
- fsspec has no async `mv`, nor builds a sync one from `_mv`: each class wraps its `_mv` itself.
- A `mv` of a single file runs the family's move too, even when not recursive. fsspec's own copies and deletes it,
  never calling `_mv_file`, so a storage's rename would be lost.

Their callers see only fsspec's answers, not the `Operation` behind them. The monitors of `monitor.py` show it without
changing fsspec's API. The filesystems run their operations with `common/operation.py`, which tells the monitors.

- A `Monitor` subclass overrides `started`, `yielded` and `ended`, each awaited where the operation runs.
- Each method takes the `Call`, then the item yielded or the error ending the call, then the `state`, then the
  `OperationHandler`. `started` takes no item.
- A `Call` names the method, its paths, and the filesystem called. A call to a wrapper names the filesystem wrapped.
- Its `via` names the routers it came through, the outermost first, as `current_routers()` tells them. A router enters
  `with routing(self):` around each call it passes on. The call's paths are then the inner filesystem's own.
- `with monitoring(monitor):`, or `async with`, sets a context variable. Every call within the block is watched,
  whoever makes it, and blocks nest.
- The filesystems read the monitors with `current_monitors()`, so that the context variable stays the interface's own.
- fsspec's sync methods see the block too: `sync()` runs the call in a copy of the caller's context.
- A monitor stops a call with `op.cancel()`. The call then raises `CallCancelled`, an Exception, since fsspec's callers
  expect a call to finish or raise, and an `asyncio.CancelledError` would end fsspec's `sync()` with no answer.
- fsspec's own `callback=` is a monitor too. `copy`, `mv` and `rm` add one per path done. `_get_file` and `_put_file`
  set the size in bytes and add each chunk, as fsspec's own filesystems do.
- Only the calls run by a strategy are watched. A call fsspec answers itself, e.g. a copy that is not recursive, is not.

The strategies themselves take no callbacks: their callers iterate the `Operation`. Monitors exist for the callers of
fsspec's methods, who cannot.

Or they wrap an existing filesystem as one, to use the strategies and monitors without writing a class:

- `AsyncTreeWrapper` and `AsyncFlatWrapper` wrap an async filesystem. Its coroutines are awaited on its own loop, and
  the wrapper runs on that loop too.
- `SyncTreeWrapper` and `SyncFlatWrapper` wrap a sync one. Each of its calls runs in a thread, within the limits.
- A wrapper is light and not cached: it is made on demand, e.g. to watch one call, and dropped after.
- Its data methods, such as `_cat_file`, are the wrapped filesystem's own, under `io_limit`.
- A flat wrapper lists a prefix with the wrapped `find` of its folder, and deletes a batch with its `rm`.
- A wrapper takes its limits and its `storage_profile` as arguments.
- Nothing in the strategies chooses a wrapper for a filesystem: the developer does.

`monitored(fs)`, in `monitor.py`, wraps a filesystem as the developers listed in `known_implementations` chose:

- A filesystem inheriting `Monitored`, such as a `TreeFileSystem`, is returned as it is.
- Any other is wrapped as the entry of its class, or of the nearest class it inherits, says. An entry gives the
  family, whether it is async, the two limits, the profile, and a flat wrapper's `rm_batch_size`.
- `implementations=` replaces the interface's table with the caller's own.
- A class no entry names raises `ValueError`. With `wrap_unknown_fs=True`, it is wrapped the way right on any storage
  instead: as a tree, one request at a time, with an unknown profile.
- It imports a family's wrappers only when it wraps a filesystem with them, so a copy may leave out the other family.

Choosing the class chooses the family: a developer wanting one strategy of another overrides that method, as above.

Each family's class has `_transfer(path1, other, path2, *, copy_file=None)`, declared on `_StrategyFileSystem`. It
returns the source family's transfer strategy, not yet started, so that a caller copying between storages needs to
know neither family:

- `other` is any async fsspec filesystem. Its `_info`, `_makedirs` and `_rm_file` are the destination's operations.
- Its `storage_profile` is the destination's profile. A filesystem without one is unknown, `StorageProfile()`.
- `copy_file` streams through both `open_async` by default. A caller may give a server-side copy instead.
- The requests hold the source's `operation_limit`, as `_copy`'s do.

#### Routers

Some namespaces span several storages, each with a filesystem of its own, e.g. a cloud's volumes. A router passes each
call to the filesystem of the path's storage. `AsyncRoutingFileSystem`, in `routing.py`, is its base: a plain
async fsspec filesystem, knowing no family and importing none.

A subclass implements three methods:

- `_route(path)` gives the inner filesystem and its path, or None above every storage. It keeps a trailing "/".
- `_outer(fs, path)` gives the router's path of an inner one, so that every name answered is the router's.
- `_roots(path)`, a coroutine, lists a folder above the storages, as `_ls` with `detail=True` does.

It may override two hooks:

- `_same_storage(fs1, path1, fs2, path2)` names the one filesystem copying or moving between two storages' paths
  itself. It returns that filesystem and both paths as its own, or None for a transfer. By default, it returns
  `(fs1, path1, path2)` only when `fs1 is fs2`.
  - Two filesystems may differ yet share a storage, e.g. through one client. The subclass then maps the other's path
    to the one it returns, as no namespace is assumed shared.
  - `path1` may be a list, of one call's sources. The paths it returns keep their shape.
- `_server_copy(fs1, path1, fs2, path2)` copies a file across without the data passing through the router. It returns
  False where it cannot, and the file is streamed instead. It raises on any other failure.

How it passes calls on:

- The primitives, such as `_info`, `_cat_file` and `_open`, call the inner filesystem's own, under its inner path.
- The recursive methods, such as `_find`, `_glob`, `_rm` and `_copy`, call the inner one's own too, so that its
  strategies run.
- The paths of one call, e.g. a list given to `rm`, are grouped by storage, and the groups run at once.
- Above the storages, `_ls` and `_info` answer from `_roots`. `find`, `du` and `walk` are fsspec's own walk, a folder at
  a time, handing each storage's folder to its filesystem.
- A glob whose literal prefix routes goes whole to that storage. Any other is fsspec's own, over `_roots`.
- A change above the storages raises PermissionError.
- An inner filesystem's error gets a note naming the router's path. Its `filename`, if the inner path, becomes the
  router's.
- fsspec makes no sync `mv` or `rmdir`, so the router has its own, as the families' classes do.
- `test_routing.py` fails on any fsspec method the router neither passes on nor lists as inherited on purpose.

A copy or move between two storages is the source's `_transfer`, so the source's family lists it:

- Its `copy_file` tries `_server_copy`, then `streaming_copy`, under the router's `transfer_limit`.
- An inner filesystem without `_transfer`, a plain fsspec one, gets fsspec's own copy, a `_cp_file` per file.
- A move deletes each source file as soon as its copy succeeds, then the source folders left empty.
- The call is the router's own, as a monitor sees it.

A list of sources in several storages is copied or moved per storage, as a list in each:

- Each root goes into the destination under its name, as a family's copy of a list places it.
- Two roots of one name in different storages are checked by the router, as no inner filesystem sees both. The first
  in path order goes, and each other raises FileExistsError.
- A pattern matching in several storages is expanded first. Each storage's matches are placed from their own common
  prefix.

The rest of a router:

- Its inner filesystems are async, on the router's loop. A sync one raises TypeError: wrap it, e.g. in a
  `SyncTreeWrapper`.
- A subclass makes them when first needed, with `_backend(key, make)`, e.g. one per volume. `close()` closes them.
- Pickling keeps the constructor's arguments, as fsspec's does, so a copy makes its inner filesystems again.
- It has no `storage_profile`: `storage_profile_of(path)` gives the storage's. Its only limit is
  `transfer_concurrency`.
- Its transaction is its own. `_open` passes `autocommit` on, and fsspec adds each file to the router's transaction.
  The commit goes file by file, so it is not atomic across storages.
- In the transaction, `_pipe_file` and `_put_file` write through `open`, in a thread, so that their files join it.
  Outside, they are the inner filesystem's own.
- `open_async` takes no part in the transaction, as fsspec's knows none. Its files are the inner filesystem's own,
  written as they close. Neither does a copy between two storages, which streams through `open_async`.
- Each call it passes on runs within `routing(self)`, so that a monitor's `Call.via` names it.

A router of volumes might read as below. Its catalogs and schemas are above the storages. Each volume's filesystem
takes full paths, so that one volume's filesystem can copy to another's paths:

```python
class VolumesFileSystem(AsyncRoutingFileSystem):
    protocol = "volumes"

    def _route(self, path):
        parts = path.split("/", 3)  # catalog/schema/volume/path
        if len(parts) < 3:
            return None
        fs = self._backend(tuple(parts[:3]), lambda: self._volume(*parts[:3]))
        return fs, path  # The volume's filesystem takes full paths.

    def _outer(self, fs, path):
        return path

    async def _roots(self, path):
        names = await self.client.list_children(path)  # Catalogs, or schemas.
        return [{"name": n, "type": "directory", "size": 0} for n in names]

    def _same_storage(self, fs1, path1, fs2, path2):
        # One API copies between volumes. Their paths are full, so fs1 takes path2 as it is.
        return (fs1, path1, path2) if fs1.workspace == fs2.workspace else None
```

A router of an account's containers, each with its own client, might route on the first segment. Its server copy is
the storage's copy from a URL, refused across accounts:

```python
class AccountFileSystem(AsyncRoutingFileSystem):
    def _route(self, path):
        container, _, inner = path.partition("/")
        if not container:
            return None
        return self._backend(container, lambda: self._container(container)), inner

    def _outer(self, fs, path):
        return f"{fs.container}/{path}".rstrip("/")

    async def _roots(self, path):
        return [{"name": c, "type": "directory", "size": 0} async for c in self.list_containers()]

    async def _server_copy(self, fs1, path1, fs2, path2):
        if fs1.account != fs2.account:
            return False
        await fs2.copy_from_url(fs1.url(path1), path2)
        return True
```

#### The tree strategies

The strategies in `tree/` walk with the walk's private engine, `_listings`, which yields each folder's listing. They
come in two layers, by what they send:

| Strategies | They send | They build on |
|---|---|---|
| `tree_find`, `tree_du`, `tree_glob` | Listings, and at most one `info` for the root | `_find_listings` |
| `tree_rm`, `tree_cp`, `tree_mv`, `tree_transfer` | Requests changing the storage, besides the listings | `_TreeJobs` |

`tree_transfer` copies between two storages. It takes the source's `ls` and `info`, the destination's operations
with a `dst_` prefix, and a `copy_file` moving one file across. Its profile is the destination's. It shares `tree_cp`'s
jobs: `_Placing` asks the destination's `info`, and refuses no path into itself, as the storages are apart.

`tree_walk` is `_listings` itself, made public. It raises for a missing or file root, as fsspec's `walk` does.

`_find_listings`, in `tree/find.py`, treats the root as fsspec's `_find` and `_glob` do. A missing root finds nothing, and a
file root yields its own `info` entry when asked. A folder root yields its own entry when asked, e.g. for `withdirs`.
The strategies that only read never handle the root themselves.

Those changing a tree, `tree_rm`, `tree_cp` and `tree_mv`, run their requests with a second private engine, `_Jobs` in
`common/jobs.py`. It starts the jobs for each item found, e.g. a listed folder's file deletions, and runs them along
with the search still in flight. It also drains them on `cancel()`, and waits for them on closing. Each family feeds
it from a source of its own: `_TreeJobs`, in `tree/_jobs.py`, with the walk's listings. A strategy subclasses that,
saying what a listing queues, how a job runs, and what its outcome is. A copy or move places its paths with
`_Placing`, in `common/placement.py`.

They take glob patterns, as fsspec's `rm`, `copy` and `mv` do. A pattern is expanded with `tree_glob` first, and
the folders it matches are then walked at once. The expansion is not streamed into the jobs: `tree_cp` and `tree_mv`
place paths relative to the common prefix of all the matches, as fsspec does, which is known only once the glob ends.
The glob lists only the folders down to the pattern's depth, so the bulk of the walk still overlaps with the jobs.

Their names start with `tree_`, e.g. `tree_rm` and `tree_glob`. Code using fsspec already has `rm` and `glob`, as
methods and modules. The prefix keeps an import from shadowing them, and tells the reader which strategy is used.

`tree_walk` is meant to replace fsspec's `walk`, so it already has two of its options:

- `topdown=False` yields a folder only after all the folders below it.
- `on_error` takes fsspec's values: `"omit"`, `"raise"` or a callable. Its default, None, yields each folder's error, as every
  strategy does.

#### The flat strategies

The strategies in `flat/` are the counterparts of `tree/` for flat namespaces, such as object stores. A flat storage
lists every key under a prefix, at any depth, in one paginated request. They take that listing as `ils`, a function
called as `ils(prefix)` and giving an async iterator of entries:

- The prefix is raw text, e.g. `data/2024-`, not only a folder. So `flat_glob` lists only the keys starting as its
  pattern does, before the first wildcard.
- `ils` yields entries, not pages, so that its meaning stays simple. A strategy never sees the pages, and holds one at a
  time.
- `ils` yields each key once, as object stores list them. Remembering every key listed, to drop repeats, would hold the
  whole listing. A key listed twice is copied, moved, deleted or counted twice; only a batched `flat_mv` drops repeats.
- The listing holds no permit of `concurrency_limit`: its pages come one after another. The other requests, such as an
  `info`, hold one.
- Folders are derived from the keys, as fsspec's filesystems derive them for object stores. A key ending with "/" is a
  folder's marker.
- An error of the listing ends the operation: there is no other listing to go on with.

fsspec has no such listing, so `SimulatedFileSystem` has its own, `_ils`, counting a request per page. Test a flat
strategy on `flat_fs`, against fsspec's own method on it.

`flat_glob` matches with `tree_glob`'s compiled patterns, so the two never answer differently.

`flat_walk` yields what `tree_walk` yields, from one listing rather than one per folder: fsspec's `walk` lists a folder
per request even on object stores. Its order differs, as a listing is in no order of folders:

- Top down, each folder comes before what is in it, but its files may come after its subfolders'.
- Bottom up, the files come as listed, and the folders once the listing ends, the deepest first. Only then is a folder
  known to be done, and only the folders' names wait, never the listing.
- `maxdepth` and `descend` leave paths out, but save no request: the listing gives all the keys anyway.

`flat_rm`, `flat_cp`, `flat_transfer` and `flat_mv` run on the same engine as the tree's, `_Jobs`, fed by `_FlatJobs` in
`flat/_jobs.py` with the keys under each root:

- Object stores delete many keys per request. So `flat_rm` takes `rm_files(paths)` and a `batch_size`, and sends each
  batch as soon as the listing has given its keys. A batch that fails yields one error, noting its keys.
- `flat_cp` copies each key as it is listed, placed as `tree_cp` places it. A folder's marker is not copied: flat
  storages need no folder for the files in it.
- `flat_transfer` is `flat_cp` between two storages, taking its operations as `tree_transfer` does. The destination
  may be a tree, so it makes each destination folder once, before its first key, unless the profile says no folder is
  needed. A folder's marker makes its folder, so that a tree keeps it even when empty.
- `flat_mv` moves each key by `mv_file`, or copies it and deletes the keys copied in batches. Which is given decides,
  and the profile's `renames_files` if both are. Unknown, batches are right on any storage. Cancelled, it still deletes the keys copied, so that no file is
  left in both places: its `stopping` keeps their batches, which the engine would otherwise drop.

`SimulatedFileSystem` has `_rm_files` too, a batch deletion counting one request, as S3's DeleteObjects does.

#### Storage profiles

Storages answer the same request differently. Writing a file into a missing folder creates the parents on some, and
raises `FileNotFoundError` on others. A strategy works on all of them, and a `StorageProfile` lets it send fewer
requests.

- A strategy takes `profile=` as a required argument, so it works with any storage. `StorageProfile()` knows
  nothing. A filesystem holds one for all its calls, as its `storage_profile` class field.
- A field left None is unknown. The strategy then takes the way that is right on any storage, at the cost of requests.
- So a profile only saves requests. It changes the outcome only if it is wrong, and each docstring says how.
- The library never detects a storage's behaviour: the developer declares it.

The fields, and the requests each saves:

| Field              | Saves                                                                              |
|--------------------|------------------------------------------------------------------------------------|
| `parents_required` | False: the `makedirs` of each folder with files, in `tree_cp`, `tree_mv` and the transfers. |
| `implicit_folders` | True: those, and the `rm_folder` of each folder, in `tree_rm` and `tree_mv`.       |
| `renames_files`    | True: the data copied by `flat_mv`, given both ways, as it moves each key by `mv_file`. |
| `renames_folders`  | True: all requests in a folder moved whole by one `mv_file`, in `tree_mv`.         |

A field may imply another: implicit folders need no parents made. A profile contradicting itself raises `ValueError`.

A fact gets a field only once a strategy saves requests, or data copied, with it. Facts that cost nothing to handle as they come have
none. An example is a copy refused because the file exists: `tree_cp` then deletes the old file and copies again.

A storage with a native operation, such as a rename of a whole folder, has no field either. A strategy takes it as an
operation instead, and falls back on its own way where it fails:

- `tree_mv` takes `mv_file`, e.g. a storage's rename. If it moves a folder too, the profile's `renames_folders` says
  so, and each folder is first moved whole, onto a path that does not exist.
- A folder merging into an existing one, or whose move fails, is walked and moved file by file, as when unknown.
- A `TreeFileSystem` subclass overrides `_mv_file` with its rename, and declares `renames_folders`.
- `tree_rm` takes `rm_tree`, optional, a storage's recursive delete of a folder. Each folder root is first deleted
  whole by it, and a root it fails on is walked, as without it.
- A `TreeFileSystem` subclass defines `_rm_tree(path)` with its recursive delete. Its `rm` stays an `Operation`,
  watched by the monitors and `callback=`, which overriding `_rm` would lose.

### Copied, not only installed

The strategies are on PyPI, but we recommend copying their modules into your own package, e.g. as
`_contrib/strategy`. A filesystem's behaviour then never changes with a release of this library.

The interface is not copied: `fsspec-alt-strategies-common` is installed, once per project.

- Every copy then shares one `OperationHandler`, one `CallCancelled` and one `StorageProfile`, so that callers catch and
  pass them whatever copy they come from.
- Its monitors are one context variable: a `monitoring` block watches the calls of every copy.
- It changes rarely, and keeps compatible: copies of different versions use it at once.

So the code stays easy to copy:

- The package imports itself only with relative imports, so a copy works under any name.
- It depends on nothing but fsspec, the interface and the standard library.
- Each file starts with its SPDX license header, which copies keep.
- Each strategy depends on few shared modules, and its module docstring names them.
- The package's `__init__.py` re-exports `monitored` only, which needs both families' wrappers. No other
  `__init__.py` imports a module, so a copy may leave out those it does not use. Users import from the modules,
  e.g. `fsspec_alt_strategies.tree.find`. The interface's `__init__.py` re-exports its public API.

`packages/impl/tests/test_copyable.py` checks the first three rules. It also imports a copy of the package under
another name.

The version is in `_version.py`, which hatch-vcs writes when the package is built. A copy taken from a wheel or an
installed package keeps the version it was copied at. A copy of the source tree has none, and reports "0.0.0".

### Async first

Strategies are async functions, and the filesystem classes are `AsyncFileSystem`s. A sync filesystem gets them through
`SyncTreeWrapper` or `SyncFlatWrapper`, which run its calls in threads. Sync versions of the strategies are written only
when asked for.

### File objects

File objects are, with `Operation`, the places for classes, as a file keeps state between calls.

- `files/` has abstract classes, by what the storage allows. A filesystem subclasses one, implements its requests,
  and returns it from `_open`. Their base classes are private: their names must not suggest fsspec's own.
  - `AdaptiveReadableFile` and `AdaptiveWritableFile` are for storages reading byte ranges and writing in parts.
    The writer aborts its upload on any failure, and after an exception in its `with` block. A failed part raises
    OSError, whether it fails while writing or closing.
- Their requests are the storage's own. They never call fsspec's `_cat_file` or `_put_file`: many filesystems
  implement these with their file objects.
- Their requests run on an asyncio event loop, such as the filesystem's `loop`. Their sync methods wait on it through
  fsspec's `sync`, and their async ones, such as `_read`, run on it.
- They adapt to how a file is read or written, without settings per file. Requests cost money and time, so they send
  as few as the reads and writes allow.
- Each request holds their `concurrency_limit`, an `asyncio.Semaphore`, while it runs. The caller passes one to
  choose the limit's scope: a file, a filesystem, or the process. A write waits for it, so a shared limit also bounds
  the memory of the parts in flight across files.
- How they split work is planned apart from the concurrency limit, which may be shared. `spread` is how many requests a large
  read or write is split into, and caps the bytes read ahead at `spread` blocks of `max_block_size`. An adaptive file given no limit gets its
  own, of `spread`.
- They follow Python's file API, not the strategy API: errors are raised.
- A file read in many requests reads one version: `AdaptiveReadableFile` keeps to `version`, or the first response's.
  `_check_version` checks every response's, e.g. its ETag, and raises OSError with `errno.ESTALE` on another.
- fsspec's `open_async` returns a file whose methods are awaited. `files/fsspec_async.py` has `AsyncFileAdapter`,
  giving our files that API, so that `open` and `open_async` share a filesystem's requests. Its methods await the
  files' async counterparts, such as `_read`, which `_AsyncFile` declares. The files keep only the sync API themselves.
- `SyncFileAdapter`, in the same module, gives that API to any other sync file object, e.g. a storage SDK's own. It
  runs the file's blocking calls in threads with `asyncio.to_thread`, one at a time. A cancelled call waits for its
  thread, as a thread cannot be stopped.
- They know nothing of fsspec's transactions. How a file stays invisible until commit depends on the storage, so
  `files/fsspec_transaction.py` holds it in a `FileTransaction`, an ABC with four steps: `start_staging()` opens
  the file written to, `end_staging()` ends the writing on closing, then `commit()`, or `discard()` instead. One
  `TransactionFile` takes any of them. A filesystem's `_open` returns it when `autocommit` is False. A commit that
  fails can still be discarded.
- The strategies provided are:
  - `TempPathTransaction`, writing to a hidden temporary path, renamed on commit, for any file;
  - `LocalBufferTransaction`, writing to a local buffer, written to the file on commit, for any file;
  - `AdaptiveWritableTransaction`, holding an adaptive file's last request, which makes it visible, until commit.
- The file objects know nothing of transactions. `AdaptiveWritableTransaction` stands in for the file's two last
  requests, on that file only. A storage with a way of its own gets a subclass of `FileTransaction`.

### Observable and cancellable

Long operations are iterated, so the caller sees each step, and its `state` so far. The caller can cancel one, which
then ends cleanly.

### Logging

The library tells its callers by what it returns and raises, not by logging.

- Strategies and file objects log nothing in their routine work. A backend logs its own requests, in its hooks.
- A file object logs a WARNING, with `logging.getLogger(__name__)`, only for an error no caller can be told of. That
  is an abort or discard run by the garbage collector, which fails.
- An abort failing after another error adds a note to that error, which is raised: it is not logged.
- A file collected while open, or a `TransactionFile` collected undecided, emits a `ResourceWarning`, as Python's
  own files do.
- No exception is swallowed silently, and `CancelledError` never is.
- A caller wanting a log of operations opts in with `LoggingMonitor`, a monitor of the common package. It logs each
  operation's start and end, and each item yielded at DEBUG.

## Releases

A release is a version tag pushed to the repository. The version of both packages comes from the tag.
