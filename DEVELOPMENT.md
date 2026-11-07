# Development

## Layout

| Path                            | Contents                                                       |
|---------------------------------|----------------------------------------------------------------|
| `src/fsspec_alt_strategies/`    | The package.                                                   |
| `.../control.py`                | `Operation` and `State`: the strategy API. `Monitor` and `monitoring`, watching the operations behind the filesystems' fsspec methods. |
| `.../tree/`                     | The strategies walking a folder tree, a folder per listing.    |
| `.../tree/walk.py`              | `tree_walk`, the concurrent folder walk under the tree strategies.  |
| `.../tree/find.py`              | `tree_find`, a find listing several folders at a time.              |
| `.../tree/du.py`                | `tree_du`, a tree's size from its listings alone.                   |
| `.../tree/glob.py`              | `tree_glob`, a glob listing only the folders it can match in.       |
| `.../tree/rm.py`                | `tree_rm`, a recursive delete that deletes while it walks.          |
| `.../tree/cp.py`                | `tree_cp`, a recursive copy that copies while it walks.        |
| `.../tree/mv.py`                | `tree_mv`, a recursive move: a `mv_file` per file, or a rename per root. |
| `.../tree/filesystem.py`        | `TreeFileSystem`, an abstract async filesystem whose recursive methods are the tree strategies. |
| `.../tree/_jobs.py`             | The private engine's tree side: it feeds `common`'s with the walk's listings. |
| `.../flat/`                     | The strategies listing a flat namespace by key prefix, as object stores do. |
| `.../flat/walk.py`              | `flat_walk`, a walk of one listing of a prefix, its folders derived from the keys. |
| `.../flat/find.py`              | `flat_find`, a find streaming one listing of a prefix.          |
| `.../flat/du.py`                | `flat_du`, the size of the files under a prefix, from the listing alone. |
| `.../flat/glob.py`              | `flat_glob`, a glob listing only the keys starting as its pattern does. |
| `.../flat/rm.py`                | `flat_rm`, a recursive delete in batches, sent as the listing goes on. |
| `.../flat/cp.py`                | `flat_cp`, a recursive copy that copies each key as it is listed. |
| `.../flat/mv.py`                | `flat_mv`, a recursive move: a `mv_file` per key, or a copy per key and a deletion per batch. |
| `.../flat/filesystem.py`        | `FlatFileSystem`, an abstract async filesystem whose recursive methods are the flat strategies. |
| `.../flat/_keys.py`             | The private listing under them: a prefix's keys, and the folders they imply. |
| `.../flat/_jobs.py`             | The private engine's flat side: it feeds `common`'s with the keys of each root. |
| `.../common/`                   | Private machinery the families share, so that none imports another. |
| `.../common/filesystem.py`      | What the filesystem classes share: the two limits, and data moved through file objects. |
| `.../common/monitors.py`        | How an operation is run to its end, as the monitors and fsspec's `callback=` watch it. |
| `.../common/jobs.py`            | The private engine running a strategy's jobs, e.g. deletions, while what to change is found, and the queue of requests under a concurrency limit, which the walk uses too. |
| `.../common/placement.py`       | Where a copy or move puts each path, as fsspec's `copy` does.   |
| `.../common/operations.py`      | The types of the storage's operations, such as `ls` and `info`. |
| `.../common/paths.py`           | Paths and depths, as the strategies take them.                  |
| `.../common/patterns.py`        | Glob patterns, matched as fsspec's `_glob` matches them, by both globs. |
| `.../common/states.py`          | The states shared by counterparts, e.g. `FindState` of `tree_find` and `flat_find`. |
| `.../profile.py`                | `StorageProfile`, what a storage does that fsspec's API leaves open. |
| `.../files/`                    | Base classes for a filesystem's file objects, by what its storage allows. |
| `.../files/adaptive.py`         | `AdaptiveReadableFile` and `AdaptiveWritableFile`, for range reads and writes in parts. |
| `.../files/buffered.py`         | `LocallyBufferedFile`, for storages transferring only whole files. |
| `.../files/fsspec_transaction.py` | Files written in a transaction, by strategy: a temporary path, a local buffer, or the storage's own. |
| `.../files/fsspec_async.py`     | `AsyncFileAdapter` and `SyncFileAdapter`, giving file objects the API of those `open_async` returns. |
| `.../files/_async_file.py`      | The private machinery under them: the loop, the concurrency limit, the requests. |
| `src/fsspec_alt_strategies_testing/` | `SimulatedFileSystem`, for tests and benchmarks: a package apart, so that copies leave it out. |
| `tests/`                        | The tests, in the package's layout, with the shared fixtures in `tests/conftest.py`. |
| `.github/workflows/ci.yaml`     | Lint and tests, on pull requests and pushes to `main`.         |

The only runtime dependency is fsspec. The version comes from the git tag, through hatch-vcs.

## Setup

All Python commands go through [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # the package, editable, with the dev dependencies
uv run pre-commit install        # ruff and file checks before each commit
```

## Tests

```bash
uv run pytest                    # the tests
uv run pytest --cov              # with coverage of src/fsspec_alt_strategies
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
- Test folders have no `__init__.py`. Share code through `conftest.py` or helpers at the top of `tests/`.

## Design

The choices below shape the whole project. Keep to them, or change them on purpose.

### A third party to fsspec

The library is not part of fsspec. It offers alternatives, and never patches fsspec or its filesystems.

### Strategies are functions

Each algorithm is a stateless function. It takes a filesystem and the operation's arguments, and keeps no state between
calls.

- Users can call a function on any fsspec filesystem, without subclassing.
- A base filesystem class wraps the functions as its methods, for implementations that want them by default.
- A function only uses fsspec's public and documented private API, e.g. `_ls` and `_info` on async filesystems. So it
  works on any implementation.

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
- They are required: a strategy guesses nothing, e.g. which request deletes a folder.
- Paths come without a protocol, as the operations take them. The method wrapping a strategy strips it first.

- `concurrency_limit` is an `asyncio.Semaphore`, of which each request holds a permit while it runs. Shared, it limits
  several operations, and the file objects, together: the caller chooses its scope.
- It is required, with no default. The right limit depends on the storage, high enough to hide each request's latency
  and low enough not to be throttled, so the caller sets it. fsspec's `batch_size` does not fit: it is how many
  coroutines fsspec gathers at once, sized from the limit of open files.
- Errors in the arguments, such as a bad `maxdepth`, are raised by the call.
- Errors ending the whole operation are raised by the iteration, as fsspec raises them. An example is a missing root,
  which raises `FileNotFoundError`.

#### Operations

An `Operation` does its work as it is iterated, a few requests ahead of the caller.

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
- `_cat_file`, `_pipe_file`, `_get_file` and `_put_file` go through `open_async`'s files, so data moves as the file
  classes move it.
- The failures of a strategy are raised: one as it is, several in an ExceptionGroup. A folder that fails to list is
  yielded empty by a walk, as fsspec's `walk` yields it, and `on_error` handles every folder's failure, not only the
  root's as fsspec's does.
- fsspec has no async `mv`, nor builds a sync one from `_mv`: each class wraps its `_mv` itself.

Their callers see only fsspec's answers, not the `Operation` behind them. The monitors of `control.py` show it without
changing fsspec's API. The filesystems run their operations with `common/monitors.py`, which tells the monitors.

- A `Monitor` subclass overrides `started`, `yielded` and `ended`, each awaited where the operation runs.
- `with monitoring(monitor):`, or `async with`, sets a context variable. Every call within the block is watched,
  whoever makes it, and blocks nest.
- fsspec's sync methods see the block too: `sync()` runs the call in a copy of the caller's context.
- A monitor stops a call with `op.cancel()`. The call then raises `CallCancelled`, an Exception, since fsspec's callers
  expect a call to finish or raise, and an `asyncio.CancelledError` would end fsspec's `sync()` with no answer.
- fsspec's own `callback=` is a monitor too. `copy`, `mv` and `rm` add one per path done. `_get_file` and `_put_file`
  set the size in bytes and add each chunk, as fsspec's own filesystems do.
- Only the calls run by a strategy are watched. A call fsspec answers itself, e.g. a copy that is not recursive, is not.

The strategies themselves take no callbacks: their callers iterate the `Operation`. Monitors exist for the callers of
fsspec's methods, who cannot.

Choosing the class chooses the family: a developer wanting one strategy of another overrides that method, as above.

#### The tree strategies

The strategies in `tree/` walk with the walk's private engine, `_listings`, which yields each folder's listing. They
come in two layers, by what they send:

| Strategies | They send | They build on |
|---|---|---|
| `tree_find`, `tree_du`, `tree_glob` | Listings, and at most one `info` for the root | `_find_listings` |
| `tree_rm`, `tree_cp`, `tree_mv` | Requests changing the storage, besides the listings | `_TreeJobs` |

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

`flat_rm`, `flat_cp` and `flat_mv` run on the same engine as the tree's, `_Jobs`, fed by `_FlatJobs` in
`flat/_jobs.py` with the keys under each root:

- Object stores delete many keys per request. So `flat_rm` takes `rm_files(paths)` and a `batch_size`, and sends each
  batch as soon as the listing has given its keys. A batch that fails yields one error, noting its keys.
- `flat_cp` copies each key as it is listed, placed as `tree_cp` places it. A folder's marker is not copied: flat
  storages need no folder for the files in it.
- `flat_mv` moves each key by `mv_file`, or copies it and deletes the keys copied in batches. Which is given decides,
  and the profile's `renames_files` if both are. Unknown, batches are right on any storage. Cancelled, it still deletes the keys copied, so that no file is
  left in both places: its `stopping` keeps their batches, which the engine would otherwise drop.

`SimulatedFileSystem` has `_rm_files` too, a batch deletion counting one request, as S3's DeleteObjects does.

#### Storage profiles

Storages answer the same request differently. Writing a file into a missing folder creates the parents on some, and
raises `FileNotFoundError` on others. A strategy works on all of them, and a `StorageProfile` lets it send fewer
requests.

- A strategy takes `profile=` as an argument, so it works with any storage. A filesystem wrapping it may hold one for
  all its calls.
- A field left None is unknown. The strategy then takes the way that is right on any storage, at the cost of requests.
- So a profile only saves requests. It changes the outcome only if it is wrong, and each docstring says how.
- The library never detects a storage's behaviour: the developer declares it.

The fields, and the requests each saves:

| Field              | Saves                                                                              |
|--------------------|------------------------------------------------------------------------------------|
| `parents_required` | False: the `makedirs` of each folder with files, in `tree_cp` and `tree_mv`.       |
| `implicit_folders` | True: those, and the `rm_folder` of each folder, in `tree_rm` and `tree_mv`.       |
| `renames_files`    | True: the data copied by `flat_mv`, given both ways, as it moves each key by `mv_file`. |

A field may imply another: implicit folders need no parents made. A profile contradicting itself raises `ValueError`.

A fact gets a field only once a strategy saves requests, or data copied, with it. Facts that cost nothing to handle as they come have
none. An example is a copy refused because the file exists: `tree_cp` then deletes the old file and copies again.

A storage with a native operation, such as a rename of a whole folder, has no field either. A strategy takes it as an
operation instead, and falls back on its own way where it fails:

- `tree_mv` takes `rename=`, and moves each root whole with it, onto a path that does not exist.
- A root merging into an existing folder, or whose rename fails, is walked and moved file by file, as without it.
- `TreeFileSystem` passes its `_rename`, which a subclass implements only if its storage has one.

### Copied, not only installed

The package is on PyPI, but we recommend copying its modules into your own package, e.g. as `_contrib/strategy`. A
filesystem's behaviour then never changes with a release of this library.

So the code stays easy to copy:

- The package imports itself only with relative imports, so a copy works under any name.
- It depends on nothing but fsspec and the standard library.
- Each file starts with its SPDX license header, which copies keep.
- Each strategy depends on few shared modules, and its module docstring names them.
- No `__init__.py` imports the modules, so a copy may leave out those it does not use. Users import from the modules,
  e.g. `fsspec_alt_strategies.tree.find`.

`tests/test_copyable.py` checks the first three rules. It also imports a copy of the package under another name.

The version is in `_version.py`, which hatch-vcs writes when the package is built. A copy taken from a wheel or an
installed package keeps the version it was copied at. A copy of the source tree has none, and reports "0.0.0".

### Async first

Strategies target `AsyncFileSystem` first. Sync filesystems may get some features, such as observability, later.

### File objects

File objects are, with `Operation`, the places for classes, as a file keeps state between calls.

- `files/` has abstract classes, by what the storage allows. A filesystem subclasses one, implements its requests,
  and returns it from `_open`. Their base classes are private: their names must not suggest fsspec's own.
  - `AdaptiveReadableFile` and `AdaptiveWritableFile` are for storages reading byte ranges and writing in parts.
  - `LocallyBufferedFile` is for storages transferring only whole files: one download, local reads and writes, and
    one upload if it changed. It never uploads after an exception in its `with` block.
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
  `TransactionFile` takes any of them. A filesystem's `_open` returns it when `autocommit` is False.
- The strategies provided are:
  - `TempPathTransaction`, writing to a hidden temporary path, renamed on commit, for any file;
  - `LocalBufferTransaction`, writing to a local buffer, written to the file on commit, for any file;
  - `AdaptiveWritableTransaction`, holding an adaptive file's last request, which makes it visible, until commit;
  - `LocallyBufferedTransaction`, closing a locally buffered file, which uploads it, only on commit.
- The file objects know nothing of transactions. `AdaptiveWritableTransaction` stands in for the file's two last
  requests, on that file only. A storage with a way of its own gets a subclass of `FileTransaction`.

### Observable and cancellable

Long operations are iterated, so the caller sees each step, and its `state` so far. The caller can cancel one, which
then ends cleanly.

## Releases

A release is a version tag pushed to the repository. The version comes from the tag.
