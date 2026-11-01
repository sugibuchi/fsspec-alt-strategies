# Development

## Layout

| Path                            | Contents                                                       |
|---------------------------------|----------------------------------------------------------------|
| `src/fsspec_alt_strategies/`    | The package.                                                   |
| `.../control.py`                | `Operation`, `State` and `collect`: the strategy API.          |
| `.../tree/`                     | The strategies walking a folder tree, a folder per listing.    |
| `.../tree/walk.py`              | `tree_walk`, the concurrent folder walk under the tree strategies.  |
| `.../tree/find.py`              | `tree_find`, a find listing several folders at a time.              |
| `.../tree/du.py`                | `tree_du`, a tree's size from its listings alone.                   |
| `.../tree/glob.py`              | `tree_glob`, a glob listing only the folders it can match in.       |
| `.../tree/rm.py`                | `tree_rm`, a recursive delete that deletes while it walks.          |
| `.../tree/cp.py`                | `tree_cp`, a recursive copy that copies while it walks.        |
| `.../tree/mv.py`                | `tree_mv`, a recursive move: a copy deleting as it goes.       |
| `.../tree/_jobs.py`             | The private engine running a strategy's jobs, e.g. deletions, while the walk goes on. |
| `.../profile.py`                | `StorageProfile`, what a storage does that fsspec's API leaves open. |
| `.../files/`                    | Base classes for a filesystem's file objects, by what its storage allows. |
| `.../files/adaptive.py`         | `AdaptiveReadableFile` and `AdaptiveWritableFile`, for range reads and writes in parts. |
| `.../files/buffered.py`         | `LocallyBufferedFile`, for storages transferring only whole files. |
| `.../files/transaction.py`      | Files written in a transaction, by strategy: a temporary path, a local buffer, or the storage's own. |
| `.../files/_async_file.py`      | The private machinery under them: the loop, the concurrency limit, the requests. |
| `.../testing.py`                | `SimulatedFileSystem`, for tests and benchmarks.               |
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
- Tests run on `SimulatedFileSystem`, from the public `fsspec_alt_strategies.testing`. It counts the requests a real
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
- A strategy calls only the filesystem's primitives, such as `_ls`, `_info` and `_rm_file`. So picking one strategy
  never changes the behaviour of another.
- A strategy needing another, or a native operation of the storage, takes it as a parameter.

### The strategy API

Every strategy follows the same shape, so that developers learn it once.

#### Calling

A strategy is a plain function returning an `Operation`. The filesystem comes first, then the operation's arguments,
then keyword-only options:

```python
limit = asyncio.Semaphore(32)  # Shared by all that uses the storage, files included.
async with tree_rm(fs, "data/2024", concurrency_limit=limit) as op:
    async for path in op:
        if isinstance(path, Exception):
            print("failed:", path)
print(op.state.files_deleted)
```

- `concurrency_limit` is an `asyncio.Semaphore`, of which each request holds a permit while it runs. Shared, it limits
  several operations, and the file objects, together: the caller chooses its scope. Without one, an operation has its
  own, of 32.
- That default is not fsspec's `batch_size`, which is how many coroutines fsspec gathers at once, sized from the limit
  of open files. Requests to one storage need a limit of their own: high enough to hide each request's latency, and
  low enough not to be throttled.
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

`collect(op)` runs an operation to its end. It returns what was yielded, split into a list of items and a list of
errors, in the order yielded. Strategies yield in the order the storage answers, so `collect` does not sort: fsspec's
`_find` and `_glob` do.

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
            _, errors = await collect(tree_rm(self, path, concurrency_limit=self.limit))
            if errors:
                raise ExceptionGroup("rm failed", errors)
            return
        return await super()._rm(path, recursive=recursive, **kwargs)
```

- The override can choose a strategy per instance, e.g. from the storage's settings.
- fsspec makes the sync method, `rm`, from the async one, `_rm`. So sync callers get the strategy too.
- The library adds no mixins or descriptors: there is no machinery to learn.

#### The tree strategies

The strategies in `tree/` walk with the walk's private engine, `_listings`, which yields each folder's listing. They
come in two layers, by what they send:

| Strategies | They send | They build on |
|---|---|---|
| `tree_find`, `tree_du`, `tree_glob` | Listings, and at most one `_info` for the root | `_scan` |
| `tree_rm`, `tree_cp`, `tree_mv` | Requests changing the storage, besides the listings | `_Jobs` |

`tree_walk` is `_listings` itself, made public. It raises for a missing or file root, as fsspec's `walk` does.

`_scan`, in `tree/walk.py`, treats the root as fsspec's `_find` and `_glob` do. A missing root finds nothing, and a
file root yields its own `_info` entry when asked. A folder root yields its own entry when asked, e.g. for `withdirs`.
The strategies that only read never handle the root themselves.

Those changing a tree, `tree_rm`, `tree_cp` and `tree_mv`, run their requests with a second private engine, `_Jobs` in
`tree/_jobs.py`. It starts each folder's jobs, e.g. its files' deletions, as soon as the folder is listed, and runs
them along with the listings still in flight. It also drains them on `cancel()`, and waits for them on closing. A
strategy subclasses it, saying what a listing queues, how a job runs, and what its outcome is.

They take glob patterns, as fsspec's `rm`, `copy` and `mv` do. A pattern is expanded with `tree_glob` first, and
the folders it matches are then walked at once. The expansion is not streamed into the jobs: `tree_cp` and `tree_mv`
place paths relative to the common prefix of all the matches, as fsspec does, which is known only once the glob ends.
The glob lists only the folders down to the pattern's depth, so the bulk of the walk still overlaps with the jobs.

Their names start with `tree_`, e.g. `tree_rm` and `tree_glob`. Code using fsspec already has `rm` and `glob`, as
methods and modules. The prefix keeps an import from shadowing them, and tells the reader which strategy is used.

`tree_walk` is meant to replace fsspec's `walk`, so it already has two of its options:

- `topdown=False` yields a folder's entries only after those of all the folders below it.
- `raise_on_error=True` raises a folder's error instead of yielding it. With filtering, it will implement fsspec's
  `on_error`.

#### Storage profiles

Storages answer the same request differently. Writing a file into a missing folder creates the parents on some, and
raises `FileNotFoundError` on others. A strategy works on all of them, and a `StorageProfile` lets it send fewer
requests.

- A strategy takes `profile=` as an argument, so it works with any filesystem class, without subclassing.
- Without one, it uses the filesystem's `storage_profile` attribute, which a filesystem class may set for all callers.
- A field left None is unknown. The strategy then takes the way that is right on any storage, at the cost of requests.
- So a profile only saves requests. It changes the outcome only if it is wrong, and each docstring says how.
- The library never detects a storage's behaviour: the developer declares it.

A fact gets a field only once a strategy saves requests with it. Facts that cost nothing to handle as they come have
none. An example is a copy refused because the file exists: `tree_mv` then deletes the old file and copies again.

A storage with a native operation, such as a rename of a whole folder, has no field either. The developer uses that
operation instead of the strategy.

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
  read or write is split into, and `max_readahead` caps the bytes read ahead.
- They follow Python's file API, not the strategy API: errors are raised.
- They know nothing of fsspec's transactions. How a file stays invisible until commit depends on the storage, so
  `files/transaction.py` holds it in a `FileTransaction`, an ABC with four steps: `start_staging()` opens the
  file written to, `end_staging()` ends the writing on closing, then `commit()`, or `discard()` instead. One
  `TransactionFile` takes any of them. A filesystem's `_open` returns it when
  `autocommit` is False. Two strategies are provided: `TempPathTransaction` writes to a hidden temporary path,
  renamed on commit, and `LocalBufferTransaction` writes to a local buffer, written to the file on commit.
- A storage with a way of its own, such as completing a multipart upload only on commit, gets a subclass of
  `FileTransaction`. The package's files help it with three private steps, `_stage`, `_publish` and `_unstage`, which
  closing runs as stage then publish.

### Observable and cancellable

Long operations are iterated, so the caller sees each step, and its `state` so far. The caller can cancel one, which
then ends cleanly.

## Releases

A release is a version tag pushed to the repository. The version comes from the tag.
