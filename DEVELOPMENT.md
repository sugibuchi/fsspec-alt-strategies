# Development

## Layout

| Path                            | Contents                                                       |
|---------------------------------|----------------------------------------------------------------|
| `src/fsspec_alt_strategies/`    | The package.                                                   |
| `.../control.py`                | `Operation`, `State` and `collect`: the strategy API.          |
| `.../tree/`                     | The strategies walking a folder tree, a folder per listing.    |
| `.../tree/walk.py`              | `walk`, the concurrent folder walk under the tree strategies.  |
| `.../tree/find.py`              | `find`, a find listing several folders at a time.              |
| `.../tree/du.py`                | `du`, a tree's size from its listings alone.                   |
| `.../tree/glob.py`              | `glob`, a glob listing only the folders it can match in.       |
| `.../tree/rm.py`                | `rm`, a recursive delete that deletes while it walks.          |
| `.../file.py`                   | File objects whose buffering adapts to how they are read or written. |
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
async with rm(fs, "data/2024", batch_size=32) as op:
    async for path in op:
        if isinstance(path, Exception):
            print("failed:", path)
print(op.state.files_deleted)
```

- `batch_size` caps the calls running at a time, as in fsspec. None takes fsspec's default.
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
    async def _rm(self, path, recursive=False, **kwargs):
        if recursive:
            _, errors = await collect(rm(self, path, batch_size=32))
            if errors:
                raise ExceptionGroup("rm failed", errors)
            return
        return await super()._rm(path, recursive=recursive, **kwargs)
```

- The override can choose a strategy per instance, e.g. from the storage's settings.
- fsspec makes the sync method, `rm`, from the async one, `_rm`. So sync callers get the strategy too.
- The library adds no mixins or descriptors: there is no machinery to learn.

#### The tree strategies

The strategies in `tree/` walk with the walk's private engine, `_listings`, which yields each folder's listing.

`walk` is meant to replace fsspec's `walk`, so it already has two of its options:

- `topdown=False` yields a folder's entries only after those of all the folders below it.
- `raise_on_error=True` raises a folder's error instead of yielding it. With filtering, it will implement fsspec's
  `on_error`.

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

- `file.py` has abstract classes for reading and writing. A filesystem subclasses one, implements its requests, and
  returns it from `_open`.
- Their requests run on an asyncio event loop, such as the filesystem's `loop`. Their sync methods wait on it through
  fsspec's `sync`, and their async ones, such as `_read`, run on it.
- They adapt to how a file is read or written, without settings per file. Requests cost money and time, so they send
  as few as the reads and writes allow.
- They follow Python's file API, not the strategy API: errors are raised.

### Observable and cancellable

Long operations are iterated, so the caller sees each step, and its `state` so far. The caller can cancel one, which
then ends cleanly.

## Releases

A release is a version tag pushed to the repository. The version comes from the tag.
