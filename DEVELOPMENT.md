# Development

## Layout

| Path                            | Contents                                                       |
|---------------------------------|----------------------------------------------------------------|
| `src/fsspec_alt_strategies/`    | The package.                                                   |
| `.../control.py`                | `Stop`, `OperationStopped`, `Result`, `State`: the strategy API. |
| `.../walk.py`                   | `walk`, the concurrent folder walk under the tree strategies.  |
| `.../testing.py`                | `SimulatedFileSystem`, for tests and benchmarks.               |
| `tests/`                        | The tests, with the shared fixtures in `tests/conftest.py`.    |
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
  storage would get, in `stats`, and can delay each by a latency.
- Its two modes are an object store, where folders are only prefixes, and a hierarchical namespace, with real
  folders. Test a strategy on both, with the fixture `any_fs`.
- The fixtures `flat_fs`, `hierarchical_fs` and `any_fs` are empty async instances. Set up files with `add_file` and
  `add_dir`, which make no request.
- Assert the requests a strategy makes, e.g. `fs.stats.calls == {"list": 1, "delete": 3}`. They are what the
  strategy is for.
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

A strategy is an async function. The filesystem comes first, then the operation's arguments, then keyword-only
options:

```python
result = await tree_rm(fs, "data/2024", batch_size=32, callback=on_progress)
```

- `batch_size` caps the calls running at a time, as in fsspec. None takes fsspec's default.
- Errors before any work starts are raised, as fsspec raises them. An example is a missing root, which raises
  `FileNotFoundError`.

#### Attaching

Developers attach a strategy by overriding the fsspec method themselves, in a line or two:

```python
class MyFileSystem(AsyncFileSystem):
    async def _rm(self, path, recursive=False, **kwargs):
        if recursive:
            return await tree_rm(self, path, batch_size=32)
        return await super()._rm(path, recursive=recursive, **kwargs)
```

- The override can choose a strategy per instance, e.g. from the storage's settings.
- fsspec makes the sync method, `rm`, from the async one, `_rm`. So sync callers get the strategy too.
- The library adds no mixins or descriptors: there is no machinery to learn.

#### Callbacks

A strategy takes an optional `callback`. It is called with a snapshot of the operation's state after each step. Each
strategy's docstring says what its steps are, e.g. a folder listed or a batch deleted.

- The state is a frozen dataclass of counts, such as files done and bytes done, and the path last worked on.
- The callback may be a plain or an async function. It runs on the event loop, so it must return quickly.
- Returning None continues. Returning a `Stop` object cancels the operation, e.g. `return Stop(cleanup=True)`.
- Returning anything else raises `TypeError`, so that a callback returning e.g. `True` by mistake is caught.
- On a stop, the strategy starts no new calls and waits for those already sent. A call cut off midway would leave its
  outcome unknown.
- From a sync method, the callback runs in fsspec's IO thread, not the caller's.

#### Stop

`Stop` is a frozen dataclass of options for how to stop:

- `cleanup`: whether to undo what the operation did, where it can, e.g. delete the files a copy wrote. False by
  default, which keeps what is done.
- `raise_error`: whether the strategy raises `OperationStopped` once stopped, instead of returning. False by default,
  which returns the partial result.
- `reason`: an optional text, kept in the result for logs and messages.

Each strategy's docstring says what `cleanup` undoes for it. Work that cannot be undone, such as a deletion, is kept.
A strategy never leaves a partly written file, with or without `cleanup`. New options are added to `Stop` with
defaults, so that callbacks keep working.

#### Results

Without a stop, a strategy returns a result object, even when some paths fail. A stop returns the partial result, or
raises it, as its `raise_error` says:

```python
def on_progress(state):
    if user_pressed_cancel:
        return Stop()                   # tree_rm returns, with result.complete False
    if state.errors:
        return Stop(raise_error=True)   # tree_rm raises OperationStopped: fail fast
```

A result has these fields:

- `complete` is False after a stop or a failure.
- `stop` is the `Stop` a callback returned, or None.
- `errors` maps each failed path to its exception.
- `raise_for_errors()` raises them as an `ExceptionGroup`, for callers that want an exception.
- Each strategy's result adds its own fields, e.g. the files deleted.

`OperationStopped` carries the partial result as its `result`. When paths have failed, it is raised from their
`ExceptionGroup`, so tracebacks show them.

Errors on single paths are recorded, never raised on their own. A caller wanting to fail fast stops from the callback,
as the state counts the errors so far.

`walk` is the one exception to this API. It is an async iterator that the caller drives, and takes no callback.
Breaking out of its loop stops it. The strategies built on it report and stop as above.

Cancelling the task itself, e.g. with `asyncio.timeout`, raises `CancelledError` as usual. The strategy then cleans up
and never swallows it, but the partial result is lost.

### Copied, not only installed

The package is on PyPI, but we recommend copying its modules into your own package, e.g. as `_contrib/strategy`. A
filesystem's behaviour then never changes with a release of this library.

So the code stays easy to copy:

- The package imports itself only with relative imports, so a copy works under any name.
- It depends on nothing but fsspec and the standard library.
- Each file starts with its SPDX license header, which copies keep.
- Each strategy depends on few shared modules, and its module docstring names them.
- The package's `__init__.py` imports none of them, so a copy may leave out the modules it does not use. Users import
  from the modules, e.g. `fsspec_alt_strategies.control`.

`tests/test_copyable.py` checks the first three rules. It also imports a copy of the package under another name.

The version is in `_version.py`, which hatch-vcs writes when the package is built. A copy taken from a wheel or an
installed package keeps the version it was copied at. A copy of the source tree has none, and reports "0.0.0".

### Async first

Strategies target `AsyncFileSystem` first. Sync filesystems may get some features, such as observability, later.

### Observable and cancellable

Long operations take a callback. It is called with the operation's intermediate state. Its return value can ask the
operation to stop, which then ends cleanly.

## Releases

A release is a version tag pushed to the repository. The version comes from the tag.
