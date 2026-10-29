# Development

## Layout

| Path                            | Contents                                                       |
|---------------------------------|----------------------------------------------------------------|
| `src/fsspec_alt_strategies/`    | The package.                                                   |
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
- The fixture `memfs` is a cleared memory filesystem. `async_memfs` wraps it as an async filesystem, with fsspec's
  `AsyncFileSystemWrapper`.
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

### Copied, not only installed

The package is on PyPI, but we recommend copying its modules into your own package, e.g. as `_contrib/strategy`. A
filesystem's behaviour then never changes with a release of this library.

So the code stays easy to copy:

- The package imports itself only with relative imports, so a copy works under any name.
- It depends on nothing but fsspec and the standard library.
- Each file starts with its SPDX license header, which copies keep.
- Each strategy depends on few shared modules, and its module docstring names them.

### Async first

Strategies target `AsyncFileSystem` first. Sync filesystems may get some features, such as observability, later.

### Observable and cancellable

Long operations take a callback. It is called with the operation's intermediate state. Its return value can ask the
operation to stop, which then ends cleanly.

## Releases

A release is a version tag pushed to the repository. The version comes from the tag.
