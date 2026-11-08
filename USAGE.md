# Usage

How to get the library's features, and how to use them. See the
[README](https://github.com/sugibuchi/fsspec-alt-strategies#readme) for why they exist.

## Provided features

### File tree operation strategies

The strategies walk, find, glob, measure, delete, copy and move file trees. They come in two families, by a
storage's namespace:

- the `tree` family, in `tree/`, for storages with a hierarchical namespace, a tree of folders, such as Azure Data
  Lake Storage Gen2, SFTP or HDFS;
- the `flat` family, in `flat/`, for storages with a flat namespace, keys whose folders are only prefixes, such as
  S3.

Each family has the same strategies, named after it, e.g. `tree_rm` and `flat_rm`. They come in three forms:

- **Functions**, such as `tree_rm` or `flat_find`, for built-in, specific strategies: you call one with the storage's
  operations it uses.
- **Abstract filesystem classes**, `TreeFileSystem` and `FlatFileSystem`, for a new filesystem implementation: you
  implement the storage's primitives, and the recursive methods are the strategies.
- **Wrappers**, in `wrapper.py`, for existing filesystem implementations: they give a filesystem the strategies
  without changing it. `monitored()` picks the wrapper for a known filesystem.

### Monitor

A `Monitor` watches the operations behind fsspec's methods, such as `fs.rm(...)`, path by path. It sees each path
done, each error and the progress, and it can cancel the operation. It is part of the interface package, so one
`monitoring` block watches every copy of the strategies in a project.

### Async file object

The file objects in `files/` adapt their buffering to how a file is used. `AsyncFileAdapter` gives a sync file the
API of the files `open_async()` opens, and fsspec's transactions work with any file object.

### Filesystem router

`AsyncRoutingFileSystem`, in `router.py`, spans several storages, e.g. a cloud's volumes. It routes each path to the
storage holding it, and copies or moves between storages with the source's strategy.

## Installation

### The interface package, from PyPI

The interface holds the operations' control and states, the monitors, the storage profiles and the known
implementations. It is installed, never copied:

```bash
pip install fsspec-alt-strategies
```

Every copy of the strategies in a project, of any version, then shares one interface. An exception such as
`CallCancelled` is the same class whichever copy raised it.

### The implementation package, by copying its files

We recommend copying the modules of `fsspec_alt_strategies_impl` you use into your own package, e.g. as
`_contrib/strategy`. Your filesystem's behaviour then never changes with a release of this library. The modules
import each other relatively, so a copy works under any name. Keep their license headers.

It is also on PyPI, as `fsspec-alt-strategies-impl`, to try it before copying.

### The modules to copy for each feature

| Feature                         | Modules to copy                                                            |
|---------------------------------|----------------------------------------------------------------------------|
| A family's functions            | `tree/` or `flat/`, and `common/`                                          |
| A family's abstract class       | `tree/` or `flat/`, and `common/`; `files/` for its file objects, if used  |
| Wrappers and `monitored()`      | `wrapper.py`, `monitor.py`, `__init__.py`, `tree/`, `flat/`, `common/` and `files/` |
| Monitor                         | Nothing: it is in the interface package                                    |
| Async file objects              | `files/`                                                                   |
| Filesystem router               | `router.py`, `common/`, and the family of each storage copied or moved from |

A monitor watches only filesystems whose methods run the strategies: the abstract classes' subclasses and the
wrappers.

## Usage

### Watch an existing filesystem

`monitored()` wraps a filesystem it knows as the family that suits it. A `Monitor` then sees each operation behind
fsspec's methods:

```python
import fsspec
from fsspec_alt_strategies_impl import monitored
from fsspec_alt_strategies import Monitor, monitoring


class Progress(Monitor):
    async def yielded(self, call, item, state, op):
        if isinstance(item, Exception):
            op.cancel()  # Stop at the first failure: the call raises CallCancelled.
        print(call.method, state)


fs = monitored(fsspec.filesystem("file"))
with monitoring(Progress()):
    fs.rm("data/2024", recursive=True)
```

The library logs none of its routine work. To log each operation's start and end, use `LoggingMonitor`:
`with monitoring(LoggingMonitor(logging.getLogger("myapp"))):`. It logs each item yielded at DEBUG.

The known filesystems are those in `known_implementations`, by class: the local filesystem, s3fs, gcsfs, adlfs,
fsspec's memory, SFTP and HDFS filesystems. Each entry says its family, whether it is async, its concurrency limits
and its profile. Pass your own table to `monitored()`, or `wrap_unknown_fs=True` to wrap any other filesystem the way
right on every storage: as a tree, one request at a time, with nothing known of the storage.

A wrapper is light and not cached: make one when you need it, and drop it after.

### Subclass a family's filesystem

`TreeFileSystem` and `FlatFileSystem` are abstract async filesystems whose recursive methods are the strategies. You
implement the storage's primitives, and declare its profile:

```python
from fsspec_alt_strategies_impl.tree.filesystem import TreeFileSystem
from fsspec_alt_strategies import StorageProfile


class MyFileSystem(TreeFileSystem):
    protocol = "my"
    storage_profile = StorageProfile()  # Nothing known: right on any storage.

    async def _ils(self, path, **kwargs): ...  # A folder's entries, as they come.
    async def _info(self, path, **kwargs): ...
    # ... and the other primitives each class declares abstract.


fs = MyFileSystem(operation_concurrency=32, io_concurrency=8)
```

### Call a strategy

A strategy is a plain function, taking the storage's operations it uses:

```python
import asyncio
from fsspec_alt_strategies_impl.tree.find import tree_find

find = tree_find("data", ils=fs._ils, info=fs._info, concurrency_limit=asyncio.Semaphore(32))
async with find as op:
    async for entry in op:
        if isinstance(entry, Exception):
            print("failed:", entry)
print(op.state.found)
```

### Choosing

None of these is right for every storage, and the library never picks one for you.

- **The family** follows how the storage lists. A storage listing a folder per request, even an object store such as
  Azure Blob Storage through adlfs, suits `tree`. One listing a key prefix in one paginated request suits `flat`.
- **The profile** only saves requests, or data copied. A field left unknown is right on any storage. A field declared
  wrongly gives wrong results: declare only what you know.
- **The concurrency limits** are yours to set. A filesystem unsafe to call from several threads, such as SFTP, needs
  them at 1.

### Costs to know

- A flat wrapper lists a prefix with the wrapped filesystem's `find`, as fsspec has no listing by prefix. That listing
  is held in memory, and covers the prefix's whole folder. A subclass overriding `_iexpand` with the storage's own listing
  by prefix streams it instead.
- A monitor is awaited where the operation runs, so a slow monitor slows the call. For a sync call, it runs on
  fsspec's own loop, in its thread.
- `known_implementations` is shared by every copy in a project. Change it with care, or pass your own table.
- Only the calls a strategy runs are watched. A call fsspec answers itself, e.g. a copy that is not recursive, is not.

### Choosing with numbers

`fsspec_alt_strategies_impl_testing.SimulatedFileSystem` is an in-memory async filesystem. It counts the requests a real
storage would get, and delays each by a latency and a bandwidth you set. It models an object store, or a storage with a
hierarchical namespace. Run a strategy and fsspec's default on it, then compare their requests and times. It is a
package of its own, so a copy of the strategies leaves it out.

```python
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

fs = SimulatedFileSystem(hierarchical=True, latency=0.02)
fs.add_file("data/2024/01/a.parquet", b"...")
fs.rm("data", recursive=True)
print(fs.stats.calls)  # Counter({'delete': 1})
```
