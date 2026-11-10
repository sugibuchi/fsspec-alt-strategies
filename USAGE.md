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

Each family replaces fsspec's recursive methods, such as `find`, `rm` and `copy`, with its strategies. They come in
two forms:

- **Abstract filesystem classes**, `TreeFileSystem` and `FlatFileSystem`, for a new filesystem implementation: you
  implement the storage's primitives, and the recursive methods are the strategies.
- **Wrappers**, in `wrapper.py`, for existing filesystem implementations: they give a filesystem the strategies
  without changing it. `monitored()` picks the wrapper for a known filesystem.

### Monitor

A `Monitor` gives the progress of file tree operations in real time. It watches the operations behind fsspec's
methods, such as `fs.rm(...)`, path by path. It sees each path done, each error and the progress, and it can cancel
the operation.

### Async file object

The file objects in `files/` adapt their buffering to how a file is used. The package also provides adapters,
which add fsspec's file features to an existing file-like object, e.g. an SDK's own:

- `SyncFileAdapter` gives a sync file, such as an SDK's handle, the API of the files `open_async()` opens;
- the transactions of `fsspec_transaction.py` make its writes take effect only when fsspec's transaction commits.

### Filesystem router

`AsyncRoutingFileSystem`, in `router.py`, spans several storages, e.g. a cloud's volumes. It routes each path to the
storage holding it, and copies or moves files between different storages.

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

Its command line copies it, with its `cli` extra. Run it from your project, with the folder to copy into:

```bash
uv run --with "fsspec-alt-strategies-impl[cli]" fsspec-alt-strategies-impl copy src/mypkg/_contrib/strategy
```

It copies the package as installed. It replaces a previous copy, and refuses any other folder that is not empty.
Delete the modules you do not use afterwards.

### The modules to copy for each feature

| Feature                         | Modules to copy                                                            |
|---------------------------------|----------------------------------------------------------------------------|
| A family's abstract class       | `tree/` or `flat/`, and `common/`; `files/` for its file objects, if used  |
| Wrappers and `monitored()`      | `wrapper.py`, `monitor.py`, `__init__.py`, `tree/`, `flat/`, `common/` and `files/` |
| Monitor                         | Nothing: it is in the interface package                                    |
| Async file objects              | `files/`                                                                   |
| Filesystem router               | `router.py`, `common/`, and the family of each storage copied or moved from |

A monitor watches only filesystems whose methods run the strategies: the abstract classes' subclasses and the
wrappers.

## Usage

### File tree operation strategies

#### Abstract filesystem classes

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

The profile may also be a property, e.g. of the account the filesystem connects to. A method of fsspec's own can
come back, e.g. `_rm = AsyncFileSystem._rm`, and a storage's native operation can override a method.

#### Wrappers

`monitored()` wraps a filesystem it knows as the family that suits it:

```python
import fsspec
from fsspec_alt_strategies_impl import monitored

fs = monitored(fsspec.filesystem("file"))
fs.rm("data/2024", recursive=True)  # Deletes by the tree family's strategy.
```

The known filesystems are those in `known_implementations`, by class: the local filesystem, s3fs, gcsfs, adlfs,
fsspec's memory, SFTP and HDFS filesystems. Each entry says its profile, whose s_type is its family, and its
concurrency limits. The wrapper is async if the filesystem is. Pass your own table to `monitored()`, or `wrap_unknown_fs=True` to wrap any other filesystem the way
right on every storage: as a tree, one request at a time, with nothing known of the storage.

An entry's `native` lists the recursive methods passed to the filesystem as they are, e.g. `["mv"]`. Use it where the
filesystem's own method uses the storage better, as gcsfs's `mv` renames a folder in a bucket with a hierarchical
namespace. Such a call is not streamed by the strategies, and a monitor sees no step of it.

A wrapper is light and not cached: make one when you need it, and drop it after.

#### Choosing

None of these is right for every storage, and the library never picks one for you.

- **The family** follows the storage's namespace. A hierarchical namespace, a tree of folders, suits `tree`. A flat
  namespace, keys whose folders are only prefixes, suits `flat`. An exception is a filesystem listing a flat namespace
  one folder per request, such as adlfs on Azure Blob Storage: it suits `tree`.
- **The profile** only saves requests, or data copied. A field left unknown is right on any storage. A field declared
  wrongly gives wrong results: declare only what you know.
- **The concurrency limits** are yours to set. A filesystem unsafe to call from several threads, such as SFTP, needs
  them at 1.

#### Costs to know

- A flat wrapper lists a prefix with the wrapped filesystem's `find`, as fsspec has no listing by prefix. That listing
  is held in memory, and covers the prefix's whole folder. A subclass overriding `_iexpand` with the storage's own listing
  by prefix streams it instead.
- `known_implementations` is shared by every copy in a project. Change it with care, or pass your own table.

#### Choosing with numbers

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

### Monitor

A `Monitor` sees each operation behind fsspec's methods, on a filesystem running the strategies. It is given the
operation's `OperationHandler`, to stop it:

```python
from fsspec_alt_strategies import Monitor, monitoring


class Progress(Monitor):
    async def yielded(self, call, item, state, op):
        if isinstance(item, Exception):
            op.cancel()  # Stop at the first failure: the call raises CallCancelled.
        print(call.method, state)


with monitoring(Progress()):
    fs.rm("data/2024", recursive=True)
```

The library logs none of its routine work. To log each operation's start and end, use `LoggingMonitor`:
`with monitoring(LoggingMonitor(logging.getLogger("myapp"))):`. It logs each item yielded at DEBUG.

- A monitor is awaited where the operation runs, so a slow monitor slows the call. For a sync call, it runs on
  fsspec's own loop, in its thread.
- Only the calls a strategy runs are watched. A call fsspec answers itself, e.g. a copy that is not recursive, is not.

### Async file object

#### Adaptive files

The adaptive files give a filesystem ready-made file objects, fast for both random and bulk sequential access,
with no parameter to tune by hand:

- scattered reads fetch small blocks, and long sequential reads ever larger ones, fetched ahead;
- writes go in one request for a small file, and in parts growing with the bytes written for a large one;
- large reads and writes are split into requests run at once.

They suit object stores like AWS S3 best, which serve several byte ranges of one object at once.

A subclass of an adaptive file implements the storage's requests, and leaves the buffering to the file:

- `AdaptiveReadableFile` implements `_fetch_range`, which reads a byte range;
- `AdaptiveWritableFile` implements `_upload_all`, for a small file, and the multipart upload's requests:
  `_start_multipart_upload`, `_upload_part`, `_complete_multipart_upload` and `_abort_multipart_upload`.

An `AdaptiveWritableFile` opened with mode "xb" creates the file only if none is at the path. Its `exclusive` is then
True, and the request making the file visible sends the storage's conditional write.

#### Async file support

A filesystem's `open_async` returns its own files through `AsyncFileAdapter`. It returns a sync file, such as an
SDK's handle, through `SyncFileAdapter`:

```python
from fsspec_alt_strategies_impl.files.fsspec_async import SyncFileAdapter


async def open_async(self, path, mode="rb", **kwargs):
    return SyncFileAdapter(self.client.open(path, mode))
```

#### Transaction support

`TransactionFile` wraps a file-like object, adding fsspec's file transactions to an existing file implementation.
The file's writes become visible only when the transaction commits, and are dropped if it is discarded.

How a file stays invisible until commit depends on the storage. So a `TransactionFile` takes a `FileTransaction`, a
strategy of file-level transactions. The package provides these:

- `TempPathTransaction` writes to a hidden temporary path, renamed on commit. It suits any storage, best one renaming
  in one request.
- `LocalBufferTransaction` writes to a local buffer, written to the file on commit. Other clients see nothing until
  then, but the file's bytes are held locally.
- `AdaptiveWritableTransaction` holds back an adaptive file's last request, which makes it visible, until commit. It
  costs nothing more than writing without a transaction.

A storage with a way of its own, such as a staged upload, subclasses `FileTransaction`.

A filesystem's `_open` returns a `TransactionFile` when fsspec asks for no autocommit:

```python
from fsspec_alt_strategies_impl.files.fsspec_transaction import LocalBufferTransaction, TransactionFile


def _open(self, path, mode="rb", autocommit=True, **kwargs):
    if autocommit:
        return MyFile(self, path, mode)
    return TransactionFile(LocalBufferTransaction(path, lambda p: MyFile(self, p, mode)))
```

Its files then join fsspec's transactions:

```python
with fs.transaction:
    with fs.open("data/a.csv", "wb") as f:
        f.write(b"...")
    # Nothing is written yet.
# Written now, as the transaction commits.
```

### Filesystem router

`AsyncRoutingFileSystem` is abstract. A subclass implements three methods:

- `_route(path)` gives the inner filesystem and its path, or None for a path above every storage;
- `_outer(fs, path)` gives the router's path of an inner one;
- `_roots(path)` lists a folder above the storages.

`transfer_concurrency` bounds the copies and moves between storages at a time. Override `_cross_fs_cp_file` or
`_cross_fs_mv_file` for a storage with a faster way, such as a server-side copy.
