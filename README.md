# fsspec-alt-strategies

Alternative, filesystem-independent strategies for [fsspec](https://github.com/fsspec/filesystem_spec) filesystems, and
a way to watch and stop what they do.

> **Status:** pre-alpha. The API may change.

fsspec's recursive operations, such as `find`, `rm` and `copy`, list a whole tree before they act on it. That suits
object stores, which list every key under a prefix in one paginated request. It is slow on storages with a hierarchical
namespace, which list one folder per request, such as Azure Data Lake Storage Gen2, SFTP or HDFS. And once called, such
an operation cannot be watched or stopped until it ends.

This library offers other ways of doing the same things. It is a third-party project, not part of fsspec.

## What is in it

### File tree operations for hierarchical namespaces

The `tree` strategies suit storages listing one folder per request, such as Azure Data Lake Storage Gen2, SFTP or HDFS.
They list several folders at a time: a walk, find, du, a glob listing only the folders it can match in, and a recursive
delete, copy and move. Their counterparts in `flat` suit object stores, which list a key prefix in one paginated request.

Storage profiles declare what a storage does, such as renaming a folder in one request. The strategies then send fewer
requests.

### Streaming through large trees

A delete, copy or move acts on what it finds while the walk goes on, rather than after listing the whole tree. A
folder's listing streams in parts, so that a folder of hundreds of thousands of files is never held whole in memory. The
first files are deleted or copied after the first part of the first listing.

### File objects that adapt to how a file is used

The file objects for cloud storages adapt their buffering to the access pattern:

- scattered reads fetch small blocks;
- the longer a run of sequential reads, the bigger the blocks, fetched further ahead, several at a time;
- a small file is written in one request, a larger one in parts that grow with the bytes written.

fsspec's transactions work with any file object, including an SDK's own.

### Observability and control of bulk operations

A strategy returns an operation, iterated step by step:

- it yields each path done, and the error of each path that failed, while the others go on;
- it reports its progress as it goes;
- it stops as an asyncio task does: cancelled, it finishes what is in flight.

Monitors watch the operations behind fsspec's methods, such as `fs.rm(...)`, file by file, without changing their API.

### Wrappers for existing filesystems

The wrappers give an existing fsspec filesystem the strategies, observability and control above. `monitored()` picks the
wrapper for a known filesystem, such as s3fs or the local one.

`TreeFileSystem` and `FlatFileSystem` are the classes to subclass for a new filesystem. A router,
`AsyncRoutingFileSystem`, spans several storages, e.g. a cloud's volumes, and copies between them with the source's
strategy.

## Two packages

| Package                        | What it holds                                                    | How you get it |
|--------------------------------|------------------------------------------------------------------|----------------|
| `fsspec-alt-strategies`        | The interface: operations' control and states, monitors, storage profiles, known implementations | Installed |
| `fsspec-alt-strategies-impl`   | The strategies, filesystem classes, wrappers and file objects    | Copied, or installed |

We recommend copying the strategies' modules you use into your own package, e.g. as `_contrib/strategy`. Your
filesystem's behaviour then never changes with a release of this library. The modules import each other relatively,
so a copy works under any name. Keep their license headers.

The interface is installed, never copied:

```bash
pip install fsspec-alt-strategies
```

So every copy in a project, of any version, shares one interface: one `monitoring` block watches the calls of all of
them, and an exception such as `CallCancelled` is the same class whichever copy raised it.

A copy keeps what the modules it uses import:

- each family, `tree/` or `flat/`, needs `common/` and `files/`;
- the router, `router.py`, needs `common/`, and the family of each storage it copies or moves from, imported only
  then;
- the wrappers, `wrapper.py`, and so `monitored()`, need both families.

## Ways to use it

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

## Choosing

None of these is right for every storage, and the library never picks one for you.

- **The family** follows how the storage lists. A storage listing a folder per request, even an object store such as
  Azure Blob Storage through adlfs, suits `tree`. One listing a key prefix in one paginated request suits `flat`.
- **The profile** only saves requests, or data copied. A field left unknown is right on any storage. A field declared
  wrongly gives wrong results: declare only what you know.
- **The concurrency limits** are yours to set. A filesystem unsafe to call from several threads, such as SFTP, needs
  them at 1.

## Costs to know

- A flat wrapper lists a prefix with the wrapped filesystem's `find`, as fsspec has no listing by prefix. That listing
  is held in memory, and covers the prefix's whole folder. A subclass overriding `_iexpand` with the storage's own listing
  by prefix streams it instead.
- A monitor is awaited where the operation runs, so a slow monitor slows the call. For a sync call, it runs on
  fsspec's own loop, in its thread.
- `known_implementations` is shared by every copy in a project. Change it with care, or pass your own table.
- Only the calls a strategy runs are watched. A call fsspec answers itself, e.g. a copy that is not recursive, is not.

## Choosing with numbers

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

## License

Apache License 2.0. See [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
