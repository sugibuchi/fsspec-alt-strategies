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

- **Strategies,** in two families, by how a storage lists:
  - `tree`, for storages listing one folder per request: a concurrent walk, and find, du, a pruning glob, and a
    recursive delete, copy and move that act while the walk goes on;
  - `flat`, for object stores listing a key prefix in one paginated request: the same, streaming that listing, with
    deletions in batches.
- **Operations:** a strategy returns an operation, iterated step by step. It yields what it did and the error of each
  path that failed, reports its progress, and stops as an asyncio task does.
- **Filesystem classes,** `TreeFileSystem` and `FlatFileSystem`, whose fsspec methods are a family's strategies, with
  one concurrency limit for requests and one for data.
- **A router,** `AsyncRoutingFileSystem`, the base of a filesystem spanning several storages, e.g. a cloud's volumes.
  It passes each call to the storage's own filesystem, and copies between storages with the source's strategy.
- **Wrappers** that give an existing fsspec filesystem those methods, and `monitored()`, which picks the wrapper for a
  known filesystem.
- **Monitors,** which watch the operations behind fsspec's methods, such as `fs.rm(...)`, without changing their API.
- **Storage profiles,** where you declare what a storage does, such as renaming a folder in one request, so that the
  strategies send fewer requests.
- **File objects** that adapt their requests to how a file is read or written, and fsspec's transactions for any file
  object, including an SDK's own.

## Two packages

| Package                        | What it holds                                                    | How you get it |
|--------------------------------|------------------------------------------------------------------|----------------|
| `fsspec-alt-strategies-common` | The interface: operations' control and states, monitors, storage profiles, known implementations | Installed |
| `fsspec-alt-strategies`        | The strategies, filesystem classes, wrappers and file objects    | Copied, or installed |

We recommend copying the strategies' modules you use into your own package, e.g. as `_contrib/strategy`. Your
filesystem's behaviour then never changes with a release of this library. The modules import each other relatively,
so a copy works under any name. Keep their license headers.

The interface is installed, never copied:

```bash
pip install fsspec-alt-strategies-common
```

So every copy in a project, of any version, shares one interface: one `monitoring` block watches the calls of all of
them, and an exception such as `CallCancelled` is the same class whichever copy raised it.

A copy keeps what the modules it uses import:

- each family, `tree/` or `flat/`, needs `common/` and `files/`;
- the router, `common/routing.py`, needs `common/`, and the family of each storage it copies or moves from, imported
  only then;
- `monitored()` needs the family of each filesystem it wraps, and imports it only then, so a copy may keep one family.

## Ways to use it

### Watch an existing filesystem

`monitored()` wraps a filesystem it knows as the family that suits it. A `Monitor` then sees each operation behind
fsspec's methods:

```python
import fsspec
from fsspec_alt_strategies import monitored
from fsspec_alt_strategies_common import Monitor, monitoring


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
from fsspec_alt_strategies.tree.filesystem import TreeFileSystem
from fsspec_alt_strategies_common import StorageProfile


class MyFileSystem(TreeFileSystem):
    protocol = "my"
    storage_profile = StorageProfile()  # Nothing known: right on any storage.

    async def _ls(self, path, detail=True, **kwargs): ...
    async def _info(self, path, **kwargs): ...
    # ... and the other primitives each class declares abstract.


fs = MyFileSystem(operation_concurrency=32, io_concurrency=8)
```

### Call a strategy

A strategy is a plain function, taking the storage's operations it uses:

```python
import asyncio
from fsspec_alt_strategies.tree.find import tree_find

find = tree_find("data", ls=fs._ls, info=fs._info, concurrency_limit=asyncio.Semaphore(32))
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
  is held in memory, and covers the prefix's whole folder. A subclass overriding `_ils` with the storage's own listing
  by prefix streams it instead.
- A monitor is awaited where the operation runs, so a slow monitor slows the call. For a sync call, it runs on
  fsspec's own loop, in its thread.
- `known_implementations` is shared by every copy in a project. Change it with care, or pass your own table.
- Only the calls a strategy runs are watched. A call fsspec answers itself, e.g. a copy that is not recursive, is not.

## Choosing with numbers

`fsspec_alt_strategies_testing.SimulatedFileSystem` is an in-memory async filesystem. It counts the requests a real
storage would get, and delays each by a latency and a bandwidth you set. It models an object store, or a storage with a
hierarchical namespace. Run a strategy and fsspec's default on it, then compare their requests and times. It is a
package of its own, so a copy of the strategies leaves it out.

```python
from fsspec_alt_strategies_testing import SimulatedFileSystem

fs = SimulatedFileSystem(hierarchical=True, latency=0.02)
fs.add_file("data/2024/01/a.parquet", b"...")
fs.rm("data", recursive=True)
print(fs.stats.calls)  # Counter({'delete': 1})
```

## License

Apache License 2.0. See [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
