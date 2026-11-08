# Using fsspec-alt-strategies

How to use the strategies, and how to choose among them. See the
[README](https://github.com/sugibuchi/fsspec-alt-strategies#readme) for what they are, and how to install or copy them.
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
