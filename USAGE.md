# Usage

How to get the library's features, and how to use them. See the
[README](https://github.com/sugibuchi/fsspec-alt-strategies#readme) for why they exist.

## Provided features

### File tree operation strategies

The strategies walk, find, glob, measure, delete, copy and move file trees. Each lists through a storage handler,
which yields entries as the storage is scanned, and acts on them as they arrive. They come in two families, by what a
storage's directories are:

- the `tree` family, in `tree/`, for storages whose directories are real, such as Azure Data Lake Storage Gen2, SFTP
  or HDFS. It deletes and renames directories, and can scan them a directory at a time, several at once;
- the `flat` family, in `flat/`, for storages whose directories are only prefixes of keys, such as S3. It lists by
  prefix, and deletes keys in batches.

Each family's handler, `TreeFileOperationHandler` or `FlatFileOperationHandler`, runs its strategies. `BaseFileSystem`
answers fsspec's recursive methods, such as `find`, `rm` and `copy`, with its handler's strategies. It runs on a
storage handler, in one of two forms:

- **Your own handler**, for a new filesystem implementation: you implement the storage's primitives in a subclass of
  `TreeFileOperationHandler` or `FlatFileOperationHandler`, and the filesystem implements fsspec's API on it.
- **Wrappers**, in `wrapper/`, for existing filesystem implementations: a handler calling the filesystem, and a
  filesystem serving its paths. `monitored()` picks the wrapper for a known filesystem.

s3fs, gcsfs and adlfs have handlers of their own, which use what each library has beyond fsspec's API:

- `S3fsHandler` streams a listing page by page, narrowed to a prefix, and copies a file without asking its size first;
- `GcsfsHandler` lists by a prefix, not the whole directory, and deletes in GCS's batches of 100;
- `AdlfsHandler` lists a tree in one paginated request, not one per directory.

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

`RoutingFileSystem`, in `router.py`, spans several storages, e.g. a cloud's volumes. Its `RoutingHandler` routes each
path to the storage holding it. It copies or moves files between different storages, leaving no part of a file.

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

Its command line copies it, with its `cli` extra. Run it from your project, with the directory to copy into:

```bash
uv run --with "fsspec-alt-strategies-impl[cli]" fsspec-alt-strategies-impl copy src/mypkg/_contrib/strategy
```

It copies the package as installed. It replaces a previous copy, and refuses any other directory that is not empty.
Delete the modules you do not use afterwards. Keep the `LICENSE` file it copies: the Apache License asks that the
code goes with it.

### The modules to copy for each feature

| Feature                         | Modules to copy                                                                    |
|---------------------------------|------------------------------------------------------------------------------------|
| A family's handler and filesystem | `tree/` or `flat/`, and `common/`; `files/` for its file objects             |
| Wrappers and `monitored()`      | `wrapper/`, `monitor.py`, `__init__.py`, `tree/`, `flat/`, `common/` and `files/` |
| Monitor                         | Nothing: it is in the interface package                                            |
| Async file objects              | `files/`                                                                           |
| Filesystem router               | `router.py`, `common/`, `files/`, and the family of each storage routed to         |

A monitor watches only filesystems whose methods run the strategies: `BaseFileSystem`'s subclasses, the wrapper and
the router.

## Usage

### File tree operation strategies

#### Storage handlers

A storage handler holds what is particular to a storage: its primitives, its paths, its profile, its limits and its
clients. Subclass `TreeFileOperationHandler` or `FlatFileOperationHandler`, and implement the primitives each declares
abstract:

```python
from fsspec_alt_strategies import UNKNOWN
from fsspec_alt_strategies_impl.tree.handler import TreeFileOperationHandler


class MyHandler(TreeFileOperationHandler):
    protocol = "my"
    root = ""
    profile = UNKNOWN  # The default: nothing known, right on any storage.

    def __init__(self, account, *, io_concurrency=8):
        super().__init__(operation_concurrency=32, io_concurrency=io_concurrency)
        self.account = account

    def verify_path(self, path):
        # Refuse what the storage would, without a request.
        if "\\" in path:
            raise ValueError(f"Not a path of the storage: {path!r}")

    async def ils(self, entry, **kwargs):
        # A directory's entries, as they come, e.g. a page at a time.
        ...

    async def info(self, entry, **kwargs): ...
    # ... and rm_file, rmdir, makedirs, cp_file, mv_file, open_file, open_async_file.
```

- A primitive takes an entry for an item that exists, and a path for one to be made. `open_file` and
  `open_async_file` take an entry, to read or write: a file to write may not exist yet, its entry holding its name,
  and its final `"size"` where the caller knows it. An entry is a dict whose
  `"name"` alone is sure. The other keys, e.g. a size or an id, are there when a listing found them, so a primitive
  needs no request to ask them again.
- `makedirs`, `cp_file` and `mv_file` take `missing=True` where nothing is at the path, as a copy into a destination it
  found missing knows. A storage asking before each creation, as one never refusing a duplicate must, skips that.
- They take `parent` too, as do `cp_tree` and `mv_tree`, and a write's entry as `"parent"`: the entry of the path's
  parent, where the caller holds it. `makedirs` returns the entry of the directory it made, or None, so that a copy
  passes it on to what goes inside. A storage addressing items by id skips resolving the parent's path.
- The strategies build paths only by the handler's `join` and `parent`, each verified by `verify_path`. `join` takes
  several paths, as `os.path.join` does, and refuses a segment such as "..".
  So a name another storage listed, e.g. one holding "..", can never leave the directory it is copied into.
- `open_file(entry, mode, *, loop, autocommit)` returns a file with Python's sync API, for fsspec's sync `open`, and
  `open_async_file(entry, mode)` one whose methods are awaited, for `open_async` and the strategies. Mode is "rb",
  "wb" or "xb". Both raise NotImplementedError by default: a handler implements what it supports, e.g. with the
  adaptive files below.
- A tree handler with a recursive listing of its own implements `ils_recursive`. Its profile's `recursive_order`
  says the listing's order: "depth_first", which covers name order, or "parents_first", which only guarantees a
  directory before what is below it. Unknown, it is taken as "parents_first", which costs the walk memory and
  streaming: declare "depth_first" where it holds.
- Without its own, a tree is listed a directory per request, several at once, by `tree.listing._tree_ils`. It is
  depth first whatever the profile says, so it needs no `recursive_order`. A handler whose storage may refuse its own
  listing falls back on it.
- A flat handler implements `ils_recursive(entry, prefix=...)`, every key below a directory starting with the prefix,
  in name order.
- `rm_tree`, `mv_tree` and `cp_tree`, a storage's own operations on a whole directory, are optional. The profile
  declares them.
- A handler whose storage deletes several files by one request, e.g. an object store, overrides the `rm_files`
  strategy with `rm_files._rm_by_request(self, entries, request)`, and declares `rm_batch_size`: a recursive `rm` or
  `mv` then deletes its files in batches. By default, each file is deleted by `rm_file`.
- Likewise, a handler whose storage copies or moves several files by one request overrides the `cp_files` or
  `mv_files` strategy with `cp_files._cp_by_request` or `_mv_by_request`, and declares `cp_batch_size` or
  `mv_batch_size`. A recursive `cp` or `mv` within the storage then sends its files in batches. By default, each file
  goes by `cp_file` or `mv_file`.
- The family's strategies are its methods: `walk`, `find`, `glob`, `du`, `rm`, `rm_files`, `cp`, `cp_files`, `mv` and
  `mv_files`. Each returns an
  operation the filesystem runs. Override one to choose another strategy, e.g. one of your own, with the same
  arguments.

A handler is made without a request. `await handler.open()` makes its clients on the running loop, and loads its
profile, once. A storage whose profile only a request can find, e.g. whether an account has a hierarchical namespace,
overrides `async def load_profile(self)`. A load that fails raises, and the next operation tries again.

Two mixins in `common/clients.py` hold a handler's clients:

- `SyncClientSupport` runs a sync client's blocking calls in threads of the handler's own, by `_run_in_thread`.
- `AsyncClientSupport` holds an async client, e.g. an aiohttp session, which belongs to the loop it was made on. A
  handler implements `create_clients`, making them in fields of its own, and `close_clients`, closing those. Its
  requests call `await self.ensure_clients()` first, which makes them on the running loop, once, and again on another
  loop. `aclose()` closes them on their loop: awaited there, or sent there as a task from another loop.

```python
from fsspec_alt_strategies_impl.common.clients import AsyncClientSupport


class MyHandler(AsyncClientSupport, TreeFileOperationHandler):
    async def create_clients(self):
        self.session = aiohttp.ClientSession()

    async def close_clients(self):
        await self.session.close()

    async def info(self, entry, **kwargs):
        await self.ensure_clients()
        async with self.session.get(...) as response:
            ...
```

It takes no path: the profile is one for the whole storage. A storage whose behaviour differs by path is several
handlers under a router. On GCS, for example, buckets differ in having a hierarchical namespace: make one handler per
bucket, and a router for the whole namespace.

#### Filesystems on a handler

`BaseFileSystem` implements fsspec's whole API on a handler. Make one on a handler, or give the storage a class of its
own, so that fsspec makes it from a URL:

```python
from fsspec_alt_strategies_impl.common.filesystem import BaseFileSystem

fs = BaseFileSystem(MyHandler("mine"), asynchronous=True)


class MyFileSystem(BaseFileSystem):
    protocol = MyHandler.protocol
    root_marker = MyHandler.root
    _strip_protocol = MyHandler.strip_protocol
    _get_kwargs_from_urls = MyHandler.options_from_url

    def __init__(self, account, *, io_concurrency=8, **kwargs):
        super().__init__(MyHandler(account, io_concurrency=io_concurrency), **kwargs)
```

A storage's class takes its storage options, makes its handler from them, and names its paths as its handler's. A
URL naming more than a path, e.g. an account or a host, gives those options by the handler's `options_from_url`,
named as the class's `_get_kwargs_from_urls`, so that `fsspec.open(url)` needs no more. The class overrides nothing
else: what the storage does is the handler's.

Its signature names each of its handler's options, and passes fsspec's on as keywords. `BaseFileSystem` names
fsspec's options: `asynchronous`, `loop`, `use_listings_cache`, `listings_expiry_time` and `max_paths`. Any other
option raises TypeError, e.g. one misspelled, which fsspec's own filesystems would silently ignore. fsspec's
`batch_size` is not taken: the handler's `io_concurrency` bounds the paths handled at a time.

`ls` keeps each listing in fsspec's `dircache`, as fsspec's implementations do: pass `refresh=True` to list again, or
`use_listings_cache=False` to keep none. Only `ls` uses it. The recursive operations always list the storage, and
`info` always asks it. Any change made through the filesystem drops the listings kept. A change by another client is
seen once a change is made, or with `refresh=True` or `invalidate_cache()`.

#### Wrappers

`monitored()` wraps a filesystem it knows by the handler that suits it:

```python
import fsspec
from fsspec_alt_strategies_impl import monitored

fs = monitored(fsspec.filesystem("file"))
fs.rm("data/2024", recursive=True)  # Deletes by the tree family's strategy.
```

It returns a `FileSystemWrapper`, on a handler calling the filesystem: an `FsspecTreeHandler` or an
`FsspecFlatHandler`, as the filesystem's family. It takes the wrapped filesystem's URLs. Make one directly:

```python
from fsspec_alt_strategies_impl.wrapper.filesystem import FileSystemWrapper
from fsspec_alt_strategies_impl.wrapper.tree import FsspecTreeHandler

fs = FileSystemWrapper(
    FsspecTreeHandler(
        fsspec.filesystem("file"),
        operation_concurrency=8,
        io_concurrency=8,
        storage_profile=UNKNOWN,
    )
)
```

The known filesystems are those in `known_implementations`, by class: the local filesystem, s3fs, gcsfs, adlfs,
fsspec's memory, SFTP and HDFS filesystems. Each entry names its handler class, e.g. "FsspecTreeHandler",
"FsspecFlatHandler" or "S3fsHandler".

- A generic handler, "FsspecTreeHandler" or "FsspecFlatHandler", takes the rest from the entry: its profile, its
  concurrency limits, and its batch sizes. Each left out is the one right on any storage: unknown, or 1.
- A dedicated handler, e.g. "S3fsHandler", declares its own as class attributes. Its entry gives none, and
  `monitored()` refuses one that does. Subclass it to change them.

A generic handler calls the filesystem's async methods if it has them. Pass your own table to
`monitored()`, or `wrap_unknown_fs=True` to wrap any other filesystem. That wraps it the way right on every storage:
by "FsspecTreeHandler", one request at a time, with nothing known of the storage.

A filesystem whose profile is found at runtime, e.g. per account, needs a handler of its own. It subclasses a
wrapper's handler, overriding `load_profile()` and the primitives the storage does better itself. Examples are
`mv_tree` renaming a directory, `cp_tree` copying one, or `rm_tree` deleting one whole. The strategies then use them
where the profile says so, streamed and monitored.

An entry's optional `handler` names a handler class of its family, in place of the generic one:

- a bare class name is a handler built into the strategies, e.g. `"S3fsHandler"`, which the entries of s3fs, gcsfs
  and adlfs name;
- a full name is a class of your own, e.g. `"mypkg.handlers.MyHandler"`, imported when wrapping.

A wrapper is light and not cached: make one when you need it, and drop it after.

#### Choosing

None of these is right for every storage, and the library never picks one for you.

- **The family** follows what a storage's directories are: `tree` where they are real, `flat` where they are only
  prefixes of keys. A filesystem whose keys have no listing by prefix but one directory per request, such as adlfs on
  Azure Blob Storage, suits `tree` too.
- **One recursive listing usually wins.** Both families list a whole tree in one paginated request where the storage
  has one. On Azure, a find of 340 directories took 1 request instead of 342. A storage with none, such as SFTP, is
  scanned a directory at a time, several at once.
- **Pruning trades requests for listing less.** A glob or `maxdepth` on a `tree` filesystem lists only the directories
  it can match, one per request. That wins on a large tree whose pattern has a wildcard early. It loses on a small
  subtree, which one recursive listing would cover.
- **The profile** picks the way within a family, e.g. deleting a directory in one request where the storage can. It only
  saves requests, or data copied. A field left unknown is right on any storage. A field declared wrongly gives wrong
  results: declare only what you know.
- **The concurrency limits** are the handler's, yours to set. A filesystem unsafe to call from several threads, such as
  SFTP, needs them at 1.
- **`io_concurrency` bounds memory too.** Each read or write in flight holds a block or a part, shared by all the
  handler's files. So the data in flight is at most `io_concurrency` times the file class's `max_block_size`, e.g.
  16 × 16 MiB = 256 MiB of writes, however many files are open. Each file written also fills one part. fsspec's
  methods on many paths, such as `cat`, `pipe`, `get` and `put`, handle `io_concurrency` paths at a time, unless given
  a `batch_size`.
- **`cat_ranges` merges nearby ranges.** The ranges of a file at most `max_gap` apart, 1 MiB by default, are fetched
  in one request with the gap, up to 64 MiB, and a large one in parallel parts. Zarr and kerchunk read through it: a
  shard's chunks then cost a few requests, not one each.

#### Costs to know

- The generic flat handler lists a prefix with the wrapped filesystem's `find`, as fsspec has no listing by prefix in
  pages. That listing is held in memory, and covers the prefix's whole directory. A handler overriding `ils_recursive`
  with the storage's own listing by prefix streams it instead, as `S3fsHandler` does.
- The generic tree handler scans a tree a directory per request, as fsspec has no recursive listing in pages. A
  handler implementing `ils_recursive` with the storage's own recursive listing lists it in one, as `AdlfsHandler`
  does.
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

Its `TreeView` and `FlatView` are handlers of it, to run a strategy on, each primitive replaceable.

### Monitor

A `Monitor` is given each operation's `OperationHandler`, to stop it:

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
- Only the calls a strategy runs are watched. A call fsspec answers itself is not, e.g. a copy of a pattern that is
  not recursive. A single file's copy or move, recursive or not, goes by the strategy.

### Async file object

#### Adaptive files

The adaptive files give a handler ready-made file objects, fast for both random and bulk sequential access, with no
parameter to tune by hand:

- scattered reads fetch small blocks, and long sequential reads ever larger ones, fetched ahead;
- writes go in one request for a small file, and in parts growing with the bytes written for a large one;
- large reads and writes are split into requests run at once.

They suit object stores like AWS S3 best, which serve several byte ranges of one object at once.

A subclass of an adaptive file implements the storage's requests, and leaves the buffering to the file:

- `AdaptiveReadableFile` implements `_fetch_range`, which reads a byte range;
- `AdaptiveWritableFile` implements `_upload_all`, for a small file, and the multipart upload's requests:
  `_start_multipart_upload`, `_upload_part`, `_complete_multipart_upload` and `_abort_multipart_upload`.
- A storage choosing the part size as the upload starts, e.g. a Box upload session, has the writer's
  `_start_multipart_upload` set `part_size`: every part but the last then has that size, the last the rest.
- A storage needing the bytes in order, e.g. for the whole file's SHA-1, reads them in `_written`, called with each
  write: the parts may be uploaded out of order.

A handler's `open_file` returns them as they are, made on its `loop`, and its `open_async_file` through
`AsyncFileAdapter`, with its `io_limit` as their `concurrency_limit`:

```python
from fsspec.asyn import sync

from fsspec_alt_strategies_impl.files.fsspec_async import AsyncFileAdapter


def open_file(self, entry, mode="rb", *, loop, autocommit=True, **kwargs):
    return sync(loop, self._adaptive, entry, mode)


async def open_async_file(self, entry, mode="rb", **kwargs):
    return AsyncFileAdapter(await self._adaptive(entry, mode))
```

An
`AdaptiveWritableFile` opened with mode "xb" creates the file only if none is at the path. Its `exclusive` is then
True, and the subclass checks it by the storage's conditional write where there is one.

The file decides which requests to send, and when. Whether a write is atomic is the storage's:

- where an upload is invisible until completed, as on S3, Google Cloud Storage or Azure, a file is written whole, or not
  at all. Declare `atomic_writes=True` in the handler's profile;
- where a file is visible while it is written, as on DBFS, a failed write may leave part of it.

#### Async file support

A handler whose storage gives sync files of its own, e.g. an SDK's handle, returns them from `open_file` as they are,
and from `open_async_file` through `SyncFileAdapter`, which runs their blocking calls in a thread:

```python
from fsspec_alt_strategies_impl.files.fsspec_async import SyncFileAdapter


def open_file(self, entry, mode="rb", *, loop, autocommit=True, **kwargs):
    return self.client.open(entry["name"], mode)


async def open_async_file(self, entry, mode="rb", **kwargs):
    return SyncFileAdapter(self.client.open(entry["name"], mode))
```

A handler reading or writing a file by one request, e.g. one download, overrides `cat_file`, `pipe_file`, `get_file`
or `put_file`, holding a permit of its `io_limit` around the request:

```python
async def cat_file(self, entry, start=None, end=None, **kwargs):
    async with self.io_limit:
        return await self.client.download(entry["name"], start, end)
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
  costs nothing more than writing without a transaction, but needs a storage where an upload is invisible until
  completed. Where the last part completes the upload, as in a resumable upload, the file's class declares
  `last_part_completes = True`, and the transaction holds that part back too.

A storage with a way of its own, such as a staged upload, subclasses `FileTransaction`.

Transactions are the handler's: its `open_file` is given `autocommit=False` within one. A handler supporting them
returns a `TransactionFile` then, e.g. on an `AdaptiveWritableTransaction` for an adaptive writer, or on another
strategy for files of its own. One ignoring `autocommit` writes as it would outside a transaction:

```python
from fsspec_alt_strategies_impl.files.fsspec_transaction import LocalBufferTransaction, TransactionFile


def open_file(self, entry, mode="rb", *, loop, autocommit=True, **kwargs):
    path = entry["name"]
    if autocommit or mode == "rb":
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

A `RoutingHandler` says where each path of a namespace goes. Implement the first three of these methods, and the
fourth if some paths are above every storage:

- `verify_path(path)` checks the router's own paths;
- `route(path)` gives the handler of the path's storage and its path there, or None for a path above every storage;
- `outer(handler, path)` gives the router's path of a path of one of its handlers;
- `roots(entry)` lists a directory above the storages. By default, it raises FileNotFoundError.

```python
from fsspec_alt_strategies_impl.router import RoutingFileSystem, RoutingHandler


class Lake(RoutingHandler):
    protocol = "lake"

    def verify_path(self, path): ...

    def route(self, path):
        volume, _, inner = path.partition("/")
        if not volume:
            return None  # The root, above every volume.
        return self.inner(volume, lambda: VolumeHandler(volume)), inner

    def outer(self, handler, path):
        return f"{handler.volume}/{path}".rstrip("/")


fs = RoutingFileSystem(Lake(transfer_concurrency=8), asynchronous=True)
```

- `route` makes each handler once, by `inner`, without a request. The routing handler's `aclose()` closes them,
  and its sync `close()` releases them at once.
- Each path is checked twice: by the routing handler, then its path in the storage by the storage's handler.
- `route` may give another routing handler, which routes the path again. The names answered are the outermost
  router's.
- A storage's class for the router follows the families' convention: it names its paths as its handler's, and makes
  its handler from its storage options. `RoutingFileSystem` names fsspec's options as the families' filesystems do.

`transfer_concurrency` bounds the files copied between storages at a time. Each file is streamed by default, a block
at a time:

- On a storage whose profile declares `atomic_writes`, it is written in place. Elsewhere, it is written to a temporary
  name beside the destination, then moved there, so that no reader sees part of a file.
- Its size is checked against the source's. A file of another size raises OSError, and leaves nothing.
- A move deletes the source file once its copy succeeds.

Override `cp_across` or `mv_across` for a storage with a faster way, such as a copy on the server. Each takes the
source file's entry, with its size, as the listing found it.
