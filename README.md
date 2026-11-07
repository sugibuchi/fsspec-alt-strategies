# fsspec-alt-strategies

Alternative, filesystem-independent strategies for [fsspec](https://github.com/fsspec/filesystem_spec) filesystems.

> **Status:** pre-alpha. The API may change. So far, there are:
>
> - a concurrent tree walk, and these strategies built on it: find, du, a pruning glob, and a recursive delete, copy
>   and move;
> - for flat namespaces, such as object stores, strategies streaming one listing of a key prefix: a walk, find, du, a glob
>   listing only the keys starting as its pattern does, a delete in batches, and a copy and move;
> - abstract async filesystems, one per family, whose recursive methods are its strategies, with two concurrency
>   limits: one for file operations, one for reading and writing data;
> - storage profiles, where a developer declares what a storage does, such as whether writes need the parent folder;
> - abstract file classes that adapt their requests to how a file is read or written:
>   - a reader that pins the version it reads, failing if the file changes meanwhile;
>   - a writer that uploads in parts, including to storages taking parts only in order, such as Google Cloud Storage;
> - an abstract file class for storages transferring only whole files, which works on a local copy and can create a
>   file atomically;
> - fsspec's transactions for any file object, including an SDK's own, through strategies such as a temporary path or
>   a local buffer;
> - an adapter giving these file objects the API of the files that fsspec's `open_async()` returns.
>
> The rest below is the plan.

fsspec's design largely assumes object stores with a flat key space. Most of its tree operations expand a whole subtree
by listing it first. That is not always efficient on filesystems with a hierarchical namespace, such as Azure Data Lake
Storage Gen2 or Databricks volumes.

This library offers other ways of doing the same things. It is a third-party project, not part of fsspec.

## Planned features

- **Tree operations** for filesystems with a hierarchical namespace: listing, finding, deleting and copying trees.
- **Glob** that lists only the folders where a pattern can still match.
- **An abstract file class** that adapts its buffering to how a file is read. It serves both random reads of small
  ranges and bulk sequential reads, without tuning per file.
- **Observability:** operations iterated step by step, reporting their state so far. The caller can cancel one at any
  step, and it then ends cleanly.

Async filesystems come first. Some features, such as observability, may come to sync filesystems too.

## Ways to use it

Each fsspec feature may have several strategies. You pick those that suit your storage: none is right for every
storage.

- Call a strategy as a function, with its arguments and the storage's operations it uses, e.g. `ls=fs._ls`.
- Subclass a family's abstract filesystem, `TreeFileSystem` or `FlatFileSystem`, whose methods call the same functions.
  You implement the storage's primitives, such as `_ls` and `open_async`.

The strategies are on PyPI, as `fsspec-alt-strategies`, but we recommend copying the modules you use into your own
package, e.g. as `_contrib/strategy`. Your filesystem's behaviour then never changes with a release of this library.
The modules use relative imports only, so a copy works under any name. Keep their license headers.

The interface they share is a package of its own, installed rather than copied:

```bash
pip install fsspec-alt-strategies-common
```

It holds the operations' control and states, the monitors and `StorageProfile`, so that every copy in a project shares one of each. The copies
depend on it and on fsspec alone.

## Choosing with numbers

`fsspec_alt_strategies_testing.SimulatedFileSystem` is an in-memory async filesystem. It counts the requests a real
storage would get, and delays each by a latency and a bandwidth you set. It models an object store, or a storage with a hierarchical
namespace. Run a strategy and fsspec's default on it, then compare their requests and times. It is a package of its
own, installed along with the strategies, so a copy of them leaves it out.

```python
from fsspec_alt_strategies_testing import SimulatedFileSystem

fs = SimulatedFileSystem(hierarchical=True, latency=0.02)
fs.add_file("data/2024/01/a.parquet", b"...")
fs.rm("data", recursive=True)
print(fs.stats.calls)  # Counter({'delete': 1})
```

## License

Apache License 2.0. See [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
