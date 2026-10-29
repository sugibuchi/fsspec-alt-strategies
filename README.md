# fsspec-alt-strategies

Alternative, filesystem-independent strategies for [fsspec](https://github.com/fsspec/filesystem_spec) filesystems.

> **Status:** pre-alpha. Nothing below is implemented yet. This is the plan.

fsspec's design largely assumes object stores with a flat key space. Most of its tree operations expand a whole subtree
by listing it first. That is not always efficient on filesystems with a hierarchical namespace, such as Azure Data Lake
Storage Gen2 or Databricks volumes.

This library offers other ways of doing the same things. It is a third-party project, not part of fsspec.

## Planned features

- **Tree operations** for filesystems with a hierarchical namespace: listing, finding, deleting and copying trees.
- **Glob** that walks only the folders a pattern can match, with patterns compiled to an NFA.
- **An abstract file class** that adapts its buffering to how a file is read. It serves both random reads of small
  ranges and bulk sequential reads, without tuning per file.
- **Observability:** callbacks that report an operation's intermediate state. A callback's return value can cancel the
  operation.

Async filesystems come first. Some features, such as observability, may come to sync filesystems too.

## Ways to use it

Each fsspec feature may have several strategies. You pick those that suit your storage: none is right for every
storage.

- Call a strategy as a function, with the filesystem and its arguments.
- Subclass a provided base filesystem class, whose methods call the same functions.

The package is on PyPI, but we recommend copying the modules you use into your own package, e.g. as
`_contrib/strategy`. Your filesystem's behaviour then never changes with a release of this library. The modules use
relative imports only and depend on nothing but fsspec, so a copy works under any name. Keep their license headers.

## Choosing with numbers

`fsspec_alt_strategies.testing.SimulatedFileSystem` is an in-memory async filesystem. It counts the requests a real
storage would get, and delays each by a latency you set. It models an object store, or a storage with a hierarchical
namespace. Run a strategy and fsspec's default on it, then compare their requests and times.

```python
from fsspec_alt_strategies.testing import SimulatedFileSystem

fs = SimulatedFileSystem(hierarchical=True, latency=0.02)
fs.add_file("data/2024/01/a.parquet", b"...")
fs.rm("data", recursive=True)
print(fs.stats.calls)  # Counter({'delete': 1})
```

## License

Apache License 2.0. See [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
