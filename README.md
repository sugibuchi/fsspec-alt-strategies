# fsspec-alt-strategies

[![PyPI - Version](https://img.shields.io/pypi/v/fsspec-alt-strategies?color=blue&label=fsspec-alt-strategies)](https://pypi.org/project/fsspec-alt-strategies/)
[![PyPI - Version](https://img.shields.io/pypi/v/fsspec-alt-strategies-impl?color=blue&label=fsspec-alt-strategies-impl)](https://pypi.org/project/fsspec-alt-strategies-impl/)
[![Python versions](https://img.shields.io/pypi/pyversions/fsspec-alt-strategies-impl)](https://pypi.org/project/fsspec-alt-strategies-impl/)
[![License](https://img.shields.io/github/license/sugibuchi/fsspec-alt-strategies)](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE)
[![CI](https://github.com/sugibuchi/fsspec-alt-strategies/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/sugibuchi/fsspec-alt-strategies/actions/workflows/ci.yaml)
[![CodeQL](https://github.com/sugibuchi/fsspec-alt-strategies/actions/workflows/codeql.yaml/badge.svg?branch=main)](https://github.com/sugibuchi/fsspec-alt-strategies/actions/workflows/codeql.yaml)
[![codecov](https://codecov.io/gh/sugibuchi/fsspec-alt-strategies/graph/badge.svg)](https://codecov.io/gh/sugibuchi/fsspec-alt-strategies)

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

## Using it

[USAGE.md](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/USAGE.md) shows how:

- to watch an existing filesystem's operations, with `monitored()` and a `Monitor`;
- to subclass `TreeFileSystem` or `FlatFileSystem` for a new storage;
- to call a strategy directly;
- to choose a family, a profile and the concurrency limits, with the costs to know and a simulated storage to measure.
## License

Apache License 2.0. See [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
