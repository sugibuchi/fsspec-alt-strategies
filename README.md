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

## What it offers

### Faster file tree operations

Finding, deleting, copying or moving a whole folder tree is faster, on storages with folders and on object stores
alike. Many folders are listed at once, and what you tell the library about your storage saves further requests.

### Work on huge folders

Files are deleted, copied or moved as soon as they are found, not after the whole tree is listed. A folder of hundreds
of thousands of files never needs to fit in memory.

### Progress you can see, and stop

You can follow a long operation file by file, see which files failed while the others go on, and stop it at any time.
This works with the familiar fsspec calls, such as `fs.rm(path, recursive=True)`.

### Faster reads and writes

Files read and write faster in cloud storages, adapting to how each file is used. Writes can wait for fsspec's
transaction to commit, even with an SDK's own file objects.

### Copies between storages

One filesystem can span several storages, and copy or move files between them.

### Works with what you have

Existing fsspec filesystems, such as s3fs or the local one, get these features with a single call. A new filesystem
gets them by implementing a few basic operations.

## Getting started

The library has two packages:

- `fsspec-alt-strategies`, installed with `pip install fsspec-alt-strategies`;
- `fsspec-alt-strategies-impl`, which we recommend copying into your own package, so that its behaviour changes only
  when you choose.

[USAGE.md](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/USAGE.md) explains what to copy for each
feature, and how to use each one.

## License

Apache License 2.0. See [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
