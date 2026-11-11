# fsspec-alt-strategies

[![PyPI - Version](https://img.shields.io/pypi/v/fsspec-alt-strategies?color=blue&label=fsspec-alt-strategies)](https://pypi.org/project/fsspec-alt-strategies/)
[![PyPI - Version](https://img.shields.io/pypi/v/fsspec-alt-strategies-impl?color=blue&label=fsspec-alt-strategies-impl)](https://pypi.org/project/fsspec-alt-strategies-impl/)
[![Python versions](https://img.shields.io/pypi/pyversions/fsspec-alt-strategies-impl)](https://pypi.org/project/fsspec-alt-strategies-impl/)
[![License](https://img.shields.io/github/license/sugibuchi/fsspec-alt-strategies)](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE)
[![CI](https://github.com/sugibuchi/fsspec-alt-strategies/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/sugibuchi/fsspec-alt-strategies/actions/workflows/ci.yaml)
[![CodeQL](https://github.com/sugibuchi/fsspec-alt-strategies/actions/workflows/codeql.yaml/badge.svg?branch=main)](https://github.com/sugibuchi/fsspec-alt-strategies/actions/workflows/codeql.yaml)
[![codecov](https://codecov.io/gh/sugibuchi/fsspec-alt-strategies/graph/badge.svg)](https://codecov.io/gh/sugibuchi/fsspec-alt-strategies)

An alternative implementation of the file operations behind [fsspec](https://github.com/fsspec/filesystem_spec)'s
filesystem interface.

> **Status:** alpha. The API may change.

The library aims to handle large cloud storages efficiently, such as data lakes holding millions of files across many
directories. It is a third-party project, not part of fsspec.

## Principles

### Streaming first

- Its file operations list with one primitive, `_ils`. fsspec's own operations list every path first, with
  `expand_path`. `_ils` is an async generator instead: it scans the storage step by step, and yields each path as it
  is found.
- The operations built on `_ils` act on its paths as they arrive, as far as they can. So a large file tree is processed
  without ever being held in memory whole.

### Hierarchical namespaces, used

- Where a storage lists by directory, some operations scan the directories recursively, several at once. That can beat a
  scan of a flat list of files, e.g. for a pattern pruning most of the tree.
- A storage's own operations on a whole tree, such as the atomic move of a directory, plug in where the storage has
  them.

### One implementation for different storages

- fsspec's abstract filesystem classes give default file operations, but many fsspec implementations override them,
  each for its own storage's behaviour.
- This library declares a storage's behaviour in a profile instead. From the profile declared for a filesystem, its
  operations choose the fitting algorithm.

### Adaptive file objects

- The adaptive file classes are the skeleton of a file-like object whose buffers resize as the data is read or
  written.
- They fetch and send a file's blocks on an event loop, in the background and in parallel, as far as they can.

### Observability and control

- Follow a long operation file by file, see which files failed, and stop it at any time, through the familiar fsspec
  calls.

## It also offers

- **Copies between storages:** one filesystem can span several storages, and copy or move files between them.
- **Transactions:** writes can wait for fsspec's transaction to commit.
- **Works with what you have:** existing fsspec filesystems, such as s3fs, get these features with a single call, and
  a new one by implementing a few basic operations.

## Getting started

The library has two packages:

- `fsspec-alt-strategies`, installed with `pip install fsspec-alt-strategies`;
- `fsspec-alt-strategies-impl`, which we recommend copying into your own package, so that its behaviour changes only
  when you choose.

[USAGE.md](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/USAGE.md) explains what to copy for each
feature, and how to use each one.

## Compatibility

- **Python:** 3.11 to 3.14, each tested in CI.
- **fsspec:** 2024.2.0 or later.
- **Dependencies:** none but fsspec. The command line copying the strategies needs click, with the `cli` extra.
- **Filesystems:** any fsspec filesystem, async or sync. Those known to `monitored()` are the local filesystem,
  s3fs, gcsfs, adlfs, and fsspec's memory, SFTP and HDFS filesystems.

## Development, security and license

- [DEVELOPMENT.md](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/DEVELOPMENT.md) explains how to work
  on the library: its design, setup, tests and releases.
- [SECURITY.md](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/SECURITY.md) explains how to report a
  vulnerability.
- The library is licensed under the Apache License 2.0. See
  [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
