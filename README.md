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

- Its file operations list with one primitive, a storage's listing. fsspec's own operations list every path first,
  with `expand_path`. The listing is an async iterator instead: it scans the storage step by step, and yields each
  path as it is found.
- The operations act on the paths as they arrive, as far as they can. So a large file tree is processed without ever
  being held in memory whole.

### Hierarchical namespaces, used

- Where a storage lists by directory, some operations scan the directories recursively, several at once. That can
  beat a scan of a flat list of files, e.g. for a pattern pruning most of the tree.
- A storage's own operations on a whole tree, such as the atomic move of a directory, plug in where the storage has
  them.

### One implementation for different storages

- fsspec's abstract filesystem classes give default file operations, but many fsspec implementations override them,
  each for its own storage's behaviour.
- This library declares a storage's behaviour in a profile instead. The operations adapt their requests to the
  profile, e.g. a move renames where the storage renames.
- A storage has one profile. Where its behaviour differs by path, e.g. from one bucket to another, it is served as
  several storages behind a router, each with its own profile. So no operation needs exceptions by path.
- The same goes for files: one abstract file object class works well across different types of workloads, without
  parameters fine-tuned for each.

### Observability and control

- Follow a long operation file by file, see which files failed, and stop it at any time, through the familiar fsspec
  calls.

### AI oriented

- We assume that AI assistants will do much of the work of developing and maintaining fsspec implementations.
- The library offers material that gives an AI assistant enough context to do that work. That includes its documented
  design and conventions, the docstrings of the primitives an implementation writes, and a simulated filesystem to test
  against.

## Offered by this package

### A framework for new filesystem classes

- A new filesystem implements a storage handler: a few basic operations on single files, and the checks of its paths.
  The library gives the rest of fsspec's API on it.
- Every path is checked before the storage is accessed. A copy places each file below its destination, so a name
  holding ".." can never leave it.
- Every filesystem answers and fails as fsspec's do, and keeps fsspec's listings cache. Each of its calls can be
  monitored.
- A storage changing many files by one request, e.g. a batch delete, plugs that request in. The library sends it the
  files in batches.
- A developer chooses another algorithm for one operation by overriding it, e.g. a storage's own glob.
- A router spans several storages under one filesystem, each served by its own handler. It copies and moves files
  between them, leaving no part of a file, and checks each file's size.

### Adaptive file objects

- The adaptive file classes aim at good read and write performance across access patterns, from random reads of short
  ranges to bulk sequential reads. No parameter or buffering strategy needs tuning for each workload.
- They resize their buffers as the data is read or written.
- They fetch and send a file's blocks on an event loop, in the background and in parallel, as far as they can.

### Client SDK support

- Many storages are reached through a client library, sync or async. A pluggable mixin creates a storage's client and
  shuts it down, the same way for both kinds.
- Some client libraries give file-like objects of their own to read and write a storage's files. So transactions come
  as adapters around any file-like object, with a few predefined transaction strategies to choose from.

### Wrappers for existing fsspec implementations

- Existing fsspec filesystems, such as s3fs, gcsfs and adlfs, get these features with a single call.
- Each known one is wrapped with its storage's profile and limits. s3fs, gcsfs and adlfs each get a wrapper of their
  own, which uses what their library does beyond fsspec's API to save requests.

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
- **Dependencies:** none but fsspec, beside the library's own `fsspec-alt-strategies`. The command line copying the
  strategies needs click, with the `cli` extra.
- **Filesystems:** any fsspec filesystem, async or sync. Those known to `monitored()` are the local filesystem,
  s3fs, gcsfs, adlfs, and fsspec's memory, SFTP and HDFS filesystems.

## Development, security and license

- [DEVELOPMENT.md](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/DEVELOPMENT.md) explains how to work
  on the library: its design, setup, tests and releases.
- [SECURITY.md](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/SECURITY.md) explains how to report a
  vulnerability.
- The library is licensed under the Apache License 2.0. See
  [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
