# fsspec-alt-strategies-impl

Alternative, filesystem-independent strategies for fsspec: tree operations, glob and file buffering.

The strategies are meant to be copied into a project, e.g. as `_contrib/strategy`, not installed. They depend on fsspec
and on [fsspec-alt-strategies](https://pypi.org/project/fsspec-alt-strategies/), the interface they share,
which is installed.

See the [project's README](https://github.com/sugibuchi/fsspec-alt-strategies#readme) for what they are, and when each
wins.

## License

Apache License 2.0. See [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
