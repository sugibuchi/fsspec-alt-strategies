# fsspec-alt-strategies-common

The interface the strategies of [fsspec-alt-strategies](https://github.com/sugibuchi/fsspec-alt-strategies) share with
their callers:

- `OperationHandler` and `State`: the control and the progress of the operation a strategy returns.
- `Monitor` and `monitoring`, watching the operations behind a filesystem's fsspec methods.
- `routing` and `current_routers`, telling a monitor's `Call` the routers it came through, as its `via`.
- `StorageProfile`, what a storage does that fsspec leaves open.
- The states the strategies report, such as `FindState`.

The strategies are copied into a project. This package is installed once, so that every copy in a project shares one
interface, and one set of monitors.

```bash
pip install fsspec-alt-strategies-common
```

## License

Apache License 2.0. See [LICENSE](https://github.com/sugibuchi/fsspec-alt-strategies/blob/main/LICENSE).
