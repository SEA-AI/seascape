import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        metavar="I/N",
        help="run every Nth test, from the Ith: CI's jobs split the suite this way",
    )


def _shard(config: pytest.Config) -> tuple[int, int] | None:
    shard = config.getoption("shard")
    if shard is None:
        return None
    index, _, count = shard.partition("/")
    if not (index.isdigit() and count.isdigit() and 1 <= int(index) <= int(count)):
        raise pytest.UsageError(f"--shard {shard}: give I/N, with I from 1 to N")
    return int(index), int(count)


def pytest_configure(config: pytest.Config) -> None:
    _shard(config)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    shard = _shard(config)
    if shard is None:
        return
    index, count = shard
    # Round robin in collection order: a test's parametrisations, which tend to cost
    # alike, land on different shards.
    mine = [n % count == index - 1 for n in range(len(items))]
    config.hook.pytest_deselected(
        items=[i for i, m in zip(items, mine, strict=True) if not m]
    )
    items[:] = [i for i, m in zip(items, mine, strict=True) if m]
