"""Options shared by the test suite."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--render",
        action="store_true",
        help="add the render checks to the run; -m render runs them alone. They need "
        "Cycles",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--render"):
        return
    skip = pytest.mark.skip(reason="needs --render")
    for item in items:
        if item.get_closest_marker("render"):
            item.add_marker(skip)
