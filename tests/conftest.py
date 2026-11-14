"""Shared fixtures. Every test runs on CPU, locally and on the pod: uv run python -m pytest -q"""

import pytest

from abstention.config import DEFAULT_CONFIG, load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config(DEFAULT_CONFIG)
