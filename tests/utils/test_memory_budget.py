from __future__ import annotations

import pytest

from squidpy._utils import memory_budget


@pytest.mark.parametrize(
    ("working_memory", "expected"),
    [
        (1000, 1000 * 2**20),  # plain numbers are MiB, as scikit-learn's working_memory
        (0.5, 2**19),
        ("2000", 2000 * 2**20),
        ("4096B", 4096),
        ("16GB", 16 * 10**9),
        ("512MiB", 512 * 2**20),
        ("1.5TB", 15 * 10**11),
        (" 2 gib ", 2 * 2**30),
    ],
)
def test_memory_budget_parses(working_memory, expected):
    assert memory_budget(working_memory) == expected


@pytest.mark.parametrize("working_memory", ["16 gigs", "GB", "-1GB", "0B", 0, -5])
def test_memory_budget_rejects(working_memory):
    with pytest.raises(ValueError, match="working_memory"):
        memory_budget(working_memory)


def test_memory_budget_default():
    assert memory_budget(None) == 2**30
