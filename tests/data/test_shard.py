"""Channel sharding (--shard i/n): a disjoint, complete, balanced round-robin partition."""
from __future__ import annotations

import os

import pytest

from pfns4neurostim.data.channels import iter_channels, parse_shard, shard_size


class TestParse:
    def test_valid(self) -> None:
        assert parse_shard("0/4") == (0, 4)
        assert parse_shard("3/4") == (3, 4)

    @pytest.mark.parametrize("bad", ["4/4", "-1/4", "a/b", "1", "1/0", "1/2/3", ""])
    def test_invalid(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_shard(bad)


class TestShardSize:
    @pytest.mark.parametrize("total", [0, 1, 7, 18, 100])
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
    def test_sizes_sum_to_total_and_are_balanced(self, total: int, n: int) -> None:
        sizes = [shard_size(total, (i, n)) for i in range(n)]
        assert sum(sizes) == total
        assert max(sizes) - min(sizes) <= 1

    def test_none_is_everything(self) -> None:
        assert shard_size(18, None) == 18


@pytest.mark.skipif(not os.path.isdir("data/monkeys"), reason="raw NHP data not available")
def test_real_channels_are_partitioned_exactly() -> None:
    kwargs = dict(dataset="nhp", subjects=[0, 1, 3], emgs=[0, 1, 2, 3])
    everything = [c.label for c in iter_channels(**kwargs)]
    shards = [[c.label for c in iter_channels(**kwargs, shard=(i, 3))] for i in range(3)]
    flat = [label for shard in shards for label in shard]
    assert sorted(flat) == sorted(everything) and len(flat) == len(set(flat))
    assert [len(s) for s in shards] == [shard_size(len(everything), (i, 3)) for i in range(3)]
