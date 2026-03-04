#!/usr/bin/env python

from lerobot.scripts.lerobot_value_infer import ContiguousDistributedEvalSampler


def test_contiguous_distributed_eval_sampler_has_contiguous_shards_with_tail_padding():
    dataset_size = 10
    world_size = 3

    shards: list[list[int]] = []
    for rank in range(world_size):
        sampler = ContiguousDistributedEvalSampler(
            dataset_size=dataset_size, num_replicas=world_size, rank=rank
        )
        shard = list(iter(sampler))
        shards.append(shard)
        assert len(shard) == len(sampler) == 4

    assert shards[0] == [0, 1, 2, 3]
    assert shards[1] == [4, 5, 6, 7]
    assert shards[2] == [8, 9, 9, 9]

    covered = set()
    for shard in shards:
        covered.update(shard)
    assert covered == set(range(dataset_size))


def test_contiguous_distributed_eval_sampler_handles_more_ranks_than_samples():
    dataset_size = 2
    world_size = 4

    shards: list[list[int]] = []
    for rank in range(world_size):
        sampler = ContiguousDistributedEvalSampler(
            dataset_size=dataset_size, num_replicas=world_size, rank=rank
        )
        shard = list(iter(sampler))
        shards.append(shard)
        assert len(shard) == len(sampler) == 1

    assert shards[0] == [0]
    assert shards[1] == [1]
    assert shards[2] == [1]
    assert shards[3] == [1]

    covered = set()
    for shard in shards:
        covered.update(shard)
    assert covered == set(range(dataset_size))
