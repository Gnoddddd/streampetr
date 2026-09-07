from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from analysis.care3d_p2a_association import (
    assignment_rows,
    hungarian_with_unmatched,
    oracle_diagnostics,
)
from analysis.care3d_p2a_execution import (
    assert_shards_partition,
    build_shared_protocol_dataset,
    cheap_train_oracle_diagnostics,
    shard_scene_frame,
)


def test_scene_shards_are_disjoint_exhaustive_and_ordered():
    frame = pd.DataFrame({"scene": [f"s{i}" for i in range(17)]})
    assert_shards_partition(frame, num_shards=4)
    shards = [
        shard_scene_frame(frame, num_shards=4, shard_index=i)
        for i in range(4)
    ]
    values = [item for shard in shards for item in shard.scene.tolist()]
    assert sorted(values, key=lambda x: int(x[1:])) == frame.scene.tolist()
    assert len(values) == len(set(values)) == len(frame)
    assert shards[0].scene.tolist() == ["s0", "s4", "s8", "s12", "s16"]


def test_scene_shard_max_scenes_is_local_to_each_worker():
    frame = pd.DataFrame({"scene": [f"s{i}" for i in range(10)]})
    shard = shard_scene_frame(
        frame, num_shards=3, shard_index=1, max_scenes=2
    )
    assert shard.scene.tolist() == ["s1", "s4"]


def test_shared_protocol_dataset_reuses_infos_and_clones_registered_pipeline():
    class LoadSomething:
        pass

    class ApplyPartialObservation:
        def __init__(self, schedule_file=None, training=False, seed=7):
            self.schedule_file = schedule_file
            self.training = bool(training)
            self.seed = int(seed)
            self.schedule = object() if schedule_file else None

    class DummyCompose:
        def __init__(self, transforms):
            self.transforms = list(transforms)

    clean_transform = ApplyPartialObservation(
        schedule_file=None,
        training=False,
        seed=7,
    )
    base = SimpleNamespace(
        data_infos=[{"token": "a"}, {"token": "b"}],
        pipeline=DummyCompose([LoadSomething(), clean_transform]),
        test_mode=True,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            test=SimpleNamespace(
                pipeline=[
                    {"type": "LoadSomething"},
                    {
                        "type": "ApplyPartialObservation",
                        "schedule_file": None,
                        "training": False,
                        "seed": 7,
                    },
                ],
                test_mode=True,
            )
        )
    )

    shared = build_shared_protocol_dataset(
        base,
        config,
        "/tmp/frozen_protocol.json",
    )

    assert shared is not base
    assert shared.data_infos is base.data_infos
    assert shared.pipeline is not base.pipeline
    assert shared.pipeline.transforms is not base.pipeline.transforms
    assert shared.pipeline.transforms[0] is not base.pipeline.transforms[0]
    assert shared.pipeline.transforms[1] is not clean_transform
    assert (
        shared.pipeline.transforms[1].__class__
        is clean_transform.__class__
    )
    assert shared.pipeline.transforms[1].schedule_file == "/tmp/frozen_protocol.json"
    assert shared.pipeline.transforms[1].schedule is not None
    assert base.pipeline.transforms[1].schedule_file is None
    assert base.pipeline.transforms[1].schedule is None
    assert config.data.test.pipeline[1]["schedule_file"] is None
    assert shared.test_mode is True


def test_fast_train_diagnostics_preserve_selection_relevant_outcomes():
    torch.manual_seed(17)
    cost = torch.rand(7, 900)
    cost[0, :20] = float("inf")
    oracle = np.asarray([21, 50, 100, 200, 300, 400, 500], dtype=np.int64)
    assignment = hungarian_with_unmatched(cost, max_cost=0.55)

    full = assignment_rows(
        assignment,
        oracle,
        oracle_diagnostics(cost, oracle),
    )
    fast = assignment_rows(
        assignment,
        oracle,
        cheap_train_oracle_diagnostics(cost, oracle),
    )

    for key in ("selected_query", "exact_match", "wrong_match", "unmatched"):
        assert np.array_equal(full[key], fast[key])
    assert np.allclose(
        full["selected_cost"], fast["selected_cost"],
        rtol=0.0, atol=0.0, equal_nan=True,
    )
    assert np.allclose(
        full["oracle_cost"], fast["oracle_cost"],
        rtol=0.0, atol=0.0, equal_nan=True,
    )
    assert np.array_equal(
        full["oracle_geometry_eligible"],
        fast["oracle_geometry_eligible"],
    )


def test_fast_train_diagnostics_do_not_claim_rank_or_margin():
    cost = torch.full((2, 900), 0.3)
    oracle = np.asarray([1, 2], dtype=np.int64)
    fast = cheap_train_oracle_diagnostics(cost, oracle)
    assert fast["oracle_rank"].tolist() == [901, 901]
    assert np.isnan(fast["correct_vs_best_wrong_margin"]).all()
