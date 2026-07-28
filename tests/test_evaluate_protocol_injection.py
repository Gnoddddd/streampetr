"""Protocol injection must follow the transform, not a fixed list index."""

from __future__ import annotations

import pytest
from mmcv import Config

from tools.evaluate import resolve_protocol_cfg_path


def _config(pipeline):
    return Config(dict(data=dict(test=dict(pipeline=pipeline))))


def test_protocol_injection_standard_pipeline():
    config = _config(
        [
            dict(type="LoadMultiViewImageFromFiles"),
            dict(type="ApplyPartialObservation"),
        ]
    )
    assert (
        resolve_protocol_cfg_path(config)
        == "data.test.pipeline.1.schedule_file"
    )


def test_protocol_injection_reordered_pipeline():
    config = _config(
        [
            dict(type="ApplyPartialObservation"),
            dict(type="LoadMultiViewImageFromFiles"),
        ]
    )
    assert (
        resolve_protocol_cfg_path(config)
        == "data.test.pipeline.0.schedule_file"
    )


def test_protocol_injection_nested_pipeline():
    config = _config(
        [
            dict(
                type="MultiScaleFlipAug3D",
                transforms=[
                    dict(type="ApplyPartialObservation"),
                ],
            )
        ]
    )
    assert (
        resolve_protocol_cfg_path(config)
        == "data.test.pipeline.0.transforms.0.schedule_file"
    )


def test_protocol_injection_missing_transform_is_error():
    with pytest.raises(ValueError, match="found 0"):
        resolve_protocol_cfg_path(_config([dict(type="Collect3D")]))


def test_protocol_injection_duplicate_transform_is_error():
    with pytest.raises(ValueError, match="found 2"):
        resolve_protocol_cfg_path(
            _config(
                [
                    dict(type="ApplyPartialObservation"),
                    dict(
                        type="Wrapper",
                        transforms=[dict(type="ApplyPartialObservation")],
                    ),
                ]
            )
        )
