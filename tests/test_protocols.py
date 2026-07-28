from protocols.partial_observation import ProtocolEvent, ProtocolSchedule


def test_wildcard_and_scene_events_merge():
    schedule = ProtocolSchedule(
        {
            "*": [ProtocolEvent(1, 3, failed_cameras=["CAM_BACK"])],
            "scene": [ProtocolEvent(2, 2, lost_cameras=["CAM_FRONT"])],
        }
    )
    state = schedule.state_for("scene", 2, ["CAM_FRONT", "CAM_BACK"])
    assert state["failed_cameras"] == ["CAM_BACK"]
    assert state["lost_cameras"] == ["CAM_FRONT"]


def test_protocol_rejects_invalid_range_and_severity(tmp_path):
    import json
    import pytest

    with pytest.raises(ValueError):
        ProtocolEvent(start_frame=5, end_frame=4)
    with pytest.raises(ValueError):
        ProtocolEvent(start_frame=0, end_frame=1, fog={"CAM_FRONT": 1.2})

    path = tmp_path / "bad_version.json"
    path.write_text(json.dumps({"version": 2, "scenes": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        ProtocolSchedule.from_json(str(path))
