import numpy as np

from datasets.corruption import ApplyPartialObservation, CAMERA_NAMES, apply_camera_crash


def images():
    return [np.full((8, 8, 3), 100 + index, dtype=np.uint8) for index in range(6)]


def test_camera_crash_does_not_modify_input():
    original = images()
    output = apply_camera_crash(original, [0, 3])
    assert output[0].sum() == 0
    assert output[3].sum() == 0
    assert original[0].sum() > 0


def test_pipeline_is_deterministic_by_sample():
    transform = ApplyPartialObservation(
        training=True,
        seed=17,
        camera_crash_prob=1.0,
        max_failed_cameras=2,
        frame_lost_prob=0.0,
        dark_prob=0.0,
        fog_prob=0.0,
        motion_blur_prob=0.0,
    )
    base = dict(img=images(), sample_idx="token", scene_token="scene", frame_idx=3)
    first = transform({key: value if key != "img" else images() for key, value in base.items()})
    second = transform({key: value if key != "img" else images() for key, value in base.items()})
    assert np.array_equal(first["camera_online_mask"], second["camera_online_mask"])
    assert int((first["camera_online_mask"] == 0).sum()) in (1, 2)


def test_motion_blur_fallback_preserves_shape(monkeypatch):
    import datasets.corruption as corruption

    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    monkeypatch.setattr(corruption, "cv2", None)
    output = corruption.apply_motion_blur(image, 0.7)
    assert output.shape == image.shape
    assert output.dtype == image.dtype


def test_environment_can_disable_random_corruption(monkeypatch):
    transform = ApplyPartialObservation(
        training=True,
        seed=1,
        camera_crash_prob=1.0,
        max_failed_cameras=6,
        frame_lost_prob=1.0,
        dark_prob=1.0,
        fog_prob=1.0,
        motion_blur_prob=1.0,
    )
    monkeypatch.setenv("EVIDENCE3D_DISABLE_RANDOM_CORRUPTION", "1")
    images = [np.full((4, 4, 3), 127, dtype=np.uint8) for _ in range(6)]
    result = transform(
        {"img": images, "sample_idx": "token", "scene_token": "scene", "frame_idx": 2}
    )
    assert np.all(result["camera_online_mask"] == 1.0)
    assert np.all(result["camera_quality"] == 1.0)
    assert np.all(result["camera_fresh_mask"] == 1.0)
