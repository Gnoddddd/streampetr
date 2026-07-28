from pathlib import Path


def test_required_project_structure_and_config_contract():
    root = Path(__file__).resolve().parents[1]
    required = [
        "configs/evidence_conserving/mini_debug.py",
        "datasets/corruption.py",
        "datasets/observability.py",
        "datasets/nuscenes_wrapper.py",
        "models/observability_head.py",
        "models/ternary_objectness.py",
        "models/evidence_ledger.py",
        "models/temporal_update.py",
        "protocols/camera_crash.py",
        "protocols/frame_lost.py",
        "scripts/train_mini.sh",
        "tools/train.py",
    ]
    assert all((root / path).is_file() for path in required)
    config = (root / "configs/evidence_conserving/mini_debug.py").read_text(encoding="utf-8")
    assert "EvidenceConservingStreamPETRHead" in config
    assert "ApplyPartialObservation" in config
    assert "PETRMultiheadAttention" in config
    assert "PETRMultiheadFlashAttention" not in config
