from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_care3d_p1_empty_scene_safe import _read_counted_csv


def test_headerless_empty_object_csv_is_valid_only_for_zero_marker_rows(tmp_path: Path):
    path = tmp_path / "scene.objects.csv"
    path.write_text("", encoding="utf-8")
    frame = _read_counted_csv(path, 0, kind="object")
    assert frame.empty


def test_headerless_empty_object_csv_is_rejected_for_nonzero_marker_rows(tmp_path: Path):
    path = tmp_path / "scene.objects.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="marker declares 1 rows"):
        _read_counted_csv(path, 1, kind="object")


def test_nonempty_csv_must_match_marker_row_count(tmp_path: Path):
    path = tmp_path / "scene.frames.csv"
    pd.DataFrame({"seed": [42, 2027], "base_fp": [1, 2]}).to_csv(path, index=False)
    frame = _read_counted_csv(path, 2, kind="frame")
    assert len(frame) == 2
    with pytest.raises(RuntimeError, match="row-count mismatch"):
        _read_counted_csv(path, 1, kind="frame")
