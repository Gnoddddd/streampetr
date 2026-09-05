import ast
from pathlib import Path


def test_trace_has_no_parameter_or_buffer_registration():
    source = Path("analysis/camera_attention_trace.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "register_parameter" not in calls
    assert "register_buffer" not in calls


def test_trace_returns_original_head_result_object():
    source = Path("analysis/camera_attention_trace.py").read_text(encoding="utf-8")
    assert "return result" in source
    assert "result[\"all_cls_scores\"]" in source
