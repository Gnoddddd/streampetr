"""Frozen confirmatory helpers for CARE-3D P2-A2-R1-C0.

This module contains no fitting, calibration, model-selection, or threshold-
selection path.  Scene identity is handled separately from fixed-prediction
evaluation, and oracle identity is consumed only after arbiter decisions have
been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from analysis.care3d_p2a2_r1_arbiter import FrozenIdentityArbiter
from analysis.care3d_p2a2_r1_features import MODEL_FEATURE_COLUMNS, finite_model_matrix
from analysis.care3d_p2a2_lineage import FROZEN_P2A0_CONFIG
from analysis.care3d_p2a_association import PROTOCOLS


SOURCE_HEAD = "d499dbb34c44fc91ece1444580ff37ef40ce939e"
FROZEN_ARTIFACT_SHA256 = (
    "cdfc0b1e6f967498b2d394a3698bf062d93653448b177698ca1f3b9653c64e69"
)
FROZEN_MODEL_FEATURE_COLUMNS_SHA256 = (
    "561b40b715a67e25defde12532c908de696bc9c1a6fc4e32a08c118a31b76af9"
)
TAU_PREFERENCE = 0.52
TAU_DEFER = 0.88
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 314159
SCENE_CLUSTER_COLUMNS = ("scene_token",)
INSTANCE_CLUSTER_COLUMNS = ("scene_token", "instance_token")
CONFIRMED_STATUS = "CONFIRMED_P2A2_R1_FROZEN_ARBITER"
NOT_CONFIRMED_STATUS = "NOT_CONFIRMED_P2A2_R1_FROZEN_ARBITER"
REPAIRED_STATUS = "SOURCE_REPAIRED_TESTED_AWAITING_CONFIRMATORY_REREVIEW"
COHORT_CLEAN_STATUS = "HELDOUT_CONFIRMATORY_COHORT_CLEAN"
COHORT_STOP_STATUS = "STOP_CONFIRMATORY_COHORT_NOT_CLEAN"
EXPECTED_OFFICIAL_VAL_SCENES = 150
FROZEN_P2A0 = FROZEN_P2A0_CONFIG.as_dict()

KEY_COLUMNS = (
    "scene_token",
    "instance_token",
    "anchor_frame_idx",
    "target_frame_idx",
    "protocol",
)
FORBIDDEN_RUNTIME_COLUMNS = frozenset({
    "gt",
    "ground_truth",
    "oracle_query_index",
    "oracle_query",
    "clean_future",
    "future_clean",
    "target_clean_query_index",
    "protocol",
    "fault_type",
    "severity",
    "severity_id",
    "scene_threshold",
    "protocol_threshold",
})
FORBIDDEN_COHORT_SELECTION_COLUMNS = frozenset({
    "protocol",
    "fault_type",
    "severity",
    "severity_id",
    "label",
    "outcome",
    "exact",
    "wrong",
    "unmatched",
    "selected_query",
    "oracle_query_index",
})

DISCIPLINE_TRUE = ("frozen_model_used", "frozen_thresholds_used")
DISCIPLINE_FALSE = (
    "model_refit",
    "threshold_search",
    "feature_search",
    "model_search",
    "calibration_search",
    "protocol_used_as_feature",
    "protocol_specific_threshold",
    "probe_val_read",
    "probe_test_read",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_feature_columns_sha256() -> str:
    payload = json.dumps(
        list(MODEL_FEATURE_COLUMNS), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FrozenConfirmatoryArbiter:
    """Validated immutable artifact plus its pure NumPy replay object."""

    artifact_path: Path
    artifact_sha256: str
    metadata: Mapping[str, Any]
    runtime: FrozenIdentityArbiter


def _validate_head(value: object, name: str) -> Mapping[str, Sequence[float]]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"frozen {name} is not a mapping")
    width = len(MODEL_FEATURE_COLUMNS)
    for field, expected in (
        ("scaler_mean", width),
        ("scaler_scale", width),
        ("coef", width),
        ("intercept", 1),
    ):
        array = np.asarray(value.get(field), dtype=np.float64)
        if array.shape != (expected,) or not np.isfinite(array).all():
            raise RuntimeError(f"frozen {name}.{field} layout changed")
        if field == "scaler_scale" and np.any(array <= 0):
            raise RuntimeError(f"frozen {name} has non-positive scale")
    return value


def load_frozen_arbiter(
    path: Path | str,
    *,
    expected_sha256: str = FROZEN_ARTIFACT_SHA256,
) -> FrozenConfirmatoryArbiter:
    """Load the exact artifact after a checksum-first hard validation."""
    artifact_path = Path(path)
    observed = sha256_file(artifact_path)
    if observed != str(expected_sha256):
        raise RuntimeError(
            "frozen arbiter SHA256 mismatch: "
            f"expected={expected_sha256}, observed={observed}"
        )
    metadata = json.loads(artifact_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise RuntimeError("frozen arbiter schema changed")
    if metadata.get("model_feature_columns") != list(MODEL_FEATURE_COLUMNS):
        raise RuntimeError("frozen arbiter feature columns changed")
    if model_feature_columns_sha256() != FROZEN_MODEL_FEATURE_COLUMNS_SHA256:
        raise RuntimeError("source MODEL_FEATURE_COLUMNS checksum changed")
    if metadata.get("model_feature_columns_sha256") != FROZEN_MODEL_FEATURE_COLUMNS_SHA256:
        raise RuntimeError("artifact MODEL_FEATURE_COLUMNS checksum changed")
    if float(metadata.get("tau_preference", np.nan)) != TAU_PREFERENCE:
        raise RuntimeError("frozen tau_preference must remain 0.52")
    if float(metadata.get("tau_defer", np.nan)) != TAU_DEFER:
        raise RuntimeError("frozen tau_defer must remain 0.88")
    if metadata.get("probe_val_read") is not False:
        raise RuntimeError("frozen artifact indicates probe_val access")
    if metadata.get("probe_test_read") is not False:
        raise RuntimeError("frozen artifact indicates probe_test access")
    preference = _validate_head(metadata.get("preference_model"), "Preference Head")
    ambiguity = _validate_head(metadata.get("ambiguity_model"), "Ambiguity Head")
    runtime = FrozenIdentityArbiter(
        preference_model=preference,
        ambiguity_model=ambiguity,
        tau_preference=TAU_PREFERENCE,
        tau_defer=TAU_DEFER,
    )
    return FrozenConfirmatoryArbiter(artifact_path, observed, metadata, runtime)


def _scene_tokens(frame: pd.DataFrame, name: str) -> list[str]:
    if "scene_token" not in frame.columns:
        raise RuntimeError(f"{name} lacks scene_token")
    tokens = frame.scene_token.astype(str).tolist()
    if not tokens or any(not value for value in tokens):
        raise RuntimeError(f"{name} has empty scene identity")
    if len(tokens) != len(set(tokens)):
        raise RuntimeError(f"{name} contains duplicate scene_token")
    return tokens


def heldout_lineage_audit(
    official_val_manifest: pd.DataFrame,
    official_val_scene_tokens: Iterable[str],
    fitting_manifests: Mapping[str, pd.DataFrame],
    *,
    required_sources: Sequence[str],
    official_val_manifest_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select official-val scenes using identity/provenance only.

    ``fitting_manifests`` must enumerate every P2 fitting, tuning, selection,
    feature-selection, and failure-redesign cohort.  It may contain scene
    identity and split provenance only; raw outcomes are rejected.
    """
    forbidden = FORBIDDEN_COHORT_SELECTION_COLUMNS & set(official_val_manifest.columns)
    if forbidden:
        raise RuntimeError(f"official-val cohort manifest contains outcomes: {sorted(forbidden)}")
    if not {"scene_token", "split"} <= set(official_val_manifest.columns):
        raise RuntimeError(f"{COHORT_STOP_STATUS}: official-val manifest requires scene_token and split")
    official = _scene_tokens(official_val_manifest, "official-val manifest")
    splits = set(official_val_manifest.split.astype(str))
    if splits != {"official_val"}:
        raise RuntimeError(f"candidate cohort is not official_val only: {sorted(splits)}")
    official_metadata = [str(token) for token in official_val_scene_tokens]
    if (
        len(official) != EXPECTED_OFFICIAL_VAL_SCENES
        or len(official_metadata) != EXPECTED_OFFICIAL_VAL_SCENES
        or len(set(official_metadata)) != EXPECTED_OFFICIAL_VAL_SCENES
    ):
        raise RuntimeError(
            f"{COHORT_STOP_STATUS}: official nuScenes val must contain exactly "
            f"{EXPECTED_OFFICIAL_VAL_SCENES} unique scenes"
        )
    official_identity_exact = set(official) == set(official_metadata)
    if not official_identity_exact:
        raise RuntimeError(f"{COHORT_STOP_STATUS}: official-val scene-token set mismatch")
    missing = sorted(set(required_sources) - set(fitting_manifests))
    extra_required = sorted(set(fitting_manifests) - set(required_sources))
    if missing or extra_required:
        audit = {
            "status": COHORT_STOP_STATUS,
            "reason": "lineage source registry is incomplete or unpreregistered",
            "missing_required_sources": missing,
            "unregistered_sources": extra_required,
            "probe_val_read": False,
            "probe_test_read": False,
        }
        return pd.DataFrame(columns=("scene_token", "split")), audit

    official_set = set(official)
    used: set[str] = set()
    source_rows: list[dict[str, Any]] = []
    for source in required_sources:
        frame = fitting_manifests[source]
        forbidden = FORBIDDEN_COHORT_SELECTION_COLUMNS & set(frame.columns)
        if forbidden:
            raise RuntimeError(f"lineage manifest {source} contains outcomes: {sorted(forbidden)}")
        tokens = _scene_tokens(frame, f"lineage manifest {source}")
        token_set = set(tokens)
        overlap = sorted(official_set & token_set)
        used.update(token_set)
        source_rows.append({
            "source": source,
            "scenes": len(token_set),
            "official_val_intersection_count": len(overlap),
            "official_val_intersection_scene_tokens": overlap,
        })

    heldout = [token for token in official if token not in used]
    cohort = pd.DataFrame({"scene_token": heldout, "split": "official_val"})
    clean = bool(heldout) and not (set(heldout) & used)
    audit = {
        "status": COHORT_CLEAN_STATUS if clean else COHORT_STOP_STATUS,
        "reason": "held-out identity proven" if clean else "no clean official-val scenes",
        "selection_fields": ["scene_token", "split"],
        "official_val_expected_scenes": EXPECTED_OFFICIAL_VAL_SCENES,
        "official_val_candidate_scenes": len(official),
        "official_val_identity_exact_match": official_identity_exact,
        "official_val_manifest_sha256": str(official_val_manifest_sha256),
        "excluded_used_scenes": len(official_set & used),
        "confirmatory_scenes": len(heldout),
        "required_sources": list(required_sources),
        "source_intersections": source_rows,
        "heldout_intersection_count": len(set(heldout) & used),
        "cohort_outcome_blind": True,
        "probe_val_identity_provenance_only": True,
        "probe_val_read": False,
        "probe_test_read": False,
    }
    return cohort, audit


def require_clean_cohort(audit: Mapping[str, Any]) -> None:
    if audit.get("status") != COHORT_CLEAN_STATUS:
        raise RuntimeError(COHORT_STOP_STATUS)
    if audit.get("heldout_intersection_count") != 0:
        raise RuntimeError(COHORT_STOP_STATUS)
    if (
        audit.get("official_val_expected_scenes") != EXPECTED_OFFICIAL_VAL_SCENES
        or audit.get("official_val_candidate_scenes") != EXPECTED_OFFICIAL_VAL_SCENES
        or audit.get("official_val_identity_exact_match") is not True
    ):
        raise RuntimeError(COHORT_STOP_STATUS)
    manifest_sha = audit.get("official_val_manifest_sha256")
    if not (
        isinstance(manifest_sha, str)
        and len(manifest_sha) == 64
        and all(character in "0123456789abcdef" for character in manifest_sha)
    ):
        raise RuntimeError(COHORT_STOP_STATUS)
    if audit.get("probe_val_read") is not False or audit.get("probe_test_read") is not False:
        raise RuntimeError(COHORT_STOP_STATUS)


def validate_runtime_feature_frame(frame: pd.DataFrame) -> None:
    forbidden = FORBIDDEN_RUNTIME_COLUMNS & set(frame.columns)
    if forbidden:
        raise RuntimeError(f"forbidden arbiter runtime input(s): {sorted(forbidden)}")
    if tuple(frame.columns) != tuple(MODEL_FEATURE_COLUMNS):
        raise RuntimeError("arbiter runtime feature columns/order changed")


def reconstruct_full_population(
    eligible_rows: pd.DataFrame,
    disagreement_evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Join disagreement evidence onto every frozen eligible-object row.

    Agreement rows remain in the output with missing model evidence because
    they bypass both heads. The disagreement key set must match exactly.
    """
    validate_full_population_keys(eligible_rows)
    validate_full_population_keys(disagreement_evidence)
    identity_columns = (
        "p2a0_selected_query", "lineage_child_query", "oracle_query_index"
    )
    missing_base = set(identity_columns) - set(eligible_rows.columns)
    missing_evidence = set(MODEL_FEATURE_COLUMNS) - set(disagreement_evidence.columns)
    if missing_base or missing_evidence:
        raise RuntimeError(
            "full-population reconstruction columns are incomplete: "
            f"base={sorted(missing_base)}, evidence={sorted(missing_evidence)}"
        )
    disagreement = (
        eligible_rows.p2a0_selected_query.to_numpy(np.int64)
        != eligible_rows.lineage_child_query.to_numpy(np.int64)
    )
    expected = set(map(
        tuple, eligible_rows.loc[disagreement, list(KEY_COLUMNS)].to_numpy()
    ))
    observed = set(map(
        tuple, disagreement_evidence.loc[:, list(KEY_COLUMNS)].to_numpy()
    ))
    if expected != observed:
        raise RuntimeError("disagreement evidence does not exactly cover eligible disagreements")
    overlap = [column for column in disagreement_evidence.columns if column in eligible_rows.columns]
    overlap = [column for column in overlap if column not in KEY_COLUMNS]
    if overlap:
        raise RuntimeError(f"disagreement evidence overlaps base columns: {sorted(overlap)}")
    output = eligible_rows.merge(
        disagreement_evidence,
        how="left",
        on=list(KEY_COLUMNS),
        validate="one_to_one",
        sort=False,
    )
    matrix = output.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(np.float64)
    if disagreement.any() and not np.isfinite(
        finite_model_matrix(output.loc[disagreement, MODEL_FEATURE_COLUMNS])
    ).all():
        raise RuntimeError("disagreement model evidence is not finite")
    if np.isfinite(matrix[~disagreement]).any():
        raise RuntimeError("agreement rows unexpectedly received model evidence")
    if len(output) != len(eligible_rows):
        raise RuntimeError("full-population reconstruction dropped eligible rows")
    return output


def run_frozen_runtime(
    model_features: pd.DataFrame,
    p2a0_candidates: Sequence[int],
    lineage_candidates: Sequence[int],
    frozen: FrozenConfirmatoryArbiter,
) -> pd.DataFrame:
    """Apply agreement bypass and the frozen heads to disagreements only."""
    validate_runtime_feature_frame(model_features)
    a = np.asarray(p2a0_candidates, dtype=np.int64)
    lineage = np.asarray(lineage_candidates, dtype=np.int64)
    if a.ndim != 1 or lineage.shape != a.shape or len(model_features) != len(a):
        raise ValueError("confirmatory runtime inputs are not aligned")
    disagreement = a != lineage
    output = pd.DataFrame({
        "p_lineage": np.full(len(a), np.nan, dtype=np.float64),
        "p_both_wrong": np.full(len(a), np.nan, dtype=np.float64),
        "arbiter_selected_query": a.copy(),
        "arbiter_decision": np.full(len(a), "AGREEMENT", dtype=object),
    })
    if disagreement.any():
        matrix = finite_model_matrix(model_features.loc[disagreement])
        decided = frozen.runtime.predict(matrix, a[disagreement], lineage[disagreement])
        for column in output.columns:
            output.loc[disagreement, column] = decided[column].to_numpy()
    if not output.loc[~disagreement, ["p_lineage", "p_both_wrong"]].isna().all().all():
        raise RuntimeError("agreement rows called a frozen head")
    if not np.array_equal(
        output.loc[~disagreement, "arbiter_selected_query"].to_numpy(np.int64),
        a[~disagreement],
    ):
        raise RuntimeError("agreement bypass modified an agreement row")
    return output


def attach_offline_outcomes(
    frozen_decisions: pd.DataFrame,
    p2a0_candidates: Sequence[int],
    oracle_query_indexes: Sequence[int],
) -> pd.DataFrame:
    """Attach exact/wrong/unmatched labels only after runtime has completed."""
    output = frozen_decisions.copy()
    selected = output.arbiter_selected_query.to_numpy(np.int64)
    baseline = np.asarray(p2a0_candidates, dtype=np.int64)
    oracle = np.asarray(oracle_query_indexes, dtype=np.int64)
    if selected.shape != baseline.shape or oracle.shape != baseline.shape:
        raise ValueError("offline outcome inputs are not aligned")
    if np.any(oracle < 0):
        raise RuntimeError("offline oracle identity must be a real query")
    output["p2a0_exact"] = (baseline == oracle).astype(np.int8)
    output["p2a0_wrong"] = ((baseline >= 0) & (baseline != oracle)).astype(np.int8)
    output["p2a0_unmatched"] = (baseline == -1).astype(np.int8)
    output["arbiter_exact"] = (selected == oracle).astype(np.int8)
    output["arbiter_wrong"] = ((selected >= 0) & (selected != oracle)).astype(np.int8)
    output["arbiter_unmatched"] = (selected == -1).astype(np.int8)
    for prefix in ("p2a0", "arbiter"):
        total = output[[f"{prefix}_exact", f"{prefix}_wrong", f"{prefix}_unmatched"]].sum(axis=1)
        if not (total == 1).all():
            raise RuntimeError(f"{prefix} population accounting is not exhaustive")
    return output


def confirmatory_point_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if len(frame) == 0:
        raise ValueError("confirmatory protocol has no eligible rows")
    agreement = frame.p2a0_selected_query.to_numpy(np.int64) == frame.lineage_child_query.to_numpy(np.int64)
    p_exact = frame.p2a0_exact.to_numpy(bool)
    p_wrong = frame.p2a0_wrong.to_numpy(bool)
    exact = frame.arbiter_exact.to_numpy(bool)
    wrong = frame.arbiter_wrong.to_numpy(bool)
    unmatched = frame.arbiter_unmatched.to_numpy(bool)
    decision = frame.arbiter_decision.astype(str).to_numpy()
    selected = frame.arbiter_selected_query.to_numpy(np.int64)
    baseline = frame.p2a0_selected_query.to_numpy(np.int64)
    result: dict[str, float | int] = {
        "rows": int(len(frame)),
        "agreement_rows": int(agreement.sum()),
        "disagreement_rows": int((~agreement).sum()),
        "p2a0_exact_rate": float(p_exact.mean()),
        "p2a0_wrong_rate": float(p_wrong.mean()),
        "p2a0_unmatched_rate": float(frame.p2a0_unmatched.astype(float).mean()),
        "arbiter_exact_rate": float(exact.mean()),
        "arbiter_wrong_rate": float(wrong.mean()),
        "arbiter_unmatched_rate": float(unmatched.mean()),
        "delta_exact": float(exact.mean() - p_exact.mean()),
        "delta_wrong": float(wrong.mean() - p_wrong.mean()),
        "wrong_reduction": float(p_wrong.mean() - wrong.mean()),
        "wrong_repaired_count": int((p_wrong & exact).sum()),
        "correct_broken_to_wrong_count": int((p_exact & wrong).sum()),
        "correct_broken_to_unmatched_count": int((p_exact & unmatched).sum()),
        "wrong_to_unmatched_count": int((p_wrong & unmatched).sum()),
        "wrong_remaining_wrong_count": int((p_wrong & wrong).sum()),
        "lineage_switch_count": int((decision == "LINEAGE").sum()),
        "lineage_switch_rate": float((decision == "LINEAGE").mean()),
        "defer_count": int((decision == "DEFER").sum()),
        "defer_rate": float((decision == "DEFER").mean()),
        "agreement_modified_count": int((selected[agreement] != baseline[agreement]).sum()),
    }
    if result["agreement_rows"] + result["disagreement_rows"] != result["rows"]:
        raise RuntimeError("agreement/disagreement accounting is not exhaustive")
    if result["agreement_modified_count"] != 0:
        raise RuntimeError("agreement bypass modified an agreement row")
    if int(p_wrong.sum()) != sum(int(result[name]) for name in (
        "wrong_repaired_count", "wrong_to_unmatched_count", "wrong_remaining_wrong_count"
    )):
        raise RuntimeError("wrong-row repair accounting is inconsistent")
    correct_remaining = int((p_exact & exact).sum())
    if int(p_exact.sum()) != correct_remaining + sum(int(result[name]) for name in (
        "correct_broken_to_wrong_count", "correct_broken_to_unmatched_count"
    )):
        raise RuntimeError("correct-row break accounting is inconsistent")
    return result


def paired_confirmatory_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_columns: Sequence[str],
    cluster_population: Sequence[Any] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fixed-prediction paired cluster bootstrap with frozen constants."""
    columns = tuple(cluster_columns)
    if columns not in (SCENE_CLUSTER_COLUMNS, INSTANCE_CLUSTER_COLUMNS):
        raise ValueError("cluster key must be scene or (scene, instance)")
    row_keys = list(frame.loc[:, list(columns)].itertuples(index=False, name=None))
    if cluster_population is None:
        population = list(dict.fromkeys(row_keys))
    else:
        population = [value if isinstance(value, tuple) else (value,) for value in cluster_population]
    if not population or len(population) != len(set(population)):
        raise RuntimeError("bootstrap cluster population is empty or duplicated")
    positions = {key: index for index, key in enumerate(population)}
    if any(key not in positions for key in row_keys):
        raise RuntimeError("bootstrap row lies outside cluster population")
    names = ("delta_exact", "delta_wrong", "arbiter_exact", "arbiter_wrong", "arbiter_unmatched")
    values = np.column_stack([
        frame.arbiter_exact.to_numpy(float) - frame.p2a0_exact.to_numpy(float),
        frame.arbiter_wrong.to_numpy(float) - frame.p2a0_wrong.to_numpy(float),
        frame.arbiter_exact.to_numpy(float),
        frame.arbiter_wrong.to_numpy(float),
        frame.arbiter_unmatched.to_numpy(float),
    ])
    sums = np.zeros((len(population), len(names)), dtype=np.float64)
    counts = np.zeros(len(population), dtype=np.int64)
    for row, key in enumerate(row_keys):
        position = positions[key]
        sums[position] += values[row]
        counts[position] += 1
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty((BOOTSTRAP_REPLICATES, len(names)), dtype=np.float64)
    valid = 0
    redraws = 0
    while valid < BOOTSTRAP_REPLICATES:
        count = min(64, BOOTSTRAP_REPLICATES - valid)
        draws = rng.integers(0, len(population), size=(count, len(population)))
        denominators = counts[draws].sum(axis=1)
        numerators = sums[draws].sum(axis=1)
        good = denominators > 0
        accepted = int(good.sum())
        if accepted:
            samples[valid:valid + accepted] = numerators[good] / denominators[good, None]
            valid += accepted
        redraws += int((~good).sum())
    rows = []
    points = values.mean(axis=0)
    for index, name in enumerate(names):
        rows.append({
            "metric": name,
            "point": float(points[index]),
            "ci_low": float(np.percentile(samples[:, index], 2.5)),
            "ci_high": float(np.percentile(samples[:, index], 97.5)),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        })
    return pd.DataFrame(rows), redraws


def evaluate_confirmatory_gate(
    protocol_metrics: pd.DataFrame,
    scene_bootstrap: pd.DataFrame,
    instance_bootstrap: pd.DataFrame,
    discipline: Mapping[str, bool],
) -> tuple[pd.DataFrame, str]:
    discipline_pass = all(discipline.get(name) is True for name in DISCIPLINE_TRUE)
    discipline_pass &= all(discipline.get(name) is False for name in DISCIPLINE_FALSE)
    rows = []
    for protocol in PROTOCOLS:
        metrics = protocol_metrics.loc[protocol_metrics.protocol.astype(str) == protocol]
        if len(metrics) != 1:
            raise RuntimeError(f"confirmatory Gate requires one row for {protocol}")
        metric = metrics.iloc[0]
        scene = scene_bootstrap.loc[
            scene_bootstrap.protocol.astype(str) == protocol
        ].set_index("metric")
        instance = instance_bootstrap.loc[
            instance_bootstrap.protocol.astype(str) == protocol
        ].set_index("metric")
        gates = {
            "C1_arbiter_wrong_le_0_10": bool(float(metric.arbiter_wrong_rate) <= 0.10),
            "C2_arbiter_unmatched_le_0_01": bool(float(metric.arbiter_unmatched_rate) <= 0.01),
            "C3_delta_exact_positive": bool(float(metric.delta_exact) > 0.0),
            "C4_scene_delta_exact_ci_low_positive": bool(float(scene.loc["delta_exact", "ci_low"]) > 0.0),
            "C5_instance_delta_exact_ci_low_positive": bool(float(instance.loc["delta_exact", "ci_low"]) > 0.0),
            "C6_delta_wrong_negative": bool(float(metric.delta_wrong) < 0.0),
            "C7_scene_delta_wrong_ci_high_negative": bool(float(scene.loc["delta_wrong", "ci_high"]) < 0.0),
            "C8_instance_delta_wrong_ci_high_negative": bool(float(instance.loc["delta_wrong", "ci_high"]) < 0.0),
            "C9_agreement_unmodified": bool(int(metric.agreement_modified_count) == 0),
            "C10_frozen_discipline": bool(discipline_pass),
        }
        rows.append({"protocol": protocol, **gates, "protocol_pass": bool(all(gates.values()))})
    gate = pd.DataFrame(rows)
    passed = bool(len(gate) == len(PROTOCOLS) and gate.protocol_pass.astype(bool).all())
    return gate, CONFIRMED_STATUS if passed else NOT_CONFIRMED_STATUS


def default_discipline() -> dict[str, bool]:
    return {
        "frozen_model_used": True,
        "frozen_thresholds_used": True,
        "model_refit": False,
        "threshold_search": False,
        "feature_search": False,
        "model_search": False,
        "calibration_search": False,
        "protocol_used_as_feature": False,
        "protocol_specific_threshold": False,
        "probe_val_read": False,
        "probe_test_read": False,
    }


def validate_full_population_keys(frame: pd.DataFrame) -> None:
    missing = set(KEY_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"confirmatory rows lack keys: {sorted(missing)}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("confirmatory full population contains duplicate keys")
    if set(frame.protocol.astype(str)) - set(PROTOCOLS):
        raise RuntimeError("confirmatory protocol set changed")


def source_smoke(frozen: FrozenConfirmatoryArbiter) -> dict[str, Any]:
    features = pd.DataFrame(
        np.zeros((2, len(MODEL_FEATURE_COLUMNS))), columns=MODEL_FEATURE_COLUMNS
    )
    decisions = run_frozen_runtime(features, [7, 4], [7, 8], frozen)
    if decisions.loc[0, "arbiter_decision"] != "AGREEMENT":
        raise RuntimeError("source smoke agreement bypass failed")
    if decisions.loc[0, ["p_lineage", "p_both_wrong"]].notna().any():
        raise RuntimeError("source smoke agreement called a head")
    return {
        "status": REPAIRED_STATUS,
        "frozen_artifact_sha256": frozen.artifact_sha256,
        "tau_preference": TAU_PREFERENCE,
        "tau_defer": TAU_DEFER,
        "agreement_bypass_exact": True,
        "protocols": list(PROTOCOLS),
        "probe_val_read": False,
        "probe_test_read": False,
    }
