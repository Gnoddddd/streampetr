"""Pure utilities for counterfactual residual construction and prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)
CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
CLASS_DIM = 10
CENTER = slice(10, 13)
LOG_SIZE = slice(13, 16)
YAW = 16
VELOCITY = slice(17, 19)
GEOMETRY = slice(10, 19)


def wrap_yaw(value):
    return np.arctan2(np.sin(value), np.cos(value))


def independent_match(
    gt_centers: np.ndarray,
    gt_labels: np.ndarray,
    pred_centers: np.ndarray,
    pred_labels: np.ndarray,
    max_distance: float = 4.0,
) -> Dict[int, int]:
    if not len(gt_centers) or not len(pred_centers):
        return {}
    distance = np.linalg.norm(
        gt_centers[:, None, :3] - pred_centers[None, :, :3],
        axis=-1,
    )
    mismatch = gt_labels[:, None] != pred_labels[None, :]
    rows, columns = linear_sum_assignment(distance + 0.5 * mismatch)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns)
        if distance[row, column] <= max_distance
    }


def residual_target(full: dict, available: dict) -> np.ndarray:
    return np.concatenate([
        np.asarray(full["logits"]) - np.asarray(available["logits"]),
        np.asarray(full["center"]) - np.asarray(available["center"]),
        np.log(np.maximum(full["size"], 1e-6))
        - np.log(np.maximum(available["size"], 1e-6)),
        np.asarray([wrap_yaw(full["yaw"] - available["yaw"])]),
        np.asarray(full["velocity"]) - np.asarray(available["velocity"]),
    ]).astype(np.float32)


def fault_key(camera_online: np.ndarray, elapsed: int) -> str:
    missing = np.flatnonzero(np.asarray(camera_online) < 0.5)
    if len(missing) == 1:
        family = "single"
    elif len(missing) == 2 and (
        (int(missing[1]) - int(missing[0])) % len(CAMERA_NAMES) in (1, 5)
    ):
        family = "adjacent_double"
    elif len(missing) == 2:
        family = "nonadjacent_double"
    elif len(missing) == 3:
        family = "three_camera"
    else:
        family = "clean_or_other"
    duration = elapsed if elapsed in (1, 3, 5) else "other"
    return f"{family}_duration_{duration}"


@dataclass
class ResidualPredictors:
    global_mean: np.ndarray
    type_means: Dict[str, np.ndarray]
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    ridge: Ridge
    mlp: torch.nn.Module

    def predict(self, name: str, x: np.ndarray,
                keys: Optional[Sequence[str]] = None) -> np.ndarray:
        if name == "Z0":
            return np.zeros((len(x), len(self.global_mean)), dtype=np.float32)
        if name == "Z1":
            if keys is None:
                raise ValueError("Z1 requires fault keys")
            return np.stack([
                self.type_means.get(key, self.global_mean) for key in keys
            ]).astype(np.float32)
        scaled_x = self.x_scaler.transform(x)
        if name == "L":
            scaled_y = self.ridge.predict(scaled_x)
        elif name == "M":
            with torch.no_grad():
                scaled_y = self.mlp(
                    torch.from_numpy(scaled_x.astype(np.float32))
                ).cpu().numpy()
        else:
            raise KeyError(name)
        return self.y_scaler.inverse_transform(scaled_y).astype(np.float32)


class FixedResidualMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
        )

    def forward(self, value):
        return self.network(value)


def fit_predictors(
    x: np.ndarray,
    y: np.ndarray,
    keys: Sequence[str],
    scenes: Sequence[str],
    seed: int = 2026,
) -> tuple[ResidualPredictors, dict]:
    unique_scenes = sorted(set(scenes))
    if len(unique_scenes) != 8:
        raise ValueError(f"expected 8 mini-train scenes, got {len(unique_scenes)}")
    fit_scenes = set(unique_scenes[:6])
    fit_mask = np.asarray([scene in fit_scenes for scene in scenes])
    stop_mask = ~fit_mask
    if not fit_mask.any() or not stop_mask.any():
        raise ValueError("fixed internal split is empty")

    x_scaler = StandardScaler().fit(x[fit_mask])
    y_scaler = StandardScaler().fit(y[fit_mask])
    x_fit = x_scaler.transform(x[fit_mask]).astype(np.float32)
    y_fit = y_scaler.transform(y[fit_mask]).astype(np.float32)
    x_stop = x_scaler.transform(x[stop_mask]).astype(np.float32)
    y_stop = y_scaler.transform(y[stop_mask]).astype(np.float32)

    ridge = Ridge(alpha=1.0).fit(x_fit, y_fit)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FixedResidualMLP(x.shape[1], y.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_fit),
        torch.from_numpy(y_fit),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        generator=generator,
    )
    x_stop_tensor = torch.from_numpy(x_stop)
    y_stop_tensor = torch.from_numpy(y_stop)
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, 201):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            stop_loss = float(
                torch.nn.functional.mse_loss(
                    model(x_stop_tensor), y_stop_tensor
                ).item()
            )
        if stop_loss < best_loss - 1e-6:
            best_loss = stop_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 20:
                break
    if best_state is None:
        raise RuntimeError("MLP early stopping never recorded a state")
    model.load_state_dict(best_state)
    model.eval()

    type_means = {}
    key_array = np.asarray(keys)
    for key in sorted(set(keys)):
        selected = fit_mask & (key_array == key)
        if selected.any():
            type_means[key] = y[selected].mean(axis=0)
    predictors = ResidualPredictors(
        global_mean=y[fit_mask].mean(axis=0),
        type_means=type_means,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        ridge=ridge,
        mlp=model,
    )
    return predictors, {
        "fit_scenes": sorted(fit_scenes),
        "early_stop_scenes": sorted(set(unique_scenes) - fit_scenes),
        "fit_instances": int(fit_mask.sum()),
        "early_stop_instances": int(stop_mask.sum()),
        "mlp_best_epoch": best_epoch,
        "mlp_early_stop_loss": best_loss,
    }


def residual_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target

    def component(index) -> tuple[float, float, float]:
        true = target[:, index]
        pred = prediction[:, index]
        return (
            float(np.mean(np.abs(pred - true))),
            float(np.sqrt(np.mean((pred - true) ** 2))),
            float(r2_score(true, pred, multioutput="variance_weighted")),
        )

    mae, rmse, r2 = component(GEOMETRY)
    class_mae, class_rmse, class_r2 = component(slice(0, CLASS_DIM))
    center_mae, center_rmse, center_r2 = component(CENTER)
    yaw_error = wrap_yaw(error[:, YAW])
    yaw_mae = float(np.mean(np.abs(yaw_error)))
    yaw_rmse = float(np.sqrt(np.mean(yaw_error ** 2)))
    yaw_r2 = float(r2_score(target[:, YAW], prediction[:, YAW]))
    velocity_mae, velocity_rmse, velocity_r2 = component(VELOCITY)
    true_geometry = target[:, GEOMETRY]
    pred_geometry = prediction[:, GEOMETRY]
    denominator = (
        np.linalg.norm(true_geometry, axis=1)
        * np.linalg.norm(pred_geometry, axis=1)
    )
    cosine = np.divide(
        np.sum(true_geometry * pred_geometry, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-8,
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "class_mae": class_mae,
        "class_rmse": class_rmse,
        "class_r2": class_r2,
        "center_mae": center_mae,
        "center_rmse": center_rmse,
        "center_r2": center_r2,
        "yaw_mae": yaw_mae,
        "yaw_rmse": yaw_rmse,
        "yaw_r2": yaw_r2,
        "velocity_mae": velocity_mae,
        "velocity_rmse": velocity_rmse,
        "velocity_r2": velocity_r2,
        "direction_cosine": float(np.mean(cosine)),
    }


def advantage_auroc(labels: np.ndarray, prediction: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return math.nan
    score = np.linalg.norm(prediction[:, GEOMETRY], axis=1)
    if np.allclose(score, score[0]):
        return 0.5
    return float(roc_auc_score(labels, score))
