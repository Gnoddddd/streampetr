"""3D observability field for multi-camera object queries.

The first implementation follows the research plan strictly: observability is
computed from camera online state, camera quality, and geometric frustum
coverage.  It is not a confidence or prediction-reliability map.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


def default_camera_correlation(num_cameras: int = 6) -> Tensor:
    """Return a lightweight correlation prior for the nuScenes camera ring.

    The diagonal is one. Adjacent views receive a modest correlation, opposite
    views are treated as almost independent. The matrix can be overridden from
    the model config when learned/empirical correlations are available.
    """

    corr = torch.eye(num_cameras, dtype=torch.float32)
    if num_cameras == 6:
        # Converter order used by StreamPETR:
        # FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT.
        adjacent_pairs = (
            (0, 1), (0, 2), (1, 5), (2, 4), (3, 4), (3, 5),
        )
        for i, j in adjacent_pairs:
            corr[i, j] = corr[j, i] = 0.35
        # The two front-side and two back-side views share degradation modes.
        corr[1, 2] = corr[2, 1] = 0.15
        corr[4, 5] = corr[5, 4] = 0.15
    return corr


class GeometricObservabilityHead(nn.Module):
    """Project 3D query centers into cameras and estimate observability.

    Args:
        num_cameras: Expected number of camera views.
        min_depth: Minimum positive camera depth.
        boundary_softness: Pixel-domain softness of image-boundary gates.
        depth_temperature: Sharpness of the positive-depth gate.
        correlation_matrix: Optional camera evidence-correlation matrix.
        learned_residual: Whether to blend a small learned query-feature gate.
        embed_dims: Query feature size when ``learned_residual`` is enabled.
        residual_weight: Maximum contribution of the learned gate. The default
            is zero so the mini implementation remains geometry-first.
    """

    def __init__(
        self,
        num_cameras: int = 6,
        min_depth: float = 0.1,
        boundary_softness: float = 8.0,
        depth_temperature: float = 4.0,
        correlation_matrix: Optional[Sequence[Sequence[float]]] = None,
        enable_correlation_discount: bool = False,
        learned_residual: bool = False,
        embed_dims: int = 256,
        residual_weight: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.min_depth = float(min_depth)
        self.boundary_softness = max(float(boundary_softness), 1e-3)
        self.depth_temperature = float(depth_temperature)
        self.enable_correlation_discount = bool(
            enable_correlation_discount
        )
        self.learned_residual = bool(learned_residual)
        self.residual_weight = float(residual_weight)
        self.eps = float(eps)

        corr = (
            torch.as_tensor(correlation_matrix, dtype=torch.float32)
            if correlation_matrix is not None
            else default_camera_correlation(self.num_cameras)
        )
        if corr.shape != (self.num_cameras, self.num_cameras):
            raise ValueError(
                f"correlation_matrix must be {self.num_cameras}x"
                f"{self.num_cameras}, got {tuple(corr.shape)}"
            )
        self.register_buffer("camera_correlation", corr, persistent=True)

        self.query_gate: Optional[nn.Module]
        if self.learned_residual:
            self.query_gate = nn.Sequential(
                nn.Linear(embed_dims, embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims // 2, 1),
            )
        else:
            self.query_gate = None

    @staticmethod
    def _as_layered(query_xyz: Tensor) -> Tuple[Tensor, bool]:
        if query_xyz.ndim == 3:
            return query_xyz.unsqueeze(0), True
        if query_xyz.ndim != 4:
            raise ValueError(
                "query_xyz must have shape [B,Q,3] or [L,B,Q,3], "
                f"got {tuple(query_xyz.shape)}"
            )
        return query_xyz, False

    def _image_hw_tensor(
        self,
        image_hw: Union[Tensor, Tuple[int, int], Sequence[int]],
        batch_size: int,
        num_cameras: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        hw = torch.as_tensor(image_hw, device=device, dtype=dtype)
        if hw.ndim == 1 and hw.numel() == 2:
            hw = hw.view(1, 1, 2).expand(batch_size, num_cameras, 2)
        elif hw.ndim == 2 and hw.shape == (num_cameras, 2):
            hw = hw.unsqueeze(0).expand(batch_size, num_cameras, 2)
        elif hw.ndim == 3 and hw.shape[:2] == (batch_size, num_cameras):
            pass
        else:
            raise ValueError(
                "image_hw must be [2], [N,2], or [B,N,2], got "
                f"{tuple(hw.shape)}"
            )
        return hw

    @staticmethod
    def _state_tensor(
        value: Optional[Tensor],
        batch_size: int,
        num_cameras: int,
        device: torch.device,
        dtype: torch.dtype,
        default: float,
    ) -> Tensor:
        if value is None:
            return torch.full(
                (batch_size, num_cameras), default, device=device, dtype=dtype
            )
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape != (batch_size, num_cameras):
            raise ValueError(
                f"camera state must be [B,N]={batch_size,num_cameras}, "
                f"got {tuple(value.shape)}"
            )
        return value.clamp(0.0, 1.0)

    def forward(
        self,
        query_xyz: Tensor,
        lidar2img: Tensor,
        image_hw: Union[Tensor, Tuple[int, int], Sequence[int]],
        camera_online_mask: Optional[Tensor] = None,
        camera_quality: Optional[Tensor] = None,
        camera_fresh_mask: Optional[Tensor] = None,
        query_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute observability and compact evidence provenance.

        Returns tensors with a decoder-layer dimension. If the input omitted
        that dimension, it is removed from all returned values.
        """

        xyz, squeeze_layer = self._as_layered(query_xyz)
        output_dtype = xyz.dtype
        cpu_half = (
            xyz.device.type == "cpu"
            and output_dtype == torch.float16
        )
        if cpu_half:
            # PyTorch 1.9 lacks several CPU Half kernels used by projection
            # and boundary gates. Keep the public dtype while computing the
            # stateless observability path in fp32.
            xyz = xyz.float()
            if query_features is not None:
                query_features = query_features.float()
        if xyz.shape[-1] != 3:
            raise ValueError("query_xyz last dimension must be 3")
        layers, batch_size, num_queries, _ = xyz.shape

        lidar2img = torch.as_tensor(
            lidar2img, device=xyz.device, dtype=xyz.dtype
        )
        if lidar2img.ndim != 4 or lidar2img.shape[-2:] != (4, 4):
            raise ValueError(
                "lidar2img must have shape [B,N,4,4], got "
                f"{tuple(lidar2img.shape)}"
            )
        if lidar2img.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between queries and lidar2img")
        num_cameras = lidar2img.shape[1]
        if num_cameras != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {num_cameras}"
            )

        hw = self._image_hw_tensor(
            image_hw, batch_size, num_cameras, xyz.device, xyz.dtype
        )
        online = self._state_tensor(
            camera_online_mask,
            batch_size,
            num_cameras,
            xyz.device,
            xyz.dtype,
            1.0,
        )
        quality = self._state_tensor(
            camera_quality,
            batch_size,
            num_cameras,
            xyz.device,
            xyz.dtype,
            1.0,
        )
        fresh = self._state_tensor(
            camera_fresh_mask,
            batch_size,
            num_cameras,
            xyz.device,
            xyz.dtype,
            1.0,
        )

        ones = torch.ones_like(xyz[..., :1])
        xyz_h = torch.cat((xyz, ones), dim=-1)
        # [L,B,N,Q,4]
        projected = torch.einsum("bnij,lbqj->lbnqi", lidar2img, xyz_h)
        depth = projected[..., 2]
        safe_depth = depth.clamp_min(self.eps)
        u = projected[..., 0] / safe_depth
        v = projected[..., 1] / safe_depth

        height = hw[..., 0].view(1, batch_size, num_cameras, 1)
        width = hw[..., 1].view(1, batch_size, num_cameras, 1)
        softness = self.boundary_softness

        depth_gate = torch.sigmoid(
            (depth - self.min_depth) * self.depth_temperature
        )
        left_gate = torch.sigmoid(u / softness)
        right_gate = torch.sigmoid((width - 1.0 - u) / softness)
        top_gate = torch.sigmoid(v / softness)
        bottom_gate = torch.sigmoid((height - 1.0 - v) / softness)
        geometry = depth_gate * left_gate * right_gate * top_gate * bottom_gate

        state = (online * quality).view(1, batch_size, num_cameras, 1)
        per_camera_lbnq = (geometry * state).clamp(0.0, 1.0)
        per_camera = per_camera_lbnq.permute(0, 1, 3, 2).contiguous()
        # Probability that at least one view supports the query.
        observability = 1.0 - torch.prod(1.0 - per_camera, dim=-1)

        if self.query_gate is not None and query_features is not None:
            qf, qf_squeezed = self._as_layered(query_features)
            if qf_squeezed != squeeze_layer or qf.shape[:3] != xyz.shape[:3]:
                raise ValueError("query_features must align with query_xyz")
            learned = torch.sigmoid(self.query_gate(qf).squeeze(-1))
            weight = min(max(self.residual_weight, 0.0), 1.0)
            observability = (1.0 - weight) * observability + weight * learned

        support_sum = per_camera.sum(dim=-1, keepdim=True)
        source_vector = per_camera / support_sum.clamp_min(self.eps)
        fresh_ratio = (
            per_camera
            * fresh.view(1, batch_size, 1, num_cameras)
        ).sum(dim=-1) / support_sum.squeeze(-1).clamp_min(self.eps)
        fresh_ratio = torch.where(
            support_sum.squeeze(-1) > self.eps,
            fresh_ratio,
            torch.zeros_like(fresh_ratio),
        )

        if self.enable_correlation_discount:
            weights = per_camera
            numerator = weights.sum(dim=-1).pow(2)
            diagonal = weights.pow(2).sum(dim=-1)
            correlation = self.camera_correlation.to(
                device=weights.device, dtype=weights.dtype
            )
            pairwise = torch.einsum(
                "lbqm,mn,lbqn->lbq",
                weights,
                correlation,
                weights,
            )
            off_diagonal = (pairwise - diagonal).clamp_min(0.0)
            effective_count = numerator / (
                diagonal + off_diagonal + self.eps
            )
            effective_count = torch.where(
                support_sum.squeeze(-1) > self.eps,
                effective_count.clamp(0.0, float(num_cameras)),
                torch.zeros_like(effective_count),
            )
        else:
            # This tensor is diagnostic only on the disabled path. The
            # adapter passes ``None`` to the temporal update so no S2.4
            # matrix or N_eff value participates in evidence accumulation.
            effective_count = torch.ones_like(observability)

        output = {
            "observability": observability.clamp(0.0, 1.0),
            "per_camera": per_camera,
            "source_vector": source_vector,
            "fresh_ratio": fresh_ratio.clamp(0.0, 1.0),
            "effective_count": effective_count,
            "projected_uvd": torch.stack((u, v, depth), dim=-1).permute(
                0, 1, 3, 2, 4
            ),
        }
        if squeeze_layer:
            output = {key: value.squeeze(0) for key, value in output.items()}
        if cpu_half:
            output = {
                key: value.to(output_dtype)
                if torch.is_tensor(value) and value.is_floating_point()
                else value
                for key, value in output.items()
            }
        return output
