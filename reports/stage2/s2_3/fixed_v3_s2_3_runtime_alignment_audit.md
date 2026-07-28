# Evidence3D S2.3 runtime and alignment audit

## Legacy novelty

- Shape/dtype/range: `[B,Q]`, evidence dtype, clamped to `[0,1]`.
- Formula: `fresh_ratio + (1-fresh_ratio) * (1-cos(current_source, prior_source))`.
- S2.2 reads `legacy_provenance`; normalized source-ledger `provenance` is not
  fed back into the legacy evidence path.
- It multiplies `observability * novelty * effective_count` immediately before
  the positive and negative alpha/beta increments.

## Query identity

- Non-DN decoder layout is 644 base queries followed by 256 propagated queries.
- The propagated segment comes from `memory_embedding[:, :256]`.
- StreamPETR commits one Top-K index tensor to feature, center, velocity and
  ego-pose memory. EvidenceLedger consumes that same tensor for alpha, beta,
  source, feature, geometry and semantic references.
- A new/base query is never paired with a historical feature. A pair is valid
  only when it is in the propagated segment and the corresponding committed
  reference is valid.
- DEFER zeros feature/center/velocity memory and commits an invalid S2.3
  reference, so it cannot become a valid feature pair next frame.

## Coordinate alignment

- StreamPETR converts historical centers from global memory to current ego
  coordinates with `ego_pose_inv` before temporal alignment.
- S2.3 stores `[x,y,z,yaw,vx,vy]`. It applies the same global-to-current
  transform before comparison and current-to-global transform after commit.
- Size is available in the current regression tensor but has no native
  StreamPETR temporal-memory counterpart and is therefore not used.

## Available prediction state

- Geometry: center, size, encoded yaw, velocity and propagated reference point.
- Probability: sigmoid class distribution, three-state distribution, ledger
  presence probability and unobserved probability.
- Before S2.3 there was no aligned class/ternary history, semantic entropy,
  geometry residual or strict query-pair diagnostic.

## Runtime state

All scene-local ledger and S2.3 reference buffers are non-persistent buffers.
They migrate with `model.to(device)`, are omitted from PyTorch and MMCV
checkpoints, and can only be saved/restored through
`export_runtime_state()`/`load_runtime_state()`. Scene changes, batch-size
changes and `reset_memory()` clear them.

## Acceptance evidence

- Unit/regression tests: 86 passed.
- N0/off versus track: 24/24 development-protocol prediction files are
  tensor-exact.
- Track and active collapse checks: pass.
- 50-iteration checkpoints: no runtime-state keys.
- No change was made under `repos/StreamPETR`.
