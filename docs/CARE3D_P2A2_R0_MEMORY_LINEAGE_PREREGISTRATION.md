# CARE-3D P2-A2-R0 Explicit Memory-Lineage Association

## Status and scope

P2-A2-R0 is a development pre-gate on the frozen 419-scene `probe_train`
cohort. It is not confirmatory validation. `probe_val` and `probe_test` are not
exposed by the extraction CLI and must remain unread. No network is trained and
no P2-A0 parameter, gate, cohort, fault protocol, checkpoint, or query-collision
rule is changed.

The prerequisite state is:

- P2-A0 decision: `NO_GO_CARE3D_P2A_ONLINE_QUERY_ASSOCIATION`
- P2-A1 status: `LOCKED_P2A0_NO_GO`
- frozen P2-A0 configuration: `g0.4_e0.4_c0.2_t0.45`
- weights `(geometry, embedding, class) = (0.4, 0.4, 0.2)`
- unmatched threshold `0.45`

## Primary method

The primary method is `lineage_first_hybrid`. For clean online anchor frame
`t`, it reproduces StreamPETR's `post_update_memory` proposal ordering:

```python
rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
_, topk_indexes = torch.topk(rec_score, head.topk_proposals, dim=1)
```

For anchor query `q`, if `topk_indexes[j] == q`, its strict one-step child at
`t+1` is `head.num_query + j`, or `644 + j`. If `q` is absent, lineage is
unavailable. This is only a `t -> t+1` ancestry statement; memory slots are not
treated as persistent identities across multiple frames.

Lineage child selection accepts only the clean anchor query and anchor-frame
scores. GT, the clean `t+1` output, `target_clean_query_index`, and the oracle
query cannot enter the online assignment. Oracle query identity is attached
after assignment for offline evaluation only.

## Frame-level one-to-one rule

Within each `(scene_token, target_frame_idx, protocol)` group:

1. Assign every available lineage child and assert uniqueness.
2. Reserve those fault-frame query indices.
3. Run the frozen P2-A0 cost and Hungarian-with-unmatched rule for unavailable
   rows, with reserved columns removed from the feasible pool.
4. Assert all matched hybrid query indices are unique.

The reported baselines are `p2a0_frozen` and `lineage_only`. Lineage-only is
unmatched whenever ancestry is unavailable.

## Engineering invariant

The excluded engineering scene must establish the query layout
`644 + 256 = 900`, `topk_proposals == 256`, and reproduce the actual memory
prefix:

```python
recomputed_rec_memory = topk_gather(outs_dec[-1], topk_indexes)
```

`recomputed_rec_memory` must be `torch.equal` to
`head.memory_embedding[:, :256]`; the recorded maximum absolute difference must
be zero. The required smoke status is
`P2A2_R0_LINEAGE_SMOKE_PASSED`.

## Metrics and paired bootstrap

For Blur, Crash, and Dark separately, report lineage coverage and conditional
outcomes; frozen P2-A0 and hybrid exact/wrong/unmatched rates; wrong repairs,
correct breaks, correct-to-unmatched transitions, net exact gain, wrong repair
fraction, and break fraction. Diagnostics are stratified read-only by anchor
query origin, oracle query origin, and target frame. Queries `0..643` are
`current`; `644..899` are `propagated`.

Use 5,000 repetitions for both `scene_token` and
`instance_token`/trajectory cluster bootstraps of the paired row-wise deltas:

- `delta_exact = hybrid_exact - p2a0_exact`
- `delta_wrong = hybrid_wrong - p2a0_wrong`

Ordinary row bootstrap is not permitted.

## Development gate

A protocol passes only when all conditions hold:

1. hybrid wrong-match rate is at most `0.10`;
2. scene-cluster paired `delta_exact` 95% CI lower bound is above zero;
3. instance-cluster paired `delta_exact` 95% CI lower bound is above zero;
4. hybrid unmatched rate is at most `0.01`;
5. `probe_val_read == false`;
6. `probe_test_read == false`.

At least two of three protocols produce `GO_P2A2_R0_MEMORY_LINEAGE`; otherwise
the decision is `NO_GO_P2A2_R0_MEMORY_LINEAGE`. A No-Go does not modify or
reinterpret the frozen P2-A0 gate.
