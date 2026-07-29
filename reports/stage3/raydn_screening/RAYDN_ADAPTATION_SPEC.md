# RayDN adaptation specification

## Sources and license

- Paper: Liu et al., *Ray Denoising: Depth-aware Hard Negative Sampling for
  Multi-view 3D Object Detection*, ECCV 2024,
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06549.pdf
- Official implementation: https://github.com/LiewFeng/RayDN, audited at
  commit `cdb8c2cf72b4b1f1a768f2e1371224436bcc4635`.
- The official repository is Apache-2.0 licensed. It is based on StreamPETR
  and has the same MMDetection3D/MMCV/PyTorch dependency family already used
  by this project. No new runtime dependency is introduced here.

## Fixed sampling rule

For a GT center at camera depth \(d\) and box dimensions \(w,h,l\):

\[
r=(w+h+l)/6,\quad z_i=d+r(2x_i-1)k,\quad
x_i\sim\mathrm{Beta}(8,2),\quad k=3.
\]

The shifted depth is back-projected through the inverse `lidar2img` matrix.
Exactly five ray queries are constructed for each GT. The query with minimum
absolute depth offset is the positive and inherits the GT class; the remaining
four are background. If the closest offset lies outside the existing DN
positive radius `split=0.75`, it is resampled inside that radius. One RayDN
group is used. These are the official R50 settings; there is no second setting
or search.

## Query construction and mask

The five ray queries are appended to StreamPETR's existing five standard DN
copies. Existing object-query count remains 644 and propagated-query count
remains 256. Matching queries cannot attend any DN query. Standard DN copies
are isolated from other DN copies. The five depths of the single RayDN group
may interact with each other, but not with standard DN groups. Temporal
attention extends the same mask and prevents propagated/matching queries from
reading DN state.

Ray queries are removed before prediction output and memory write-back.
Consequently they cannot alter query count, Top-K capacity, inference graph or
temporal runtime state.

## Loss

Ray queries reuse StreamPETR's existing DN loss path and `dn_weight=1.0`:

- sigmoid focal classification loss;
- normalized 3D box L1 regression loss with the existing code weights.

The positive query targets its GT class; the four hard negatives target
background. No new head, loss, threshold, teacher or curriculum is added.

## Project-side integration

- `models/ray_denoising.py` supplies the sampler and a registered
  `RayDNStreamPETRHead` subclass for B0.
- `EvidenceConservingStreamPETRHead` calls the same helper for M1.
- `enable_ray_denoising=False` directly executes the pre-existing
  `prepare_for_dn` path.
- RayDN is active only when both `model.training` and the switch are true.
- Configuration and transient Python references add no parameter or buffer;
  state-dict and checkpoint key sets are unchanged.
- `repos/StreamPETR` is neither edited nor vendored.

## Training/inference distinction

Training adds GT-derived ray queries, an attention mask and ordinary DN losses.
Evaluation always follows the original graph, even when the training config
sets RayDN true. It therefore adds no inference latency or persistent state.

## Fair screening protocol and pre-registered decision

Groups are limited to B0, B0+RayDN, M1 and M1+RayDN. All start from
`outputs/stage2/s2_2_source_ledger_debug_50/iter_50.pth`, seed 2026, the same
mini split/order, optimizer, LR, batch size, FP16 with fixed loss scale 512,
gradient clipping, 644
object queries and 50 iterations. B0 ignores the checkpoint's Evidence3D-only
keys; shared detector tensors are identical at initialization.

A RayDN group may enter 200 iterations only if, relative to its corresponding
baseline:

1. mean Crash5/Crash10/Compound NDS improves by at least 0.002;
2. Clean NDS drops by no more than 0.001;
3. at least two of the three fault protocols improve;
4. false positives or false memory writes clearly decrease;
5. no NaN, Inf or OOM occurs;
6. M1 conservation and source-mass violation counts remain zero.

The thresholds and fixed formula above are frozen before smoke or training.
