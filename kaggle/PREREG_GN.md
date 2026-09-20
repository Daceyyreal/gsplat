# Pre-registration: E0, a Gauss-Newton metric for the shN codebook (`bench/gn-vq`)

Written on 2026-09-19, before any E0 code exists and before any E0 result. This file fixes the
questions, the definitions and the decision rules. It is not edited after results exist. Any later
change goes into a new, dated section below the original text and says why.

## Questions

- **E0 / G0:** Does a per-splat Gauss-Newton (GN) metric on the shN coefficients predict how much
  shN vector quantization changes the rendered images?
- **E0, exploratory:** What does that metric look like (spectrum, relation to existing weights), and
  what does a GN-weighted refinement of the codebook do?
- **E1 / G1** (a later run, stated here so it is fixed first): Does GN-weighted vector quantization
  (GN-VQ) beat `lloyd_wopa_area` at equal size?

## Fixed setup

- **Scenes and checkpoints:** MipNeRF360 garden and bicycle, the training-#2 checkpoints used by
  runs 1-5 (MCMC, 1,000,000 Gaussians, data factor 4, `mcmc.sh` settings, LPIPS VGG). Every row
  records the checkpoint sha1.
- **Views:** train views = `simple_trainer`'s train split, test views = its val split
  (`test_every = 8`), with the same cameras and intrinsics as its eval.
- **Sort and codebook size:** seed-0 PLAS order (the cached run-2 order), K = 65,536 centroids,
  k-means seed 0.
- **The three configs** (the run-3 names in brackets):

  | E0 name | Clustering | Run-3 config |
  |---|---|---|
  | `upstream_l1` | TorchPQ manhattan (L1 assignment, mean update), the library default before #1063 | `manhattan_log` |
  | `plain_l2` | unweighted Lloyd (L2 assignment, mean update) = library `weighted_kmeans` without weights | `lloyd_w1` |
  | `lloyd_wopa_area` | Lloyd with weights `sigmoid(opacity) * exp(sum of the two largest log-scales)` | `lloyd_wopa_area` |

  The float centroids and labels come from the run-3 clustering cache for that checkpoint, sort and
  seed when it is restored and its key matches. Otherwise they are recomputed with the same code and
  seed; each row records which.
- **Quantization and files:** the unchanged `PngCompression` path: 6-bit codebook with one scalar
  min/max, uint16 labels, both inside `shN.npz`, other parameters as PNG. The codebook is fed to the
  library writer; the clustering is the only thing that differs between configs.

## The GN metric

Each rendered pixel p of view v (channel ch) is `I_p = sum_i w_ip * col_i,ch`. Here `w_ip` is splat
i's alpha-blending weight at p, `col_i,ch = max(SH_i,ch(d_iv) + 0.5, 0)`, and
`d_iv = normalize(mean_i - campos_v)`, as in gsplat. Linearizing in the 15 shN coefficients of one
channel (SH bands 1-3, `y(d)` = the 15 basis values), and dropping cross-splat terms and the clamp:

- `M_i = sum_v s_iv * y(d_iv) y(d_iv)^T` (15 x 15, the same for all three channels), with
  `s_iv = sum_p w_ip^2` over the pixels of view v.

**Estimating `s_iv`:** render zero features `[N, 17]` with `sh_degree=None`, background 0, and the
eval's rasterize settings (`rasterize_mode`, `near_plane`, `far_plane`, `packed`, camera model).
Backpropagate a pixel gradient that is independent Rademacher +-1 per pixel on channels 0-15 and 1
on channel 16. Channel 16's gradient is `f_iv = sum_p w_ip`. The mean of the squared gradients of
channels 0-15 is a 16-probe Hutchinson estimate of `s_iv`. Probe noise comes from a fixed seed (0).

Accumulated per splat over train views, only where the splat is visible (radius > 0):

- `M_i`: upper triangle, fp32, chunked over visible splats.
- `F_i = sum_v f_iv`.
- **C3DGS-style weight:** `max over k = 1..15 of (1/V) sum_v |f_iv * y_k(d_iv)|` (V = number of train
  views; invisible views contribute 0).
- **Clamp fraction:** the fraction of visible (splat, view, channel) triples whose full degree-3 SH
  evaluation plus 0.5 is below 0. This is logged only; the metric does not mask them.

17 channels is a compiled channel count in this gsplat build (`GSPLAT_NUM_CHANNELS` default list).

## G0 (judged in E0 by the notebook, from the result rows)

For each scene and config c, with `Delta_i = c_i - q(label_i)`: the splat's original shN minus its
**dequantized** centroid, as the unchanged decoder returns it.

- **Predicted:** `P_c = sum_i sum_rgb Delta_i^T M_i Delta_i / (3 * total train pixels)`.
- **Measured:** `D_c = mean over pixels and the 3 channels of (I_q - I_orig)^2`, on train views
  (`D_c^train`) and on test views (`D_c^test`). `I_q` is the render with **only shN** replaced by its
  decoded version; every other parameter is the uncompressed original. Both images come from the
  eval's render call and are clamped to [0, 1] as the eval does.

**G0 passes** when, on garden **and** bicycle:

1. the three configs sorted by `P_c` come out in the same order as sorted by `D_c^train`, and in the
   same order as sorted by `D_c^test` (ascending; exact float ties broken by the table order above);
   **and**
2. `0.5 <= P_c / D_c^train <= 2` for all three configs.

Reported alongside, not part of the rule: the same comparisons on unclamped renders,
`P_c / D_c^test`, and GT PSNR / SSIM / LPIPS for each config, both for the full compressed pipeline and
with only shN swapped.

### Validity checks (G0 is judged only if all hold; otherwise it is reported as invalid, not failed)

- **SH basis:** `bench/gn/sh_basis.py` against gsplat's CUDA `spherical_harmonics` on random directions
  and coefficients: max abs error < 1e-5.
- **Toy exactness:** a toy scene with at most 256 splats and a small image. Exact `s_i = sum_p w_ip^2`
  comes from an identity-feature render; the Hutchinson estimate uses 64 probes (four 16-probe
  renders). It must be within 5% relative. That is judged on the sum over splats: at 64 probes a
  single splat's estimate has a relative standard deviation of up to sqrt(2/64) = 18%, and `P_c` is a
  sum over splats. Per-splat errors are reported. Channel 16 must equal the identity render's
  `sum_p w_ip` to 1e-4 relative.
- **Render parity:** the direct `rasterization` call that the GN pass uses gives the same image as the
  runner's eval render for a train view (max abs difference <= 1e-6).
- **Reproduction:** if the run-3 `lloyd_wopa_area` seed-0 row exists, the E0 full-pipeline row for that
  config has the same PSNR (|difference| <= 1e-6 dB) and the same raw bytes. (TorchPQ reproduces only to
  about 0.002 dB across sessions, so `upstream_l1` is reported against run 3, not checked.)

## Exploratory in E0 (no decision)

a. **Spectrum:** per-splat `eigvalsh(M_i)`: participation ratio `(sum lambda)^2 / sum lambda^2`, and the
   top-1 and top-3 energy fractions. Histograms and CSV, unweighted and weighted by `tr(M_i)`, over
   splats with `tr(M_i) > 0`.
b. **Spearman rank correlations** (average ranks for ties) between `tr(M_i)`, the `opacity_area`
   weight (the bench function `tilequant_run3.cluster_weights`), `F_i` and the C3DGS-style weight. They
   are computed over all splats and over splats with `tr(M_i) > 0`.
c. **Predictivity:** the G0 quantities above, plus GT metrics.
d. **GN refine**, warm-started from the `lloyd_wopa_area` seed-0 float codebook and labels, K = 65,536.
   Three iterations of:
   - **Shortlist:** the top 64 centroids by L2 distance (chunked). The "wopa-weighted" L2 of the plan
     scales all of a splat's distances by the same weight, so its ranking is plain L2.
   - **Rerank:** exact Mahalanobis cost `sum_rgb (x_i - q)^T M_i (x_i - q)` over the shortlist **plus the
     splat's current centroid**, so that an assignment step never increases the objective. Splats
     with `tr(M_i) = 0` take their L2-nearest centroid.
   - **Update per channel:** `q = (sum M + eps * tr(sum M) / 15 * I)^-1 sum M c`, with eps = 1e-4 and
     sums over a cluster's splats, solved in float64. A cluster with `tr(sum M) = 0` keeps its previous
     centroid.

   Then the unchanged `PngCompression` quantization and file write. Reported:
   - exact bytes per file, with `shN.npz` split into its centroid and label members (upstream stores
     the labels inside `shN.npz`, not as a PNG);
   - test PSNR / SSIM / LPIPS, `P`, `D^train` and `D^test`;
   - the GN objective per iteration;
   - shortlist recall against exhaustive Mahalanobis search, at each iteration, on 50,000 random
     splats with `tr(M_i) > 0` (the fraction whose exhaustive argmin is in their top-64 L2 shortlist).

   `M` comes from the train views, so train-view numbers are in-sample and test views are
   out-of-sample.

## G1 (judged in E1, not in E0)

GN-VQ against `lloyd_wopa_area`, at equal total bytes (raw bytes of the compressed directory within
+-0.5% of `lloyd_wopa_area` at the same seed): at least +0.05 dB mean test PSNR over 3 k-means seeds,
no seed with a negative PSNR difference, on both scenes. The exact GN-VQ variant and its size-matching
procedure are fixed in writing before the E1 run.

## Amendment 1 (2026-09-19, during implementation, before any E0 result)

Only the toy exactness check changes. G0, the other validity checks and everything above stay as
written.

**Why:** on a CPU copy of the renderer (`bench/gn/toy_render.py`), the toy layout I first wrote put
the 64-probe summed estimate 5% or more off the exact sum in 9.9% of probe draws (99 of 1,000;
`bench/gn/toy_noise.json`). In that layout, splats of very different size and opacity cluster in the
middle of the image, so a few large splats dominate the sum and share probes. A validity check should
catch implementation errors, not probe noise.

**Changes:**

- **Toy scene** (`gn_metric.toy_scene`): 256 splats of similar size and opacity, spread uniformly over
  a 128 x 96 view at depth 3 +- 0.3. With the scene the CUDA check uses (seed 0), 86.8% of covered
  pixels still blend two or more splats. Over 1,000 simulated probe draws the summed estimate is off
  by 1.2% on average, 3.7% at the 99th percentile and 4.95% at most; no draw reaches 5%
  (`bench/gn/toy_noise.json`, from `bench/gn/toy_noise.py`). I chose the layout from these CPU
  simulations only, before any CUDA run.
- **Unchanged:** the pre-registered test itself, the summed 64-probe estimate within 5% of the exact
  sum.
- **Added, stricter:** a deterministic check. Per splat, the gradient-based estimate must equal the
  Hutchinson formula `mean_c (sum_p w_pi r_pc)^2`, evaluated with the identity-render weights and the
  same probe images, to 1e-4 relative. This tests the gradient plumbing exactly, independent of probe
  noise. (Splats covering less than 1e-3 of the largest footprint are excluded from the per-splat
  relative errors.)

## Amendment 2 (2026-09-19, before any E0 result and before the code change it describes)

This replaces the G0 rule and exploratory item d above. The G0 validity checks, Amendment 1 and G1 are
unchanged.

**Why:** a strict ranking of 3 configs at one K can flip on near-ties. The quantities differ little:
run 3's bicycle `lloyd_wopa_area` gain over the baseline was +0.030 dB mean over 3 k-means seeds and
+0.027 dB at seed 0, and `lloyd_w1` was -0.023 dB at seed 0 (FINDINGS section 4). One flipped near-tie
would decide G0 on noise. So G0 now uses more codebooks and exempts measured ties.

### Codebooks (per scene, k-means seed 0)

The three configs (`upstream_l1` = TorchPQ manhattan, `plain_l2` = `lloyd_w1`, `lloyd_wopa_area`),
each at K in {4096, 16384, 65536}, give 9 codebooks per scene. K = 65,536 is the run-3 default and
reuses the run-3 caches as before. K = 4,096 and 16,384 are clustered with the same code and seed.
Writing, measuring and the definitions of `P`, `D^train` and `D^test` are unchanged.

### G0 rule (replaces the rule above)

- **Ratio:** `0.5 <= P / D^train <= 2`, with the clamped measurement, for all 9 codebooks of each scene.
- **Ranking, checked only within each K:** there are 3 config pairs per K, each checked on train
  views and on test views, for both scenes: 2 x 3 x 3 x 2 = 36 pair checks.
  - A pair (a, b) is a **tie** on a view set if `|D_a - D_b| / min(D_a, D_b) < 0.05` (measured,
    clamped). Ties are exempt.
  - A non-tied pair **agrees** if `P_a < P_b` exactly when `D_a < D_b`. Equal `P` does not agree.
- **Verdict**, in this order:
  - `incomplete` if a row is missing;
  - `invalid` if a validity check failed;
  - `fail` if the ratio rule fails or any non-tied pair disagrees;
  - `inconclusive` if every non-tied pair agrees but there are fewer than 6 non-tied pairs in total;
  - otherwise `pass`.
- **Reported, not part of the rule:** the same computation on unclamped renders, and `P / D^test`.
- **Reproduction check:** unchanged; it uses the `lloyd_wopa_area` row at K = 65,536.

### Exploratory item d (replaces the shortlist GN refine)

- **Exact Mahalanobis assignment.** `M_i` is shared by the three channels, so
  `d(i, k) = sum_ch (c_i^ch - q_k^ch)^T M_i (c_i^ch - q_k^ch) = const_i + u_i . v_k`, with 165-dim
  vectors:
  - `u_i = [-2 M_i c_i^R, -2 M_i c_i^G, -2 M_i c_i^B, triu(M_i) with off-diagonals x2]`;
  - `v_k = [q_k^R, q_k^G, q_k^B, triu(sum_ch q_k^ch q_k^ch^T)]`.

  The argmin over k is one chunked fp32 matrix product over splats (no fp16, no TF32), in gsplat's shN
  `[N, 15, 3]` layout. Coordinates are shifted by the codebook mean, which leaves the distances
  unchanged. A splat keeps its current centroid unless the new one is strictly closer by the direct
  formula, and splats with `tr(M_i) = 0` take their L2-nearest centroid.
- **Implementation checks:** a CPU test (N = 2,000, K = 256, random data including rank-deficient
  `M_i`) compares the achieved minimum distances against brute force, allowing ties. On the GPU, the
  job compares lifted against direct (float64) minimum distances on 10,000 real splats with
  `tr(M_i) > 0`: within 1e-4 relative, or the job stops before the refine. The G0 rows are written
  before this step.
- **Diagnostic:** at every assignment step, the share of exact argmins inside the plain-L2 top-64
  shortlist, over all splats and over splats with `tr(M_i) > 0`.
- **Two refine variants.** Both warm-start from the `lloyd_wopa_area` K = 65,536 seed-0 codebook and
  labels, run 3 iterations of assignment then update, and use `mu = 1e-4 * tr(sum M) / 15` per cluster:
  - (a) **ridge to zero:** `q = (sum M + mu I)^-1 sum M c`;
  - (b) **proximal:** `q = (sum M + mu I)^-1 (sum M c + mu q_old)`.

  Both are solved in float64 per channel. A cluster with `tr(sum M) = 0` keeps its centroid.
- **Logged for each variant:** the GN objective on unquantized centroids after every assignment and
  every update step, asserted non-increasing for (b) to 1e-6 relative; the objective after
  `PngCompression`'s centroid quantization (= `P` of its row); bytes, total and per `shN.npz` member;
  and train and test PSNR / SSIM / LPIPS. Both variants are extra rows of the predictivity table
  (`gn_refine_ridge`, `gn_refine_prox`) and are **excluded from G0**.

### Also logged (exploratory)

- Per scene and view set, the per-channel fraction of pixels where the original eval render is below
  0 or above 1 before clamping.
- Train-view PSNR / SSIM / LPIPS for every row, computed the way `Runner.eval` does on the test views.

## Amendment 3 (2026-09-19, before any E0 result and before the code change it describes)

This replaces the G0 verdict of Amendment 2, adds one validity check, and changes three checks of
exploratory item d. The ranking itself (pairs, ties, agreement), the other validity checks, Amendment 1
and G1 are unchanged. Amendments 1 and 2 stay as written above.

**Why:** the GN model is block-diagonal: `M_i` holds only splat i's own terms, and the cross-splat terms
are dropped. Neighbouring splats that share a centroid have correlated residuals, so where they overlap
the cross terms push the measured error above the predicted one, by up to the effective overlap (the
number of splats blending at a pixel). This is strongest at K = 4,096, where the most neighbours share
each centroid. The predicted/measured ratio therefore measures calibration, not ranking ability, and
G0 asks about ranking.

### G0 verdict (replaces the verdict of Amendment 2)

- **Ranking only.** The pair checks are those of Amendment 2: within each K, the 3 config pairs on train
  and test views of both scenes (36 pair checks); a pair is a tie if its clamped measured errors differ
  by less than 5% relative, and ties are exempt; a non-tied pair agrees if `P_a < P_b` exactly when
  `D_a < D_b`, and equal `P` does not agree (it counts as misordered).
- **Verdict**, in this order:
  - `incomplete` if a row is missing;
  - `invalid` if a validity check failed;
  - `fail` if any non-tied pair is misordered;
  - otherwise `inconclusive` if there are fewer than 6 non-tied pairs;
  - otherwise `pass`.
- **Reported, not part of the verdict,** for all 9 codebooks of each scene:
  - `P / D^train`, clamped and unclamped, each flagged **calibrated** when it is within 0.5-2x
    (inclusive); also `P / D^test`, clamped and unclamped, without a flag, because `P` is a train-view
    quantity;
  - `D^train_unclamped / P - 1`, the **cross/diagonal term ratio**;
  - the ranking on unclamped measurements, as in Amendment 2.
- **Degenerate cases** (none expected with real data):
  - an empty validity dict counts as `invalid`: the selftest results were never recorded;
  - a pair with `min(D) = 0` is a tie only if both are 0, because the relative difference is undefined;
    a pair with one zero is non-tied and judged like any other;
  - `D^train = 0` gives a ratio of +inf when `P > 0` (undefined when `P = 0` too), which is not
    calibrated; the cross/diagonal ratio is +inf when `P = 0 < D`, undefined when both are 0. Neither
    affects the verdict.
- **If G0 is inconclusive,** it is re-judged in E1 over the non-GN rungs only: upstream L1
  (`upstream_l1`), `lloyd_w1` (`plain_l2`), `lloyd_wopa_area` and a rung with C3DGS-style weights, with
  the same rule applied to all pairs of those rungs. E1's rungs and K values are written down before the
  E1 run, together with G1's variant.

### Added validity check: end-to-end exactness

- **Toy scene:** 48 splats, one per 16 x 16 pixel cell of an 8 x 6 grid over the 128 x 96 toy view, at
  depth about 3. They are small enough that no pixel receives weight from two splats (checked on the identity
  render), and their colours keep `SH + 0.5` in (0, 1) at every splat and view, for the original and the
  perturbed coefficients. So every rendered value is in [0, 1): covered pixels in (0, 1), uncovered
  pixels exactly 0 in both renders. Two views (the toy camera and a slightly translated one). The scene is
  drawn with a CPU random generator and then moved to the device, so the CPU test and the CUDA check use
  the same scene.
- **Prediction:** exact `s_iv = sum_p w_ip^2` from the identity-feature render (as in the toy check of
  Amendment 1), accumulated into `M_i` by the GN pass's accumulator; `P` from `predicted_dmse`.
- **Measurement:** shN plus a random perturbation, uniform with amplitude 0.1 per coefficient (fixed seed);
  `D` from `measure_dmse` on the SH render (gsplat's `rasterization` with `sh_degree=3` in the notebook).
- **Pass:** `|P - D_unclamped| <= 1e-4 * D_unclamped`, and the preconditions above hold.
- **Reported, not asserted:** an overlapping variant (the same splats with 5x larger scales, so neighbours
  blend) with the same perturbation: its `P / D_unclamped` and cross/diagonal ratio. Also the same
  overlapping scene with one perturbation shared by all splats, the fully correlated case of the "Why"
  above.
- It runs in the notebook's smoke-test cell with the SH and toy checks, and it is a G0 validity check. If
  it fails, the notebook stops before the scene jobs. A CPU test runs it on the brute-force CPU renderer.

### Exploratory item d: changed checks

- **Lifted check on real splats (replaces "within 1e-4 relative" of Amendment 2).**
  - Sample: 10,000 random splats of the scene (seed 0). The criterion is evaluated over the sampled
    splats with `tr(M_i) > 0`.
  - Per splat: `d_min_i` = the exhaustive float64 direct minimum; `excess_i` = the direct distance at the
    lifted fp32 argmin minus `d_min_i`; `m` = the codebook mean that the fp32 shift already uses;
    `scale_i = |c_i - m|^2_{M_i} + tr(M_i) * max_k |q_k - m|^2` (the first term summed over the three
    channels, the second over all 45 coordinates).
  - **Pass** needs both `sum excess / sum d_min <= 1e-4` and `excess_i <= 1e-4 * scale_i` for every
    evaluated splat.
  - **Why:** a criterion relative to `d_min_i` fails on fp32 rounding at near-ties where `d_min_i` is
    close to 0, which rank-deficient `M_i` produce: centroids that differ from `c_i` only in the null space
    of `M_i` are all at distance 0 or nearly so. `scale_i` bounds the magnitudes that enter the fp32
    product, so the per-splat test measures rounding on the scale where it arises. The aggregate test
    bounds the effect on the objective.
  - Logged: the worst per-splat `excess_i / scale_i`, the number of splats with
    `d_min_i < 1e-3 * scale_i`, and the number of `tr(M_i) = 0` splats in the sample.
  - The record carries a criterion version. On resume, a record with another version is re-run.
  - If the check fails, only the refines are skipped. The job goes on (the G0 rows are already written),
    and so does the notebook, to G0 and the bundle.
- **Proximal refine (replaces "asserted non-increasing for (b)").**
  - The logged objective is the per-splat direct-formula distances, summed in float64 (divided by
    `3 * total train pixels`, as `P`).
  - If it rises by more than 1e-6 relative at any step, the rise is logged and the proximal variant's rows
    are marked invalid. The variant still runs its 3 iterations and its row is written with the flag; the
    remaining rows and the bundle follow.
- **Clusters with `tr(sum M) = 0`** (empty, or all members unseen) keep `q_old` in both variants.
- **Still stopping the run early:** the SH, toy and render-parity checks, and now the end-to-end check.

## Amendment 4 (2026-09-20, before any E0 result and before the code change it describes)

A correction of the toy check's input (Amendment 1), the same fix for the end-to-end check's input
(Amendment 3), end-to-end preconditions read from gsplat's own render, and one report-only
diagnostic. No pre-registered test, threshold or verdict changes. Amendments 1-3 stay as written above.
Nothing has run on a GPU.

**Why:**

- `gn_metric.toy_scene` drew its random numbers with `torch.Generator(device=device)`. On CUDA that is
  a different generator from the CPU one, so the CUDA toy check would have rendered a different draw of
  the layout than the scene `bench/gn/toy_noise.py` simulated on the CPU. Amendment 1 says the
  simulated scene is the scene the CUDA check uses; the code did not match it.
- Drawing on the CPU and moving to the GPU is not enough. A fresh draw cannot be reproduced bitwise
  across machines: torch's CPU `randn` (float32, 16 or more values) takes a different code path per
  dispatched CPU capability (an AVX2 SIMD approximation of log/sin/cos under AVX2 dispatch, scalar
  libm calls under DEFAULT or AVX512 dispatch, and libm differs between platforms). On the machine
  that ran the simulation, the same torch and seed with DEFAULT instead of AVX2 dispatch change 5,037
  of the toy scene's 15,104 values in their last bits (max abs difference 1.76e-6), and its hash with
  them. The end-to-end scene (Amendment 3, "drawn with a CPU random generator") has the same problem.
  Kaggle's CPU and torch build are not known in advance.

**Fix: committed scene fixtures.**

- `gn_metric.toy_scene` now draws with the CPU generator and the same seed, then moves the tensors to
  the device, as `e2e_scene` does.
- The scenes the checks use are committed as tensors (float32, little-endian `.npz`), each with a
  metadata file recording the hash, the CPU capability (`torch.backends.cpu.get_cpu_capability()`),
  the torch version and the platform it was drawn under:
  - `bench/gn/fixtures/toy_scene_seed0.npz`: `toy_scene(256, seed=0)`, drawn on the CPU of the machine
    that ran the simulation (AVX2 dispatch, its default; torch 2.11.0+cpu, Windows).
  - `bench/gn/fixtures/e2e_scene_seed0.npz`: the 48 splats of `e2e_scene(seed=0)` and the uniform
    +-0.1 shN perturbation of the end-to-end check, drawn the same way. The overlapping variants are
    derived from it as before (scales x5; the shared perturbation is the first splat's).
- **Scene hash:** SHA-256 over the tensors in sorted key order; for each tensor, the bytes of
  `"<key>:<shape>:float32-le;"` (shape as a Python tuple) followed by its values as float32
  little-endian.
  - toy scene: `1bb442ee09b6d8417384f5aad10e019a3ebab15458141ef685c873d45f06e441`
  - end-to-end scene and perturbation:
    `103c99e07bf582e8def6068efa103994041da1fdda4d9ca91523b178177ac757`
- **Provenance of the toy fixture.** Re-derived with `gn_metric.py` from commit `c69ba388`, the commit of
  `toy_noise.py` and `toy_noise.json` (the probe simulation was not re-run): it gives the same hash as
  the fixture. The two scene-only fields of `toy_noise.json` (blending fraction and visible splats)
  reproduce exactly from it. They reproduce from the DEFAULT-dispatch draw as well, so they cannot
  pin the dispatch; the simulation ran with the machine's default dispatch (AVX2), and no override was
  recorded.
- **The CUDA checks load the fixtures and assert the hashes before rendering,** then move the tensors to
  the GPU. A mismatch stops the whole run.
- **CPU provenance test:** re-draws both scenes. It requires a bitwise match when the fixture's CPU
  capability and torch version both match the machine, and otherwise a max abs difference of at most
  1e-5, reporting the difference in capability.
- **Unchanged:** the toy layout, the probe count (64), the 5% rule and the per-splat plumbing and
  all-ones channel checks (1e-4); the end-to-end tolerance (1e-4 relative) and its perturbation.

**End-to-end preconditions, read from gsplat's own render.** Before the exactness is judged, the check
verifies on the renders it compares:

- no pixel gets nonzero weight from two splats in the identity render;
- `SH + 0.5 > 0`: each splat's rendered colour, recovered as its SH-render pixel value divided by its
  identity-render weight at its strongest pixel, which only it covers, is `max(SH + 0.5, 0)` as gsplat
  computed it. It must lie in (0, 1), the range Amendment 3 states;
- every rendered value at a covered pixel is in (0, 1), and every uncovered pixel is exactly 0 (the
  background), so clamping to [0, 1] changes nothing.

If a precondition fails, the run stops with a message that the exactness claim does not apply to the
scene. That is not reported as a mismatch between prediction and measurement.

**Report-only diagnostic, outside every verdict.** In the toy check, from the exact identity-render
weights `W` (splats x pixels, per view) and before the check is judged: the exact relative standard
deviation of the probe estimate of `S = sum w^2`,

`sigma_rel = sqrt( sum_views 2 (||W W^T||_F^2 - sum_p (sum_i w_ip^2)^2) / n_probes ) / S`

(the variance of a Rademacher quadratic form), and the implied false-fail probability of the 5% rule
under a normal approximation, `erfc(0.05 / (sqrt(2) sigma_rel))`. Both are logged and stored. They do
not enter the toy check, G0 or any other verdict. A CPU test checks the formula against Monte Carlo on
a small `W`.

## Amendment 5 (2026-09-20, before any E1 code; E0 has finished and G0 passed)

E0 is closed: G0 passed, and its numbers are in `kaggle/FINDINGS.md` section 8 from the committed
bundle `kaggle/gn_e0/gn/`. This amendment fixes E1 before any E1 code exists: the GN-VQ variant that
G1 judges, the size-matching procedure that G1's last sentence delegates, the secondary comparisons,
the exploratory rows and what every row logs. **G1's threshold and wording stay exactly as written
above.** G0, the validity checks and Amendments 1-4 are unchanged.

### The codec's shN centroid quantizer, which E1's clip depends on

Read from `gsplat/compression/png_compression.py` (`_compress_kmeans` / `_decompress_kmeans`), the
unchanged path E0 and E1 both write through:

- **one global scalar min/max over the whole `[K, 45]` codebook**, not per dimension:
  `mins = min(centroids) + 1e-6`, `maxs = max(centroids)`;
- **6 bits**, so 64 levels: `round((c - mins) / (maxs - mins) * 63)`, stored as uint8 column-major
  inside `shN.npz`; the step is `(maxs - mins) / 63`;
- dequantization is `c' = q / 63 * (maxs - mins) + mins`.

So one extreme centroid coordinate coarsens the step for all `K * 45` coordinates at once. That is the
mechanism FINDINGS section 8 records as a hypothesis for the E0 refines' loss, and the clip below is
its test.

### a. GN-VQ: the variant G1 judges

Per scene (garden, bicycle), at K = 65,536:

1. **Warm start:** the `lloyd_wopa_area` codebook and labels at K = 65,536, k-means seeds 0, 1 and 2,
   from the run-3 clustering caches. A seed whose cache is missing or whose key does not match is
   reclustered with the same code and seed, and every row records which source it used.
2. **Exact Mahalanobis Lloyd,** iterating:
   - **assignment:** the lifted fp32 assignment of Amendment 2, with its guard (a splat keeps its
     current centroid unless the new one is strictly closer by the direct float64 formula; splats with
     `tr(M_i) = 0` take their L2-nearest centroid);
   - **update:** ridge to zero, `q = (sum M + mu I)^-1 sum M c` with
     `mu = 1e-4 * tr(sum M_k) / 15` per cluster, solved in float64 per channel. A cluster with
     `tr(sum M_k) = 0` keeps `q_old`.
   - **clip, after every update:** every centroid coordinate is clipped to the warm-start codebook's
     range, in the form the codec's quantizer uses, which by the section above is the single global
     interval `[min(C_0), max(C_0)]` of the warm-start codebook `C_0`. A clipped update is kept for a
     cluster **only if it lowers that cluster's own objective** (the sum of the direct Mahalanobis
     distances of its members); otherwise that cluster keeps `q_old`. So the objective cannot rise,
     and the codebook's range cannot grow beyond the warm start's, which keeps the quantizer step no
     coarser than the warm start's.
3. **Stop** when an iteration lowers the objective by less than 1e-3 relative, or after 10 iterations,
   whichever comes first.
4. **Quantize** the centroids with the codec's own quantizer (above), then run **one final exact
   assignment** against the dequantized codebook, with the same guard.
5. **Write** with the unchanged library writer, and assert that re-quantizing the written codebook
   gives the same codes as step 4 used.

### b. Size matching for G1

This is the size-matching procedure that G1's last sentence leaves to be fixed in writing, and it
**replaces the lower bound of G1's parenthetical**. Bytes means the total compressed size as G1 defines
it: the raw bytes of the compressed directory (`size_bytes`). Per scene and seed, GN-VQ against
`lloyd_wopa_area` at the same seed:

- GN-VQ **at most 0.5% larger**: compare test PSNR directly. Being smaller earns no credit.
- GN-VQ **more than 0.5% larger**: that seed counts as **negative**.

**Why the lower bound goes:** a row that is both smaller and better dominates `lloyd_wopa_area`, so
allowing it cannot manufacture a PSNR win; the win has to come from PSNR at no byte cost. Keeping the
bound two-sided would instead score a dominating row as a failure, and E0's refines already came out
1.2-4.4% smaller than `lloyd_wopa_area` (FINDINGS section 8), so this is the likely case, not a corner
one.

**Per-seed dominance flag, reported and never part of any verdict:** GN-VQ bytes <= `lloyd_wopa_area`
bytes **and** GN-VQ test PSNR >= `lloyd_wopa_area` test PSNR, at the same scene and seed.

### c. G1 threshold: unchanged

As written above: at least +0.05 dB mean test PSNR over the 3 k-means seeds, no seed with a negative
PSNR difference, on both scenes. Nothing in this amendment changes that sentence.

### d. Secondary comparisons: reported, not gating

- GN-VQ against **`tr(M)`-weighted Lloyd**, and GN-VQ against **C3DGS-style-weighted Lloyd**: same K,
  same seeds, same size rule as in b. Both weights are the ones E0 correlated against `tr(M)`
  (`gn_spearman_<scene>.csv`); the Lloyd code, the writer and the measurement are unchanged.
- **Rate-distortion curves** for GN-VQ and `lloyd_wopa_area` over K in {4,096, 16,384, 65,536}, seed 0,
  both scenes, with **BD-rate** between the two curves. This exists so that a seed which trades PSNR
  for bytes shows up as a rate-distortion result, not only as a negative G1 seed.

Neither can pass or fail G1.

### e. Exploratory rows: seed 0 only, not judged

- GN-VQ **without the clip** of step a.2.
- GN-VQ **without the final quantized assignment** of step a.4.

### Logged for every E1 row

- predicted `P` and measured `D` on train and test views, clamped and unclamped, as in E0;
- the quantizer's **range and step** actually used by that row's write (`mins`, `maxs`,
  `(maxs - mins) / 63`);
- the **fraction of centroid coordinates outside the warm-start range** before clipping;
- the **objective before and after quantization**, and the objective per iteration with the number of
  clusters whose clipped update was rejected;
- bytes, total and per `shN.npz` member, and test PSNR / SSIM / LPIPS plus train PSNR. Train-view SSIM
  and LPIPS are dropped: no rule uses them.
