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

## Amendment 6 (2026-09-21, before any E1 run and before the code change it describes)

Two exploratory rows are added to E1. Nothing else changes: G1's threshold and wording, the GN-VQ
variant of Amendment 5 a, the size matching of Amendment 5 b, the secondary comparisons of
Amendment 5 d and the exploratory rows of Amendment 5 e all stay exactly as written above. G0, the
validity checks and Amendments 1-4 are unchanged. E1 has not run.

### The rows

Two more runs of the GN-VQ variant defined in Amendment 5 a, differing from it in one number only,
the ridge `eps` of the per-cluster update `mu = eps * tr(sum M_k) / 15`:

- **`gn_vq_eps1e3`**: `eps` = 1e-3;
- **`gn_vq_eps1e2`**: `eps` = 1e-2.

The pre-registered GN-VQ of Amendment 5 a keeps `eps` = 1e-4, the value E0's refines used.

Both rows are at **k-means seed 0 and K = 65,536, on both scenes** (garden and bicycle). Warm start,
assignment, clip, acceptance rule, stopping rule, final quantization, final assignment, writer and
measurement are identical to Amendment 5 a in every other respect.

**They are exploratory and not judged.** They are not part of G1 (Amendment 5 c), not part of either
secondary comparison (Amendment 5 d), and not part of the rate-distortion curves or the BD-rate. They
join the exploratory rows of Amendment 5 e: reported only. G1 remains GN-VQ (`eps` = 1e-4) against
`lloyd_wopa_area` at K = 65,536 over seeds 0-2, and nothing here can pass or fail it.

E1 therefore has 21 rows per scene instead of 19.

### Why the ridge, and why these two values

The ridge is the knob for two risks at once, and E0 measured a symptom of each.

**Overfitting to train-view directions.** `M_i` is accumulated over the train views, so a direction of
shN that no train view constrains is a direction the update is free to move in. Larger `mu` pulls
those directions toward 0 instead. E0 saw the weighting advantage shrink out of sample: on bicycle at
K = 65,536, the measured shN-only dMSE gap between `plain_l2` and `lloyd_wopa_area` was **34.51% on
train views and 23.16% on test views** (`kaggle/gn_e0/gn/gn_results_bicycle.csv`:
`measured_train_clamped` 1.3701e-4 vs 1.0186e-4, `measured_test_clamped` 1.5018e-4 vs 1.2195e-4). A
codebook fitted harder on train-view `M` is not guaranteed to keep its advantage on test views, and
test PSNR is what G1 judges.

**Centroid extrapolation widening the quantizer's global range.** As recorded under Amendment 5, the
codec quantizes the whole `[K, 45]` shN codebook with **one global scalar min/max at 6 bits**, so a
single extreme coordinate coarsens the step for every coordinate. A weakly constrained cluster is
exactly where an update can throw a coordinate far out. Amendment 5 a's clip is one answer to this;
the ridge is the other, and it acts before the clip rather than after, so the two are worth
separating.

1e-3 and 1e-2 are one and two orders of magnitude above the pre-registered 1e-4, which is the range
over which `mu` goes from negligible against a typical cluster's `tr(sum M_k) / 15` to comparable
with it. No result is claimed for them, and no threshold is attached.

### Also logged

For these two rows **and for the pre-registered GN-VQ row**: the **final codebook's own** quantizer
min, max and step - the range the codec's quantizer actually uses when that row is written - recorded
**alongside the warm-start codebook's** min, max and step, so the two can be read side by side
without recomputing either. This is a logging addition only; Amendment 5's "Logged for every E1 row"
list is otherwise unchanged, and nothing reads these fields in a verdict.

## Amendment 7 (2026-09-21, after E1's results and before any E2 code)

E1 has run and its bundle is committed unchanged in `kaggle/gn_e1/gn1/`; its numbers are in
`kaggle/FINDINGS.md` section 9. This amendment records G1's outcome, states the deviation the project
takes because of it, and fixes E2 - its variant, scenes, configs and two verdicts - before any E2 code
exists. G0, G1, the validity checks and Amendments 1-6 are unchanged.

### a. G1 failed as pre-registered

G1 (as originally written, with the size matching of Amendment 5 b) **failed on both scenes, on the size
rule, on all six seeds** (`gn1_g1.json`, verdict `fail`). At K = 65,536 every GN-VQ seed was more than
0.5% larger than `lloyd_wopa_area` at the same seed: +2.37% on garden, +0.98% to +1.01% on bicycle.
Under Amendment 5 b each such seed counts as negative whatever its PSNR, so every seed is negative. The
test PSNR differences were positive on every seed (+0.195 to +0.201 dB garden, +0.087 to +0.091 dB
bicycle) and the means cleared +0.05 dB, but that does not change the verdict.

**G1 is not amended.** Its wording, threshold and size rule stay as written, and its verdict stays
`fail`.

### b. The deviation, stated as one

G1 was the gate for GN-VQ. With G1 failed, the pre-registered plan gives no licence to continue.
**The project continues anyway, and this is a deviation from that plan, decided after E1's results
were known.** It rests on the evidence of the rate-distortion secondary that Amendment 5 d
pre-registered for exactly this case ("so that a seed which trades PSNR for bytes shows up as a
rate-distortion result, not only as a negative G1 seed"): at seed 0, bicycle's BD-rate of GN-VQ against
`lloyd_wopa_area` was -9.84%, and on garden every GN-VQ point was above every `lloyd_wopa_area` point in
PSNR, so no BD-rate exists there and the whole GN-VQ curve lies above the baseline's (FINDINGS section
9).

That evidence is weaker than a gate: it is one seed, three points per curve, and two scenes that
GN-VQ was designed and tuned on. So from here on:

- **garden and bicycle are development scenes.** Everything learned on them, including this
  amendment's choice of variant, is development, and no verdict below counts them;
- **confirmation moves to held-out scenes** that played no part in designing or tuning GN-VQ.

### c. The E2 variant

**E1's GN-VQ (Amendment 5 a) with ridge `eps` = 1e-2 and `max_iters` = 20; everything else exactly as
in Amendment 5 a:** warm start from `lloyd_wopa_area` at the same K and k-means seed; the lifted exact
assignment with its guard; the ridge update `q = (sum M + mu I)^-1 sum M c` with
`mu = eps * tr(sum M_k) / 15` per cluster; the clip to the warm-start codebook's global range with
per-cluster acceptance; the stopping rule (relative drop below 1e-3, or `max_iters`); the codec's
quantizer, the final exact assignment against the dequantized codebook, and the unchanged library
writer with the re-quantization assertion.

Reasons, from E1's files:

- **eps = 1e-2 had the best development trade-off of E1's GN-VQ rows.** The ridge rows are single
  points at K = 65,536 and seed 0, not curves, so this is a comparison of points. Against the
  pre-registered eps = 1e-4 at the same seed, eps = 1e-2 was 1.77% smaller on garden at -0.0061 dB
  and 1.02% smaller on bicycle at +0.0033 dB, the smallest clipped GN-VQ row at K = 65,536 on both
  scenes; on
  bicycle it was also smaller than `lloyd_wopa_area` (-0.01%) at +0.0901 dB. On garden it was still
  +0.55% larger than `lloyd_wopa_area`. The un-clipped ablation was smaller still but gave up
  0.0987 / 0.0341 dB against the clipped variant, and it is not chosen.
- **K <= 16,384 hit the 10-iteration cap:** GN-VQ at K = 4,096 and 16,384 stopped on `max_iters` on
  both scenes, while every K = 65,536 row stopped on the relative-drop rule at iteration 9 or 10. E2's
  grid adds K = 1,024. The relative-drop rule is unchanged, so a curve point that converges earlier
  still stops earlier.
- Not measured: eps = 1e-2 below K = 65,536. E1 ran it only at K = 65,536.

Because this variant was chosen on garden and bicycle after seeing their results, it is judged only
on the held-out scenes.

### d. Scenes and inputs

**Gate set, 9 held-out scenes:** the 7 MipNeRF360 scenes of `mcmc.sh` other than garden and bicycle
(stump, bonsai, counter, kitchen, room, treehill, flowers) and the 2 Tanks & Temples scenes of
`mcmc_tt.sh` (train, truck).

**Development scenes, garden and bicycle:** they also run, queued after all 9 held-out scenes. Their
rows and comparisons are reported and **excluded from every verdict**.

Each scene uses the MCMC 1M checkpoint that runs 4-5 measured, and no other. The E2 job computes the
checkpoint's sha1 and refuses to run on a mismatch. The expected values are the `ckpt_sha1` of the
scene's rows in `kaggle/run5/tilequant/run5_results.csv`:

| Scene | Set | Dataset, data factor | Checkpoint sha1 |
|---|---|---|---|
| stump | held-out | MipNeRF360, 4 | `52715bdb81d53bf3793b4052e6ed656458d74fb3` |
| bonsai | held-out | MipNeRF360, 2 | `60868efab7cfd97dada0ea6badcb8a2700c7d7c8` |
| counter | held-out | MipNeRF360, 2 | `50464c36ef6e1b0c2da3012bc7c8b3feb1a410c2` |
| kitchen | held-out | MipNeRF360, 2 | `8e5e31d4a8d25f458ac55f95a72f442bde2c0f83` |
| room | held-out | MipNeRF360, 2 | `843339232b480503676f8cd4e7dd1cea9ad78749` |
| treehill | held-out | MipNeRF360, 4 | `66fcadcf3298034c06227e69e6381e5a539f6980` |
| flowers | held-out | MipNeRF360, 4 | `d0ea4a76881fb22b0c2848903cb0cce0001b06d5` |
| train | held-out | Tanks & Temples, 1 | `15394ef18333d9edd61facf46874238b45e84de7` |
| truck | held-out | Tanks & Temples, 1 | `4b9c9babe37cf16db99d286be8e6af805a776d6c` |
| garden | development | MipNeRF360, 4 | `e1ac1e31dde161dac584ff107198217b5e90e1db` |
| bicycle | development | MipNeRF360, 4 | `122a280de4da86448d1581e9fb27f90ae6f71b8c` |

Data factors are those of `mcmc.sh` and `mcmc_tt.sh`, as in runs 4-5; test views are the runner's
evaluation split, as in every earlier run. Each scene uses its cached seed-0 PLAS sort order (runs 1-2
for garden and bicycle, run 4 for the other MipNeRF360 scenes, run 5 for Tanks & Temples); a missing
order stops the run rather than being rebuilt. The GN metric `M` is computed from each scene's train
views with E0's code and probe seed 0; garden and bicycle may reuse E0's cache, whose key pins the
checkpoint, the render settings and the views.

### e. Configs

Per scene, each config at **K in {1,024, 4,096, 16,384, 65,536}, k-means seed 0 only**:

- `upstream_l1`: TorchPQ manhattan k-means, 100 iterations, the upstream default path (E0's config);
- `lloyd_wopa_area`: the library's weighted Lloyd with opacity x footprint-area weights (runs 3-5,
  E0, E1);
- `lloyd_trace`: the library's weighted Lloyd with `tr(M_i)` weights (E1's secondary);
- `gn_vq`: the variant of c, warm-started from `lloyd_wopa_area` at the same K;

plus one `uncompressed` row: 17 rows per scene. Every codebook is written with the unchanged library
writer and measured with the full compressed pipeline on the test views, as in E0 and E1.

A clustering at K = 65,536 may come from a run-3 cache (garden, bicycle) or a run-4 cache (the other
MipNeRF360 scenes) when its key, the checkpoint sha1 with the sort order, matches; otherwise it is
recomputed with the same code, and every row records its source. Run 5 found the library's Lloyd and
the benchmark's Lloyd rows identical on all 18 rows it compared. TorchPQ reproduces only to about
0.002 dB.

**Why seed 0 only:** in E1 the paired G1 difference moved by at most 0.0058 dB across seeds 0-2
(garden; 0.0040 dB on bicycle). The per-config PSNR spread was larger for some configs: up to
0.0087 dB for `lloyd_trace` on garden and 0.0075 dB for GN-VQ on bicycle. The verdicts below compare
whole curves on nine scenes, not single seeds, and the per-scene thresholds are signs, not margins.

### f. G2a (the gate): GN-VQ against `lloyd_wopa_area` on the held-out scenes

**Curves.** Per scene and config, the four points `(size_bytes, PSNR)` at K = 1,024, 4,096, 16,384 and
65,536, seed 0: raw bytes of the compressed directory and test PSNR of the full compressed pipeline.

**BD-rate** of a curve `new` against a curve `ref` (Bjontegaard): fit `log10(bytes)` of each curve as a
polynomial of degree 3 in PSNR (with four points, the exact interpolating cubic); integrate both over
the common PSNR interval `[lo, hi]`, `lo` = the larger of the two lowest PSNRs and `hi` = the smaller of
the two highest; BD-rate = `(10^((I_new - I_ref) / (hi - lo)) - 1) x 100%`. It is **defined** only if
`hi > lo` and the result is finite. (This is `g1.bd_rate` at degree 3; E1 reported degree 2 on three
points, the same exact-interpolation construction.)

**BD-PSNR** of `new` against `ref`: fit PSNR of each curve as a polynomial of degree 3 in
`log10(bytes)`; integrate both over the common `log10(bytes)` interval; BD-PSNR =
`(J_new - J_ref) / (hi - lo)` in dB. It is **defined** only if that interval has positive length and
the result is finite.

**Per scene,** GN-VQ against `lloyd_wopa_area`:

1. if any of the scene's 8 rows (4 K x 2 configs) is missing, the scene is **incomplete**;
2. else, if the BD-rate is defined, the scene **wins** if the BD-rate is below 0;
3. else (no PSNR overlap), if the BD-PSNR over the overlapping byte range is defined, the scene
   **wins** if the BD-PSNR is above 0;
4. else (neither defined), the scene counts as a **loss**.

**G2a passes if at least 8 of the 9 held-out scenes win AND the mean BD-rate, over the held-out scenes
where it is defined, is at most -5%.** If no held-out scene has a defined BD-rate, the mean does not
exist and that condition is **not met**. The verdict is `incomplete` if any held-out scene is
incomplete; otherwise `pass` or `fail`. Garden and bicycle are never read.

### g. H2b (reported, with its own verdict, not gating): the matrix against the scalar

The same per-scene rule, with the same fallbacks, for GN-VQ against `lloyd_trace`: **H2b holds if the
scene wins on at least 7 of the 9 held-out scenes.** It has no mean condition. Its verdict is
`incomplete`, `pass` or `fail`, reported next to G2a's; it does not affect G2a. This is the claim that
the per-splat matrix `M_i` does better than the scalar `tr(M_i)` weight alone, which E1 left open: a
post-hoc look at E1 (FINDINGS section 9) had `lloyd_trace` beating `lloyd_wopa_area` at about equal
bytes on all six seeds.

### h. Also reported, never part of a verdict

- BD-rate and BD-PSNR of GN-VQ against `upstream_l1`, and against each baseline even where the verdict
  did not need BD-PSNR;
- at every K, GN-VQ's PSNR difference and byte ratio against each of the three baselines, and the
  dominance flag (bytes `<=` and PSNR `>=`);
- every comparison above for garden and bicycle, labelled as development;
- per row, the same logging as E1 (Amendments 5 and 6): `P` and measured `D` on train and test views,
  the written codebook's quantizer range and step and the warm start's, the fraction of coordinates
  outside the warm range, the objective before and after quantization and per iteration, the
  iterations and the stopping reason, clusters rejected by the clip, bytes total and per `shN.npz`
  member, test PSNR / SSIM / LPIPS and train PSNR.

### i. Checks that stop or skip, as in E1

The CUDA smoke tests, per-scene render parity and the writer-code assertion stop the run (or the
scene) as in E1. A failed lifted-assignment check skips that scene's GN-VQ rows, which leaves the
scene, and so G2a and H2b, `incomplete`. G2 has no validity dict, for the reason G1 has none: its
checks stop the run instead of qualifying the verdict.

## Amendment 8 (2026-09-21, before any E2 data existed and before the code change it describes)

E2 has not run: no E2 row, bundle or log exists anywhere. This amendment changes one clause of G2a
(Amendment 7 f), the mean condition, so that the mean is always defined, and it adds exploratory rows
on the development scenes. Everything else in Amendment 7 stands as written: the variant, the scenes
and pinned checkpoints, the configs, the per-scene win rule with its BD-PSNR fallback and its loss
case, the count of at least 8 wins out of 9, the -5% threshold, H2b's rule, the reported extras and
the stopping checks. G0, G1 and Amendments 1-7 are unchanged.

### a. G2a's mean condition (replaces the mean clause of Amendment 7 f)

Amendment 7 f read: "the mean BD-rate, over the held-out scenes where it is defined, is at most -5%.
If no held-out scene has a defined BD-rate, the mean does not exist and that condition is not met."
That clause is replaced by:

**The mean is taken over all 9 held-out scenes.** Each scene enters with one term:

- a scene **with a defined BD-rate** (GN-VQ against `lloyd_wopa_area`, as Amendment 7 f defines it)
  enters with that BD-rate;
- a scene **without one** enters with a substitute. A BD-rate is undefined when the two curves share
  no PSNR range, that is, when one curve lies entirely above the other; a non-finite BD-rate is treated
  the same way. The substitute is the first of these that applies:
  - **a.** if some `gn_vq` point has PSNR >= the baseline's best PSNR at fewer bytes than the
    baseline's best point: **-(1 - bytes_gn / bytes_base) x 100%**, with `bytes_base` the bytes of the
    baseline's best point and `bytes_gn` the bytes of the cheapest such `gn_vq` point;
  - **b.** if some baseline point has PSNR >= `gn_vq`'s best PSNR at fewer bytes than `gn_vq`'s best
    point: **+(bytes_gn / bytes_base - 1) x 100%**, with `bytes_gn` the bytes of `gn_vq`'s best point
    and `bytes_base` the bytes of the cheapest such baseline point;
  - **c.** otherwise: **0%**.

Definitions: a curve's **best point** is its point with the highest PSNR (a tie, not expected with
measured PSNRs, goes to the point with fewer bytes); "fewer bytes" is strict; the **cheapest** point
is the one with the fewest bytes. Points are the four `(size_bytes, PSNR)` rows at K = 1,024, 4,096,
16,384 and 65,536, seed 0, as in Amendment 7 f.

**G2a passes if at least 8 of the 9 held-out scenes win AND this mean is at most -5%.** The win rule
is unchanged: a scene's outcome still comes from its BD-rate, else its BD-PSNR, else it is a loss. The
substitutes feed only the mean. As before, a missing row makes its scene incomplete and the verdict
`incomplete`; no mean is computed then.

**Why:**

- As Amendment 7 had it, a held-out set on which every scene looked like garden in E1 would have
  failed G2a with 9 of 9 wins: no scene would have had a defined BD-rate, so the mean condition could
  not be met.
- Neither substitute extrapolates. Each compares two measured points, the lower curve's best point and
  the cheapest point of the higher curve that reaches its PSNR, and is the smallest magnitude of rate
  difference those points guarantee at that quality: in a, `gn_vq` is measured to reach the
  baseline's best PSNR at `bytes_gn`, where the baseline needed `bytes_base`; in b, the reverse. No
  curve is fitted or extended.
- With E1's garden points (`kaggle/gn_e1/gn1/gn1_g1.json`), substitute a gives **-9.77%**: `gn_vq` at
  K = 4,096 (14,803,113 bytes, 27.0065 dB) reaches `lloyd_wopa_area`'s best (K = 65,536, 16,405,132
  bytes, 26.9631 dB). E1's bicycle points have a defined BD-rate (-9.84%) and would enter with it.

**Same construction elsewhere.** H2b's verdict has no mean condition and is unchanged, but its
reported mean BD-rate (GN-VQ against `lloyd_trace` over the 9 held-out scenes) uses the same terms and
substitutes. The `upstream_l1` comparison is reported per scene and has no mean; its per-scene
substitute is reported next to its BD-rate and BD-PSNR, and nothing aggregates it.

### b. Exploratory rows on the development scenes, not judged

Amendment 7 c chose eps = 1e-2 from single points at K = 65,536. To give that choice a curve on the
development scenes:

- **`gn_vq_eps1e4`:** E2's GN-VQ variant (Amendment 7 c) with ridge eps = 1e-4, E1's pre-registered
  value, and everything else identical: `max_iters` = 20, warm start from `lloyd_wopa_area` at the same
  K, seed 0, the clip, the final quantized assignment and the writer check;
- at K = 1,024, 4,096, 16,384 and 65,536, **on garden and bicycle only**: 8 rows;
- **queued after everything else:** they run in a final phase of the notebook, after every E2 scene
  job has finished, under the same start cutoff; they never delay a pre-registered row.

They are **not judged**: G2a and H2b never read them, and neither does any reported comparison of the
held-out scenes. Reported only, for garden and bicycle: their curve against `lloyd_wopa_area`, and
`gn_vq` (eps 1e-2) against them, each with BD-rate, BD-PSNR and the substitute above. Nothing learned
from them can change E2's variant or verdicts.

## Amendment 9 (2026-09-22, after E2's results and before any E2b code)

E2 has run. Its bundle is committed unchanged in `kaggle/gn_e2/gn2/` (`9287eb47`), a post-hoc
sensitivity analysis of its BD measures in `kaggle/gn_e2/bd_sensitivity.json` (`4d143c4b`), and its
numbers are in `kaggle/FINDINGS.md` section 10 (`09bb2f69`). **G2a passed as pre-registered and is not
amended.** H2b passed as computed; FINDINGS section 10 counts it as 8 of 9, because treehill's win is an
artifact of the cubic fit. This amendment changes how BD measures are computed from now on (a), and
fixes E2b (b), an exploratory run with a pre-stated success criterion and no gate. G0, G1, G2a, H2b, the
validity checks and Amendments 1-8 are unchanged.

### a. BD-rate and BD-PSNR from E2b on: the same quantity, computed better conditioned

- **The pre-registered quantity is unchanged:** Amendment 7 f's BD-rate and BD-PSNR, the degree-3
  polynomial through each curve's four points (degree n - 1 for n < 4 points), integrated over the
  common interval.
- **Only its computation changes.** `g2.bd_rate` / `g2.bd_psnr` fit with `np.polyfit` on the uncentred
  axes (E2's PSNRs are 21.59-32.31 dB, its log10 bytes 7.13-7.23), which is badly conditioned: on E2's
  39 comparisons the bundle's BD-rates are up to 6.09e-4 percentage points from the exact interpolating cubic
  (computed in 50-digit arithmetic), a recomputation on another machine up to 1.16e-3 pp, and the two
  differ by up to 1.77e-3 pp (`bd_sensitivity.json`). From E2b on, every BD-rate and BD-PSNR is
  computed with the domain-scaled fit, `numpy.polynomial.Polynomial.fit`, which maps the abscissa onto
  [-1, 1] before fitting; in code, `g2.bd_rate_scaled` and `g2.bd_psnr_scaled`. A test checks both
  against the 50-digit exact cubic on all of E2's curves to 1e-8.
- **E2's recorded values stand.** `gn2_g2.json` is the verdict of record; `g2.bd_rate`, `g2.bd_psnr` and
  `g2.judge_e2` are not changed.
- E2b's curves have two K each and its criterion uses no BD measure, so this applies to any BD value
  computed later, including any re-analysis.

### b. E2b: an isotropic floor on the GN metric (exploratory, not gated)

**Why.** E2's ridge regularizes only the centroid update: `mu = eps * tr(sum M_k) / 15` pulls a solved
centroid toward 0 in the directions its cluster's summed metric does not constrain. The assignment uses
the pure `M_i`. A direction in the null space of `M_i` costs nothing there, so a splat can be assigned a
centroid whose shN differs arbitrarily in directions its training cameras do not cover, and test cameras
can see those directions. The symptom in E2 is a train-view gain that does not transfer to test views:
at K = 65,536, GN-VQ's test PSNR gain over `lloyd_trace` divided by its train PSNR gain is lowest on
**treehill (-1.06), flowers (0.32) and train (0.47)**, against 0.52-0.90 on the other eight scenes
(FINDINGS section 10, post hoc). `lloyd_trace` uses the same `M`, reduced to its trace, so this compares
the full matrix with its scalar part on the same views. A floor `rho * tr(M_i) / 15 * I` (`tr(M_i) / 15`
is the mean eigenvalue of `M_i`) makes every direction cost at least `rho` times that mean, in the
assignment as well as the update.

**Scenes.** Treehill, flowers and train, the three lowest ratios above; **garden is the control**, a
development scene where GN-VQ's test gain transferred (ratio 0.72). Stump is not used. **From this
amendment on, treehill, flowers and train are development scenes**, like garden and bicycle: E2's
verdicts, which counted them as held out, stand, and no later held-out claim may use them. Each scene
uses its Amendment 7 checkpoint (sha1 pinned there; the job refuses any other), its data factor (4 for
treehill, flowers and garden; 1 for train) and its seed-0 PLAS order.

**b.a. Variant.** E2's GN-VQ (Amendment 7 c: ridge eps = 1e-2, at most 20 iterations, the 1e-3
relative-drop stopping rule, the clip to the warm-start range with per-cluster acceptance, the codec's
quantizer, the final exact assignment against the dequantized codebook, the writer-code assertion), with
`M_i` replaced everywhere inside GN-VQ by the floored metric

    M'_i = M_i + rho * tr(M_i) / 15 * I

- in the assignment (the lifted argmin and its guard) **and** in the update, whose ridge then uses
  `tr(sum M'_k) = (1 + rho) tr(sum M_k)`; also in the clip's per-cluster acceptance, the stopping rule's
  objective and the final quantized assignment. Splats with `tr(M_i) = 0` keep a zero metric and take
  their L2-nearest centroid, as before;
- `rho` in {0, 1e-3, 1e-2, 1e-1}. At `rho = 0` the floored metric is `M_i` exactly, and the variant is
  E2's `gn_vq`;
- warm start: `lloyd_wopa_area` at the same K and seed 0, the codebook E2 used, restored from **E2's
  notebook output**: `gn2_work/<scene>/clusters/lloyd_wopa_area_k<K>_s0.pt`, and for K = 65,536 on
  treehill, flowers and garden the run-3 / run-4 cache E2 copied to `tilequant/e2_kmeans/<scene>/`.
  Fallbacks, in order: the run-3 / run-4 cache in the run-5 output (K = 65,536, MipNeRF360 scenes);
  otherwise reclustered with E2's code and seed. Every row records its warm start's source;
- also logged for every row: the GN objective on the unfloored `M_i` (in the units of `P`) before and
  after quantization, next to the floored objective GN-VQ minimizes.

**b.b. Training-view cross-validation (selects `rho`).**

- The train views are E2's: the runner's train split, in the order E0's `camera_views` lists them,
  indexed 0, 1, 2, ... **`M_even`** is E0's GN pass (probe seed 0) over the even-indexed train views
  only (0, 2, 4, ...), with those views' pixels as the objective's total pixels.
- For each `rho`, GN-VQ with the floored metric built from `M_even`, from the warm start above. The
  codebook is written and decoded with the unchanged library writer, and **scored by the
  render-vs-render dMSE of the decoded shN on the odd-indexed train views** (1, 3, 5, ...): only shN
  swapped, clamped renders, E0's `measure_dmse`, as E2's measured `D`.
- **`rho_cv`** (per scene and K) is the `rho` with the lowest score; an exact tie goes to the smaller
  `rho`. **No test view is used for the selection.**
- Reported for these codebooks and not used by the selection: the same dMSE on the test views, bytes,
  and `P` on `M_even`.

**b.c. The full-`M` check.** For each `rho` > 0, GN-VQ with the floored metric built from E2's
full-train-view `M` (`gn_cache/<scene>.pt` from E2's notebook output, used only if its cache key
matches; otherwise recomputed with E0's code, which the row records), then written, decoded and
evaluated as E2's rows: `P`, measured `D` on train and test views, test PSNR / SSIM / LPIPS of the full
compressed pipeline, train PSNR, bytes. **The `rho = 0` full-`M` codebook is E2's `gn_vq` row itself**,
read from `kaggle/gn_e2/gn2/` and not recomputed. **`rho_cv`'s codebook** is the full-`M` codebook at
`rho_cv` (E2's `gn_vq` row when `rho_cv = 0`).

**b.d. Grid.** K in {4,096, 65,536}, k-means seed 0, the four scenes: per scene and K, 4
cross-validation codebooks and 3 full-`M` codebooks, 14 GN-VQ rows per scene. The comparators are E2's
committed `lloyd_wopa_area`, `lloyd_trace` and `gn_vq` rows at the same scene and K; nothing E2 measured
is measured again.

**b.e. Success criteria, stated in advance.** Test PSNR is the full compressed pipeline's (E2's `PSNR`
column). **The floor works if both hold:**

1. on **treehill**, at K = 4,096 **and** at K = 65,536, `rho_cv`'s codebook has a higher test PSNR than
   E2's `lloyd_trace` row at the same K; **and**
2. on **garden**, at K = 4,096 **and** at K = 65,536, `rho_cv`'s codebook's test PSNR is at least E2's
   `gn_vq` (`rho = 0`) test PSNR at the same K minus 0.02 dB.

Differences are rounded to 9 decimals before they are compared, as in run 4. The result is `works`,
`does not work`, or `incomplete` if a row either condition needs is missing. Flowers and train are
reported with the same quantities and enter neither condition. This is E2b's own criterion: it is not a
gate, and it changes nothing about G2a, H2b or any E2 row.

For reference, what the criterion is up against, from E2's rows: on treehill, `lloyd_trace` has 23.2465
dB at K = 4,096 and 23.2746 dB at K = 65,536, against E2's `gn_vq` at 23.1988 and 23.2302 dB. One caveat
known before E2b: the selection scores dMSE against the uncompressed render, while the criterion is PSNR
against the ground truth, and the two can disagree. On treehill at K = 4,096, E2's `gn_vq` has the lower
test dMSE (1.8965e-4 against `lloyd_trace`'s 2.2533e-4) and the lower test PSNR.

**b.f. Reported, never part of the criterion.**

- Per scene and K, the **Spearman rank correlation across the four `rho` values between the
  cross-validation codebooks' odd-train-view dMSE and the full-`M` codebooks' test dMSE** (at `rho = 0`,
  E2's `gn_vq` row's `measured_test_clamped`), with average ranks for ties (`diagnostics.spearman`).
  With four values it takes only a few values; no threshold is attached.
- The same Spearman with the cross-validation codebooks' own test dMSE.
- Per scene, K and `rho`: bytes, test PSNR, dMSE, iterations, stopping reason, clip rejections and both
  objectives; `rho_cv` and its codebook's test PSNR and bytes against E2's `lloyd_wopa_area`,
  `lloyd_trace` and `gn_vq` rows.

**b.g. Checks, as in E2.** The checkpoint sha1 pins before any install and again in each job; the CUDA
smoke tests; render parity per scene; the writer-code assertion on every row. The lifted-assignment
check (Amendment 3's criterion, 10,000 splats, on the K = 65,536 warm start) runs once per scene for
every metric the scene's GN-VQ rows use: the floored `M_even` at each `rho`, and the floored full `M` at
each `rho` > 0 (E2 checked the unfloored full `M`). A failed check skips the rows using that metric,
which leaves them missing. E2b writes its own files (`gn2b/`) and never E2's.

## Amendment 10 (2026-09-25, before any E2b data exists)

E2b has not run: no E2b row, bundle or log exists anywhere. This amendment changes E2b's scenes, adds a
fidelity criterion judged on the quantity `rho` is selected on, keeps Amendment 9's PSNR criteria as
reported items, and makes E2b run its own `rho = 0` full-`M` codebook to check that it reproduces E2's.
Everything else in Amendment 9 b stands: the variant, the `rho` grid, the cross-validation, the grid of K,
the warm starts, the lifted checks. Amendment 9 a (the BD computation), G0, G1, G2a, H2b and Amendments
1-8 are unchanged.

### a. Why: the measure that picked Amendment 9's scenes

Amendment 9 picked its scenes by the ratio of GN-VQ's test to train **PSNR gain** over `lloyd_trace`.
PSNR is measured against the ground truth, so a change in it also contains the cross term between the
quantization error and the uncompressed model's own error, `2 <I_q - I_orig, I_orig - I_gt>`, which can
have either sign. The render-vs-render error against the uncompressed model (the measured shN-only dMSE,
E2's `measured_*_clamped` columns) leaves that term out: it measures how faithfully a codebook
reproduces the model, which is what GN-VQ optimizes and what E2b's cross-validation scores.

Measured that way, from E2's rows (`kaggle/gn_e2/gn2/gn2_results_<scene>.csv`), as GN-VQ's clamped dMSE
divided by `lloyd_trace`'s, train views then test views:

| Scene | K = 4,096 | K = 65,536 |
|---|---|---|
| treehill | 0.3559 -> 0.8417 | 0.3684 -> 1.1253 |
| stump | 0.4575 -> 0.7458 | 0.3810 -> 0.6797 |
| flowers | 0.4911 -> 0.7058 | 0.4143 -> 0.6611 |
| room | 0.2919 -> 0.3701 | 0.3150 -> 0.5115 |
| train | 0.3625 -> 0.4239 | 0.3526 -> 0.4177 |
| truck | 0.3359 -> 0.3679 | 0.3015 -> 0.3316 |
| counter | 0.5205 -> 0.5375 | 0.4585 -> 0.4802 |
| bonsai | 0.3857 -> 0.3958 | 0.3736 -> 0.3859 |
| kitchen | 0.5228 -> 0.5293 | 0.4857 -> 0.4938 |
| garden (development) | 0.4445 -> 0.4665 | 0.3828 -> 0.4085 |
| bicycle (development) | 0.4561 -> 0.6569 | 0.4336 -> 0.6326 |

- On most scenes the test ratio is close to the train ratio. The largest increase from train to test
  (test ratio minus train ratio) among the held-out scenes is on **treehill, stump and flowers, at both
  K** (+0.4858, +0.2883, +0.2147 at K = 4,096; +0.7568, +0.2986, +0.2468 at K = 65,536). That difference is
  the measure this amendment uses. Measured as a quotient (test ratio over train ratio) the same three
  lead at K = 4,096, but at K = 65,536 room (1.624) ranks above flowers (1.596).
- Train, which Amendment 9 chose, has a small increase (+0.0614 and +0.0651). **Amendment 9's switch to
  train was based on the PSNR ratio, which by this analysis is the wrong measure of generalization.**
- The cross term is visible in E2's rows: on treehill at K = 4,096, GN-VQ's test dMSE is 0.8417 of
  `lloyd_trace`'s, more faithful to the model, yet its test PSNR is 0.0477 dB lower.

### b. Scenes (replaces Amendment 9 b, "Scenes")

**Treehill, flowers and stump**, the three held-out scenes where the dMSE ratio grows most from train
to test; **garden stays the control**. **Train is dropped from E2b.** Stump uses its Amendment 7
checkpoint (`52715bdb81d53bf3793b4052e6ed656458d74fb3`), data factor 4 and seed-0 PLAS order. From this
amendment on, treehill, flowers and stump are development scenes. Train stays one, as Amendment 9
declared: that choice was made from E2's results, and dropping train from E2b does not undo it.

### c. E2b runs its own `rho = 0` full-`M` codebook (changes Amendment 9 b.c and b.d)

For every scene and K, E2b also runs GN-VQ at `rho = 0` with E2's full `M` and evaluates it like the other
full-`M` rows, so each scene has 16 GN-VQ rows (per K, 4 cross-validation and 4 full-`M` codebooks), and
the lifted check also runs on the unfloored full `M` (8 per scene).

- **Reproduction check.** Per scene and K, this row against E2's committed `gn_vq` row: the difference in
  test PSNR, the relative difference in test dMSE (`measured_test_clamped`) and the difference in bytes.
  Status: `identical` if all three are exactly equal; `within_tolerance` if |dPSNR| <= 1e-3 dB and the
  relative dMSE difference is at most 1e-3; otherwise `not_reproduced`, which is **flagged**. The check
  applies when the row used E2's `M` (`m_source` = the restored cache) and E2's warm start (E2's own
  clustering caches); otherwise the status is `inputs_differ`, with the differences still reported.
  Nothing is gated on it.
- **Which `rho = 0` row each item uses.** Amendment 9's items stay exactly as Amendment 9 wrote them: its
  PSNR criteria and its Spearman correlation take E2's `gn_vq` row as the `rho = 0` full-`M` codebook.
  This amendment's items (d, e) take **E2b's own** `rho = 0` row, which shares the realization of `M`
  and the run with the other E2b rows. The Spearman correlation is also reported with E2b's own
  `rho = 0` row.

### d. Fidelity criterion (new), judged on test dMSE, the quantity `rho` is selected on

Per scene and K, **R = test dMSE of the full-`M` codebook at `rho_cv` / test dMSE of E2's committed
`lloyd_trace` row** (both `measured_test_clamped`). `rho_cv` is Amendment 9 b.b's; at `rho_cv = 0` the
codebook is E2b's own `rho = 0` row. **The floor works on fidelity if both hold:**

1. R is below E2b's own `rho = 0` value of R in **at least 5 of the 6** (scene, K) cells of treehill,
   flowers and stump; **and**
2. **treehill's R at K = 65,536 is below 1.**

**Garden control, reported with it:** at K = 4,096 and at K = 65,536, garden's test dMSE at `rho_cv` is
at most 5% above its own `rho = 0` row's.

Comparisons are made on differences rounded to 9 decimals ("below" is strict, "at most" inclusive). A
missing row makes the result `incomplete`. With `rho_cv = 0` a cell is not below its own `rho = 0`
value.

### e. Amendment 9's PSNR criteria, reported alongside

Amendment 9 b.e's two conditions (treehill against `lloyd_trace`, garden within 0.02 dB) are computed
exactly as written and reported next to the fidelity criterion, with the cross-term caveat of (a): a
change in test PSNR mixes fidelity to the model with the model's own error. **Neither criterion gates
anything**; E2b stays exploratory, and E2's verdicts are unchanged.

### f. Note (2026-09-26, before any E2b data exists)

Added to Amendment 10 before any E2b row, bundle or log exists:

Any claim that the floor works requires both the fidelity criterion and the garden control to hold. The
verdict fields stay as implemented. This governs how FINDINGS and any write-up describe the result. If
the garden row is missing, no such claim is made.

## Amendment 11 (2026-09-26, after E2b's results and before any E2c code)

E2b has run. Its bundle is committed unchanged in `kaggle/gn_e2b/gn2b/` (`a3c0099a`), and its numbers are
in `kaggle/FINDINGS.md` section 11 (`cc2c3b95`, re-checked by `bench/gn/check_s11.py`). Under Amendment
10 f the floor works in fidelity terms, and not by the PSNR criterion. This amendment pre-registers
**E2c, a gated run** of the floor with the strength chosen per scene and K by cross-validation, on the
five scenes no decision has used yet. G0, G1, G2a, H2b, E2b's criteria and Amendments 1-10 are unchanged.

### a. Scenes

**Bonsai, counter, kitchen, room and truck, all five gate scenes.** They are the only scenes with a
pinned checkpoint that no tuning decision has used: garden and bicycle became development scenes in
Amendment 7, treehill, flowers and train in Amendment 9, and stump in Amendment 10. Each scene uses its
Amendment 7 d checkpoint (the job refuses any other sha1), its data factor (2 for the four MipNeRF360
scenes, 1 for truck) and its cached seed-0 PLAS order.

| Scene | Dataset, data factor | Checkpoint sha1 (Amendment 7 d) |
|---|---|---|
| bonsai | MipNeRF360, 2 | `60868efab7cfd97dada0ea6badcb8a2700c7d7c8` |
| counter | MipNeRF360, 2 | `50464c36ef6e1b0c2da3012bc7c8b3feb1a410c2` |
| kitchen | MipNeRF360, 2 | `8e5e31d4a8d25f458ac55f95a72f442bde2c0f83` |
| room | MipNeRF360, 2 | `843339232b480503676f8cd4e7dd1cea9ad78749` |
| truck | Tanks & Temples, 1 | `4b9c9babe37cf16db99d286be8e6af805a776d6c` |

### b. Method: `gn_vq_cvfloor`

- **The variant:** E2's GN-VQ (Amendment 7 c: ridge eps = 1e-2, at most 20 iterations, the 1e-3
  relative-drop stopping rule, the clip to the warm-start range with per-cluster acceptance, the codec's
  quantizer, the final exact assignment against the dequantized codebook, the writer-code assertion),
  with `M_i` replaced everywhere inside GN-VQ by the floored metric of Amendment 9 b.a,

      M'_i = M_i + rho * tr(M_i) / 15 * I

  in the assignment and in the update (whose ridge then uses `tr(sum M'_k)`), the clip's acceptance, the
  stopping objective and the final quantized assignment. Splats with `tr(M_i) = 0` keep a zero metric
  and take their L2-nearest centroid. At `rho = 0` the variant is E2's `gn_vq`.
- **Selection by training-view cross-validation, as E2b defined it (Amendment 9 b.b):** the train views
  are the runner's train split in the order E0's `camera_views` lists them. **`M_even`** is E0's GN pass,
  probe seed 0, over the even-indexed train views only, with its own probe draws (not a sub-sum of E2's
  `M`) and those views' pixels as the objective's pixels. For each `rho` of the grid, GN-VQ runs with the
  floored `M_even` from the warm start (c). The codebook is written and decoded with the unchanged library
  writer and **scored by the render-vs-render dMSE of the decoded shN on the odd-indexed train views**
  (only shN swapped, clamped renders, E0's `measure_dmse`).
- **Grid: `rho` in {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3}.** **`rho_cv`**, per scene and K, is the `rho` with
  the lowest odd-view score; an exact tie goes to the smaller `rho`. No test view enters the selection.
- **The final codebook** (config `gn_vq_cvfloor`, one per scene and K): GN-VQ with the floored metric
  built from **E2's full-train-view `M`**, restored from E2's `gn_cache` (used only if its cache key
  matches), at `rho_cv`, from the same warm start. It is written, decoded and evaluated like E2's rows:
  `P`, measured `D` on train and test views, test PSNR / SSIM / LPIPS of the full compressed pipeline,
  train PSNR, bytes. It is run at every `rho_cv`, 0 included. When `rho_cv = 0`, it is E2's `gn_vq` run
  again, and its reproduction of E2's committed row is reported, as in Amendment 10 c: `identical`,
  `within_tolerance` (|dPSNR| <= 1e-3 dB and relative test-dMSE difference <= 1e-3) or `not_reproduced`,
  flagged. G2c uses E2c's own row either way.
- The cross-validation codebooks (config `gn_vq_cvfloor_cv`) get the odd-view dMSE, the render-vs-render
  test dMSE (reported, never used by the selection), `P` on `M_even` and the bytes, and not the full
  evaluation.

### c. Grid, comparators and inputs

- **K in {1,024, 4,096, 16,384, 65,536}, k-means seed 0.** Per scene: 7 cross-validation codebooks and
  1 final codebook per K, 32 GN-VQ runs.
- **Comparators:** E2's committed rows (`kaggle/gn_e2/gn2/gn2_results_<scene>.csv`) for
  `lloyd_wopa_area`, `lloyd_trace`, `upstream_l1` and `gn_vq`, at the same K, seed 0 and checkpoints.
  Nothing E2 measured is measured again.
- **Warm start:** `lloyd_wopa_area` at the same K, seed 0, the codebook E2 warm-started its `gn_vq` from,
  restored from **E2's notebook output**: `gn2_work/<scene>/clusters/lloyd_wopa_area_k<K>_s0.pt`, and
  for K = 65,536 on the four MipNeRF360 scenes the run-4 cache E2 copied to
  `tilequant/e2_kmeans/<scene>/lloyd_wopa_area_s0.pt` (truck's K = 65,536 codebook was clustered by E2
  and is in `gn2_work/`). **There is no fallback:** a missing or key-mismatched warm start or `M` stops
  the scene before any GN-VQ run. The run must have both E2's notebook output and the run-5 notebook
  output attached, and the notebook stops before any install if either is missing.

### d. G2c (the gate)

All BD measures use the domain-scaled fit of Amendment 9 a (`g2.bd_rate_scaled`, `g2.bd_psnr_scaled`),
the degree-3 Bjontegaard quantity of Amendment 7 f over the four points per curve (K = 1,024-65,536):
raw bytes of the compressed directory against test PSNR of the full compressed pipeline.

**G2c passes if all three hold:**

1. **Against `lloyd_trace`, every scene wins:** the BD-rate of `gn_vq_cvfloor` against E2's `lloyd_trace`
   is below 0 on **all 5 scenes**, by G2a's per-scene rule (Amendment 7 f, steps 2-4): if the BD-rate is
   undefined, the scene wins if the BD-PSNR is above 0; if both are undefined, it is a loss.
2. **Against `lloyd_wopa_area`, the mean BD-rate over the 5 scenes is at most -5%**, each scene entering
   with its BD-rate or, where that is undefined or not finite, Amendment 8 a's substitute (a, b or c).
3. **No harm against E2's `gn_vq`:** the BD-PSNR of `gn_vq_cvfloor` against E2's committed `gn_vq` curve
   is at least -0.01 dB **on every scene**. A scene whose BD-PSNR against `gn_vq` is undefined (the
   curves share no byte range) does not meet this condition.

Differences are rounded to 9 decimals before they are compared, as in run 4 ("below" and "above" strict,
"at least" and "at most" inclusive). A missing row makes its scene, and so G2c, **`incomplete`**; a
failed lifted check skips the rows using that metric, which leaves them missing. Verdict order
`incomplete` > `fail` > `pass`.

### e. Reported, not gating

- `rho_cv` per scene and K, and whether it is at the top of the grid (`rho = 3`); the count of cells
  with `rho_cv > 0`;
- per cell, the test dMSE (`measured_test_clamped`) of `gn_vq_cvfloor` over E2's `lloyd_trace` and over
  E2's `gn_vq`;
- per cell, the test LPIPS and SSIM of `gn_vq_cvfloor` minus E2's `gn_vq`, and its bytes against E2's
  `gn_vq`, `lloyd_trace` and `lloyd_wopa_area`;
- per scene and K, the **Spearman rank correlation across the 7 `rho` between the cross-validation
  codebooks' odd-view dMSE and their own test dMSE** (average ranks for ties, `diagnostics.spearman`);
- the three G2c comparisons' BD-rate and BD-PSNR per scene, and against `upstream_l1`; per curve, whether
  PSNR rises with K; per comparison, whether BD-rate and BD-PSNR name different curves as better (the
  case that made E2's treehill H2b win a fit artifact, FINDINGS section 10). These flags are reported
  and change no verdict;
- the reproduction status of every `rho_cv = 0` cell (b);
- per row, E2b's logging: both objectives (floored and unfloored), iterations, stopping reason, clip
  rejections, the quantizer ranges, the warm start's and `M`'s sources.

### f. Why

- **This freezes the method before E3 ports it to other codecs.** E3 is not designed yet; whatever it
  ports should be one method, fixed here, and tested on scenes it was not tuned on.
- **E2b showed that the floor closes the fidelity gap on development scenes, and that the strength must
  be selected per scene.** On treehill, flowers and stump, R at `rho_cv` was below its own `rho = 0` value
  in all 6 cells (treehill at K = 65,536: 1.1253 to 0.6602). A fixed `rho = 1e-1` would have failed
  garden's 5% control: garden's test dMSE at `rho = 1e-1` was 1.0704 times its `rho = 0` value at
  K = 65,536 (FINDINGS section 11).
- **The grid is extended because `rho_cv` sat at the top of E2b's grid (1e-1) in every gap cell**, with
  the test dMSE still falling there. As `rho` grows, the floored metric divided by `rho` tends to
  `tr(M_i) / 15 * I`, the `tr(M_i)`-weighted Euclidean distance that `lloyd_trace` minimizes (FINDINGS
  section 11, post hoc), so the extended grid runs from the full matrix toward the scalar weighting.
- **The limit, stated in advance.** In E2 these five scenes generalized well, except room: the ratio of
  `gn_vq`'s to `lloyd_trace`'s clamped test dMSE exceeded the train ratio by +0.0064 to +0.0320 on
  bonsai, counter, kitchen and truck at K = 4,096 and 65,536, and by +0.0782 and +0.1966 on room
  (computed from E2's rows; Amendment 10 a tabulates the ratios to four decimals). And E2's own `gn_vq` rows already meet conditions 1 and 2 on these scenes with
  the domain-scaled fit: BD-rate -3.10% (counter) to -6.61% (room) against `lloyd_trace`, and a mean of
  -5.93% against `lloyd_wopa_area`. So G2c would pass with `rho_cv = 0` in every cell if E2's rows
  reproduce. **E2c mainly tests that the selected floor does no harm and keeps GN-VQ's wins;** the count
  of cells with `rho_cv > 0` (e) says how much of it the floor is. Whether the floor helps on new,
  gap-prone data is a question for E3.

### g. Checks, as in E2 and E2b

The checkpoint sha1 pins before any install and again in each job; the CUDA smoke tests; render parity
per scene; the writer-code assertion on every row; the lifted-assignment check (Amendment 3's criterion,
10,000 splats, on the K = 65,536 warm start) once per metric used: the floored `M_even` at each of the 7
`rho`, and the floored full `M` at each distinct `rho_cv`. E2c writes its own files (`gn2c/`) and never
those of E0, E1, E2 or E2b.

## Amendment 12 (2026-09-27, after E3's scouting and before any E3p code)

E2c has run and the method is frozen (FINDINGS section 12, "The method, as frozen"; Amendment 11 b). E3's
scouting is in `kaggle/E3_SCOUTING.md` (`dc5345f3`). E3 itself has no design yet. This amendment fixes
the scope of **E3p, an engineering pilot** that comes before E3's pre-registration, fixes the `rho` grid E3
will use, and records what E3 will pre-register later. G0, G1, G2a, H2b, G2c, E2b's criteria and
Amendments 1-11 are unchanged.

### a. Scope of E3p

- **E3p is exploratory and produces no verdicts.** It has no gate, no criterion and no comparison that
  decides anything. It measures whether the pipeline runs at INRIA scale and what it costs, so that E3
  can be designed and pre-registered from measured parts rather than estimates. Its numbers may inform
  E3's design; no claim about the method rests on them.
- **Scenes: bicycle and train only**, both already development scenes: bicycle since Amendment 7, train
  since Amendment 9 (Amendment 10 b kept it one). E3p uses **INRIA's 30k checkpoints** of these two
  scenes (`<scene>/point_cloud/iteration_30000/point_cloud.ply` in INRIA's pretrained-models archive),
  not the gsplat MCMC checkpoints of runs 4-5.
- **Untouched:** Deep Blending (drjohnson, playroom) and the five scenes no tuning decision has used
  (bonsai, counter, kitchen, room, truck; Amendment 11 a). E3p fetches, loads, renders and measures
  nothing of theirs, INRIA checkpoints included. The other INRIA scenes (garden, stump, treehill,
  flowers) are not used either.
- **Inputs.** From the archive
  (`https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip`,
  14,660,630,999 bytes), only three members per scene are fetched, by HTTP range requests: the 30k
  `point_cloud.ply`, `cameras.json` and `cfg_args`. Each member's offset, sizes and CRC32 are pinned
  from the archive's zip directory, read on 2026-09-27, and checked on fetch; the SHA-1 of each extracted
  file is recorded.

  | Member | Local header offset | Compressed bytes | Bytes | CRC32 |
  |---|---|---|---|---|
  | `bicycle/cameras.json` | 38 | 25,147 | 77,427 | `fd1d74d3` |
  | `bicycle/cfg_args` | 25,235 | 137 | 169 | `ae8c42ba` |
  | `bicycle/point_cloud/iteration_30000/point_cloud.ply` | 1,470,979 | 1,353,363,151 | 1,520,726,124 | `ebf2474a` |
  | `train/cameras.json` | 12,050,023,231 | 38,454 | 120,800 | `9940f310` |
  | `train/cfg_args` | 12,050,061,733 | 130 | 162 | `cbf086a8` |
  | `train/point_cloud/iteration_30000/point_cloud.ply` | 12,052,819,184 | 219,441,845 | 254,575,516 | `85d6d4ca` |

  The datasets are bicycle's MipNeRF360 images and Tanks & Temples train, from the downloaders runs 4
  and 5 used.
- **What E3p measures:**
  - the uncompressed INRIA model under two evaluation protocols, test PSNR / SSIM / LPIPS for each:
    (i) this project's harness (gsplat's own downscaled images, float renders clamped to [0, 1]) and
    (ii) INRIA's protocol (the resolution in the model's `cfg_args`, the dataset's own reduced images
    loaded directly, renders quantized to 8 bits before the metrics). Protocol (ii)'s PSNR is put next
    to INRIA's published per-scene number as a sanity check only;
  - time and peak GPU memory of each step at INRIA scale: the GN pass and the size of `M`, the PLAS
    sort, TorchPQ `upstream_l1` and `lloyd_wopa_area` at K = 65,536, `gn_vq_cvfloor` at K = 65,536 (the
    7-`rho` cross-validation and the final codebook), the `PngCompression` write and the evaluation; the
    torch, CUDA and driver versions; output sizes. A step that runs out of memory is recorded with where
    it failed, and the steps that do not depend on it still run;
  - a build check of C3DGS (`KeKsBoTer/c3dgs` at `2a234af55fbe8b90c8829c1436ce80088c4b622b`, the commit
    E3's scouting read): its dependencies and CUDA extensions are installed and compiled and then
    imported, and nothing of C3DGS is run.
- **One engineering change, method-neutral:** E3p keeps one copy of `M` in GPU memory instead of the
  five E2c's job holds (E3_SCOUTING.md c). The arithmetic of `gn_vq_cvfloor` does not change; a CPU test
  checks that codebooks, labels and reports are bit-identical to E2c's code path.

### b. The `rho` grid stays {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3}, selected per K

E3 uses Amendment 11 b's grid, with `rho_cv` selected per scene and K, unchanged. E3_SCOUTING.md d
listed a 3-point grid and a per-scene selection as ways to cut the cost. Neither is adopted:

- In E2c, no cell selected 3e-1, 1 or 3 (0 of 20; FINDINGS section 12).
- But in E2b's six gap cells (treehill, flowers, stump at K = 4,096 and 65,536), `rho_cv` was 1e-1, the
  top of a grid that ended there, and the full-`M` codebook's test dMSE was still falling at that top: in
  all six cells it is lowest at `rho` = 1e-1 (`kaggle/gn_e2b/gn2b/gn2b_results_<scene>.csv`; FINDINGS
  section 11; Amendment 11 f). On gap-prone scenes, which E3's new checkpoints may be, values above 1e-1
  have not been shown to be unneeded, so the evidence does not support dropping them.

### c. Planned, each with its own pre-registration before any of its code or data

- **The C3DGS comparison.** Primary: **no fine-tuning**, for both C3DGS's own vector quantization and
  GN-VQ. Secondary: with C3DGS's 5,000-iteration quantization-aware fine-tuning, for both.
- **Cross-paper comparisons in E3 use INRIA's evaluation protocol** (protocol ii above: the dataset's own
  reduced JPEGs, renders quantized to 8 bits before the metrics).

Nothing else about E3 is decided here: its scenes, hosts, grids of K, comparators and rules are for its
own amendment, written after E3p's results and before any E3 code or data.

## Amendment 13 (2026-09-28, after E3p's results and before any E3q code)

E3p has run. Its bundle is committed unchanged in `kaggle/gn_e3p/gn3p/` (`25f2133a`) and its numbers are in
`kaggle/FINDINGS.md` section 13 (`08902af2`, re-checked by `bench/gn/check_s13.py`). E3p's C3DGS build check
failed before compiling anything: `python -m venv` could not run `ensurepip` in the session's Python. This
amendment pre-registers **E3q, an engineering smoke test of the C3DGS host**. G0, G1, G2a, H2b, G2c, E2b's
criteria and Amendments 1-12 are unchanged.

### a. Scope

- **E3q has no verdicts.** It checks that C3DGS builds, runs its own compression pipeline on an INRIA
  checkpoint, and produces numbers that can be read next to C3DGS's published ones and next to this project's
  harness. GN-VQ is not run in E3q, and nothing in it is a result about the method. Its numbers inform E3's
  C3DGS comparison (Amendment 12 c), which still gets its own pre-registration before any of its code or data.
- **One scene: train**, a development scene (Amendment 9; kept one by Amendment 10 b), with INRIA's 30k
  checkpoint through E3p's pinned archive members (Amendment 12 a) and the Tanks & Temples dataset through
  run 5's downloader. Nothing else is fetched: Deep Blending, bonsai, counter, kitchen, room and truck stay
  untouched, as do the other INRIA scenes.

### b. Build

- **C3DGS** (`KeKsBoTer/c3dgs`) at `2a234af55fbe8b90c8829c1436ce80088c4b622b`, the commit E3's scouting read,
  with its glm submodule.
- **Into the session's own Python environment, not a venv**, because Kaggle's Python lacks `ensurepip`
  (E3p). The README's route is a conda environment from `environment.yml`; its packages are installed with pip
  instead: `plyfile==0.8.1` (its pin), `tqdm`, `torch-scatter` (from the PyG wheel index for the session's
  torch and CUDA; built from source if no wheel exists), and `submodules/diff-gaussian-rasterization` and
  `submodules/weighted_distance` built with `--no-build-isolation`. **`--no-deps` on every one of them**, so
  that the session's torch, torchvision and CUDA toolkit stay as they are.
- **Every deviation from the README is recorded** in the job's output. Known before any code: the session's
  Python, torch, torchvision and CUDA toolkit instead of `python=3.8`, `pytorch-cuda=12.1` and
  `cuda-toolkit=12.1`; pip instead of conda; `--no-deps`; torch-scatter from pip rather than conda's
  `pytorch-scatter`; `--source_path` passed on the command line (the checkpoint's `cfg_args` names its
  authors' local paths); the model directory laid out from the three pinned members, without the archive's
  other files (`input.ply`, the 7k iteration).
- **Two fallbacks, fixed now, each recorded as a deviation when used:** if `plyfile==0.8.1` cannot write and
  read back a small `.ply` in the session, the current `plyfile` is installed instead; if a CUDA extension
  fails to compile and its log names a missing fixed-width integer type (`uint32_t` and the like), the build is
  retried once with `<cstdint>` force-included through compiler flags. The source is never edited.

### c. Runs

C3DGS's own `compress.py`, unchanged, with its own vector quantization and its defaults otherwise (colour and
Gaussian codebooks of 4,096, `color_compress_non_dir` on, its importance thresholds and cluster iterations,
Morton sorting), on the INRIA train 30k model and the train dataset, **twice**:

1. `--finetune_iterations 0`: no fine-tuning, the setting Amendment 12 c makes E3's primary comparison;
2. `--finetune_iterations 5000`: its default quantization-aware fine-tuning, the published setting.

Each run is started through a small wrapper that executes `compress.py` as it is and records the process's
peak GPU memory from torch's allocator. Its written `point_cloud.npz` is then decoded with C3DGS's own
`npz2ply.py` into a `.ply`, which E3p's loader reads.

### d. Reported, per run

- **Size:** the bytes of the written `point_cloud.npz`, in MiB (2^20, C3DGS's own unit) and in MB (10^6).
- **C3DGS's own evaluation:** PSNR, SSIM and LPIPS from its `render_and_eval` (`results.json`; it evaluates the
  compressed model in memory, before the `.npz` round trip, on INRIA's test split, with float renders).
- **Protocol ii from this project's harness** (Amendment 12 a, as E3p ran it) on the decoded model, if its `.ply`
  loads there; if it does not, the report says so and why. The uncompressed model's protocol ii from the same
  session sits beside it. The `.ply` carries the decoded attributes but not C3DGS's render-time fake
  quantization, so this evaluates the decoded model as `npz2ply.py` writes it.
- **Time and memory:** the run's wall time, C3DGS's own `times.json` (sensitivity, clustering, fine-tuning,
  encoding), the peak GPU memory from the wrapper, and each harness step's time and peak memory as E3p
  recorded them.

### e. Published numbers, a sanity check only

C3DGS's published train numbers (Niedermayr et al., arXiv 2401.02436v2, **Table 9, "Tanks&Temples results"**,
row train), in its own protocol and with fine-tuning: "Ours" 21.863 dB PSNR, 0.798 SSIM, 0.226 LPIPS, 13.249
"MB"; the "3D Gaussian Splatting" columns 21.770 / 0.805 / 0.217 / 242.782 "MB". The "MB" is MiB: 242.782 is
the train 30k `.ply`'s 254,575,516 bytes divided by 2^20 (Amendment 12 a's pin). E3q's numbers are placed next
to these. No criterion attaches to the comparison. For context, known before E3q: that "3D Gaussian Splatting"
train PSNR, 21.770 dB, differs from INRIA's own published 21.097 dB and from E3p's protocol ii, 21.293 dB
(FINDINGS section 13).

### f. Failures

A step that fails, out of memory or otherwise, is recorded with its error, and every step that does not need
its product still runs (a failed build still leaves the harness's uncompressed evaluation; a failed
fine-tuned run still leaves the one without fine-tuning). The notebook reports the failures; it has no
verdict to withhold.

### g. Note (2026-09-28, after E3q's attempt 1 and before any attempt-2 code)

**What attempt 1 showed** (its bundle is committed unchanged in `kaggle/gn_e3q/attempt1/gn3q/`, `59303486`): C3DGS
built and imported, and both runs, `c3dgs_ft0` and `c3dgs_ft5000`, failed in `compress.py` after 270.8 s and 273.3 s
(the wrapper's wall time). They failed in C3DGS's `utils/splats.py`, line 29, `extract_rot_scale`, called from
`compress_covariance` in `compression/vq.py`. There, `torch.linalg.eigh` on one batch of 3x3 float32 matrices was refused
by cuSOLVER: `CUSOLVER_STATUS_INVALID_VALUE` from `cusolverDnXsyevBatched_bufferSize`. This is the failure class that
`bench/gn/batched.py` fixed for E0.

- **The batch** is the 4,096 Gaussian-codebook covariances plus every splat kept above `gaussian_importance_include`,
  so it grows with the splat count. On train it is at most 4,096 + 1,026,508 = 1,030,604; attempt 1's files do not
  record the kept count.
- **`R.det()` on the same batch.** The same function then calls `R.det()` (line 34, `torch.linalg.det` through the
  tensor method) on the eigenvectors of that batch. It never ran, because `eigh` failed first.
- **No other batched linear algebra** is on the compress, fine-tuning, save or evaluation path; only single 4x4
  camera inverses run there.

**For attempt 2.** This changes Amendment 13 c's "unchanged" in this respect only:

- **The wrapper replaces two functions.** In its own process, before it runs `compress.py`, the wrapper
  (`e3q_c3dgs_run.py`) replaces `torch.linalg.eigh` and `torch.Tensor.det` with chunked versions built on
  `bench/gn/batched.py`'s `batched_linalg`:
  - at most **8,192 matrices per call**;
  - on a backend refusal, the batch is **halved**, down to one matrix;
  - **every reduction is recorded** (`linalg_fallbacks`).

  Covering `det` as well as `eigh` was Dace's decision after the diagnosis above. An input that is not a single
  batch of matrices (`[N, n, n]`) passes through unchanged.
- **C3DGS's source is not edited.** `compress.py` still runs as `__main__`, with the same arguments.
- **What is computed does not change.** Each matrix is decomposed independently, so chunking does not change what is
  computed. Two caveats:
  - a chunk of one matrix may take torch's unbatched path, a different algorithm for that matrix;
  - either way, the replacement is **recorded as a deviation** in each run's record.
- **A report-only check** runs on each chunked call. It takes a random sample of up to 4,096 of the actual input
  matrices (seed 0, from a separate generator, so C3DGS's own random stream is untouched) and compares the chunked
  GPU results with a float64 CPU reference:
  - for `eigh`: the largest absolute eigenvalue difference, the sample's largest absolute eigenvalue (for scale), and
    the largest absolute entry of the float64 reconstruction residual of the chunked eigenpairs;
  - for `det`: the largest absolute determinant difference.

  No threshold attaches to it.
- **Batch sizes are recorded.** The wrapper also records each call's batch size, so attempt 2 measures the batch on
  train.
- **Everything else in Amendment 13 stands**, f included: a later failure is recorded, and every step that does not
  need its product still runs.

## Amendment 14 (2026-09-29, after E3q's results and before any E3r code)

E3q has closed (FINDINGS section 14, `6568cf25`, re-checked by `bench/gn/check_s14.py`). The C3DGS arm's integration
design is `kaggle/E3_C3DGS_DESIGN.md` (`cab8d1d3`), which recommends a pilot before the arm's pre-registration.
This amendment pre-registers that pilot, **E3r**, and records the method extension it exercises. G0, G1, G2a, H2b,
G2c, E2b's criteria and Amendments 1-13 are unchanged.

### a. Scope

- **E3r is an engineering pilot of the C3DGS host. It is exploratory and has no verdicts.** It measures what the
  arm's pre-registration needs:
  - C3DGS's rate range under two knobs;
  - how much of the metric the colour-quantized splats carry;
  - whether GN-VQ with the floor runs inside C3DGS end to end, and what it costs.

  No claim about the method rests on its numbers. In particular, the GN-VQ row against C3DGS's own row at the same
  settings is reported as an engineering number, not as a comparison.
- **Scenes: train and bicycle**, both development scenes (Amendments 7 and 9), with INRIA's 30k checkpoints through
  E3p's pinned archive members (Amendment 12 a), and their datasets through the downloaders runs 4 and 5 used.
  - **Untouched:** the gate scenes of the planned arm (`kaggle/E3_C3DGS_DESIGN.md` section 8: bonsai, counter,
    kitchen, room and truck on INRIA checkpoints, and Deep Blending's drjohnson and playroom) and the other INRIA
    scenes. E3r fetches, loads, renders and measures nothing of theirs.
- **The host is E3q's:**
  - C3DGS at `2a234af55fbe8b90c8829c1436ce80088c4b622b`, built into the session's Python as Amendment 13 b sets
    out, with its two fallbacks;
  - its `compress.py` run unchanged through E3q's wrapper, with Amendment 13 g's chunked `eigh` and `det`;
  - its outputs decoded by `npz2ply.py` and evaluated with protocol ii (Amendment 12 a) against the uncompressed
    model measured in the same session.
- **The build and the jobs.** C3DGS is built once per session, before the scene jobs. The two scene jobs then run
  in parallel, one per T4.

### b. A method extension for this host, planned for the arm's pre-registration

- **Why:** C3DGS vector-quantizes all 48 colour values per splat, the DC coefficient included
  (`color_compress_non_dir = True`, its default and published setting). Its DC-excluded setting looks broken at
  `2a234af5` (`kaggle/E3_C3DGS_DESIGN.md` section 2, read from the code, not run). The frozen metric covers bands
  1-3 only (15 x 15; Amendment 11 b), so it has no term for a change in DC.
- **What:** the GN metric extends from 15 x 15 (bands 1-3) to 16 x 16 (bands 0-3), by the same derivation.
  - The same render, probes and per-view weights `s_iv`, with `y` the 16 basis values at `d_iv`:
    `M_i = sum_v s_iv y(d_iv) y(d_iv)^T`.
  - Its lower-right 15 x 15 block is the frozen metric, entry for entry.
  - The floor becomes `M_i + rho * tr(M_i) / 16 * I`, and the ridge `mu = eps * tr(sum M_k) / 16`.
  - The cross-validation grid {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3} and its procedure are unchanged (Amendments 11 b
    and 12 b).
- **How GN-VQ sits in this host** (the pilot's definition; the arm's pre-registration may change it):
  - **Where:** the colour call of C3DGS's `vq_features`, replaced in the wrapper's process (C3DGS's source stays
    unedited). C3DGS's own `vq_features` runs first, unchanged, so the global random streams advance exactly as in
    C3DGS's own run. Its codebook and labels are GN-VQ's warm start.
  - **Which splats:** the splats C3DGS quantizes in that run, after its pruning, with colour importance at or below
    the threshold.
  - **Their values:** C3DGS's quantizer input (`get_features`, already int8-fake-quantized; 48 values, index
    `k * 3 + channel`).
  - **The variant:** E2's GN-VQ (eps = 1e-2, at most 20 iterations, the 1e-3 relative-drop stopping rule, the clip
    to the warm-start range with per-cluster acceptance), with the floored 16 x 16 metric.
  - **The codec's quantizer:** C3DGS's per-tensor int8 quantization of the colour table, DC and AC separately, with
    the scale and zero point its two fake quantizers hold at injection. The final exact assignment is against that
    dequantized codebook. The scales C3DGS actually writes are recorded next to them, because its observers keep
    updating afterwards.
- **Selection of `rho`: SH-only cross-validation,** as the design doc recommends (its section 3, option a):
  - **The splats:** those quantized in C3DGS's own run at K = 4,096 and the default threshold (c.i's probe run).
  - **`M_even`:** the 16 x 16 metric over the even-indexed train views (probe seed 0).
  - **Scoring:** for each `rho`, GN-VQ from that run's recorded warm start, scored by the render-vs-render dMSE,
    clamped, on the odd-indexed train views:
    - the variant replaces those splats' 48 colour values with the dequantized codebook at their labels;
    - the reference has every splat's colour at C3DGS's quantizer input;
    - the geometry is the checkpoint's, and the renderer is gsplat's.
  - **`rho_cv`** is the lowest score; a tie goes to the smaller `rho`.
  - **The recorded choice:** SH-only scoring, not the full C3DGS pipeline.
- **The final codebook** is computed inside the injected run itself, at `rho_cv`, with the full-train-view 16 x 16
  metric, from that run's own warm start, on that run's own set of quantized splats. How that set differs from the
  probe run's is recorded.
- **The pilot implements and exercises the extension, and nothing is concluded from it.** Whether the arm uses it
  is for the arm's pre-registration.

### c. What the pilot measures, per scene

Every C3DGS run is `compress.py` with C3DGS's defaults except where stated, through the wrapper. Each run reports:
- the `.npz`'s bytes, in MiB and MB;
- C3DGS's own PSNR / SSIM / LPIPS;
- protocol ii's PSNR / SSIM / LPIPS of the decoded model;
- C3DGS's `times.json` parts (clustering among them), the wall time and the peak GPU memory;
- the number of pruned, kept and colour-quantized splats.

1. **C3DGS's baseline without fine-tuning at K in {1,024, 4,096, 16,384, 65,536}** (`--color_codebook_size`).
   - The K = 4,096 run is C3DGS's default and is the **probe run**. Its colour-quantizer input, its masks, codebook,
     labels and quantizer state are recorded for (iii) and (iv).
2. **The colour-importance threshold as a second rate knob:** `color_importance_include` = 0.6e-6 x 3^j,
   j = -2, ..., 2 (the default 0.6e-6 and two tripling steps each way), at K = 4,096 without fine-tuning.
   - The default point is (i)'s probe run.
   - Chosen by Dace (2026-09-29); the design doc named the knob without values.
   - Per (e), this runs **on train only**.
   - The aim is to see which knob gives enough rate range for BD-rate. The report gives each knob's byte range and
     PSNR range. No criterion attaches to them; the arm's pre-registration chooses.
3. **The colour-quantized splats:**
   - their number (and the pruned and kept numbers) in the probe run;
   - their share of the total `tr(M_i)` over all the checkpoint's splats, under the full-train-view 16 x 16 metric;
   - the same share under its 15 x 15 block (the frozen metric).
4. **GN-VQ with the floor, injected, at K = 4,096, default threshold, without fine-tuning, end to end:**
   - the row's measurements as above;
   - `rho_cv` and all 7 cross-validation scores;
   - the GN-VQ runs' iterations and times;
   - the injected set's difference from the probe run's;
   - the quantizer scales at injection and at write;
   - its time.

   Two report-only checks come with it:
   - the lifted check (Amendment 3's criterion, 10,000 splats) on the final codebook;
   - a calibration: the predicted dMSE `P` from the even-view 16 x 16 metric against the measured unclamped dMSE on
     the even train views, for the `rho_cv` cross-validation codebook. This exercises the DC terms.
5. **Train only: the same injected run with C3DGS's 5,000-iteration fine-tuning.** It checks that the pipeline
   completes and that the labels survive: the colour indices C3DGS holds just before it writes must equal the
   injected ones. It records how far fine-tuning moved the injected codebook, and gives its size, C3DGS's metrics
   and protocol ii. No baseline with fine-tuning runs in E3r.

### d. Failures

- **A failure is recorded, and the rest still runs.** A step that fails, out of memory or otherwise, is recorded
  with its error, and every step that does not need its product still runs. A failed GN pass leaves (i) and (ii); a
  failed injected run leaves every baseline row.
- **One fallback, fixed now:** a C3DGS run that fails out of GPU memory is retried once with `--data_device cpu`
  (C3DGS's own option: the images stay in host memory). The retry is recorded as a deviation. C3DGS has never run on
  bicycle's 6,131,954 splats.
- **The deadline.** A job starts no new C3DGS run once the session's deadline, less a reserve for protocol ii and
  the bundle, has passed. A run not started is recorded with that reason. The job resumes from the notebook's own
  output, and every row resumes on its own.
- **The order within a job** puts what the arm needs most first:
  1. the probe run;
  2. the harness phase (runner, the uncompressed model's protocol ii, the GN passes, (iii), the cross-validation);
  3. the injected run(s);
  4. the other K;
  5. the thresholds;
  6. protocol ii of every decoded row.

### e. The runtime estimate, before any E3r code

`bench/gn/e3r_estimate.py` builds it only from E3q's, E3p's and E2c's committed files. It is an estimate, not a
measurement. C3DGS's runs are taken as flat to linear in the splat count, its colour clustering as flat to
proportional in K, and the 16 x 16 GN-VQ as 1 to 136 / 120 times the 15 x 15 cost.

| Job | Estimate |
|---|---|
| train | 5,362-8,309 s |
| bicycle, (ii) included | 10,735-46,922 s (3.0-13.0 h) |
| bicycle, without (ii) | 9,247-38,037 s |
| session setup (restore, install, build) | 385 s |

- **(ii) on both scenes exceeds one session.** Its upper end, 13.0 h for bicycle alone, is above Kaggle's 12 h
  session, so (ii) runs on train only. That is the rule this request set.
- **With both jobs in parallel,** the session is estimated at 2.7-10.7 h.
- **What is not estimated:** C3DGS on bicycle has never run, so its memory is not estimated either. The deadline
  and resume in (d) cover the upper end.

### f. Note (2026-09-29, before any E3r code is committed and before any E3r run)

**What building E3r showed.** C3DGS's `compress.py` sets no seed. Its `render.py` calls `safe_state`
(`utils/general_utils.py`), which seeds `random`, `numpy` and `torch` with 0; `compress.py` does not. And torch's
default generator starts from a different seed in every process: two fresh processes gave different
`torch.initial_seed()` values locally, under torch 2.11 on the CPU.

- **The consequence.** Two C3DGS runs draw different random batches. That holds for any two runs: a baseline and
  an injected run, or two baselines.
- **What that does to b.** Its statement that running C3DGS's own `vq_features` first makes "the global random
  streams advance exactly as in C3DGS's own run" holds within one run, but pairs nothing across runs. The same
  goes for `kaggle/E3_C3DGS_DESIGN.md` section 2's "the geometry VQ that follows draws the same batches in both
  arms".

**For E3r:**
- **Every run is seeded.** The wrapper seeds `random`, `numpy` and `torch` (all devices) with 0 before it runs
  `compress.py` (`--seed 0`): the seeds of C3DGS's own `safe_state`. This covers every E3r run, baseline and
  injected alike. It is recorded as a deviation in each run's record. C3DGS's source is still not edited.
- **The pairing can still break.** GPU atomics can make two seeded runs differ: C3DGS's scatter-based codebook
  updates, and the backward pass of its sensitivity computation, which sets the keep thresholds' inputs. The pilot
  already records how much:
  - the injected run's quantized set against the probe run's (c.iv);
  - every run's geometry-table SHA-1.

  Equal SHA-1s mean the geometry VQ was reproduced.
- **A CPU check:** a stand-in with C3DGS's call structure, fake quantizers and random streams gives identical
  outputs across processes once seeded, and identical outputs with and without the observe and record hooks
  (`bench/gn/test_gn.py`).
- **The rest of Amendment 14 stands.** This changes neither its grids nor its order, and E3q's runs (Amendment 13)
  were unseeded.

### g. Note (2026-09-29, after E3r's code was pushed at `290e62f5` and before any E3r run): a memory check restricts bicycle

**The order of events.** Dace asked for a GPU-memory check before E3r was pushed. The request reached this session
after the push, so this note follows the code it constrains. No E3r run exists, and nothing has run on Kaggle.

**The check** (`bench/gn/e3r_memory.py`; its numbers are in `kaggle/gn_e3r_memory/e3r_memory.json`) sizes, from the
code, what each step allocates on the GPU. The inputs are the pinned train and bicycle checkpoints and their
`cameras.json`, fetched locally, with SHA-1s equal to E3p's. Every figure below is in 10^9 bytes.

- **The 16 x 16 metric:**
  - it takes 544 bytes per splat, float32 and packed (136 values), as stored;
  - the GN pass's accumulator takes 628 bytes per splat, for every splat;
  - the colour-quantized splats need the metric in this host. Their count is not known before C3DGS runs, and
    the only bound the checkpoint gives is the whole model: on bicycle, 6,131,774 of 6,131,954 splats touch a
    tile in some train view. So one device copy is at most 3.34 GB on bicycle.
- **C3DGS's own peak.** It is the backward pass of its sensitivity computation (`compress.py` `calc_importance`)
  over its worst train view. The allocations that scale with the splat count total 1,850 bytes per splat:
  - the parameters and their gradients;
  - the fake-quantized features and masks;
  - the rasterizer's per-splat state;
  - its 75-float backward buffers;
  - the hooks' temporaries.

  On top of those come the images, which sit on the GPU with `--data_device cuda`, and the binning buffers, at
  36 bytes per tile instance. The tile instances were counted with the rasterizer's own formulas: at most
  6,254,053 per train view on train and 11,798,558 on bicycle.
- **The tie to E3q.** On train this accounting gives 4.068 GB against E3q's measured 4.551 GB, explaining 89.4%.
  The rest, 0.482 GB, is not attributed; scaled per splat, it is 470 bytes each.
- **C3DGS on bicycle:**

  | Images | From the code | With train's unexplained rest per splat |
  |---|---|---|
  | on the GPU (`--data_device cuda`) | 14.16 GB | 17.05 GB |
  | on the CPU (`--data_device cpu`) | 11.80 GB | 14.68 GB |

  This is before either process's CUDA context and before allocator fragmentation: E3p's bicycle job reserved
  15.18 GB with 11.51 GB allocated. The T4 has 15.64 GB.
- **E3r's own steps.** As `290e62f5` codes them, the cross-validation and the injection hold two copies of the
  metric, and `e2b.floored_metric` makes a float64 copy of the whole metric for its trace: on bicycle 12.57 GB
  plus a 6.67 GB transient, beyond the T4. With one device copy they fit:
  - the GN pass: 8.41 GB (E3p's measured 7.99 GB plus the extra width);
  - the cross-validation's GN-VQ: 8.06 GB, and its scoring 5.82 GB;
  - the injected run's colour step, with C3DGS's own state at that point: 11.78 GB with images on the GPU, or
    9.41 GB on the CPU.

  But C3DGS's sensitivity pass comes first in every C3DGS run.

**For E3r:**
- **Mitigations, method-neutral.** The cross-validation and the injection keep one device copy of the floored
  metric: E3p's `metric_store` layout, filled slice by slice with `e2b.floored_metric` itself, with the unfloored
  objective read from host memory. The reference colours stay in host memory, and the metric's buffer is released
  during the scoring renders. A CPU test shows GN-VQ's codebook, labels and report, `P` and the lifted check bit
  for bit as the full-copy path gives them. With these, every E3r step of its own fits on both scenes.
- **No method-neutral mitigation brings C3DGS itself under about 14 GB on bicycle.** Its allocations are its own,
  and its source stays unedited. `--data_device cpu` still leaves 11.80-14.68 GB.
- **So bicycle's C3DGS steps are restricted to train:** the probe run, the injected run and the other K
  (Amendment 14 c.i, c.iii and c.iv), as Dace set the rule.
  - Train keeps everything in c.
  - Bicycle keeps its runner, the uncompressed model's protocol ii and the two 16 x 16 GN passes, which measure
    the extension's time and memory at 6.13M splats.
  - Bicycle has no C3DGS row, no cross-validation, no trace share and no calibration.
- **What the arm inherits.** On this data, C3DGS does not fit a T4 at bicycle's size. The arm's gate scenes are
  at most 3,405,153 splats (drjohnson); their own checks belong to the arm's pre-registration.
- **The runtime estimate for this configuration** (`bench/gn/e3r_estimate.py`, `estimate_g`), an estimate:

  | Part | Estimate |
  |---|---|
  | train | 5,362-8,309 s, unchanged |
  | bicycle | 492-498 s |
  | the session | 5,747-8,694 s (1.6-2.4 h) |

  Amendment 14 e's table stays as it was estimated.
- **Everything else in Amendment 14 stands.**

### h. Note (2026-09-30, before any E3r run): a pruned-splat trace count

**What.** On train only, report-only, with no criterion. The count uses two things E3r already produces:
- the probe run's recorded prune mask: C3DGS's `color_importance_n <= prune_threshold`, with `prune_threshold` 0,
  as the hooks record it;
- the full-train-view 16 x 16 GN metric the harness computes.

From them it records:
- `n_splats`;
- `n_pruned`;
- `n_pruned_tr0`, the pruned splats with `tr(M16_i) == 0` exactly, on the stored float32 trace;
- `n_tr0_all`, all splats with `tr(M16_i) == 0`.

**Why.**
- **C3DGS's side.** C3DGS prunes the splats whose colour sensitivity is zero. Its sensitivity pass renders with
  `clamp_color=False`, so DC's gradient is the constant basis value times the sum of the splat's blending weights.
  It is therefore zero exactly when the splat has zero blending weight in every train view, under INRIA's
  rasterizer.
- **gsplat's side.** Its GN pass gives such a splat `tr(M) = 0`, under gsplat's rasterizer.
- **What the count shows.** It bears on rasterizer parity (`kaggle/E3_C3DGS_DESIGN.md` section 9, item 2), which
  nothing has measured. It is the agreement on which splats are visible at all, and nothing about render parity.
- **What limits it:**
  - C3DGS's pass renders its int8-fake-quantized opacity and scales (`compress.py` `calc_importance`), not the
    stored checkpoint that gsplat's pass renders;
  - gsplat's per-view weight `s_iv` is a 16-probe Hutchinson estimate. It is zero whenever every weight is zero, but
    could in principle vanish otherwise, through a cancellation across all probes or float32 underflow.

**One count suffices.** `tr(M16_i) = (16 / 15) tr(M15_i)` for every splat (the addition theorem; HANDOFF, E3r
decisions), so a trace that is zero under one metric is zero under the other.

**Where it does not run:**
- On **bicycle** it is `not_applicable`: bicycle has no C3DGS run (Amendment 14 g).
- If the probe run or the GN pass failed, it is `not_computed`, with the reason. It never stops the job.
- It uses the probe run's mask, not the injected run's, because GPU atomics may separate the two (Amendment 14 f).

**Nothing else in Amendment 14 changes:** no grid, order, row, step or estimate.

## Amendment 15 (2026-10-01, after E3r's results and before any E4p or E4 code)

E3r has closed: its bundle is committed unchanged in `kaggle/gn_e3r/gn3r/` (`9c28cc89`), and its numbers are in
FINDINGS section 15 (`2eb091f9`, re-checked by `bench/gn/check_s15.py`, `5fda2ea7`). Two preprints that overlap GN-VQ
have been read (`kaggle/RELATED_WORK_OGC.md`, `04577841`), and the C3DGS arm has been designed as E4
(`kaggle/E4_DESIGN.md`, `7a954b0d`). Dace's E4 decisions (HANDOFF, "Related work, paper direction and E4 decisions
(2026-10-01)", `6250c1d8`) supersede the design doc where the two differ. This amendment pre-registers **E4p, a pilot
on train**, and **E4, a gated comparison inside C3DGS on seven INRIA checkpoints**. G0, G1, G2a, H2b, G2c, E2b's
criteria and Amendments 1-14 are unchanged.

### a. Scope and motivation

- **The question.** Inside C3DGS, at its default operating point, does GN-VQ with the cross-validated floor
  (`rho_cv`) give better test-view quality than:
  1. the same GN-VQ without the floor (`rho` = 0, the unfloored Gram);
  2. OGC's released Gram VQ?
- **Against the unfloored Gram.** G2c passed with the floor, but E2c's gain over E2's GN-VQ was clear only on room
  (FINDINGS section 12). Whether the floor helps on new checkpoints was left to E3 (Amendment 11 f).
- **Against OGC.** arXiv 2609.28997 ("OGC") vector-quantizes C3DGS's colours under a per-splat Gram that equals our
  `M_i` in expectation, with no floor and a ridge toward the cluster mean in the update. It reports its gain inside
  C3DGS from one unseeded run per scene (`kaggle/RELATED_WORK_OGC.md` sections 2 b, 2 g and 3).
- **OGC's code** is `github.com/moholo-founder/ogc-3dgs` at `49ccae72e75eec9877354ed72074827531f7fd79`, under PolyForm
  Noncommercial 1.0.0. E4 uses it for noncommercial academic research.
  - The session clones it at run time and checks the commit; a mismatch stops before any row.
  - Its modules are imported from that clone. No file of it is copied into this repository or into a bundle.
- **Why one forked process, repeated.** E3r found that seeded C3DGS runs are not reproduced on the GPU:
  - all 10 runs' geometry SHA-1s differed;
  - C3DGS's own colour codebook, the warm start, differed from a bit-identical input;
  - so the injected row's +0.1268 dB at +1.485% bytes could not be read as GN-VQ's effect (FINDINGS section 15).

  E4 therefore computes all its rows inside one forked C3DGS process, which shares the warm start, the keep mask and
  the geometry VQ by construction (b). And it runs every scene in 3 independent processes, so that the bar (c) is
  set against the spread between processes.
- **Not in E4** (Dace, 2026-10-01): the held-out 90-degree arc protocol, and a ridge-toward-the-mean variant of
  GN-VQ. Of `kaggle/E4_DESIGN.md` section 2's two OGC context rows, only `lam` = 1e-6 runs, as the report-only row
  2b (b); the 512-entry row does not.

### b. Rows: one forked C3DGS process

**The process.** C3DGS at `2a234af5`, built and run as in E3r:
- Amendment 13 b's build, E3q's wrapper with Amendment 13 g's chunked `eigh` and `det`, Amendment 14 f's seeding;
- the scene's INRIA 30k checkpoint;
- K = 4,096 (`--color_codebook_size`), the default colour threshold (`color_importance_include` = 0.6e-6) and
  `--finetune_iterations 0`.

C3DGS's source is not edited. The hooks extend E3r's (`kaggle/e3r_hooks.py`):
1. C3DGS's own `vq_features` runs first, unchanged. Its codebook and labels go back to C3DGS, so C3DGS's own run
   proceeds as row 1.
2. At that colour call, the hooks compute rows 2-5 and 2b from the same inputs:
   - the quantizer input (48 values per splat, int8-fake-quantized, index `k * 3 + channel`);
   - the same quantized set;
   - the same 16 x 16 metric.

   Each row's codebook and labels are held.
3. C3DGS's geometry VQ runs once, after the colour step. Every row shares its codebook and indices.
4. When C3DGS's compression returns, the model, every fake quantizer's state (observers included) and the random
   states are copied to host memory. Every row is saved before any row is evaluated (d, "When results exist"):
   1. row 1 is saved with C3DGS's own `save_npz`, as its own run does;
   2. each other row is installed into the restored copy with C3DGS's own `join_features` and `set_color_indexed`
      and saved with its `save_npz`;
   3. then each row, row 1 included, is evaluated with C3DGS's `render_and_eval`.
5. Every row's `.npz` is decoded with `npz2ply.py` and evaluated with protocol ii (Amendment 12 a), one at a time.

**Checked in every process.** A failure is a bug, and that process's rows enter nothing:
- the keep mask and the quantized set are identical across the rows;
- the geometry SHA-1 (E3r's) is identical at every row's save;
- each row's colour indices just before it is written equal that row's labels.

**The rows,** all at K = 4,096:

| Row | Colour codebook | Start | Settings |
|---|---|---|---|
| 1 `c3dgs` | C3DGS's own `vq_features` | - | C3DGS's defaults |
| 2 `ogc` | OGC's `vq.gram_kmeans` on our 16 x 16 metric | its own | its defaults, as their C3DGS host calls it (below) |
| 3 `gnvq_rho0` | GN-VQ, unfloored | row 1's codebook and labels | `rho` = 0 |
| 4 `scalar` | GN-VQ with `M_i` replaced by `tr(M_i) / 16 * I` | row 1's | `rho` has no effect (below) |
| 5 `gnvq_cv` | GN-VQ with the floor | row 1's | `rho_cv` |
| 2b `ogc_lam1e6` (context) | as row 2 | its own | as row 2, but `lam` = 1e-6 |

- **GN-VQ (rows 3-5)** is the frozen method (Amendment 11 b) as Amendment 14 b ports it to this host:
  - E2's variant: eps = 1e-2, at most 20 iterations, the 1e-3 relative-drop rule, the clip to the warm-start range
    with per-cluster acceptance;
  - the 16 x 16 metric floored as `M_i + rho * tr(M_i) / 16 * I`;
  - the final exact assignment against the dequantized codebook of C3DGS's int8 colour-table quantizer at injection;
  - one device copy of the metric (Amendment 14 g).

  The cap stays at 20 iterations. All of E3r's GN-VQ runs reached it (FINDINGS section 15), so every run's iterations
  and stopping reason are reported.
- **Row 4.** With an isotropic metric, the floor multiplies every splat's metric and the ridge by the same 1 + `rho`.
  That changes no assignment, centroid, acceptance or relative drop, so row 4 runs at `rho` = 0.
- **Row 2, OGC's VQ, called as their C3DGS host calls it** (`hosts/c3dgs_run.py:50-62` at `49ccae72`):
  `gram_kmeans(X, G, 4096, metric="gram", iters=15, device="cuda", chunk=100000)`, with `lam` and `seed` left at the
  function's defaults, 1e-3 and 0 (`vq.py:17`).
  - **The inputs and outputs:**
    - `X` is the quantizer input reshaped to `[n, 3, 16]`;
    - `G` is our 16 x 16 metric of the same splats, unpacked (`[n, 16, 16]`, float32, in host memory);
    - the codebook goes back as `C.permute(0, 2, 1).reshape(K, 48)`, and its assignment as the labels.
  - **As in their host,** the codebook reaches C3DGS unclipped and without a final assignment against the int8
    table. C3DGS's quantizer acts on it at save.
  - **Recorded why:**
    - **`lam` = 1e-3:** every released caller leaves it at 1e-3, their C3DGS host included. The code's comment says
      the ridge keeps codewords from saturating C3DGS's int8 quantizer (`vq.py:63-68`). The paper's text states 1e-6
      (p. 14; `kaggle/RELATED_WORK_OGC.md` section 2 b).
    - **15 iterations:** their host's default (`--lloyd_iters`). The function's own default is 20.
    - **Its own init:** points sampled in proportion to `tr(G_i)` (`vq.py:31-33`). `gram_kmeans` takes no initial
      codebook, and E4 does not modify their code. So row 2 differs from row 3 in init and stopping as well as in
      the regulariser: row 5 against row 2 compares the two released methods, not a ridge against a floor.
    - **Seed 0:** the function's default. It draws from its own generator, so C3DGS's global streams are untouched.
    - **Our metric, not their `A_i`:** our 16 x 16 `M_i` equals their S2 Gram in expectation
      (`kaggle/RELATED_WORK_OGC.md` section 3). Row 2 therefore reproduces their VQ, not their statistics pipeline.
      E4p measures how far the two Grams differ on train (f).
    - **CUDA and chunks of 100,000:** their host's. The function's defaults are `mps` and 150,000.
- **Row 2b, `ogc_lam1e6`, a context row, report only.** It is row 2 with `lam` = 1e-6, the value the paper's text
  states (p. 14), and it enters no primary difference.
  - **Why:** which ridge produced their tables, the paper's 1e-6 or the code's 1e-3, is unknown
    (`kaggle/RELATED_WORK_OGC.md`, open question 1).
  - **No 512-entry row** runs.
- **`rho_cv`, chosen once per scene from a probe run.**
  - **The probe run** is C3DGS's own run at K = 4,096 and the default threshold, seeded 0, with E3r's `--record`. It
    is not one of the three processes.
  - **The selection** is Amendment 14 b's SH-only cross-validation on held-out train views:
    - the probe run's quantized set, quantizer input and codebook;
    - `M_even`, the 16 x 16 metric over the even-indexed train views;
    - GN-VQ from the probe's codebook at each `rho` of {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3} (Amendments 11 b, 12 b);
    - each scored by the clamped render-vs-render dMSE on the odd-indexed train views.

    `rho_cv` is the lowest score; a tie goes to the smaller `rho`. No test view enters the selection.
  - **Every process uses that `rho_cv`.** Row 5's codebook is computed inside each process, from that process's
    row 1, with the full-train-view 16 x 16 metric (Amendment 14 b).
  - **If `rho_cv` = 0,** row 5 is row 3 by definition and is not run again. Row 3's measurements stand for row 5's
    wherever row 5 is named.
- **The metric** is computed once per scene by the harness (E3r's 16 x 16 GN passes, probe seed 0). Every process
  uses it.
- **Three processes per scene:** independent C3DGS processes seeded 0, 1 and 2. Seeding pairs nothing on the GPU
  (FINDINGS section 15); distinct seeds make the independence explicit. Each process runs every row.

### c. The primary (E4) and its bar

**The measure:** protocol ii's test-view PSNR (Amendment 12 a, as E3p-E3r computed it) of the decoded `.npz`. Rows at
K = 4,096, the default threshold, without fine-tuning.

**Two co-primary differences,** within process `p` of scene `s`:
- `D1_sp` = PSNR(`gnvq_cv`) - PSNR(`gnvq_rho0`);
- `D2_sp` = PSNR(`gnvq_cv`) - PSNR(`ogc`).

If `rho_cv` = 0 on a scene, `D1_sp` = 0 in all three of its processes, and the scene does not count as positive for
D1.

**For each difference D, over the n scenes run (d):**
- the scene mean `D_s`: the mean of `D_sp` over the scene's 3 processes;
- the mean over scenes `D_bar`: the mean of `D_s` over the n scenes;
- the within-scene variance `v_s` = sum over `p` of (`D_sp` - `D_s`)^2 / 2;
- `SD_pool` = sqrt(mean of `v_s` over the n scenes). Every scene has weight 1 (2 degrees of freedom each). A scene
  where D1 is 0 by rule enters with `v_s` = 0;
- `SE_noise` = `SD_pool` / sqrt(3 n).

**D passes if all three hold:**
1. `D_bar` > 0;
2. `D_s` > 0 on at least ceil(0.7 n) scenes: 5 of 7 if all seven run (5 of 6, 4 of 5);
3. `D_bar` > 2 x `SE_noise`.

**E4 passes if D1 and D2 both pass.**
- **At least 5 scenes:** if fewer than 5 scenes run (n < 5, after d's drops), E4 is `incomplete`, never `pass`,
  whatever D1 and D2 show.
- **Both are reported whatever the outcome,** with every part: each `D_sp`, `D_s` and `v_s`, `D_bar`, `SD_pool`,
  `SE_noise` and the count of positive scenes.
- **Rounding:** differences are rounded to 9 decimals before they are compared, as in run 4 and G2c. Every
  comparison in this section is strict.
- **`incomplete`:** a primary row (rows 2, 3 and 5, measured with protocol ii) missing in any process of a scene that
  was not dropped (d), after d's one rerun, makes E4 `incomplete`. Verdict order: `incomplete` > `fail` > `pass`.
- **What `SE_noise` measures:** the spread between processes of one scene (warm start, geometry, GPU atomics). It does
  not treat the scenes as a sample of scenes; condition 2 is the across-scene requirement.

### d. Scenes, memory and the header-read rule

- **Scenes:** INRIA's official 30k checkpoints of bonsai, counter, kitchen, room, truck, drjohnson and playroom
  (`<scene>/point_cloud/iteration_30000/point_cloud.ply`, with `cameras.json` and `cfg_args`, in Amendment 12 a's
  archive). Each member is pinned from the archive's zip directory (offset, sizes, CRC32) and checked on fetch, as in
  Amendment 12 a. The pins are recorded in g.2's note, before any E4p code.
- **Disclosure.**
  - E2 and E2c used bonsai, counter, kitchen, room and truck, and G2c gated on them, with this project's gsplat MCMC
    1M checkpoints (Amendment 7 d, Amendment 11 a) and measurements on their test views.
  - E4's checkpoints are INRIA's, which this project has not fetched, loaded or rendered (Amendments 12 a, 13 a,
    14 a).
  - Nothing in E4 is tuned on these scenes. The method was fixed in Amendment 11 before E2c ran, and E4's rows, K,
    threshold and bars were not chosen from any result on them.
  - drjohnson and playroom have never been used.
- **Memory.**
  - Every C3DGS run starts with the images on the GPU (`--data_device cuda`).
  - A run that fails out of GPU memory is retried once with `--data_device cpu` (Amendment 14 d). The retry is
    recorded as a deviation, and the scene's later runs start on the CPU.
  - **A scene whose run also fails out of memory on the CPU is dropped before any of its results exist,** and n
    shrinks. The drop is reported with the step that failed.
- **When results exist.**
  - A scene's results are the test-view measurements and the bytes of its processes' rows.
  - So that a drop precedes them, each scene runs the probe run, the harness phase (the GN passes and the
    cross-validation), and the first process's colour steps and saves before any row of that scene is evaluated.
  - The probe run's own evaluations (C3DGS's and protocol ii) are deferred too, until after the first process's
    colour steps and saves. The probe is then evaluated from its decoded `.npz`: C3DGS's evaluation of the loaded
    `.npz`, and protocol ii. This differs from E3r's in-memory evaluation of its probe. Its measurements enter no
    difference and are reported as context.
- **Failures after results exist.** An out-of-memory failure in a scene's second or third process does not drop the
  scene. Which failures are rerun depends on what they lose:
  - **A failure that loses a primary row** (rows 2, 3 and 5, measured with protocol ii), out of memory or otherwise,
    reruns that process once, whole, with the same seed. If the rerun loses a primary row too, E4 is `incomplete`
    (c).
  - **A failure in a secondary** (fine-tuning, row 1, row 4, row 2b, the bytes and index measures, C3DGS's own
    evaluation) is recorded and not rerun, and it does not change the verdict.
- **The header-read rule** (written before the read).
  - Reading a scene's camera count, image sizes and splat count (from `cameras.json` and the `.ply` header), with no
    pixel, splat value, render or metric, does not count as touching the scene.
  - Deep Blending's two scenes are read this way after this amendment is committed (g.2).
  - A dated note then records their feasibility by Amendment 14 g's memory model (`bench/gn/e3r_memory.py`'s terms,
    as `kaggle/E4_DESIGN.md` section 6 applies them). The note changes no scene, row or bar. Whether a scene runs is
    settled by the memory rule above.

### e. Secondaries: descriptive, no bars

- **Fine-tuning:** on all 3 processes of every scene, for rows 1, 2 and 5.
  - **After the primary rows,** each of the three is restored from the copy with its own table and the copy's random
    states, then:
    1. fine-tuned with C3DGS's `finetune` (5,000 iterations);
    2. saved;
    3. evaluated by C3DGS;
    4. decoded and evaluated with protocol ii.
  - **Reported:** `ogc` minus `c3dgs`, and `gnvq_cv` minus `ogc`, after fine-tuning, per process, per scene and
    pooled, each with its `SE_noise` computed as in c.
  - **A caveat stated in advance:** resetting the random states pairs the camera order, but GPU atomics separate the
    rows during fine-tuning. These differences therefore carry run-to-run noise that the primary's do not.
- **Before fine-tuning,** computed as in c, each with its `SE_noise`:
  - **`ogc` minus `c3dgs`:** the replication of arXiv 2609.28997's pre-fine-tuning gain in its C3DGS host (its
    Table 1: +0.49 dB, the mean of 9 Mip-NeRF 360 scenes, one unseeded run each), with OGC's VQ on our metric and 3
    processes per scene. Four of E4's scenes (bonsai, counter, kitchen, room) are among those 9 (its Table 12);
    truck, drjohnson and playroom are scenes its host did not report;
  - **`gnvq_cv` minus `c3dgs`;**
  - **`ogc_lam1e6` minus `ogc`.**
- **Gram against scalar:** `gnvq_rho0` minus `scalar`, computed as in c, with its `SE_noise`.
- **Bytes and indices,** for every row, fine-tuned rows included:
  - the `.npz`'s bytes, in bytes, MiB and MB;
  - each array's compressed and uncompressed size inside the `.npz`;
  - the zero-order entropy of the stored colour-index array, in bits per index;
  - the number of distinct colour indices used.
- **Also reported:**
  - per row, the mean, SD and range over the 3 processes of protocol ii's PSNR / SSIM / LPIPS, C3DGS's own metrics
    and the bytes;
  - D1's and D2's SSIM, LPIPS and bytes, computed as in c;
  - GN-VQ's logging as in E3r;
  - each step's time and peak GPU memory.
- **The threshold-sweep BD, in a separate session after E4's primary.** Fixed now:
  - `color_importance_include` = 0.6e-6 x 3^j, j = -2, -1, +1, +2, at K = 4,096 without fine-tuning;
  - one forked process per scene and point, seeded 0, with rows 1-5 and the scene's `rho_cv` reused;
  - the j = 0 point is E4's process seeded 0;
  - BD-rate and BD-PSNR of `gnvq_cv` against `gnvq_rho0`, `ogc` and `c3dgs` over the five points: `.npz` bytes
    against protocol ii's PSNR, with Amendment 9 a's domain-scaled fit.

### f. E4p: the pilot on train, before E4

- **Train only,** a development scene (Amendment 9), with INRIA's checkpoint through E3p's pinned members.
  - E4p is exploratory: it has no verdict, and no claim about the method rests on it.
  - It runs before E4 (g).
- **What it runs,** in 3 processes:
  - everything in b;
  - c's measurements: each `D_sp`, `D_s`, `v_s` and `SD_pool` for D1 and D2, with no verdict;
  - e's fine-tuning, and e's other secondaries except the threshold sweep.
- **OGC's released code against their published train row** (report only).
  - Their paper publishes one per-scene train row, Table 19 (p. 22), uniform degree reduction of the official model,
    test PSNR in dB:
    - the full model: 21.79;
    - truncation / their projection at degree 2: 21.00 / 21.73;
    - at degree 1: 20.11 / 21.44;
    - at degree 0: 19.48 / 20.01.
  - E4p runs their released code for those rows (`run_exps.py`, stage `core`) on train, from the pinned clone, with
    their evaluation, and places the numbers next to Table 19's. This checks OGC's code. The replication of their VQ
    claim is e's pre-fine-tuning `ogc` minus `c3dgs`, not E4p.
- **Their exact S2 Gram against our 16-probe `M` on train** (report only).
  - **Their `A_i`:** from their released statistics code (`plugin.observation_gram`, S2 weighting, every 8th view
    held out), from the pinned clone.
  - **Ours:** the full-train-view 16 x 16 metric.
  - **Reported**, over the splats with `tr(A_i) > 0`:
    - the relative error sqrt(sum_i ||M_i - A_i||_F^2 / sum_i ||A_i||_F^2);
    - the median, 90th and 99th percentiles of the per-splat ||M_i - A_i||_F / ||A_i||_F;
    - the ratio of the total traces;
    - the number of splats where exactly one of the two traces is zero.
  - **What the error mixes:** the probes' variance with every other difference in `kaggle/RELATED_WORK_OGC.md`
    section 3's table (rasterizer, resolution, principal point, which splats accumulate).
- **What E4p may change.**
  - Only two things: the number of processes per scene, upward; and bug fixes.
  - Each such change is a dated note before any E4 code.
  - It cannot change E4's rows, scenes, metric or bars.

### g. Order

1. This amendment, committed alone (docs).
2. Deep Blending's header read (d) and its dated note, with the seven scenes' archive pins (d).
3. E4p's code, its tests and its dry run (on the CPU stand-in, `bench/gn/dryrun/fake_c3dgs/`).
4. The E4p run and its findings.
5. E4.

Kaggle titles: **"E4p C3DGS fork pilot"** and **"E4 C3DGS gate"**.

**E4 may span several Kaggle sessions.**
- Scenes are assigned to sessions in a dated note before any E4 run, from the feasibility note's estimates (d).
- A scene's probe run, harness phase and 3 processes run in one session.

### h. Unchanged

- The frozen method (Amendment 11 b), with Amendment 14 b's 16 x 16 extension and its placement in this host.
- Protocol ii (Amendment 12 a).
- Amendment 14's rules, unless stated above: the build (13 b), the chunked `eigh` and `det` (13 g), the seeding
  (14 f), the out-of-memory retry and the deadline (14 d), one device copy of the metric (14 g). A failure is
  recorded and every step that does not need its product still runs (14 d), within the drop and `incomplete` rules of
  c and d.

### i. Note (2026-10-01, after Amendment 15 was committed at `573142ac` and before any E4p code): the header read, the archive pins and Deep Blending's feasibility

**What was read, under d's rule.** Order step g.2; nothing else was fetched, decompressed or kept.
- **The archive's zip directory** (Amendment 12 a's archive; zip64, 117 entries, central directory at 14,660,617,193,
  13,708 bytes; 14,660,630,999 bytes). It matches E3p's pins: the archive size, the directory, and all six pinned
  bicycle and train members (`e3p_inria.check_directory`, no mismatch).
- **drjohnson and playroom only:**
  - `cameras.json`, fetched whole with `e3p_inria.fetch_member` (size and CRC32 checked). Only the camera count and
    each camera's width and height were kept; the file was deleted, and its poses were not used.
  - the `.ply` member, inflated one output byte at a time up to and including `end_header`, so no splat byte was
    decompressed. Only the vertex count, the property names and the header's size were kept.
- **Not read:** any `cfg_args` content, any image, splat value, render or metric. 16 HTTP requests, 11.2 s.
- **The scripts are scratch files, not committed:** `header_read.py`, `feas7.py` and `cost7.py` (this session's
  scratchpad, `note_i/`). Their outputs are quoted here; E4p's code will carry the pins.

**The pins** (method 8, deflate, for every member; offsets are the local headers'):

| Scene | Member | Local header offset | Compressed bytes | Bytes | CRC32 |
|---|---|---|---|---|---|
| bonsai | `bonsai/cameras.json` | 2,143,065,696 | 37,076 | 116,695 | `f963b75c` |
| bonsai | `bonsai/cfg_args` | 2,143,102,821 | 136 | 167 | `72f4cedf` |
| bonsai | `bonsai/point_cloud/iteration_30000/point_cloud.ply` | 2,146,082,330 | 260,603,022 | 308,716,644 | `3088ffa4` |
| counter | `counter/cameras.json` | 2,642,254,053 | 30,633 | 95,501 | `a3f7feee` |
| counter | `counter/cfg_args` | 2,642,284,736 | 135 | 169 | `271a40f7` |
| counter | `counter/point_cloud/iteration_30000/point_cloud.ply` | 2,644,544,174 | 261,431,049 | 303,294,620 | `f324e3ad` |
| kitchen | `kitchen/cameras.json` | 7,875,310,892 | 35,453 | 111,500 | `18babec2` |
| kitchen | `kitchen/cfg_args` | 7,875,346,395 | 137 | 169 | `fc9fc092` |
| kitchen | `kitchen/point_cloud/iteration_30000/point_cloud.ply` | 7,878,777,571 | 406,623,576 | 459,380,612 | `84ed81c5` |
| room | `room/cameras.json` | 9,529,779,210 | 39,535 | 124,668 | `57dae086` |
| room | `room/cfg_args` | 9,529,818,792 | 134 | 163 | `c6b59cd5` |
| room | `room/point_cloud/iteration_30000/point_cloud.ply` | 9,531,466,904 | 329,909,932 | 395,158,780 | `1a63439c` |
| truck | `truck/cameras.json` | 13,739,353,050 | 32,225 | 100,406 | `820f7d65` |
| truck | `truck/cfg_args` | 13,739,385,323 | 130 | 162 | `6c485142` |
| truck | `truck/point_cloud/iteration_30000/point_cloud.ply` | 13,741,446,521 | 550,481,900 | 630,225,580 | `44027887` |
| drjohnson | `drjohnson/cameras.json` | 3,123,184,169 | 33,890 | 105,635 | `bcf444d6` |
| drjohnson | `drjohnson/cfg_args` | 3,123,218,111 | 131 | 167 | `a9d24034` |
| drjohnson | `drjohnson/point_cloud/iteration_30000/point_cloud.ply` | 3,124,408,130 | 735,160,560 | 844,479,476 | `66dd518a` |
| playroom | `playroom/cameras.json` | 8,654,411,652 | 29,048 | 89,710 | `e0e302d1` |
| playroom | `playroom/cfg_args` | 8,654,440,751 | 131 | 165 | `0219f3f9` |
| playroom | `playroom/point_cloud/iteration_30000/point_cloud.ply` | 8,654,988,456 | 523,887,782 | 631,438,300 | `f8b2c7e2` |

- **Checks.** Every `.ply` size equals `kaggle/E3_SCOUTING.md` a's, and 1,525 bytes plus the count's digits plus 248
  bytes per splat gives its splat count exactly.
- **Deep Blending's two `.ply` headers:** INRIA's 62 float properties in INRIA's order, a 1,532-byte header, and
  `element vertex` 3,405,153 (drjohnson) and 2,546,116 (playroom), equal to the counts derived from the sizes.
- **The `cameras.json` SHA-1s:** drjohnson `f30d572a0de95ac42492811497ce5b5232cb4df6`, playroom
  `84615f7f2988cbdcbc4f4c4aa0b9d9e279693fd3`.

**Deep Blending's cameras:**

| Scene | Cameras | Image size (every camera) | Test views, every 8th |
|---|---|---|---|
| drjohnson | 263 | 1332 x 876 | 33 |
| playroom | 225 | 1264 x 832 | 29 |

- **The loaded size is taken as the `cameras.json` size.** `cfg_args` was not read (d's rule covers counts and sizes
  only). Both widths are below 1,600, so INRIA's `-r 1` and `-r -1` (`e3p_inria.inria_image_size`) both load this
  size. A larger reduction in `cfg_args` would only shrink every image term, so the figures below are upper ends in
  that respect. E4p's and E4's code read `cfg_args` and apply the size rule.

**Feasibility, by Amendment 14 g's model as `kaggle/E4_DESIGN.md` section 6 applies it** (an estimate, not a
measurement). In GB (10^9 bytes):
- **The sensitivity pass:** C3DGS's per-splat allocations (1,850 bytes per splat), the image state (28 bytes per pixel
  of the largest view) and the binning buffers (36 bytes per tile instance), plus the images (all views, float32 RGB)
  with `--data_device cuda`.
  - Tile instances are bracketed from the two measured scenes: 11.60 (bicycle) to 11.71 (train) per pixel, and up to
    6.09 (train) per splat.
  - The upper end adds train's unexplained 470 bytes per splat.
- **The colour step** adds the larger of our metric copy plus GN-VQ's update chunk (544 bytes per splat plus
  `e3r_memory.UPDATE_CHUNK_BYTES`, 922,746,880 bytes) and OGC's assignment chunk (100,000 x 4,096 float32 scores and
  their inputs, 1.76 GB), added to the upper end. That is conservative: the two peaks are not simultaneous.
- **"x 1.14"** is E3r's reserved-over-allocated ratio (5.38 / 4.71 GB). The T4 has 15.64 GB.
- **Sizes for the five other scenes** are the design's (ceil of the full size / 2, from runs 4-5's committed metadata).
  Their `cameras.json` was not read.

| Scene | Splats | Loaded size, views | Images on CUDA | Sensitivity, images on CUDA | Sensitivity, images on CPU | + colour step, CUDA / CPU | + colour step x 1.14, CUDA / CPU | Verdict |
|---|---|---|---|---|---|---|---|---|
| bonsai | 1,244,819 | 1559 x 1039, 292 | 5.68 | 8.70-9.29 | 3.02-3.62 | 11.05 / 5.38 | 12.60 / 6.13 | CUDA likely |
| counter | 1,222,956 | 1558 x 1038, 240 | 4.66 | 7.64-8.22 | 2.98-3.56 | 9.98 / 5.32 | 11.38 / 6.07 | CUDA likely |
| kitchen | 1,852,335 | 1558 x 1039, 279 | 5.42 | 9.57-10.44 | 4.15-5.03 | 12.38 / 6.96 | 14.11 / 7.93 | CUDA tight; CPU sure |
| room | 1,593,376 | 1557 x 1038, 311 | 6.03 | 9.70-10.45 | 3.67-4.42 | 12.24 / 6.21 | 13.96 / 7.08 | CUDA tight; CPU sure |
| truck | 2,541,226 | 979 x 546, 251 | 1.61 | 6.55-8.08 | 4.94-6.47 | 10.38 / 8.77 | 11.84 / 10.00 | CUDA likely |
| drjohnson | 3,405,153 | 1332 x 876, 263 | 3.68 | 10.50-12.36 | 6.82-8.68 | 15.14 / 11.45 | 17.26 / 13.06 | CUDA unlikely at the colour step; CPU fits |
| playroom | 2,546,116 | 1264 x 832, 225 | 2.84 | 8.02-9.33 | 5.18-6.49 | 11.64 / 8.80 | 13.27 / 10.04 | CUDA likely |

- **Against the design's table:** the five rows' sensitivity columns reproduce it exactly, as do kitchen's, room's
  and truck's colour-step columns. Bonsai's and counter's colour-step columns, bounded by OGC's chunk, read 0.01 GB
  lower in the last digit (11.05 against 11.06; 9.98 against 9.99): the design states that bound as 1.76 GB, and its
  unrounded value is not recorded.
- **drjohnson.**
  - Its sensitivity pass alone fits with images on the GPU: 10.50-12.36 GB, 11.97-14.09 GB x 1.14.
  - The colour step does not, at the upper end: 15.14 GB, 17.26 x 1.14, above the T4. The lower end is 13.28 GB,
    15.14 x 1.14, at the limit.
  - **So d's retry is expected to run there:** a process that runs out of memory at its colour step is retried with
    `--data_device cpu`, which the model puts at 11.45 GB (13.06 x 1.14). The scene's later runs then start on the
    CPU (d).
- **playroom:** 11.64 GB on CUDA (13.27 x 1.14), below kitchen's and room's.
- **By this model every scene fits with images on the CPU,** so no scene is expected to be dropped. Whether a scene
  runs is still settled by d's memory rule, not by this note.
- **What the model leaves out:**
  - tile instances on an indoor Deep Blending view: the per-pixel bracket comes from bicycle and train;
  - host memory: the images under `--data_device cpu`, the host copy of the forked state, and OGC's `G` (1,024 bytes
    per quantized splat, at most 3.49 GB on drjohnson);
  - the time the CPU fallback costs.

**Time per scene** (an estimate). The method is `kaggle/E4_DESIGN.md` section 7's:
- **The base:** E3r's measured train times (`kaggle/gn_e3r/gn3r/`), flat (lower end) to linear in the splat count
  (upper end), with C3DGS's evaluation and protocol ii scaled by test-view pixels.
- **Amendment 15's scope:**
  - six rows per process: three GN-VQ runs, and two OGC runs at 15 iterations taken at our per-iteration rate (an
    assumption);
  - the probe evaluated from its decoded `.npz`;
  - fine-tuning of rows 1, 2 and 5 in all 3 processes, each with its evaluation, decode and protocol ii.
- **Not included:** the saving when `rho_cv` = 0, an out-of-memory retry, and the session setup (396.0 s in E3r).

| Scene | Download, runner, GN passes, CV, probe (s) | One process (s) | Scene: the above plus 3 processes (s) | Hours |
|---|---|---|---|---|
| bonsai | 1,567-1,796 | 3,906-4,265 | 13,285-14,591 | 3.69-4.05 |
| counter | 1,515-1,722 | 3,484-3,807 | 11,966-13,142 | 3.32-3.65 |
| kitchen | 1,552-2,418 | 3,785-5,143 | 12,906-17,849 | 3.59-4.96 |
| room | 1,581-2,176 | 4,021-4,953 | 13,642-17,035 | 3.79-4.73 |
| truck | 1,374-2,963 | 2,322-4,814 | 8,339-17,403 | 2.32-4.83 |
| drjohnson | 1,470-3,965 | 3,113-7,027 | 10,810-25,045 | 3.00-6.96 |
| playroom | 1,434-3,028 | 2,817-5,317 | 9,885-18,979 | 2.75-5.27 |

In all, 80,834-124,045 s (22.5-34.5 GPU-hours), without the threshold sweep.

**This note changes no scene, row or bar** (d). The scenes' assignment to sessions is a later dated note, before any
E4 run (g).

### ii. Note (2026-10-01, after note i (`85b43cff`) and its scripts (`15f263a7`), before any E4p code): report-only measures

**Report only.** This note adds measures to E4p and E4. It changes no row, scene, metric, bar or verdict, and nothing
in it enters Amendment 15 c's primary. No E4p or E4 code or data exists. Items a, b, c and e apply to E4p and E4;
item d to E4p only. Below, "Amendment 15 c" and "Amendment 15 e" name the amendment's sections; a bare letter names
this note's items.

**The geometry the items share,** per scene, from the runner's train split (the cameras protocol ii uses, in COLMAP's
world frame, as E3p set it up; camera-to-world rotation `R_j` and centre `p_j`, OpenCV axes):
- **the scene centre `c`:** the least-squares point nearest the training cameras' optical axes, minimizing
  sum_j ||(I - d_j d_j^T)(c - p_j)||^2 with `d_j = R_j e_z` (closed form);
- **the up axis `u`:** the normalized mean of the training cameras' up vectors, `-R_j e_y`.
- **The centre's conditioning, reported per scene:**
  - the smallest eigenvalue of sum_j (I - d_j d_j^T), divided by the number of training cameras;
  - the minimum, median and maximum distance from `c` to the training camera centres;
  - the share of a's orbit camera centres farther from `c` than the farthest training camera. A rotation about an
    axis through `c` keeps each centre's distance to `c`, so this equals the test cameras' own share.

**a. Novel-direction fidelity.**
- **The cameras:** for each test camera, synthetic cameras orbited about the axis through `c` along `u` by
  `theta` in {-40, -20, -10, +10, +20, +40} degrees (right-handed about `u`): centre `c + R_u(theta)(p - c)`,
  rotation `R_u(theta) R`, the test camera's own intrinsics and image size. `theta` = 0 is the test camera itself.
- **The renders:** protocol ii's renderer and 8-bit quantization (Amendment 12 a), for the uncompressed model and
  every row of every process: rows 1-5 and 2b, and the fine-tuned rows 1, 2 and 5. Each row is its decoded `.ply`,
  as protocol ii reads it.
- **The measure:** per row and angle, the mean over the test cameras of the PSNR of the row's render against the
  uncompressed model's render at the same camera. There is no ground truth at `theta` != 0; this measures fidelity to
  the uncompressed model, which E4's question is about, and not image quality.
- **Reported per angle, the seven angles 0 included:** D1, D2 and Amendment 15 e's differences (`ogc` minus
  `c3dgs`, `gnvq_cv` minus `c3dgs`, `ogc_lam1e6` minus `ogc`, `gnvq_rho0` minus `scalar`, and after fine-tuning `ogc`
  minus `c3dgs` and `gnvq_cv` minus `ogc`), each with `D_sp`, `D_s`, `v_s`, `SD_pool` and `SE_noise` computed as in
  Amendment 15 c.
- **A limit stated in advance:** a synthetic camera can look where no training view did, and the uncompressed model
  can be poor there; the measure compares the codecs with that model, whatever it renders.

**b. Test views by distance to the training views.**
- **The angle of a test camera:** the smallest angle, seen from `c`, between its centre and any training camera's
  centre: min_j angle(`p_t - c`, `p_j - c`).
- **Terciles:** the test cameras sorted by that angle (ties by view index), split into three groups as
  `numpy.array_split` splits them.
- **Reported per tercile:** protocol ii's PSNR per row, and D1 and D2 with `D_sp`, `D_s`, `v_s`, `SD_pool` and
  `SE_noise` as in Amendment 15 c, over that tercile's views; the terciles' angle ranges.

**c. Splat direction coverage, from the full-train-view 16 x 16 metric.**
- **Per splat with `tr(M_i)` > 0:** the effective rank exp(H), with H the entropy of the eigenvalues of
  `M_i / tr(M_i)` (float64, negative eigenvalues set to 0 before normalizing; `bench/gn/batched.py`'s chunked
  eigendecomposition). It runs from 1 (one direction) to 16.
- **Reported per scene,** over all the checkpoint's splats with `tr(M_i)` > 0 and over the colour-quantized splats of
  the probe run:
  - the number of splats with `tr(M_i)` = 0, which are left out;
  - the effective rank's minimum, 10th, 25th, 50th, 75th and 90th percentiles and maximum;
  - the share of the total `tr(M_i)` carried by the lowest-rank tercile (splats sorted by effective rank, ties by
    index, split as in b above).

**d. A power check (E4p only).**
- **From E4p,** for D1 and D2 separately: the within-scene SD `s` over E4p's 3 processes (`sqrt(v_s)`, 2 degrees of
  freedom).
- **The smallest `D_bar` that Amendment 15 c's condition 3 lets pass** with n = 7 and 3 processes, if E4's
  `SD_pool` were `s`: 2 x `s` / sqrt(21). Conditions 1 and 2 set no such threshold.
- **If E4p's observed |`D_s`| is below it,** the report proposes a process count P, the smallest with
  2 x `s` / sqrt(7 P) < |`D_s`|. Adopting it is for a dated note before any E4 code (Amendment 15 f allows only an
  increase), which also restates Amendment 15 c's `SE_noise` with P in place of 3.
- **Limits:** `s` comes from one scene and 2 degrees of freedom; the check is a planning number, not a test. If
  `rho_cv` = 0 on train, D1 is 0 by rule and its check is not computed.

**e. Cost per row.** For every process, per colour row (C3DGS's own VQ, GN-VQ at `rho` = 0, the scalar weighting,
GN-VQ at `rho_cv`, OGC, OGC at `lam` 1e-6): the wall time, the peak allocated GPU memory, and the peak host RSS of the
C3DGS process and its children during that row's computation, with the RSS at its start. Reported per process and as
the mean and range over the processes, per scene. **Also a's fidelity renders:** their wall time per row and in total
(the uncompressed model's included), and their peak allocated GPU memory and peak host RSS (process and children).

**What this note does not change:** E4's rows, scenes, metric, bars, order and Kaggle titles; Amendment 15 d's
memory and failure rules; the secondaries of Amendment 15 e, which stay as it states them.

## Amendment 16 (2026-10-02, after E4p's results and before any E4 code ran on a gate scene, and before any E4q code)

E4p ran on Kaggle ("E4p C3DGS fork pilot"); its bundle is committed unchanged in `kaggle/gn_e4p/gn4p/` (`60c4839b`) and
FINDINGS section 16 reports it (`75c53e09`, re-checked by `bench/gn/check_s16.py`, `82d70578`). This amendment
withdraws E4 (a), pre-registers **E4q, an exploratory dissection on development scenes** (b), fixes two engineering
items (c), and states what is unchanged (d). G0, G1, G2a, H2b, G2c, E2b's criteria, E4p and Amendments 1-15 with their
notes are unchanged except for E4's withdrawal.

### a. E4 is withdrawn

- **When:** before any E4 code ran on a gate scene, and before any gate-scene data was read beyond note i's header
  read (the archive's zip directory; drjohnson's and playroom's camera counts, image sizes and `.ply` headers). The
  only C3DGS-arm code is E4p's, which runs on train alone (`kaggle/gn_e4p_scene.py`, `SCENES`). No pixel, splat value,
  render or metric of bonsai, counter, kitchen, room, truck, drjohnson or playroom has been read.
- **Why,** from E4p on train, a development scene (FINDINGS section 16):
  1. **D1 is about 0 on the standard test views:** +0.0007 dB with `SE_noise` 0.0020, one process of three negative.
  2. **D2 compares unequal effective codebooks at equal K.** OGC reseeds empty clusters at the points of largest
     distortion (`vq.py:74-84` at `49ccae72`) and used all 4,096 entries in every process. GN-VQ starts from C3DGS's
     own codebook and keeps its empty entries (`bench/gn/diagnostics.py:547`), and used 2,310-2,592. OGC's `.npz` was
     275,459 bytes larger. Equal K therefore equalizes neither the effective codebook nor the rate, so PSNR at equal K
     is the wrong primary for the comparison E4 was meant to make.
- **What is withdrawn:** Amendment 15 c (the primary, its bar and E4's verdict), the seven scenes of d as E4's scenes,
  e's E4 secondaries and E4's threshold-sweep session, g.5 and the session-assignment note, and the Kaggle title
  **"E4 C3DGS gate"**. E4 has no verdict, and no result will be reported as E4's. Note ii d's proposal (14 processes
  for D1) is not adopted.
- **What stands:** E4p (Amendment 15 f, notes i and ii) and its results; note i's pins and header read, as records.
- **The gate scenes stay untouched.** No code reads the seven scenes' members beyond note i's header read until a
  later amendment registers a gate on them.

### b. E4q: a dissection on development scenes (exploratory, no verdict)

**Purpose.** E4q measures which of OGC's differences from the frozen GN-VQ moves quality and rate inside C3DGS, and how
the rows compare at equal rate. It has no verdict, no bar and no gate; its purpose is to choose the gate hypothesis.
**The new gate is a later amendment,** written after E4q's findings and before any of that gate's code or data.
**Train and treehill can never be gate scenes.**

**Scenes.**
- **train:** INRIA's 30k checkpoint through E3p's pinned members, as in E4p.
- **treehill,** a development scene since Amendment 10, if it fits:
  1. **A header read under Amendment 15 d's rule:** the camera count and image sizes from its `cameras.json`, and the
     vertex count from its `.ply` header. Its three archive members are pinned (offset, sizes, CRC32) from the archive's
     zip directory. Both go in a dated note before any E4q code.
  2. **Feasibility in that note,** by Amendment 14 g's model as note i applied it. The colour-step term is the larger
     of our metric copy plus GN-VQ's update chunk, and OGC's assignment at chunk 25,000 (c) by E4p's measured
     decomposition: 4 x 25,000 x 4,096 x 4 bytes of score buffers, the chunk's inputs (25,000 x 304 x 4 bytes) and the
     codebook's device copies (5,767,168 bytes), 1,674,567,168 bytes in all (FINDINGS section 16). The reserved-over-
     allocated factor stays E3r's 1.14 (5.38 / 4.71 GB), because E4p's reserved figures are unusable (c).
  3. **The download path:** whether treehill's dataset can be fetched through E3p's download path (the code E3p and E4p
     fetch a scene's images and COLMAP model with), checked from that code and recorded in the note with the reason.
  4. **E4q runs on train only if treehill does not fit even with images on the CPU (2), or cannot be fetched through
     E3p's download path (3).** The note records which, and why. Otherwise treehill runs, starting with its images on
     the GPU (on the CPU if the model puts the GPU above the T4's 15.64 GB), under Amendment 15 d's out-of-memory
     retry.

**Per scene, before any fork process:**
- **as in E4p:** the probe run (C3DGS at K = 4,096 and the default threshold, seeded 0, `--record`, its evaluation
  deferred), E3r's harness phase (the runner, protocol ii of the uncompressed model, the 16 x 16 GN passes), and note
  ii's geometry, reference renders and coverage;
- **`rho_cv`** by Amendment 14 b's cross-validation, as in E4p. On train it is selected again from E4q's own probe run;
  whether it equals E4p's 1e-2 is reported;
- **`lam_cv`, OGC's ridge selected the same way:**
  - `lam` in {1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1};
  - for each, `gram_kmeans(X, G_even, 4096, metric="gram", iters=15, device="cuda", chunk=25000, seed=0, lam=lam)`,
    with `X` the probe's quantizer input reshaped to `[n, 3, 16]` and `G_even` the even-view 16 x 16 metric of the
    probe's quantized splats, unpacked;
  - each codebook through C3DGS's int8 table quantizer at the probe's colour step (`bench/gn/e3r.py`'s
    `C3DGSQuantizer`, as the `rho` cross-validation quantizes its codebooks, `kaggle/gn_e3r_scene.py:581`), with OGC's
    labels;
  - each scored by the clamped dMSE on the odd-indexed train views (`e3r.colour_dmse`, the `rho` cross-validation's
    function and views);
  - `lam_cv` is the lowest score; a tie goes to the smaller `lam`. No test view enters the selection.

**The rows of one fork process.** Amendment 15 b's fork: C3DGS at `2a234af5` unedited, K = 4,096, the process's colour
threshold, no fine-tuning; every row computed at the colour call from the same inputs, every row saved before any is
evaluated, Amendment 15 b's three checks at every save, every row decoded and evaluated with protocol ii and evaluated
by C3DGS's `render_and_eval`.

| Row | Colour codebook |
|---|---|
| `c3dgs` | C3DGS's own `vq_features` (Amendment 15's row 1) |
| `gnvq_cv` | the frozen GN-VQ at `rho_cv` (Amendment 15's row 5): the ladder's starting point |
| `lad_reseed`, `lad_init_tr`, `lad_clip_part`, `lad_no_clip`, `lad_ridge_mean`, `lad_iters15`, `lad_iters50`, `lad_no_final_int8` | the ladder, below |
| `lad_all` | the ladder's factors all at once, below |
| `ogc` | Amendment 15's row 2, at chunk 25,000 (c) |
| `ogc_lamcv` | `ogc` with `lam` = `lam_cv`; if `lam_cv` = 1e-3 it is `ogc` and is not run again |

**The dissection ladder.** Each row changes one factor of `gnvq_cv` toward OGC's VQ and keeps every other: the 16 x 16
metric floored at `rho_cv`; the ridge toward zero, eps = 1e-2; C3DGS's codebook and labels as the warm start; empty
clusters keeping their entries; the clip to the warm start's range with per-cluster acceptance; at most 20 iterations
with the 1e-3 relative-drop rule; the final exact assignment against C3DGS's int8 table. "The loop" is `gn_vq`
(`bench/gn/gn_vq.py`) as `gnvq_cv` runs it.

| Row | The factor | Exactly | Code |
|---|---|---|---|
| `lad_reseed` | reseed empty clusters, as OGC (`vq.py:74-84`) | After each update, its clip and acceptance: every cluster with no member in that iteration's assignment takes the quantizer input `x_i` of a splat of largest distortion, `D_i` = the floored-metric distance of `x_i` to its centroid in the codebook that assignment used; the empty clusters take the top `D_i` in `torch.topk`'s order, one splat each. A reseeded entry is not clipped (it is a data point, on the int8 grid by construction) and has no acceptance test (it has no members). Clusters with members but a zero summed metric keep their entries, as in the loop. | the loop; `diagnostics.direct_distance` for `D_i`; new: the reseed step |
| `lad_init_tr` | OGC's init (`vq.py:21, 31-33`) | The starting codebook is K splats' quantizer inputs, drawn without replacement with probability proportional to `tr(M_i)` + 1e-12 (the unfloored metric, as OGC is given it) from a `torch.Generator` seeded 0; the starting labels are one exact assignment to it under the floored metric (`diagnostics.assign_exact`). The clip range stays the range of C3DGS's codebook, as in `gnvq_cv`, so only the start changes. | the loop; new: the draw, and the clip range passed explicitly |
| `lad_clip_part` | the clip per part | The DC coordinates (the first 3 of the 48 values, `k` = 0) are clipped to the DC range of the warm start, and the 45 AC coordinates to its AC range; acceptance per cluster as in the loop. | the loop; new: per-coordinate bounds |
| `lad_no_clip` | no clip | `gn_vq(..., clip=False)`: neither the clamp nor the acceptance test. | the loop, its existing flag |
| `lad_ridge_mean` | OGC's regulariser (`vq.py:63-73`) in place of the floor | The metric is unfloored (`rho` = 0) in every assignment, the final one included. The update is `c = (sum M_i + r I)^-1 (sum M_i x_i + r xbar)`, with `r` = 1e-3 `tr(sum M_i)` / 16 per cluster and `xbar` the members' mean of `x`, in float64, in place of the loop's `(sum M_i + mu I)^-1 sum M_i x_i`, `mu` = 1e-2 `tr(sum M_i)` / 16. Clusters with a zero summed metric keep their entries, as in the loop. | the loop; new: a ridge-to-mean mode of `diagnostics.update_centroids` |
| `lad_iters15` | OGC's iteration count | at most 15 iterations; the 1e-3 rule kept | the loop, `max_iters` |
| `lad_iters50` | more iterations | at most 50 iterations; the 1e-3 rule kept | the loop, `max_iters` |
| `lad_no_final_int8` | no final assignment against the int8 table, as OGC's host | `gn_vq(..., final_quantized_assignment=False)`: the labels are the last iteration's; C3DGS still quantizes the table at its save. | the loop, its existing flag |

- **`lad_ridge_mean` changes two things that act as one regulariser:** the floor goes, and the ridge changes form
  (toward the cluster's mean, at OGC's 1e-3, in place of toward zero at 1e-2). It counts as two changes below.

**The combined row `lad_all`:** the loop with every OGC choice at once.
- **Exactly:** `lad_init_tr`'s start (the draw seeded 0, with one exact assignment for the starting labels);
  `lad_reseed`'s reseeding; `lad_no_clip` (no clamp, no acceptance test); `lad_ridge_mean`'s metric and update (the floor
  and the eps ridge both replaced by the ridge toward the cluster mean at `lam` 1e-3); exactly 15 iterations, with no
  relative-drop stop; no final assignment against the int8 table. In its place, one exact assignment against the
  returned float codebook, as OGC's last step (`vq.py:87`). Without it the labels would come from before the last
  update, a difference of structure rather than arithmetic. So OGC's sequence is followed: 15 times assign, update,
  reseed, then one assignment.
- **Purpose:** it validates the ladder. `lad_all` minus `ogc` is the residual that the listed factors do not explain:
  the implementation arithmetic. That covers the loop's exact assignment (the lifted float32 argmin with its float64
  guard, which keeps the current label unless another is strictly closer) against OGC's expanded float32 cost and plain
  argmin; the zero-trace splats (the loop gives them their L2-nearest entry, OGC's cost is 0 for every entry and its
  argmin takes the first); and the loop's float64 update on the GPU against OGC's float32 sums and float64 solve on the
  CPU, with its clamps (`vq.py:69-71`).
- **Reported:** `lad_all` minus `ogc` in every measure, as the other differences; and the labels that agree between
  `lad_all` and `ogc`, as a count and a fraction of the quantized splats, with the number of codebook entries both use.
- **`lad_all` is not a ladder variant for the selection below,** and it does not run at the sweep's other points.

**Processes.**
- **The default point (j = 0):** two processes per scene, seeded 0 and 1, each running every row above. The second
  process is for noise: each difference is reported per process and as `D_s`, `v_s` = (`D_0` - `D_1`)^2 / 2 (1 degree
  of freedom) and `SE_noise` = sqrt(`v_s` / 2).
- **The rate sweep, train only:** `color_importance_include` = 0.6e-6 x 3^j, j = -2, -1, +1, +2, at K = 4,096; one
  process per point, seeded 0, with the rows `c3dgs`, `ogc`, `ogc_lamcv`, `gnvq_cv` and the best ladder row; the j = 0
  point is the seed-0 default process. `rho_cv` and `lam_cv` are the default point's.
- **The best ladder row, fixed now:** the highest protocol-ii test PSNR at j = 0 among the eight single-factor ladder
  rows (`lad_all` excluded), as the
  mean of the two processes, rounded to 9 decimals; a tie goes to the row with fewer changes (`lad_ridge_mean` two, every
  other row one), then to the earlier row in the ladder's order. It is chosen among ladder rows whether or not it beats
  `gnvq_cv`.
- **Order within a scene:** the probe run and the selections, then the two default processes, then the sweep (the
  best ladder row needs the default point's results). Treehill runs on the second GPU.
- **BD, report only:** per row, `.npz` bytes against protocol ii's test PSNR at the five points; BD-rate and BD-PSNR with
  Amendment 9 a's domain-scaled fit (`g2.bd_rate_scaled`, `bd_psnr_scaled`), of every row against `gnvq_cv`, of `ogc`
  and `ogc_lamcv` against `c3dgs`, and of `ogc_lamcv` against `ogc`.

**Measures, per row and process (report only):**
- **E4p's per-row record:** protocol ii (PSNR, SSIM, LPIPS, per view), C3DGS's evaluation, the `.npz`'s bytes (bytes,
  MiB, MB), each array's compressed and uncompressed size, the index entropy and distinct indices, GN-VQ's iterations,
  stopping and last relative drops, the three checks, and the per-row time, GPU peak and host RSS.
- **Note ii's a, b, c and e** (d was E4's power check), with each difference computed as above: every ladder row minus
  `gnvq_cv`; `lad_all` minus `ogc`; `gnvq_cv` minus `c3dgs`; `ogc` minus `c3dgs`; `gnvq_cv` minus `ogc`; `gnvq_cv` minus
  `ogc_lamcv`; `ogc_lamcv` minus `ogc`.
- **Fidelity also as the PSNR of the pooled MSE:** per row and angle, 10 log10(1 / m), with m the mean over the test
  cameras of each render's MSE against the uncompressed model's (note ii a's 8-bit renders on [0, 1]); and its
  differences as above.
- **The quantizer state at every save:** each row's and the probe's DC and AC scale and zero point.
- **Each table's range against its int8 grid:** for every row's codebook as installed, the minimum and maximum of its DC
  values (the first 3 of 48) and of its AC values (the other 45), each part's grid at that save,
  [scale x (-128 - zero point), scale x (127 - zero point)], and the number of codebook values outside it.
- **Codewords used per row:** the distinct codebook entries among the quantized splats' stored indices, and so the
  empty entries.
- **The OGC chunk check** (c).

**Failures, memory and the deadline:** as in E4p (Amendment 15 d's out-of-memory retry, a failed row recorded and
every row that does not need it still run; no new C3DGS run after 11.5 h minus the 1,800 s reserve). E4q has no
primary row, so no failure reruns a process.

**Not in E4q:** fine-tuning; the rows `gnvq_rho0`, `scalar` and `ogc_lam1e6` (1e-6 is a point of the `lam` grid); any
gate scene; any verdict.

**Kaggle title "E4q C3DGS dissection"**; bundle `E4q_bundle.zip`, arcname `gn4q/`.

### c. Engineering

1. **The reserved-memory recording.** E4p's processes recorded a reserved peak below their allocated peak (FINDINGS
   section 16): the hooks reset the allocator's peak statistics at every row (`kaggle/e4p_hooks.py:139`) and fold
   only the allocated peak (`:130`). From E4q on, the hooks fold `torch.cuda.max_memory_reserved()` per row the same
   way, and the wrapper records the folded value beside the allocated one (`kaggle/e3q_c3dgs_run.py:305-308`).
2. **OGC's chunk is 25,000** in E4q and in any later gate run; OGC's call is otherwise Amendment 15 b's. The chunk
   only batches the assignment, and E4q measures whether that changes anything: on train, in the seed-0 default
   process, one more `gram_kmeans` call at chunk 100,000 and `lam` 1e-3 on the same inputs (not saved, not evaluated).
   Reported: whether its labels equal those at chunk 25,000 exactly; if not, how many differ (count and fraction) and
   the two codebooks' largest absolute difference. Neither result changes a row.

### d. Unchanged

- The frozen method (Amendment 11 b) with Amendment 14 b's 16 x 16 extension and its placement in C3DGS. Its global clip
  (`bench/gn/gn_vq.py:179`) stays: the per-part clip is the ladder's `lad_clip_part`, not a fix.
- Protocol ii (Amendment 12 a).
- Amendments 1-15 and their notes, except E4's withdrawal in a. E4p's results stand as FINDINGS section 16 reports them.

### e. The runtime estimate (before any E4q code; an estimate)

From E4p's measured train parts (`kaggle/gn_e4p/gn4p/`), by note i's method: train at E4p's times; treehill flat (lower
end) to linear in the splat count (upper end, 3,783,761 / 1,026,508 = 3.69), C3DGS's evaluation, protocol ii and the
fidelity renders scaled by test-view pixels; the ladder rows at E4p's GN-VQ rate (4.99 s per iteration), `lad_iters50`
from 20 to 50 iterations, `lad_all` at 17 (15 iterations, the starting and the final assignment); OGC at chunk 25,000
taken at E4p's chunk-100,000 time (26.5 s, an assumption); treehill's fetch from E3p's bicycle fetch scaled by the
`.ply` bytes, its download at E3r's bicycle download (290.9 s). The script is `bench/gn/e4q_estimate.py`, committed with
E4q's code.

| | train | treehill (2 default processes) |
|---|---|---|
| probe, harness, `rho` and `lam` cross-validation, note ii | 1,705 s | 1,794-5,522 s |
| one default process (13 rows) | 2,468-2,618 s (the seed-0 one, with the chunk check) | 2,390-9,362 s |
| the sweep's four points (5 rows each) | 3,816-4,348 s | |
| the scene | 10,432-11,263 s | 6,575-24,247 s |
| without C3DGS's evaluation of the 12 rows after row 1 | 7,987-8,818 s | 5,203-19,191 s |

- **One session:** with the scenes on the two GPUs and setup (429.0 s in E4p), 10,861-24,676 s, 3.0-6.9 h, inside
  the 11 h start cutoff; 4.7-9.9 GPU-hours.
- **Not included:** treehill on the CPU retry, the header read, and chunk 25,000's own speed.

### i. Note (2026-10-02, after Amendment 16 was committed at `1f0b7e15` and before any E4q code): treehill's header read, pins, download path and feasibility

**What was read, under Amendment 15 d's rule** (b.1): counts and sizes only. Nothing else was fetched, decompressed or
kept.
- **INRIA's archive's zip directory** (117 entries, central directory at 14,660,617,193, 13,708 bytes; 14,660,630,999
  bytes). It matches E3p's six pins and note i's 21 (Amendment 15 i): no mismatch.
- **treehill's `cameras.json`,** fetched whole with `e3p_inria.fetch_member` (56,201 bytes, CRC32 `fc3331eb`, both
  checked; SHA-1 `1dda79c011bc188d903852bc5758b0043d337316`). Only the camera count and each camera's width and height
  were kept; the file was deleted, and its poses were not used: **141 cameras, every one 5068 x 3326**, so 18 test
  views (every 8th).
- **treehill's `.ply`,** inflated one output byte at a time up to and including `end_header`: INRIA's 62 float
  properties in INRIA's order, a 1,532-byte header, **`element vertex` 3,783,761**, equal to the count
  `kaggle/E3_SCOUTING.md` a derived from the member's size, and header plus 248 bytes per splat equals the size.
- **Not read:** any `cfg_args` content, any image, splat value, render or metric. 10 HTTP requests to INRIA's archive.

**The pins** (method 8, deflate; offsets are the local headers'):

| Member | Local header offset | Compressed bytes | Bytes | CRC32 |
|---|---|---|---|---|
| `treehill/cameras.json` | 12,388,503,807 | 18,444 | 56,201 | `fc3331eb` |
| `treehill/cfg_args` | 12,388,522,302 | 137 | 171 | `1e467944` |
| `treehill/point_cloud/iteration_30000/point_cloud.ply` | 12,389,905,633 | 829,119,730 | 938,374,260 | `a9916ad2` |

**The download path (b.3): treehill can be fetched through it.**
- **From the code:** E3p, E3r and E4p fetch a scene's dataset with `gn_e2_scene.ensure_data`, which sends a MipNeRF360
  scene to `tilequant_run4.download_scene`: only `images/`, `images_<factor>/`, `sparse/` and `poses_bounds.npy` of the
  scene, read from the scene's zip with range requests. For treehill, `SCENE_META` names the zip `360_extra_scenes`,
  `MIPNERF360_ZIPS` gives its URL, and `mcmc.sh` gives the factor 4. It is the function run 4 fetched treehill with.
- **From the zip's central directory** (4,488,140,217 bytes, 1,278 entries; names and sizes only, 4 HTTP requests):
  the downloader's own pattern selects 286 members, all with `SCENE_META`'s bytes and image count: `images/` 141 files,
  1,288,424,692 bytes; `images_4/` 141 files, 128,303,200 bytes; `sparse/` 3 files, 39,353,154 bytes;
  `poses_bounds.npy`, 19,304 bytes.

**The loaded image size is not in what the rule lets this note read.** `cameras.json` records the full 5068 x 3326;
INRIA's loader reads the image set `cfg_args` names. The rows below take `images_4` at `-r 1`, as bicycle's pinned
`cfg_args` gives for a MipNeRF360 outdoor scene (`e3p_inria.CFG_ARGS`), at ceil(full / 4) = 1267 x 832, as bicycle's
`images_4` is (4946 x 3286 to 1237 x 822); and `images_2` as a sensitivity.

**Feasibility (b.2),** by Amendment 14 g's model as note i applied it (an estimate, not a measurement), in GB (10^9
bytes), x 1.14 for E3r's reserved-over-allocated ratio, against the T4's 15.64 GB. The colour-step term is the larger
of our metric copy plus GN-VQ's update chunk for every splat (2.98 GB) and OGC's assignment at chunk 25,000
(1,674,567,168 bytes, 1.67 GB): 2.98 GB.

| Loaded size | Images on CUDA | Sensitivity, images on CUDA | Sensitivity, images on CPU | + colour step x 1.14, CUDA (lower / upper) | + colour step x 1.14, CPU (upper) |
|---|---|---|---|---|---|
| `images_4`, 1267 x 832, 141 views | 1.78 | 9.25-11.42 | 7.47-9.64 | 13.95 / 16.42 | 14.39 |
| `images_2`, 2534 x 1663 (sensitivity) | 7.13 | 16.01-17.80 | 8.88-10.67 | 21.65 / 23.69 | 15.57 |

- **Verdict (b.4): treehill runs in E4q.** It fits with images on the CPU (14.39 GB at the upper end), and it can be
  fetched through E3p's download path.
- **It starts with its images on the CPU:** with them on the GPU the upper end, 16.42 GB, is above the T4 (the lower
  end is 13.95 GB). Amendment 15 d's retry rule then has no further fallback: a CPU run out of memory drops the scene
  before any of its results.
- **The fit rests on the `images_4` assumption.** E4q's code reads treehill's `cfg_args` and applies INRIA's size rule
  before treehill's first C3DGS run; if the loaded size is not 1267 x 832, the fit above was not shown (b.4), treehill
  is not started, and the reason is recorded. At `images_2` the CPU upper end is 15.57 GB, at the limit.
- **What the model leaves out,** as in note i: tile instances on a treehill view (bracketed from bicycle and train); the
  host memory (the images under `--data_device cpu`, 1.78 GB; OGC's `G`, 1,024 bytes per quantized splat; the forked
  state's copy); and the time the CPU images cost, which Amendment 16 e did not include.

**The scripts** are `kaggle/gn_e4q_note_i/header_read.py` (network, range requests only) and `feas.py` (offline), with
their outputs `header_read.json` and `feas.json`; they are committed after this note. **This note changes no rule,
row or scene.**
