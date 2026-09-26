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
