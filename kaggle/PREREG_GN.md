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
