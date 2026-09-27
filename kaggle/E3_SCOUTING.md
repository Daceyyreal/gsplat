# E3 scouting: INRIA checkpoints and host codecs for the frozen GN-VQ

Written 2026-09-27 on `bench/gn-vq`. This is a scouting report for E3's design, which is not written
yet. Nothing here is pre-registered, no pipeline code changed, and nothing ran on Kaggle.

The method to port is **`gn_vq_cvfloor`**, frozen after E2c:
- **Metric and variant:** E2's GN-VQ (eps = 1e-2, at most 20 iterations, clip, final quantized
  assignment) with the metric `M_i + rho * tr(M_i) / 15 * I`.
- **Selection:** `rho` chosen per scene and K by training-view cross-validation over
  {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3}.
- **Definition:** `PREREG_GN.md` Amendment 11 b; results in `FINDINGS.md` section 12.

**How the sources were read.**
- **Repositories:** GitHub pages, raw files fetched to standard output, and the GitHub REST API for
  licenses and last commits (8 unauthenticated calls). Nothing was cloned.
- **Papers:** their arXiv HTML. Tables were extracted from the raw HTML, not from a summary.
- **INRIA archive:** read without downloading it. A HEAD request gave its size. HTTP range requests
  (13,708 bytes of zip directory, plus the first few KB of 14 members) gave its file list, the 13
  `cfg_args` files and one `.ply` header. They were read in memory; only a JSON listing was written, to
  the session scratchpad.
- **Estimates:** built from this repository's committed E2, E2b and E2c bundles, with the construction
  shown.

Anything not read in one of these ways is marked **unverified**.

## a. INRIA pretrained 3DGS models

**Download** (graphdeco-inria/gaussian-splatting `README.md`, top table): "Pre-trained Models (14 GB)",
`https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip`.
- **Size:** a HEAD request gives `Content-Length: 14660630999` (14.66 GB decimal) and `Last-Modified: Wed,
  05 Jul 2023 22:10:27 GMT`; range requests are accepted.
- **Contents:** the zip has 117 entries (16,755,936,764 bytes uncompressed).
- **README caveat:** "The pre-trained models were created with the release codebase. This code base has
  been cleaned up and includes bugfixes, hence the metrics you get from evaluating them will differ from
  those in the paper."
- **Source datasets** are separate. The README links "T&T+DB COLMAP (650MB)"; MipNeRF360 comes from its
  authors' site.

**Scenes: 13, one top-level directory each** (from the zip's directory):
- **MipNeRF360 (9):** bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill;
- **Tanks & Temples (2):** train, truck;
- **Deep Blending (2):** drjohnson, playroom.

**Iterations:** every scene has `point_cloud/iteration_7000/point_cloud.ply` and
`point_cloud/iteration_30000/point_cloud.ply`, plus `cfg_args` and `input.ply`. The README's training
default is `--iterations` "`30_000` by default", with saves at "`7000 30000`".

**Training flags** (each scene's `cfg_args`):
- **`eval=True` in all 13**, so the test views were held out. INRIA's reader takes every 8th image by
  sorted name (`llffhold=8`, `scene/dataset_readers.py`).
- **Images:** `images_4` for the five outdoor MipNeRF360 scenes, `images_2` for the four indoor ones,
  full-resolution `images` for T&T and Deep Blending. All with `resolution=1` and `sh_degree=3`.
- **gsplat's COLMAP parser** (`examples/datasets/colmap.py`) sorts image names and uses
  `indices % test_every == 0` with `test_every = 8`, so its test views are the same.
  - It builds downsampled images itself from the full-resolution JPEGs, where INRIA read the dataset's
    `images_4` / `images_2` directly. Absolute PSNRs can therefore differ slightly from INRIA-protocol
    numbers (unverified by how much).

**Splat counts are not published**, but they follow exactly from the file sizes:
- **Layout:** bicycle's 30k `.ply` header (read from the archive) is the standard 62 float32 properties
  per splat (`x y z`, `nx ny nz`, `f_dc_0-2`, `f_rest_0-44`, `opacity`, `scale_0-2`, `rot_0-3`: 248 bytes)
  and declares `element vertex 6131954`. That fixes the header at 1,532 bytes, i.e. 1,525 bytes plus the
  count's digits.
- **Derived:** every other file size then gives an exact integer count (26 of 26 files).

| Scene | Dataset | Splats, 30k | Splats, 7k | 30k `.ply` bytes |
|---|---|---|---|---|
| bicycle | MipNeRF360 | 6,131,954 | 3,616,103 | 1,520,726,124 |
| garden | MipNeRF360 | 5,834,784 | 4,386,142 | 1,447,027,964 |
| stump | MipNeRF360 | 4,961,797 | 3,807,536 | 1,230,527,188 |
| treehill | MipNeRF360 | 3,783,761 | 2,399,849 | 938,374,260 |
| flowers | MipNeRF360 | 3,636,448 | 2,461,555 | 901,840,636 |
| drjohnson | Deep Blending | 3,405,153 | 1,902,253 | 844,479,476 |
| playroom | Deep Blending | 2,546,116 | 1,734,607 | 631,438,300 |
| truck | Tanks & Temples | 2,541,226 | 1,732,378 | 630,225,580 |
| kitchen | MipNeRF360 | 1,852,335 | 1,684,648 | 459,380,612 |
| room | MipNeRF360 | 1,593,376 | 1,130,125 | 395,158,780 |
| bonsai | MipNeRF360 | 1,244,819 | 1,157,141 | 308,716,644 |
| counter | MipNeRF360 | 1,222,956 | 1,029,406 | 303,294,620 |
| train | Tanks & Temples | 1,026,508 | 559,263 | 254,575,516 |

**Totals and range:**
- 39,781,233 splats in the 13 scenes at 30k iterations; 9,865,765,700 bytes of 30k `.ply` files.
- **Range:** 1.03M (train) to 6.13M (bicycle). Only five scenes are above 3.5M, so "3-6M splats" covers
  the outdoor scenes and drjohnson, not the set.

**Cross-checks** (these also establish which checkpoints and units the published tables use):
- **Per-dataset means of these files:**
  - MipNeRF360: 833.89 MB decimal = 795.26 MiB, 3,362,470 splats;
  - T&T: 442.40 MB = 421.91 MiB, 1,783,867 splats;
  - Deep Blending: 737.96 MB = 703.77 MiB, 2,975,634 splats.
- **C3DGS** (Table 1, "3D Gaussian Splatting" rows: 795.26 / 421.90 / 703.77 "MB") matches the MiB means.
  Its text says "the reconstructions from Kerbl et al. [13] were used". **C3DGS's "MB" is MiB.**
- **POTR** (Table I "Baseline" row: 834 / 442 / 738 MB; 3,362 / 1,784 / 2,976 thousand splats) matches the
  decimal-MB means and the splat counts. **POTR's "MB" is 10^6 bytes**, and its baseline is this model set.
- **MesonGS++'s configs** state "Original size: 1450.3 MB" (bicycle) and "376.9 MB" (room). These are
  bicycle's and room's 30k files in MiB (1,520,726,124 / 2^20 = 1,450.28; 395,158,780 / 2^20 = 376.86).
  Its code defines `BIT2MB_SCALE = 8 * 1024 * 1024`. **MesonGS++'s "MB" is MiB.**
- **Consequence:** cross-paper size comparisons carry a 4.9% unit difference unless converted.

**Using these checkpoints in gsplat:**
- **SH layout:** INRIA's `save_ply` writes `f_rest` as `_features_rest.transpose(1, 2).flatten(start_dim=1)`,
  i.e. channel-major (`[N, 3, 15]`). gsplat's `shN` is `[N, 15, 3]`, so a loader must transpose. Scales
  (log), opacities (logit) and quaternions are stored pre-activation, as in gsplat's checkpoints.
- **Square crop:** gsplat's `PngCompression` requires a square N and crops the lowest-opacity splats
  (`png_compression.py`, `_crop_n_splats`). On these checkpoints that removes 14 (kitchen) to 3,612
  (flowers) splats, 0.001%-0.158% (counter: 1,931, 0.158%). E3a must count them.

## b. Candidate hosts

The four requested hosts, plus MesonGS (MesonGS++'s ECCV 2024 predecessor, `ShuzhaoXie/MesonGS`, last
commit 2024-10-24, license file = the INRIA Gaussian-Splatting License), which MesonGS++ supersedes.

### C3DGS: "Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis" (Niedermayr et al.)

- **Repo:** https://github.com/KeKsBoTer/c3dgs.
  - **License:** `LICENSE.md` is the INRIA "Gaussian-Splatting License" (GitHub reports NOASSERTION), for
    non-commercial research and evaluation.
  - **Last commit:** `2a234af55f`, 2025-09-17, "added darkmode and fixed navbar", a docs change; the code
    is on the same default branch, `master`.
- **Where the SH VQ happens:** `compression/vq.py`:
  - `compress_color` calls `vq_features` on the splats whose color importance is at most
    `color_importance_include` (the rest keep their own colors).
  - `vq_features` runs `VectorQuantize.update` for `steps` batches of `vq_chunk` random splats. Defaults
    (`arguments/__init__.py`, `CompressionParams`): codebook `2**12`, 100 steps, batch `2**18`, decay 0.8.
- **Distance:** the CUDA kernel `submodules/weighted_distance/weighted_distance.cu`, despite its name,
  computes **plain unweighted squared Euclidean distance** (`distance()` sums `diff * diff`) and returns
  the argmin over the codebook.
- **Weights:** only in the update. `update` is an EMA of the importance-weighted mean per codeword:
  `scatter(x * importance)` / `scatter(importance)`, `ema_inplace` with the decay.
  - The importance is `calc_importance`: the absolute gradient of `rendering.sum()` with respect to each
    SH coefficient, summed over the train views and divided by their pixels. So it is a first-order
    sensitivity, not a Gauss-Newton second moment.
  - With the default `color_weights_per_param = False` it is reduced per splat with `amax(-1)`, then
    normalized by its maximum.
- **What is quantized:**
  - With the default `color_compress_non_dir = True` the VQ vector is all 16 SH coefficients, DC included:
    48 values (`get_features.flatten(-2)`, coefficient-major, the same order as gsplat's `shN.reshape(N, -1)`).
  - With `False` it is the 15 AC coefficients (45 values), which is exactly the space the frozen GN metric
    is defined on (`tr(M_i) / 15`).
- **Centroid quantization:** `torch.ao.quantization.FakeQuantize(dtype=torch.qint8)` per tensor (8-bit,
  observed scale and zero point) on `features_dc` and `features_rest`, stored with
  `torch.quantize_per_tensor(...).int_repr()`. The indices are int32 in `np.savez_compressed`.
- **After VQ:** `finetune_iterations = 5000` of quantization-aware fine-tuning (paper: "5000 optimization
  steps"). It moves the codebook, so a GN-VQ codebook would not survive unchanged.
- **Replacing the objective with GN-VQ:**
  - **Interface:** `vq_features(features, importance, codebook_size, ...)` returns `(codebook [K, D],
    indices [N])`. A GN-VQ drop-in can return the same two tensors. It needs the per-splat `M_i` for the
    VQ'd splats instead of `importance`; the current interface takes only a per-splat scalar, so **`M_i`
    cannot be passed in without changing the call** (`compress_color`).
  - **Layout:** a `[N, 45]` feature layout matches `bench/gn`'s `x` and packed `M` (`[N, 120]`, 15x15 per
    splat shared across channels) if `color_compress_non_dir = False`. With the default (DC included), the
    metric would have to cover 16 coefficients, which is a change to the frozen method.
  - **Computing `M_i`:** C3DGS renders with the INRIA rasterizer (`submodules/diff-gaussian-rasterization`,
    with an indexed variant). `M_i` would come from `bench/gn`'s gsplat-based GN pass on the same splats;
    parity between the two rasterizers on these checkpoints is unverified.
  - **Clip:** the clip and the final quantized assignment would use C3DGS's per-tensor qint8 quantizer
    instead of gsplat's 6-bit global min / max.
  - **Selection:** the 7-`rho` cross-validation needs the even / odd train-view split and C3DGS's decode
    path.
- **Dependencies:** `environment.yml` pins `python=3.8`, `pytorch-cuda=12.1`, `cuda-toolkit=12.1`,
  `pytorch-scatter`, `plyfile=0.8.1`, and two local CUDA extensions (`diff-gaussian-rasterization`,
  `weighted_distance`).
  - **Kaggle:** both extensions are small `CUDAExtension` builds. Building them against Kaggle's torch and
    CUDA 12 on sm_75 looks likely, but is **unverified**. Kaggle's torch version was never recorded by this
    project: E0-E2c's bundles hold only the local fixture's `2.11.0+cpu`, and the notebooks install
    `torch==2.9.1` (cu126) only if the image's torch is below 2.7. `torch_scatter` needs a wheel matching
    that torch.
- **Inputs:** an INRIA model directory (the `.ply` plus `cfg_args`) and the COLMAP dataset. The paper
  states "For Mip-Nerf360, Tanks&Temples and Deep Blending the reconstructions from Kerbl et al. [13] were
  used".
- **Evaluation:**
  - `render_and_eval` in `compress.py` and `metrics.py`: test views of the INRIA split, LPIPS
    `net_type="vgg"`, PSNR and SSIM.
  - **Size:** `os.path.getsize(npz) / 1024**2`, i.e. **MiB** of the `.npz` alone, labelled "MB".
  - Resolution: the model's training images (per `cfg_args`); unverified in code.
- **Reported numbers** (arXiv 2401.02436, **Table 1**, "Quantitative comparison to 3D Gaussian Splatting.
  Size is measured in Megabytes."). One operating point per dataset (codebook 4,096), not an RD curve:

  | Dataset | 3DGS PSNR / SSIM / LPIPS / size | C3DGS PSNR / SSIM / LPIPS / size | Ratio |
  |---|---|---|---|
  | Mip-NeRF360 | 27.21 / 0.815 / 0.214 / 795.26 | 26.981 / 0.801 / 0.238 / 28.80 | 26.23 |
  | Tanks&Temples | 23.36 / 0.841 / 0.183 / 421.90 | 23.324 / 0.832 / 0.194 / 17.28 | 23.26 |
  | Deep Blending | 29.41 / 0.903 / 0.243 / 703.77 | 29.381 / 0.898 / 0.253 / 25.30 | 27.81 |

### LightGaussian (Fan et al., NeurIPS 2024)

- **Repo:** https://github.com/VITA-Group/LightGaussian.
  - **License:** `LICENSE.md` is the INRIA Gaussian-Splatting License (GitHub: NOASSERTION).
  - **Last commit:** `6676b983e7`, 2024-12-30, "fix run_train_densify_prune.sh".
- **Pipeline** (README):
  - "Prune & Recovery" (`prune_finetune.py`), then "SH distillation" (`distill_train.py`, to SH degree 2),
    then "VecTree Quantization" (`vectree/vectree.py`, `scripts/run_vectree_quantize.sh`).
  - The first two are training stages.
- **Where the SH VQ happens:** `vectree/vectree.py`, `Quantization.quantize`.
  - It vector-quantizes only the least important `vq_ratio = 0.6` of the splats (`torch.topk` on the
    global significance `imp_score.npz` from the pruning stage). The top 40% keep their SH in fp16.
  - `iteration_num = 1000` batches of `VQ_CHUNK = 80000` splats.
  - **Codebook:** `codebook_size = 2**13` (8,192), stored `.half()`. Indices are bit-packed.
- **Distance:** `vectree/vq.py` `EuclideanCodebook.forward`, `dist = -torch.cdist(flatten, embed, p = 2)`
  (Euclidean).
- **Weights:** the update's `cluster_size` and `embed_sum` are weighted by the significance, EMA decay
  0.8; dead codes are replaced by the most important samples (`k_expire = 10`).
- **Feature vector:** `feats[:, 6:6 + sh_dim]`, DC and AC together; `sh_dim` is 27 for the default
  `--sh_degree 2` (after distillation) and 48 for degree 3.
- **Replacing the objective with GN-VQ:**
  - The VQ reads a `.ply` and a significance file. The loop could be replaced by GN-VQ given `M_i` for the
    VQ'd 60%.
  - **Degree-2 metric:** after distillation the metric would be over degree-2 SH (8 AC coefficients),
    where the frozen method's `tr(M_i) / 15` is defined for 15. That is a change to the method, as is DC
    inclusion. At `--sh_degree 3` without distillation the space matches but the pipeline is not
    LightGaussian's published one.
  - **Interface:** there is no argument for a per-splat matrix.
- **Dependencies:** `environment.yml` pins `python=3.9`, `pytorch=1.12.1`, `cudatoolkit=11.6`,
  `torchvision=0.13.1`, and two submodules (`compress-diff-gaussian-rasterization`, `simple-knn`; the first
  is a git submodule not in the tree). Building against a current Kaggle image means porting from
  torch 1.12 / CUDA 11.6: **likely to need work, unverified**.
- **Inputs:** a trained 3DGS checkpoint and the dataset; the prune and distillation stages retrain.
- **Evaluation:**
  - `metrics.py` uses LPIPS `net_type="vgg"`.
  - **Size:** `vectree.py` zips `extreme_saving/` and reports `size / 1024.0 / 1024.0` as "MB", i.e. MiB of
    the zip.
- **Reported numbers** (arXiv 2311.17245, **Table 1**, "Quantitative Comparisons in Real-world Large-scale
  Scenes"). No Deep Blending column. Its 3D-GS rows (734 MB, 411 MB) do not equal the INRIA means above,
  so its baselines are not exactly these checkpoints; unit unverified.

  | Row | Mip-NeRF360 FPS / size / PSNR / SSIM / LPIPS | Tanks & Temples FPS / size / PSNR / SSIM / LPIPS |
  |---|---|---|
  | 3D-GS | 134 / 734MB / 27.21 / 0.815 / 0.214 | 154 / 411MB / 23.14 / 0.841 / 0.183 |
  | 3D-GS* | 144 / 782MB / 27.40 / 0.813 / 0.217 | 106 / 433MB / 23.66 / 0.845 / 0.178 |
  | Compressed 3D-GS* (C3DGS, rerun) | 152 / 28MB / 27.03 / 0.802 / 0.238 | 202 / 17MB / 23.54 / 0.838 / 0.189 |
  | LightGaussian | 237 / 45MB / 27.13 / 0.806 / 0.237 | 357 / 25MB / 23.44 / 0.832 / 0.202 |

### MesonGS++ (post-training, with hyperparameter search)

- **Repo:** https://github.com/mmlab-sigs/mesongs_plus.
  - **License:** `LICENSE.md` is MIT. The repository also ships INRIA-derived CUDA rasterizers under
    `splatwizard/_cmod/rasterizer/`, whose licensing is **unverified**.
  - **Last commit:** `0004153458`, 2026-05-07, "add result.xlsx".
  - Built on the SplatWizard framework (arXiv 2604.26799).
- **Pipeline** (paper Table I stages): prune, voxelize (octree), "Replace", cluster (the SH VQ), RAHT on the
  other attributes, then the hyperparameter search (0-1 ILP over bit-widths), with fine-tuning.
  - The geometry goes through MPEG G-PCC (`tmc3`).
- **Where the SH VQ happens:** `splatwizard/model_zoo/mesongs_plus/model.py`, task `vq_fe`, calls
  `meson_utils.vq_features(self._features_rest.detach().flatten(-2), self.imp, self.codebook_size,
  self.batch_size, self.steps, sh_keep_topk=..., quantize_kept=True, kept_quant_bits=8, ...)`.
  - The docstring says "borrowed from c3dgs": the same `VectorQuantize` EMA update, with importance
    weights, and the distance from `splatwizard._cmod.weighted_distance` (C3DGS's kernel, by name;
    unverified that it is identical).
  - **What is quantized:** the 45 AC coefficients only (`_features_rest`), the space the frozen metric is
    defined on.
  - The shipped MipNeRF360 script (`scripts/eval_mesongs_plus_360.sh`) sets `CODEBOOK_SIZE=4096`,
    `NUM_BITS=16` and `SH_KEEP_TOPK=1000000`, overriding the per-scene YAML's `cb: 2048`.
  - **Kept splats:** the 1,000,000 most important splats keep their full SH, block-quantized at 8 bits
    (`quantize_kept_sh`), and only the rest use the codebook. All points still take part in the
    clustering (code comment, "Step 1").
- **Replacing the objective with GN-VQ:**
  - **Interface:** the same as C3DGS (`vq_features(features, importance, ...)` returning codebook and
    indices, plus a `quant_info` dict), on `[N, 45]` features in gsplat's order. `M_i` cannot be passed
    through the current signature.
  - **Clip:** the codebook's bit-width is part of the ILP search, so the clip's "warm-start range" would
    have to be defined against whichever quantizer the search picks.
  - **Kept splats:** GN-VQ would change only the non-kept splats' SH.
- **Dependencies:**
  - README: `torch==2.4.0+cu121`, `torchvision==0.19.0`, `torch-scatter` for that torch;
    `requirements.txt` adds `vector-quantize-pytorch==1.22.0`, `open3d`, `pycolmap`, `torchac`,
    `constriction`, `point_cloud_utils`, `trimesh`, `kornia` and others.
  - `setup.py` builds many CUDA extensions under `splatwizard/_cmod/` (arithmetic coder, fused SSIM, grid
    encoder, kNN, Lanczos resampling, rANS, several rasterizers).
  - The RD evaluation needs `tmc3` built from `MPEGGroup/mpeg-pcc-tmc13` with cmake.
  - **Kaggle:** buildable in principle, but the heaviest of the three: **unverified**, and the torch pin
    (2.4) may not match Kaggle's image.
- **Inputs:** a trained 3DGS `.ply` (`INIT_CHECKPOINT=.../point_cloud/iteration_30000/point_cloud.ply`) and
  the COLMAP dataset; the script passes `--images images`.
- **Evaluation:**
  - `splatwizard/pipeline/evaluation.py`: renders rounded to 8 bits ("Quantize rendered image to 8-bit
    precision (1/255) for fair comparison with FCGS"); LPIPS `lpips.LPIPS(net='vgg')`
    (`metrics/loss_utils.py`).
  - **Size:** MiB (`BIT2MB_SCALE`).
  - **Resolution:** `--images images` loads the full-resolution folder; the resize rule that then applies
    is **unverified**, so its resolution may not match INRIA's `images_4` / `images_2`.
- **Reported numbers:** the paper's main comparison is RD curves (figures). The paper's tables are
  ablations; Table I on Mip-NeRF 360 ends at 27.20 dB / 0.8238 / 0.2402 LPIPS / 18.43 MB. The repository's
  `results.xlsx` lists 5 RD points per scene (size in MB = MiB, PSNR, SSIM, LPIPS, chosen pruning ratio).
  Its per-dataset means, **computed here from that spreadsheet**:

  | Point | Mip-NeRF 360 (9 scenes) MB / PSNR / SSIM / LPIPS | T&T (2) | Deep Blending (2) |
  |---|---|---|---|
  | 0 | 65.36 / 27.06 / 0.803 / 0.277 | 31.09 / 23.37 / 0.839 / 0.223 | 55.45 / 29.65 / 0.903 / 0.312 |
  | 1 | 57.19 / 27.02 / 0.802 / 0.279 | 27.25 / 23.36 / 0.838 / 0.224 | 48.42 / 29.64 / 0.902 / 0.313 |
  | 2 | 49.78 / 26.98 / 0.801 / 0.280 | 23.78 / 23.32 / 0.837 / 0.225 | 42.25 / 29.62 / 0.902 / 0.314 |
  | 3 | 42.16 / 26.92 / 0.799 / 0.283 | 20.40 / 23.28 / 0.835 / 0.228 | 35.90 / 29.61 / 0.902 / 0.315 |
  | 4 | 36.15 / 26.85 / 0.797 / 0.286 | 17.81 / 23.25 / 0.832 / 0.231 | 31.21 / 29.59 / 0.902 / 0.316 |

  - **LPIPS gap:** its Mip-NeRF 360 LPIPS (0.277-0.286) is well above C3DGS's 0.238 at a comparable PSNR,
    consistent with a different evaluation resolution (unverified).
  - **Size limits:** the RD size limits in each scene YAML are "based on FCGS compression results"; for
    room (26.5 / 23.3 / 20.4 / 17.3 / 15.2) the spreadsheet's sizes match them.

### POTR ("POTR: Post-Training 3DGS Compression", arXiv 2601.14821)

- **No public code found.**
  - The arXiv abstract page links none; the comments say "Submitted to IEEE TCSVT, under review".
  - Zenodo record 15077060 is a poster (one PDF, `POTR_Post-Training_3DGS_Compression.pdf`, CC-BY-4.0).
  - A web search found no repository.
  - Repo URL, license, last commit, file and function: **not available**.
- **No SH vector quantization to replace.** The method prunes by each splat's removal effect and
  "recompute[s] lighting coefficients" to raise their sparsity, then compresses losslessly (npz or zstd).
  The paper's only vector-quantization mentions are in its references.
- **Evaluation** (paper): COLMAP datasets with "every 8th image is designated for testing"; size in decimal
  MB (baseline row equals the INRIA means above).
- **Reported numbers** (**Table I**, "Quantitative comparison of our proposed codec with the baseline ... All
  fine-tuning methods use 1,000 training iterations"). PSNR / SSIM / LPIPS / size MB / splats x1,000:

  | Method | FT | Tanks and Temples | Mip-NeRF 360 | Deep Blending |
  |---|---|---|---|---|
  | Baseline | - | 23.36 / .838 / .186 / 442 / 1,784 | 27.47 / .821 / .206 / 834 / 3,362 | 29.43 / .898 / .246 / 738 / 2,976 |
  | MesonGS | No | 22.84 / .820 / .211 / 17.3 / 1,163 | 26.22 / .785 / .249 / 29.7 / 2,147 | 28.70 / .890 / .271 / 29.0 / 2,023 |
  | MesonGS-FT | Yes | 23.16 / .832 / .200 / 17.3 / 1,163 | 27.03 / .805 / .231 / 29.7 / 2,147 | 29.54 / .901 / .255 / 29.0 / 2,023 |
  | C3DGS | Yes | 23.13 / .834 / .195 / 18.3 / 1,483 | 27.16 / .811 / .226 / 30.4 / 2,973 | 29.35 / .901 / .256 / 26.7 / 2,613 |
  | LightGaussian | Yes | 22.86 / .817 / .215 / 29.1 / 607 | 26.75 / .805 / .244 / 54.5 / 1,143 | 29.16 / .894 / .261 / 47.9 / 1,012 |
  | POTR (npz) | No | 23.27 / .834 / .191 / 12.8 / 690 | 27.08 / .806 / .226 / 29.3 / 1,500 | 29.31 / .897 / .253 / 18.6 / 785 |
  | POTR (zstd) | No | size 11.3 | size 26.0 | size 16.5 |
  | POTR-FT (npz) | Yes | 23.34 / .837 / .189 / 10.6 / 594 | 27.20 / .808 / .223 / 24.1 / 1,285 | 29.44 / .902 / .250 / 13.2 / 585 |
  | POTR-FT (zstd) | Yes | size 9.36 | size 21.3 | size 11.8 |

  C3DGS in POTR's table (30.4 MB decimal on Mip-NeRF 360) is about C3DGS's own 28.80 MiB (30.20 MB): a
  rerun, not a copy.

## c. Compute and memory for `gn_vq_cvfloor` at INRIA scale (estimates)

**Everything in this section is an estimate.**
- **Base:** measured at 1,000,000 splats in E2, E2b and E2c (`kaggle/gn_e2/gn2/`, `kaggle/gn_e2b/gn2b/`,
  `kaggle/gn_e2c/gn2c/`).
- **Scaling:** linear in the splat count N for GN-VQ, the evaluation and the GN pass (the lifted GEMM, the
  centroid accumulation and the renders all touch every splat). The lifted checks are held constant (they
  sample 10,000 splats).
- **Not measured above 1M:** GN-VQ, the GN pass and the evaluation. Rendering cost need not be linear in N.
- **Script:** the construction is in `e3_estimate.py` in the scouting session's scratchpad.

**Measured parts at 1M splats (E2c layout: per K, 7 CV rows + 1 final row, 4 K, 32 GN-VQ runs per scene):**

| Part | Measured | Source |
|---|---|---|
| GN-VQ, 32 runs | 3,448-3,645 s per scene | E2c, 5 scenes (`vq_time_s`) |
| GN-VQ per (scene, K), 8 runs | 801-869 s (K = 1,024), 736-823 (4,096), 747-803 (16,384), 1,112-1,242 (65,536) | E2c |
| Lifted check | 25.9-29.8 s each, 8-11 per scene (7 `M_even` metrics + 1-4 distinct `rho_cv`) | E2c metas |
| Evaluation + overhead, 32 rows | indoor MipNeRF360 941-1,075 s; truck 373 s | E2c: job - download - GN pass - lifted checks - GN-VQ |
| Evaluation + overhead, outdoor | 508-754 s (28 x 13.0-20.6 s CV rows + 4 x 36.2-44.4 s full rows) | E2b CV rows and E2 full rows, treehill / flowers / stump |
| Deep Blending | not measured; the indoor-to-truck span, 373-1,075 s, is used | - |
| GN pass, all train views | indoor 29.7-32.8 s, truck 15.8 s, outdoor 8.4-12.8 s | E2 metas |

**Memory:**
- **`M` per copy:** 120 float32 values per splat, 480 bytes: 2.94 GB for bicycle, 0.49 GB for train,
  19.1 GB for the 13 scenes' full `M`, and the same again for `M_even`.
- **E2c's job as written** holds four copies at once: `gn["M_packed"]`, `gn_even["M_packed"]` and the
  two sorted copies `M[order]`, `M_even[order]`. `e2b.floored_metric` materializes a fifth for the metric
  in use. At bicycle's 6.13M that is about 14.7 GB of `M` alone, beyond a 16 GB T4 once splats, renders
  and buffers are added.
  - **Fix (a code change, not a method change):** keep one sorted copy of the metric in use, load
    `M_even` and `M` in turn, and floor per chunk (the same float64 add and cast, so bit-identical). That
    gives about 2 copies: 5.9 GB at bicycle.
- **Lifted GEMM:** `diagnostics.lifted_argmin` works in chunks of 2,048 splats. At K = 65,536 the fp32
  score buffer is 2,048 x 65,536 x 4 B = 512 MB, independent of N. The number of chunks grows linearly
  with N.
- **GN pass memory** at 6M splats (a 17-channel gsplat render with a backward pass) is **unverified**.
- **Disk:** the `M` caches (19.1 GB full plus 19.1 GB `M_even`) do not fit a cache-everything design. Each
  scene's caches would be written to `/tmp` and deleted after its job. Kaggle's `/kaggle/working` quota is
  **unverified** here.

**Per scene** (seconds; GN-VQ and evaluation scaled by N / 1M; totals add the lifted checks and 1.5 GN
passes, full `M` plus `M_even`, because E3 has no E2 cache for these checkpoints; downloads excluded):

| Scene | Splats | GN-VQ | Evaluation | Total |
|---|---|---|---|---|
| bicycle | 6,131,954 | 21,141-22,348 | 3,114-4,623 | 24,540-27,416 |
| garden | 5,834,784 | 20,117-21,265 | 2,964-4,399 | 23,361-26,104 |
| stump | 4,961,797 | 17,107-18,083 | 2,520-3,741 | 19,897-22,247 |
| treehill | 3,783,761 | 13,045-13,790 | 1,922-2,853 | 15,222-17,043 |
| flowers | 3,636,448 | 12,537-13,253 | 1,847-2,742 | 14,637-16,392 |
| drjohnson | 3,405,153 | 11,740-12,410 | 1,272-3,659 | 13,299-16,564 |
| playroom | 2,546,116 | 8,778-9,279 | 951-2,736 | 9,996-12,468 |
| truck | 2,541,226 | 8,761-9,262 | 949 | 9,977-10,598 |
| kitchen | 1,852,335 | 6,386-6,751 | 1,744-1,991 | 8,420-9,160 |
| room | 1,593,376 | 5,493-5,807 | 1,500-1,712 | 7,271-7,925 |
| bonsai | 1,244,819 | 4,292-4,537 | 1,172-1,338 | 5,726-6,263 |
| counter | 1,222,956 | 4,216-4,457 | 1,151-1,314 | 5,629-6,159 |
| train | 1,026,508 | 3,539-3,741 | 383 | 4,154-4,476 |

- **Per (bicycle, K), GN-VQ only (8 runs):** 4,913-5,331 s at K = 1,024, 4,511-5,048 at 4,096,
  4,583-4,926 at 16,384 and 6,817-7,618 at 65,536.
- **13 scenes:** 162,129-182,815 s, **45.0-50.8 GPU-hours**. On two T4s, longest first, the makespan is
  82,695-93,218 s (**23.0-25.9 h**), so at least three 12 h sessions with E0-E2c's 9.5 h start cutoff.
- **Single job:** bicycle alone is 24,540-27,416 s (6.8-7.6 h), inside one session's cutoff only if it
  starts first.
- **Kaggle's weekly GPU quota** is not documented here (**unverified**); the whole of E3 may take more than
  one week's quota.

**Downloads** (not in the totals):
- The 30k `.ply` files are 9.87 GB. They can be fetched per member with range requests (as `remotezip`
  does for run 4's datasets) instead of the 14.66 GB archive.
- Dataset downloads in E2 and E2c took 36.9-911.2 s per scene: the same MipNeRF360 scenes took 53.5-89.3
  s in E2 and 371.7-911.2 s in E2c.

## d. Cutting the cost

**Options that do not change the frozen method** (same inputs, same arithmetic, same selection):

1. **Memory restructuring (above):** required at bicycle's size. One metric copy in memory, the floor
   applied per chunk.
2. **Per-member downloads:** fetch only each scene's 30k `.ply` (and `cfg_args`) from the INRIA archive,
   9.87 GB instead of 14.66 GB.
3. **Scene-level sharding** across sessions with the existing queue, start cutoff and per-row resume.
   Largest scenes first, so no session ends with a 7 h job pending.
4. **Delete each scene's `M` caches after its job** rather than bundling or keeping them.
5. **Drop the CV codebooks' test-view dMSE:** it is reported only and never used by the selection
   (Amendment 11 e). This removes one test-view render per CV row (28 per scene). It changes E2c's reported
   items, not the method.

**Options that would change the method.** Each is a change to `gn_vq_cvfloor` and would need its own
pre-registration. The evidence is every `rho_cv` selected so far: 28 cells, 8 in E2b (treehill, flowers,
stump, garden at K = 4,096 and 65,536) and 20 in E2c (5 scenes at 4 K), from `gn2b_e2b.json` and
`gn2c_g2c.json`.

- **Which `rho` were selected:**

  | Value | Selected in E2b | Selected in E2c | Total |
  |---|---|---|---|
  | 0 | 1 (garden, 65,536) | 0 | 1 |
  | 1e-3 | 0 | 3 | 3 |
  | 1e-2 | 1 (garden, 4,096) | 16 | 17 |
  | 1e-1 | 6 (every gap cell; the top of E2b's grid) | 1 (room, 16,384) | 7 |
  | 3e-1, 1, 3 | not on E2b's grid | 0 | 0 |

- **Stability across K:** the same `rho_cv` at every K in 6 of 9 scenes (treehill, flowers and stump at
  both of E2b's K; bonsai, counter and truck at all four of E2c's). It varies in 3: garden (1e-2, 0),
  kitchen (1e-3, 1e-2, 1e-3, 1e-2) and room (1e-2, 1e-3, 1e-1, 1e-2).

**The options:**

1. **A 3-point grid {1e-3, 1e-2, 1e-1}.**
   - **What changes:** the grid. The CV codebooks per (scene, K) drop from 7 to 3, and the GN-VQ runs per
     scene from 32 to 16, about half of the GN-VQ time (the dominant part in section c).
   - **Evidence for:** 27 of the 28 selections are in this set, and 3e-1, 1 and 3 were never selected in
     the 20 cells where they were available.
   - **Evidence against:** the one cell outside it is garden at K = 65,536 (`rho_cv = 0`, E2b), the
     control case, where a floor raised test dMSE (FINDINGS section 11). Dropping 0 would remove the
     method's way of choosing "no floor". A 4-point grid {0, 1e-3, 1e-2, 1e-1} keeps it and still cuts the
     CV codebooks from 7 to 4.
2. **Select `rho` once per scene** (at one K) and reuse it at the other K.
   - **What changes:** the selection rule. The CV codebooks per scene drop from 28 to 7.
   - **Evidence for:** 6 of 9 scenes were stable across K.
   - **Evidence against:** in kitchen, room and garden a per-scene choice would differ from the per-K
     choice in 1-2 cells each. What that costs in dMSE is not measured.
3. **Cross-validate on a subsample of splats or views, or at a smaller K.**
   - **What changes:** the selection's inputs.
   - **Evidence:** none; no run has measured selection agreement under subsampling.
4. **Lower precision or a lower iteration cap** (fp16 `M`, TF32 in the lifted GEMM, a lower iteration cap).
   - **What changes:** the arithmetic of the frozen variant.
   - **Evidence:** none on quality. The exact assignment is pre-registered as "one chunked fp32 matrix
     product over splats (no fp16, no TF32)" (PREREG_GN.md), and the lifted check's criterion (Amendment 3)
     was set for that arithmetic.

## e. Recommendation

**Integrate C3DGS first.**
- **Smallest, best-localized objective:** one function, `compression/vq.py` `vq_features`, with a single
  call site (`compress_color`). Its output (codebook, indices) is exactly what GN-VQ produces.
- **Same feature space:** `color_compress_non_dir = False` gives the frozen method's 45-coefficient AC
  space, with gsplat's layout.
- **Closest baseline:** its objective (Euclidean assignment, scalar-importance-weighted mean) is the one
  E1-E2c measured GN-VQ against in scalar form (this repository's `lloyd_c3dgs` and `lloyd_trace`), so a
  GN-VQ-for-C3DGS row isolates the metric.
- **Inputs:** it runs on the INRIA checkpoints as published ("the reconstructions from Kerbl et al. were
  used") and reports all three datasets.
- **Build:** two small CUDA extensions against CUDA 12, versus MesonGS++'s dozen extensions plus
  `tmc3`, and LightGaussian's torch 1.12 / CUDA 11.6 stack and training stages.
- **Things to settle before any E3 code:**
  - **Protocol:** C3DGS's default VQs DC too, and it fine-tunes the codebook for 5,000 iterations. E3 must
    pre-register whether GN-VQ is compared before fine-tuning, after it, or both, and must run C3DGS's own
    baseline at `color_compress_non_dir = False` as well as at its default.
  - **Codebook:** the published operating point is a single 4,096-entry codebook, so any RD curve for
    C3DGS is E3's own.
  - **Renderer:** `M_i` from gsplat's render must be shown to match C3DGS's INRIA rasterizer on these
    checkpoints (a parity check like E0's).

MesonGS++ is the stronger second host scientifically:
- **For:** VQ on the AC coefficients only, recent code, and a published RD curve on all 13 scenes.
- **Against:** its ILP-chosen codebook precision, kept-top-1M splats, G-PCC dependency and different
  evaluation resolution make GN-VQ's effect harder to isolate.

LightGaussian's degree-2 distillation changes the metric's space. POTR has no code and no VQ.

**E3a (gsplat `PngCompression` on the INRIA checkpoints) is feasible on T4 x2 as a first step, as an
estimate:**
- **Scaled from measured parts at 1M** (E2's clustering times: TorchPQ `upstream_l1` 7.6-96.8, 26.1-28.9,
  100.8-108.3 and 398.2-425.7 s at K = 1,024-65,536; library `lloyd_wopa_area` 5.2-10.0, 25.9-41.3,
  139.2-171.4 s and 276.0-646.0 s at 65,536 across E2 and run 5; E2's per-row evaluation 34.2-114.3 s;
  run 4's PLAS sort 69-83 s):
  - **Default codec only** (K = 65,536 plus an uncompressed row): 536-737 s per 1M splats. For the 13
    scenes (39.78M): 5.9-8.1 GPU-hours, about 3.0-4.1 h on two GPUs, one session.
  - **4-K grid of `upstream_l1` and `lloyd_wopa_area`:** 1,356-2,640 s per 1M, 15.0-29.2 GPU-hours, about
    7.5-14.6 h on two GPUs, one or two sessions.
- **Unverified and to be measured first**, on bicycle (6.13M):
  - PLAS sort memory and time;
  - TorchPQ K-means memory at 6M x 45;
  - the GN pass (only if E3a includes GN-VQ);
  - Kaggle's current torch;
  - the size of the datasets' images;
  - the output quota.
- **Protocol points to fix in E3a's amendment:**
  - the loader's SH transpose;
  - the square crop (up to 3,612 splats; section a);
  - the resized-image difference against INRIA's `images_4` / `images_2`;
  - sizes reported in both MB and MiB, so every published table above can be compared.

The E3a baseline needs no host code at all. It measures the codec this project has studied on the
checkpoints every host uses, before any integration work.
