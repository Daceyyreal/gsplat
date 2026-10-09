# Related work: arXiv 2609.28997 (OGC) and arXiv 2609.15735

**Literature notes, not findings.** These notes record what two preprints and one code release say, with page,
section and file:line for every item, and how they relate to this project's GN metric and GN-VQ. Nothing here is
a result of this project, no rule changes because of them, and the notes make no claim about novelty beyond what
the quoted locations show. Written 2026-10-01, before any pre-registration of the C3DGS arm.

**Quoting.** Verbatim quotes are short fragments (at most about a dozen words) in quotation marks; everything else
is a paraphrase with its location, so check the wording against the source. Numbers are copied from the
tables. Page numbers are the printed page numbers, which match the PDF's.

## 1. Sources

| Source | What | Licence | Where |
|---|---|---|---|
| arXiv **2609.28997v1** (24 Sep 2026), K. Pietroszek (Moholo Inc.), "Only What Was Seen: Observation-Gram Compaction of View-Dependent Appearance in 3D Gaussian Splatting" ("OGC") | 22 pages, main text and supplement | CC BY-NC-SA 4.0 (arXiv abstract page) | `arxiv.org/pdf/2609.28997`, downloaded 2026-10-01 to a scratch directory, **not committed** |
| arXiv **2609.15735v1** (14 Sep 2026), T. T. Do, P. A. Chou, G. Cheung, "Transforming harmonic coefficients for 3D splat compression" | 7 pages | CC BY-NC-ND 4.0 (arXiv abstract page) | `arxiv.org/pdf/2609.15735`, as above, **not committed** |
| `github.com/moholo-founder/ogc-3dgs`, commit **`49ccae72e75eec9877354ed72074827531f7fd79`** (2026-09-21, the only commit) | OGC's code | **PolyForm Noncommercial 1.0.0** (`LICENSE`; copyright K. Pietroszek, licensee Moholo Inc.) | cloned to a scratch directory, **not a submodule, not committed** |

The repository's `docs/paper.pdf` is not byte-identical to the arXiv PDF (both 22 pages); the arXiv PDF is the one
cited below. File:line references to OGC's code are to commit `49ccae72`.

- **2026-10-09:** the repository was unavailable (HTTP 404) around 2026-10-08, and E5p attempt 1's clone failed
  (`kaggle/gn_e5p/attempt1/`). It is available again at the same commit. On 2026-10-09 `git ls-remote` gave HEAD
  `49ccae72e75eec9877354ed72074827531f7fd79`, and a fresh clone's tree is `9feebced57d11c6204077baa717528952be811ce`,
  equal to the copy cloned earlier.
- **2026-10-09:** the author states that the method is patented for commercial use. The author confirmed by email that
  the code is available for research purposes (paraphrased). The code's licence is unchanged (PolyForm Noncommercial
  1.0.0).

## 2. arXiv 2609.28997 (OGC)

### a. The Gram matrix used in the VQ

- **Definition.** Per Gaussian, A_i is the sum over training views j of a per-view scalar weight times
  Y(d_ij) Y(d_ij)^T, a 16 x 16 matrix over all 16 SH basis functions (p. 4, §3.2, Eq. (6); also Eq. (1), p. 1).
  The weight is either S1 = sum_p w_ip (the bound) or S2 = sum_p w_ip^2 (the uncorrelated estimate) (p. 4, §3.2).
- **S1 or S2:** S2. "Unless stated otherwise the Gram matrices use S(2)" (p. 6, §4 Setup). Code: `compact.py:50-54`
  accumulates `S2[idx] * (Y Y^T)` per view; `gram.py:3` records S1, S2 and the hit count per view.
- **Clamp:** ignored. §3.2 (p. 4) sets aside the clamp of the colour (their Eq. (2)) as inactive for most contributing
  Gaussians, and §5 (p. 10) lists it as a limitation. The accumulator weights are the compositing weights with no
  clamp mask (`fastrender.py:179-185`).
- **DC:** included in A_i (16 x 16). The VQ of their own pipeline uses only the AC block: G_i is "the corresponding AC
  block of A_i" (p. 5, §3.5); code `compact.py:88`, `pipeline.py` and `run_exps.py` slice `A[:, 1:m, 1:m]`. Their
  C3DGS host uses the full 16 x 16 A_i on all 16 SH rows, DC included (`hosts/c3dgs_run.py:39-53`).
- **Per channel or shared:** one A_i shared by the three colour channels; K_i has one column per channel (p. 4,
  §3.1; `vq.py:6`, distortion summed over channels).
- **Exact or estimated weights:** exact. S1 and S2 are accumulated in the compositing loop of their PyTorch
  re-implementation of the reference rasteriser (`fastrender.py:173-186`: alpha clamped at 0.99, cut below 1/255,
  compositing stopped at transmittance 1e-4). The supplement adds a patch putting the same two accumulators into
  the reference CUDA rasteriser and reports agreement in total to 0.2% on one garden view (p. 14, App. A).
- **Which views:** training cameras only: "statistics are accumulated from the training cameras only" (p. 6, §4).
  Code: `compact.py:34-39` drops every 8th camera of the sorted list. Directions run from the camera centre to the
  Gaussian centre (`compact.py:51-53`). Cameras use the image centre as principal point (`compact.py:31`; p. 14,
  App. A).

### b. Ridge, floor or regulariser in the VQ

- **Assignment:** none found. The assignment minimises the Gram distance of Eq. (11) as written (p. 5, §3.5);
  code `vq.py:42-53` uses G_i itself (`metric="gram"`). No floor on G_i was found anywhere.
- **Update: a ridge, and the paper and code disagree on its size.**
  - Paper: "the update step uses a ridge of 10^-6 tr(Σ_i G_i)/q" (p. 14, App. A).
  - Code: `vq.py:17` defaults to `lam=1e-3`; `vq.py:63-73` solves (Σ G_i + ρ I) c = Σ G_i x_i + ρ x̄ with
    ρ = lam · tr(Σ G_i)/q per cluster and x̄ the cluster's Euclidean mean. So the ridge shrinks unobserved
    components toward the cluster mean, not toward zero.
  - The code comment (`vq.py:66-68`) gives the reason: without it, codewords saturated the int8 quantiser inside
    Compressed3D.
  - Callers that pass no `lam` get 1e-3: the C3DGS host (`hosts/c3dgs_run.py:61`), `pipeline.py:94`,
    `compact.py:90` and `run_exps.py:121, 140`. Only `arc_eval.py:76-82` sets it, to 1e-3 ("regularised") and 1e-6
    ("unregularised").
  - App. N (p. 19) reports the two arc-test variants as indistinguishable, and says the regularisation matters
    inside a host quantiser (§4.1) rather than on its own.
- **The λ = 10^-3 of the paper's setup** (p. 6, §4) is the degree-reduction regulariser of Eq. (8),
  λ_i = λ tr(A_SS)/m_L, shrinking toward truncation (p. 5, §3.3). It is not the VQ's ridge.

### c. How λ is chosen

- **Projection λ:** fixed at 10^-3 after a sweep on test PSNR. App. E's Table 9 (right) gives test PSNR, mean of
  garden and kitchen, for λ from 10^-6 to 10, and "We use λ = 10^-3: it is on the flat part of the curve"
  (p. 16, App. E). App. N: "our default λ = 10^-3 is tuned for the standard split" (p. 19), which is the test split.
  In that table 10^-6 reads 28.13 dB at degree 1 against 28.09 dB at 10^-3.
- **VQ ridge:** a fixed constant (above). No sweep, no cross-validation and no per-K choice was found.
- **No per-K choice** of any regulariser was found.

### d. The held-out 90-degree arc (App. N, pp. 18-19; Table 16, p. 20)

**Protocol** (paraphrased from p. 18):
- garden and kitchen are retrained with the official trainer, holding out a contiguous 90-degree arc of the
  trajectory: cameras within ±45° of azimuth zero, in the principal frame of the camera centres;
- statistics come from the training cameras only;
- the held-out views are grouped by angular distance to the nearest training camera: near < 10°, mid 10-25°,
  far ≥ 25°. That is 4 / 17 / 28 views on garden and 12 / 25 / 44 on kitchen (Table 16 caption).

Table 16 (test PSNR, dB):

| Configuration | garden near | mid | far | kitchen near | mid | far |
|---|---|---|---|---|---|---|
| uncompressed | 22.65 | 22.66 | 21.02 | 28.41 | 22.89 | 19.27 |
| truncation, degree 1 | 21.92 | 22.40 | 21.58 | 25.46 | 22.18 | 19.47 |
| projection, degree 1 | 22.40 | 22.19 | 20.74 | 27.09 | 22.99 | 19.70 |
| truncation, degree 0 | 21.22 | 21.90 | 21.31 | 24.43 | 21.54 | 19.14 |
| projection, degree 0 | 21.46 | 21.74 | 20.90 | 25.24 | 22.07 | 19.38 |
| allocation (truncated), 9 floats | 22.71 | 22.84 | 21.38 | 27.88 | 22.86 | 19.40 |
| allocation (projected), 9 floats | 22.62 | 22.43 | 20.80 | 28.26 | 22.99 | 19.42 |
| + importance VQ | 22.45 | 22.35 | 20.83 | 27.62 | 22.85 | 19.42 |
| + Gram VQ, unregularised update | 22.56 | 22.49 | 20.92 | 27.94 | 22.94 | 19.51 |
| + Gram VQ, regularised update | 22.56 | 22.49 | 20.93 | 27.95 | 22.94 | 19.50 |
| null-space components removed | 22.63 | 22.55 | 20.98 | 28.41 | 22.84 | 19.20 |
| null-space components doubled | 22.64 | 22.53 | 20.89 | 28.41 | 22.86 | 19.26 |

**A text-table mismatch.** The text of App. N (pp. 18-19) quotes degree-2 numbers that Table 16 does not contain:
- garden near 22.52 against 22.45, and garden far 21.61 against 20.88;
- kitchen far 19.55 against 19.33.

Table 16 has degree-1 and degree-0 rows only. The text's degree-1 kitchen numbers (27.09 / 25.46 near, 19.70 /
19.47 far) do match the table.

### e. Reported cases where their metric loses on test views

- **treehill, degree 2** (p. 7, §4.2; Table 19, p. 22): truncation reads 22.32 dB, the projection 22.12 dB and the
  uncompressed model 22.27 dB. The paper calls this the one exception in 38 of 39 comparisons ("truncation to
  degree 2 (22.32 dB) beats both our projection (22.12 dB)", p. 7). At degrees 1 and 0 the projection leads on
  treehill by 0.31 and 0.16 dB.
- **The 90-degree arc, garden** (Table 16): the projection is below truncation at mid and far, at both degrees.
  - Degree 1: 22.19 against 22.40 (mid) and 20.74 against 21.58 (far).
  - Degree 0: 21.74 against 21.90 (mid) and 20.90 against 21.31 (far).
  - The text adds degree-2 cases that are not in the table: see d.
  - §5 (p. 10) says that on one of two scenes plain truncation generalises better far from the capture.
- **Compressed3D after fine-tuning: my tabulation of their Table 12** (p. 18, per scene; the means reproduce their
  Table 1's +0.49 / +0.09 / +0.32 / +0.01 dB):
  - **The matrix A_i at 4,096 entries:**
    - below their own VQ on bicycle (25.01 against 25.04);
    - below their sensitivity in the same Lloyd iterations (s_i I) on kitchen (30.44 against 30.50) and room
      (31.13 against 31.31);
    - below the scalar tr(A_i)/16 I on kitchen (30.44 against 30.51), room (31.13 against 31.18) and treehill
      (22.25 against 22.26).
  - **The rate-matched 512-entry codebook** is below their VQ on bicycle (24.96 against 25.04), bonsai (31.32
    against 31.38) and stump (26.31 against 26.34), and ties on flowers (21.22).
  - **The 2,048 and 1,024 entries** are below their VQ on bicycle (25.01, 24.95), with 2,048 tying on counter
    (28.66).
  - **Before fine-tuning,** the only case is the 512-entry codebook on treehill below the scalar tr(A_i)/16
    (22.13 against 22.16). Every matrix row is above their VQ before fine-tuning.
- **No other loss was found** in Tables 1-3, 9, 14, 17, 18 or 19. Table 6's comparisons (for example -0.39 dB
  against GSICO on Tanks&Temples, p. 9) are of their whole stack, not of the metric.

### f. VQ details

- **K:** 4,096 per degree group (p. 5, §3.7; p. 14, App. A). In code, K = min(codebook, max(16, n // 8))
  (`pipeline.py:93`, `compact.py:89`). In the C3DGS host it is C3DGS's `codebook_size`, 4,096 by default, plus
  rows at 2,048, 1,024 and 512 (Table 1, p. 6).
- **Iterations:** "Vector quantisation runs 12–15 generalised Lloyd iterations" (p. 14, App. A). Code: a fixed count
  with no tolerance test (`vq.py:55`); 12 in `pipeline.py:38`, 15 in the host (`hosts/c3dgs_run.py:76`) and
  `run_exps.py:121`.
- **Init:** data points sampled with probability proportional to tr(G_i) (p. 14, App. A; `vq.py:31-33`, multinomial
  without replacement from a seeded generator, `vq.py:21`, seed 0).
- **Empty clusters:** re-seeded at the points of largest distortion (p. 14; `vq.py:74-84`).
- **Stopping:** the iteration count only; no relative-drop rule was found.
- **Coefficients:** AC only in their pipeline, one codebook per degree group of the allocation (p. 5, §3.5; Table 18
  caption, "Vector quantisation of AC coefficients", p. 21). All 16 rows including DC in the C3DGS host
  (`hosts/c3dgs_run.py:39-53`).
- **Final assignment:** one more assignment against the returned codebook (`vq.py:87`). No assignment against a
  quantised codebook was found in `vq.py`; the host hands the float codebook to C3DGS (`hosts/c3dgs_run.py:62-66`).
- **Equal rate:** measured bytes of the host's container.
  - Table 1 gives each variant's size as a ratio to the reproduction (p. 6). The host computes size as
    `os.path.getsize(npz) / 1024**2`, so their "MB" there is MiB (`hosts/c3dgs_run.py:122, 131`).
  - The 4,096-entry matrix rows are 104-106% of the reproduction's bytes. Rate is matched by shrinking the codebook:
    512 entries give 100% after fine-tuning (p. 6, §4.1).
- **Index entropy:** Table 13 (p. 19). Their minibatch EMA k-means uses the 4,096-entry codebook unevenly (about 10
  bit zero-order entropy per index), while the generalised Lloyd assignment uses it almost uniformly (about 12 bit).
  That is the stated source of the size difference (also p. 6, §4.1).

### g. The C3DGS integration (§4.1, Table 1, p. 6; Table 12, p. 18; `hosts/c3dgs_run.py`)

- **Which C3DGS commit:** not found. The host imports C3DGS from `C3DGS_ROOT` (default `/work/c3dgs`,
  `hosts/c3dgs_run.py:24`); no commit is pinned in the paper or the repository.
- **What was replaced:** only the colour VQ (`compress_color_variant`, `hosts/c3dgs_run.py:38-67`), with C3DGS's own
  keep mask, pruning, covariance VQ, fine-tuning and container kept (`:88-131`). Four variants (`:10-14`):
  - `theirs`;
  - `theirs_lloyd` (s_i I in their Lloyd);
  - `ours_scalar` (tr(A_i)/16 I);
  - `ours_gram` (A_i).

  The paper: "We run their released code unmodified on the official pretrained models" (p. 6). The code runs the
  reproduction through the same host script with `--appearance theirs`, which calls C3DGS's functions in
  `compress.py`'s order, adding a save and an evaluation before fine-tuning (`:118-123`). Which of the two produced
  the reproduced rows is an open question.
- **Rows** (Mip-NeRF 360, 9 scenes; MB per above):

  | Row | Size (MB) | PSNR | SSIM | LPIPS |
  |---|---|---|---|---|
  | published (their sensitivity, their VQ; the 3DGS.zip survey's dataset mean) | 28.8 | 26.98 | 0.801 | 0.238 |
  | reproduced, before fine-tuning | 28.7 | 26.30 | 0.782 | 0.258 |
  | reproduced, after fine-tuning | 28.8 | 27.01 | 0.803 | 0.237 |
  | our matrix A_i, before / after (paired difference) | 30.3 / 29.9 | +0.49 / +0.09 | +0.017 / +0.004 | -0.019 / -0.007 |
  | our matrix, 512 entries, before / after | 29.0 / 28.9 | +0.32 / +0.01 | +0.012 / +0.001 | -0.012 / -0.003 |

  The 2,048- and 1,024-entry rows cover 5 scenes only (n = 5 in Table 1).
- **Fine-tuning:** both, before and after C3DGS's 5,000 iterations (`hosts/c3dgs_run.py:118-132`).
- **Seeds, repeats, variance:**
  - not found in the paper;
  - the host sets no seed: no `safe_state`, `Scene(..., shuffle=True)` at `:86`, while C3DGS's own VQ and the
    covariance VQ draw from the global generators;
  - one run per variant and scene, and no run-to-run spread is reported;
  - only `gram_kmeans` uses a fixed generator (`vq.py:21`).
- **GPU determinism:** not discussed for the host. For the statistics, App. A (p. 14) reports atomic adds, and
  projection residuals from two GPUs agreeing to 0.02%.

### h. Ablations

- **Table 18** (p. 21; AC coefficients, one 4,096-entry codebook, mean of garden and kitchen; predicted distortion
  D in units of 10^3):

  | Coefficients | VQ metric | PSNR | SSIM | LPIPS | pred. D |
  |---|---|---|---|---|---|
  | degree 3 (45) | none | 28.96 | 0.893 | 0.119 | - |
  | degree 3 (45) | Euclidean | 27.96 | 0.875 | 0.143 | 156.5 |
  | degree 3 (45) | importance-weighted (tr(G_i)/q) | 28.20 | 0.877 | 0.141 | 129.3 |
  | degree 3 (45) | Gram | 28.55 | 0.886 | 0.130 | 52.1 |
  | allocated (9) | none | 28.89 | 0.892 | 0.120 | - |
  | allocated (9) | importance-weighted | 28.02 | 0.874 | 0.143 | 145.7 |
  | allocated (9) | Gram | 28.47 | 0.885 | 0.131 | 53.9 |

  Per scene (p. 8, §4.5): plain k-means loses 0.71 / 1.30 dB on garden / kitchen, importance-weighted 0.54 / 0.98,
  Gram 0.26 / 0.56.
- **Generalised Lloyd against the host's quantiser** (Table 1, p. 6; paired, 9 scenes, before / after fine-tuning):
  - s_i I in their Lloyd: +0.21 / +0.07 dB at 103% size;
  - tr(A_i)/16 I: +0.22 / +0.06 at 102%;
  - A_i: +0.49 / +0.09 at 106% / 104%.
- **Table 14** (p. 19, 9 scenes, plain container): Gram VQ 27.02 against importance VQ 26.74 at 118.1 MB (no
  pruning), and 26.81 against 26.51 at 59.6 MB (50% pruning).
- **Table 9, left** (p. 16, degree reduction, not VQ; garden and kitchen at degree 0 / degree 1):
  - S2: 25.60 / 28.09;
  - S1: 25.55 / 28.05;
  - pixel count: 25.29 / 27.85;
  - hit (0/1): 24.99 / 27.56;
  - truncation: 24.56 / 25.83.

### i. arXiv 2609.15735 (Do, Chou, Cheung)

- **Per-splat or global Gram:** global. The Gram matrix Φ^T Φ is over all K = N·M·L basis functions jointly (§3.3,
  p. 3; Eq. (18)-(19), p. 4). The energy-preserving transform is its square root (Eq. (19)).
- **How it is approximated:** as a tensor product of an N x N spatial, an M x M directional (M = 16) and an L x L
  colour Gram (Eq. (25)-(26), p. 4). This comes from replacing the joint measure on the observed (ray, direction)
  pairs by a product measure with the same marginals (Eq. (27)-(28), p. 4).
  - The directional M x M Gram is estimated by scanning all views used for the distortion measurement (p. 4,
    §4.1).
  - A joint directional-colour KLT follows, estimated by SVD (Eq. (30), p. 4). They also report a version with the
    directional square root replaced by the identity (p. 4).
  - Reading Eq. (6)-(7) (p. 2) and (27)-(28), the spatial factor uses φ^s_n = T_n α_n, the blending weight, so the
    inner product weights by products of blending weights. That is my reading, not their statement.
- **Data:** sparse voxel splats ("we use Sparse Voxel Splats [3] as our splat representation", p. 4, §4.1), learned
  as in SVRaster, at two splat-count configurations. Datasets: Mip-NeRF 360 and NeRF Synthetic (p. 5, §4.2). No
  3DGS model is coded.
- **Quantiser:** uniform scalar quantisation after RAHT (the spatial transform) and the SH transforms; ANS entropy
  coding; the KLT and step sizes in the header (p. 4, §4.1). No VQ.
- **Headline numbers:**
  - the abstract claims gains of over 2 dB from orthonormalising before coding (p. 1);
  - §4.3 (p. 5) reports at least 1 dB from the directional-colour KLT in all configurations, and about 1 dB more
    from the splat count;
  - Table 1 (p. 5, Mip-NeRF 360): SVComp (<10M) 28.12 MB, 26.10 dB / 0.7461 / 0.290; SVComp (<5M) 16.94 MB,
    25.62 dB / 0.7187 / 0.323.
- **OGC's description of it** (OGC p. 3, Related Work) matches this: a global Gram factorised into spatial, shared
  16 x 16 directional and colour factors, no degree reduction, allocation or VQ, evaluated on sparse voxels.

## 3. Metric equivalence: this project's M_i against OGC's A_i (S2)

**What M_i estimates** (`bench/gn/gn_metric.py:3-12`, docstring):

- M_i = Σ_v s_iv y(d_iv) y(d_iv)^T over the train views v where the splat is visible:
  - y is the 15 SH basis values of bands 1-3 (frozen metric), or all 16 with `with_dc=True` (E3r,
    `gn_metric.py:296-298`);
  - d_iv is the normalised direction from the camera position to the splat mean (`gn_metric.py:296`,
    `sh_basis.py:29-30`, camera position from the view matrix at `gn_metric.py:358`).
- s_iv is one render of zero features with 17 channels, followed by one backward pass, with Rademacher ±1 on 16
  channels per pixel (`gn_metric.py:36, 229-232, 253`). Then s_iv = mean over the 16 channels of the squared
  feature gradient (`gn_metric.py:257`).
  - With `sh_degree=None` and background 0 (`gn_metric.py:190`), the gradient of channel c with respect to splat i's
    feature is Σ_p w_ip r_pc, with w_ip the compositing weight.
- **Clamp:** ignored in M. Only the fraction of negative colours is recorded (`gn_metric.py:303-305`).
- **Weighting per pixel:** w_ip^2, through the probes.
- **Views:** the runner's train set, from gsplat's COLMAP parser with every 8th image held out
  (`examples/datasets/colmap.py:459`; E3r `kaggle/gn_e3r_scene.py:488`). K is the parser's
  (`kaggle/gn_e0_scene.py:202`).

**In expectation it equals their S2 Gram.**
- E[(Σ_p w_ip r_pc)^2] = Σ_p w_ip^2, because E[r_pc r_qc] = δ_pq for independent Rademacher probes. So
  E[s_iv] = S2_iv (OGC p. 4, §3.2) and E[M_i] = Σ_v S2_iv y y^T.
- With `with_dc=True` that is OGC's A_i (Eq. (6)) for the same weights and directions. The frozen 15 x 15 M_i is
  E-equal to their AC block G_i at degree 3 (OGC p. 5; `compact.py:88`).
- E0's toy check compared the probe estimate with exact Σ_p w_ip^2 (`gn_metric.py:582-`; FINDINGS section 8).

**Every difference found:**

| Item | This project (`bench/gn`) | OGC (`ogc-3dgs` `49ccae72`) |
|---|---|---|
| Weight per view | 16-probe Hutchinson estimate of S2, unbiased, with variance (`gn_metric.py:253-257`) | exact S2 from the compositing loop (`fastrender.py:183-185`) |
| Rasteriser | gsplat CUDA, eval settings (`gn_metric.py:178-215`) | PyTorch re-implementation of INRIA's forward pass (p. 14, App. A; `fastrender.py`) |
| Alpha cut, alpha clamp, transmittance stop | 1/255, 0.99, 1e-4 (`gsplat/cuda/include/Common.h:97, 105, 106`) | 1/255, 0.99, 1e-4 (`fastrender.py:173-178`); same constants, termination conventions not compared |
| Clamp of the colour | ignored, fraction recorded | ignored (p. 4, §3.2) |
| Background | 0; does not enter M | does not enter A (weights only) |
| DC | frozen metric 15 x 15 (bands 1-3); 16 x 16 since E3r (Amendment 14 b) | A_i always 16 x 16; their VQ uses the AC block; their C3DGS host the full 16 x 16 |
| Views | train views of gsplat's parser, every 8th held out | training cameras, every 8th held out (`compact.py:34-39`) |
| Resolution | the runner's data factor (protocol i, gsplat's own resize) | the official model resolutions: `images_4` outdoor, `images_2` indoor, full for T&T and DB (p. 14, App. A) |
| Principal point | the parser's K | image centre (`compact.py:31`; p. 14) |
| Splats accumulated | visible (radius > 0) in the view (`gn_metric.py:291`) | S2 > 0 in the view (`compact.py:50`) |
| Precision | fp32 packed upper triangle | fp32 N x 256 (p. 14) |

## Differences from GN-VQ (the frozen `gn_vq_cvfloor` and E3r's injection)

| | GN-VQ (this project) | OGC's Gram VQ |
|---|---|---|
| Metric in the assignment | floored: M_i + ρ tr(M_i)/d I (Amendment 9 b, `e2b.py:90-102`) | G_i (or A_i), no floor (`vq.py:42-53`) |
| How the regulariser is chosen | ρ_cv by cross-validation on odd train views, per K, from {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3} (Amendments 11 b, 12 b) | fixed: VQ ridge 1e-3 in code / 1e-6 in the paper; projection λ = 1e-3 tuned on the test split (b, c) |
| Centroid update | ridge μ = ε tr(Σ M)/d toward 0, ε = 1e-2 (`diagnostics.py:545-576`) | ridge ρ = λ tr(Σ G)/q toward the cluster's Euclidean mean (`vq.py:63-73`) |
| Range clip with per-cluster acceptance | yes (Amendment 5) | not found |
| Final assignment against the codec's quantised codebook | yes (gsplat's 6-bit codebook; in E3r C3DGS's int8 table) | not found; float codebook handed over |
| Init | the host's own codebook as warm start (`lloyd_wopa_area` in gsplat; C3DGS's own `vq_features` in E3r) | points sampled ∝ tr(G_i) |
| Stopping | relative drop < 1e-3 or the cap (20 since E2) | fixed 12-15 iterations |
| C3DGS integration | C3DGS's `vq_features` runs first, then GN-VQ replaces its output (Amendment 14 b); every run seeded 0 (Amendment 14 f) | the colour VQ replaced outright; host unseeded; C3DGS commit not pinned |
| Pre-registration, held-out gates | G0-G2c gates on held-out scenes (FINDINGS 8-12) | not found |

## Open questions

1. **Which VQ ridge produced Tables 1, 12, 16 and 18:** the paper's 10^-6 (p. 14) or the code's default 10^-3
   (`vq.py:17`), which every caller but `arc_eval.py` uses?
2. **Which C3DGS commit** did they run, and do the "reproduced" rows come from C3DGS's `compress.py` or from
   `hosts/c3dgs_run.py --appearance theirs`?
3. **Run-to-run spread.** Their post-fine-tuning gains (+0.09 dB; +0.01 dB rate-matched) are single runs per scene
   with no seed set in the host. E3r found that even seeded C3DGS runs are not reproduced on the GPU (FINDINGS
   section 15). How large is C3DGS's own spread, against differences this size?
4. **App. N's degree-2 numbers** (22.52 / 22.45, 21.61 / 20.88, 19.55 / 19.33) are not in Table 16. Where do they
   come from?
5. **Selection on test views.** The projection λ was chosen on the standard split's test PSNR (App. E, App. N).
   Was anything in the VQ chosen on test views? Their reported VQ settings are fixed.
6. **Floor against ridge-to-mean.** Their remedy for unobserved directions is a ridge in the update toward the
   cluster mean, with no change to the assignment. Ours is a floor in both steps, toward isotropy. No experiment
   compares the two.
7. **Their int8-saturation remark** (`vq.py:66-68`). In E3r, GN-VQ's clip left no coordinate outside the warm range
   (`fraction_outside_warm_range_final` 0.0, `kaggle/gn_e3r/gn3r/`). Is the clip doing the job their ridge does?
8. **Convergence.** They run 12-15 iterations from a tr-weighted sample; E3r's 9 runs from C3DGS's codebook hit our
   20-iteration cap. Neither reports convergence at their stopping point.
9. **2609.15735 on 3DGS.** Its Gram is global and factorised, and its experiments are on sparse voxels only. Is the
   loss from the product-measure approximation (Eq. (28)) quantified anywhere? Not found.
10. **Units.** The host's sizes are MiB (`getsize / 1024**2`) under the label "MB", as with C3DGS's own tables
    (FINDINGS section 14). Are the survey rows they compare with in the same unit?
