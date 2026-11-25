# DRAFT v2 (R33 2026-11-25; R37a-e 2026-11-25, ninth session) — declaration: a training-views-only transfer ratio against OGC's lead, on the ten scenes with an OGC row (exploratory)

**Status: draft for Dace's go (R33_GO blank on 2026-11-25). Not in PREREG, not committed as a declaration, nothing
run; R37b and R37e are not yet in the code.** On go, this text is committed alone as the next PREREG amendment
(**21**, unless E8's draft takes that number first: `E8_AMENDMENT21_DRAFT.md` is named for it but is unregistered),
pushed alone, and only then is the code changed (R37b's Gram stability, R37e's `SCENES` parameter and order), the
notebook regenerated, pinned and pushed, and the run started. Code: `kaggle/drafts/r33_cv_ratio.py`; notebook:
`kaggle/drafts/gn_r33_cv_ratio.ipynb` (built by `build_r33_notebook.py`, pinned to the commit that holds the code).
v1 of this draft: `f8048cd0`.

## Disclosure

- **Not blind, and exploratory.** The ten scenes' P1, P2 and P_FT are committed (FINDINGS section 29) and have been read;
  R29 (section 32) found GN-VQ's E2 test/train ratio ranked almost as P1 (Spearman +0.9524, exact p 0.0011, n = 8), and
  this question was chosen after that result. Nothing here can confirm a predictor; it measures whether a ratio that
  reads no test view behaves like E2's on the scenes already studied.
- **Selected scenes:** the gate scenes were kept and the development scenes chosen on GN-VQ's transfer outcomes
  (section 30).
- **(R37d) C3DGS's reader opens every image, test images included, before the split.** `readColmapCameras`
  (`scene/dataset_readers.py:99` at C3DGS `2a234af`) calls `PIL.Image.open` on each image in the COLMAP model and keeps
  the handle in its `CameraInfo`; `readColmapSceneInfo` splits train and test only afterwards (`:148-150`, every 8th
  image by sorted name is a test view). `Image.open` reads the file's header (format, mode, size), not its pixels,
  which are decoded only when `Scene` builds a camera (`loadCam`); R33's wrapper drops the test infos between the two
  (`drop_test_infos`), so no test image's pixels are decoded, and no value derived from a test image's header is used
  (the camera's width and height come from COLMAP's intrinsics, `:78-79`). The disclosure is that the test files are
  opened, not that their content is read.

## The ratio (fixed now)

- **Views:** the scene's training views only (INRIA's split: C3DGS's `Scene` with `--eval`, sorted by image name).
  **Folds:** k = 4, `numpy.random.default_rng(0).permutation`, the view at permuted position p in fold p % 4. k = 4
  keeps each fold's metric on 75% of the views, close to the all-training-view metric E5-E7 used; k = 2 (E4q's
  even/odd) would halve the metric's views and cost half as much.
- **The GN metric per fold, once per scene (v2):** for each fold g, the 16 x 16 GN metric (`gn_metric.compute_gn`,
  `with_dc`, 16 probes, probe seed g) summed over fold g's views only: `M_g` (`[n, 136]`, packed). `M` is a sum over
  views (`gn_metric.py:298`), so the in-fold metric for fold f is exactly `M_in(f) = sum_{g != f} M_g`, the held-out
  metric `M_out(f) = M_f`. One GN pass over the training views per scene in all, instead of four 75% passes (v1);
  the estimator is the same, the probe draws differ.
- **Per fold f:** in-fold = the other three folds, held-out = fold f. C3DGS's sensitivity pass on the in-fold views
  only; C3DGS's colour and geometry VQ at E5's middle rate point (j = 0, C3DGS's default colour threshold, K = 4,096),
  seed 0, with E3r's recording hooks; OGC's Gram VQ by this project's derived code (`ogc_derived.gram_kmeans_ours`,
  metric "gram", 15 iterations, lam 1e-3, chunk 25,000) on the same quantizer input with `M_in(f)`; both codebooks
  through C3DGS's int8 table quantizer, substituted into the uncompressed model's colours for the colour-quantized
  splats (geometry and the other splats' colours unchanged, as E3r's and E4q's training-view scoring); each row
  rendered by C3DGS's rasterizer on every in-fold and held-out view, quantized to 8 bits, PSNR against the view's image
  as `Scene` loads it.
- **gain_in(f), gain_out(f):** the mean over the in-fold and the held-out views of PSNR(ogc_gram) - PSNR(c3dgs).
- **The scene's ratio (primary, the only primary):** r_s = the mean over folds of r_f = gain_out(f) / gain_in(f), over
  the folds with gain_in(f) > 0; a fold with gain_in(f) <= 0 is reported and left out; a scene with no such fold has no
  ratio and is reported as such. The ratio of the fold means is reported beside it.
- **Why j = 0 and colour only:** E5's middle rate point is the one rate point every E5-E7 process shares. Scoring with
  the colours substituted (no pruning, geometry VQ or `.npz` round trip) isolates the colour VQ, which is what OGC's
  metric changes; it differs from P1, which is a BD over three rate points of C3DGS's whole pipeline on test views.

## (R37a) Per-scene precision and the usefulness rule, applied before any correlation is computed

- **Reported per scene:** the per-fold ratios r_f, their number k_s (folds with gain_in > 0), r_s, and two standard
  errors of r_s:
  - **naive:** SE_naive = sd(r_f) / sqrt(k_s) (sd with k_s - 1 in the denominator);
  - **corrected (used by the rule):** SE_s = sd(r_f) x sqrt(1/k_s + n_out/n_in) = sd(r_f) x sqrt(1/4 + 1/3) =
    1.53 x SE_naive at k_s = 4 (Nadeau and Bengio's correction for resampled estimates whose training sets overlap).
    **Why corrected:** the four folds share two thirds of their in-fold views and the same model, so their ratios are
    positively correlated and the naive SE understates the fold-to-fold uncertainty (no unbiased estimator of k-fold
    variance exists, Bengio and Grandvalet 2004); the correction is a stated heuristic, not exact. With k_s = 4 each
    SE has 3 degrees of freedom, so single-scene SEs are themselves noisy: the rule uses their median.
  - A scene with k_s < 2 has no SE and is left out of the median; it keeps its r_s if k_s = 1.
- **The rule (declared now):** R33 is called **uninformative** if either
  1. fewer than **8** of the ten scenes have a ratio (k_s >= 1), or
  2. **S_between < 2 x median_s SE_s**, with S_between = 1.4826 x MAD of the scenes' r_s (the normal-consistent
     median absolute deviation; the sample SD is reported beside, not used).
- **Why this threshold:** if the scenes' true ratios have spread tau and each r_s carries sampling SE, the observed
  between-scene variance is about tau^2 + SE^2. S_between < 2 SE means tau^2 < 3 SE^2, i.e. a reliability
  tau^2 / (tau^2 + SE^2) below 0.75: more than a quarter of the ranking is fold noise, and a true rank correlation
  with P1 would be attenuated by more than sqrt(0.75) = 0.87 before n <= 10's coarse exact test. **Why MAD:** one
  scene far from the rest (E2's treehill was -1.06) would alone inflate the SD and make noise look like spread.
- **Order (fixed):** the checker computes and prints the rule's inputs and outcome first. **If R33 is uninformative, no
  correlation of r_s is computed** (the FINDINGS section says "uninformative" with the rule's numbers; the bundle stays
  committed). Only if it is informative are the correlations below computed. The same rule, with D_s's fold values
  in place of r_f, decides independently whether the secondary is read.

## (R37b) Secondary predictor: Gram stability (no VQ, no rendering; computed in the same pass)

- **Per fold f** (from the same `M_g`, so it costs no extra pass), with M-bar the per-view mean metric,
  M-bar_in,i = M_in(f)_i / n_in and M-bar_out,i = M_out(f)_i / n_out, and Q_f the colour-quantized splats of fold f
  (the rows OGC's VQ weights, `ids`):

      D_f = sum_{i in Q_f} || M-bar_in,i - M-bar_out,i ||_F  /  sum_{i in Q_f} || M-bar_in,i ||_F

  with ||A||_F over the full symmetric 16 x 16 matrix (from the packed 136: diagonal terms once, off-diagonal terms
  twice). **The scene's value:** D_s = the mean of D_f over the four folds; its fold SE as R37a's.
- **Why this form, not the unnormalized one:** `M` is a sum over views (`gn_metric.py:298`), and the held-out fold has
  a third of the in-fold views, so ||M_in - M_out|| / ||M_in|| without the per-view division would read about
  1 - 1/3 = 0.67 on every splat and measure the view count. **Why the ratio of sums:** it is the ||M-bar_in||_F-weighted
  mean of the per-splat relative differences; an unweighted mean would be dominated by splats the in-fold views barely
  see (tiny ||M-bar_in||, unbounded relative difference), which pull on OGC's codebook least. A splat seen in-fold
  and never held-out has M-bar_out = 0 and counts with relative difference 1: lost coverage is part of instability.
  **Why Frobenius:** it is the norm OGC's objective is most directly sensitive to without choosing an error direction;
  an affine-invariant (log-Euclidean) distance would need full-rank matrices, which a splat seen from few directions
  does not have (each view adds a rank-1 term s y y^T).
- **Its noise floor, reported beside:** D_f has a positive floor from the 16-probe estimate and from the held-out
  fold's few views, which is larger on scenes with fewer training views. Reported per scene: n_train; D_probe, the same
  statistic between fold 0's metric with probe seed 0 and a second pass over fold 0's views with probe seed 100 (one
  extra 25% pass per scene, 3-10 s); and, across scenes, Spearman(D_s, n_train). A D_s that tracks n_train is not read as
  Gram stability.
- **Secondary only:** expected sign negative (a less stable metric, a smaller lead); reported as Spearman(D_s, P1),
  one-sided negative, exact p, and Spearman(D_s, r_s) descriptively; not a second primary, no multiplicity claim.

## (R37c) What R33 can and cannot see

R33's held-out views are held out from the metric (C3DGS's sensitivity, the GN metric and both VQs), **not from the
model**: INRIA's checkpoints were trained on every training view and no test view (`eval=True` in all 13 models'
`cfg_args`, `kaggle/E3_SCOUTING.md:49`), so on a held-out training view the model fits as
well as on an in-fold one. The test views behind P1 and E2's ratio are different: every 8th image by sorted name
(`dataset_readers.py:148-150`; gsplat's parser does the same, `examples/datasets/colmap.py:459-461`), interleaved with
the training views and **unseen by the model and by the metric**. So E2's ratio mixes two things that R33 separates:
whether the metric transfers to views it was not computed on, at training-level model fit (R33 measures this), and how
the model itself does on views it never saw (R33 cannot see this). Part of a test-over-train PSNR-gain ratio below 1 is
arithmetic: the same reduction in quantization error is fewer dB where the model's own error on the view is larger
(gain = 10 log10((m + q_c3dgs) / (m + q_ogc)) falls as m grows). That attenuation shrinks a ratio toward 0 but cannot
make it negative; E2's treehill ratio (-1.06) needs the metric to point the wrong way on model-unseen views, which is a
metric effect, though one that may arise only where the model generalizes poorly.

**A flat R33** (uninformative by R37a, or informative with r_s near 1 everywhere and no rank relation with P1) would
mean: at training-level model fit, OGC's Gram metric transfers about equally well across these ten scenes' views, so
whatever makes E2's ratio and P1 co-vary is not metric transfer at fixed fit; the candidates left are the model's fit
to unseen views (and the dB arithmetic above), and set membership (section 30). It would **not** mean that R29's
correlation is an artefact, that a model-unseen validation ratio (E9 v3's predictor, R38) would be flat, that the
scenes' P1 differences are unpredictable, or that the metric transfers to test views. **A non-flat R33 that ranks as
P1** would say metric transfer at fixed fit co-varies with OGC's lead on these selected scenes; set membership still
cannot be separated from it.

## What is computed (after the bundle is committed)

0. **R37a's rule, first,** for r_s and, separately, for D_s.
1. Per scene: the four folds' gains, r_f, r_s, the ratio of means, SE_naive and SE_s; D_f, D_s, D_probe, n_train.
2. If r_s is informative: **Spearman's rho of r_s with P1, P2 and P_FT** (section 29's registered values) over the
   scenes with a ratio (n <= 10), exact two-sided p over all n! orderings (10! = 3,628,800, enumerated); PCHIP's
   values beside. Descriptive: Spearman's rho of r_s with E2's ratio on the eight scenes with both (section 32's).
3. If D_s is informative: Spearman(D_s, P1), one-sided negative, exact p; Spearman(D_s, r_s), descriptive.
4. Report only: no criterion, no verdict, no correction for multiplicity.

## (R37e) Run order and the smoke

- **Order:** counter first, then the other nine largest first. The notebook takes a `SCENES` parameter (a list, in run
  order; default the declared ten). **Session 1:** `SCENES = ["counter"]`. Dace checks the log for mechanics only:
  the guards installed and no `Forbidden` raised, the dropped test count (counter: every 8th of its images), the four
  fold sizes, `n_colour_quantized` per fold, OGC's iteration info, no NaN or inf, the chunked-linalg calls, timings and
  peak memory. **Session 2:** `SCENES` = the remaining nine. Both sessions run the same pinned commit (code identity).
- **Counter counts.** Its results are R33's counter results under this declaration; the per-fold gains print in its
  log, so they will have been seen before session 2, which is allowed only because the code cannot change between the
  sessions. **If a code change follows the smoke** (a crash, a mechanical fault), the fix is committed and pinned, counter
  re-runs in session 2 under the new pin with the nine, the new pin's counter is the one that counts, and the smoke's
  values are reported beside it, labelled. No value from an attempt that did not finish is read.

## The guard (no test view)

`kaggle/drafts/r33_cv_ratio.py`: `install_guards` (`:87-95`) first wraps C3DGS's scene readers (`drop_test_infos`,
`:74-84`), so the test camera infos are dropped, and counted, before `Scene` decodes any image (R37d above); then it
makes `Scene.getTestCameras`, C3DGS's `render_and_eval`, both `psnr` bindings and the LPIPS entry point raise
`Forbidden`. `guard_scene` (`:98-101`) replaces the loaded `Scene`'s `test_cameras` with `RaisingCameras` (`:62-68`),
which raises on any access. `main` installs the guards before any scene is loaded (`:182`) and wraps the `Scene` as it
is built (`:203`). Tested by `test_r33_guards_raise_on_any_test_view_access`, which also checks that `main`'s source
names no test camera. (Line numbers are v1's, `f8048cd0`; they move when R37b and R37e are implemented.)

## Code identity (back for R33, a new experiment)

Every R33 session runs one pinned commit; no file the job imports (`kaggle/drafts/r33_cv_ratio.py`,
`kaggle/e3r_hooks.py`, `kaggle/e3q_c3dgs_run.py`, `kaggle/e3p_inria.py`, `kaggle/e5_data.py`, `kaggle/e6_scenes.py`,
`bench/gn/gn_metric.py`, `bench/gn/ogc_derived.py`, `bench/gn/e3r.py`, `gsplat/`) changes between its sessions,
except as R37e's smoke rule allows (and then counter re-runs).

## Cost (estimate; nothing measured for this job)

| Step | Estimate | Source |
|---|---|---|
| GN metric, per scene: four fold passes (one full pass) + D_probe's 25% pass | 15-50 s | E5/E6 full passes 13.8-39.9 s |
| Per fold: C3DGS sensitivity on 75% of the training views | 1-2 min | E5's processes (not separately timed) |
| Per fold: C3DGS colour and geometry VQ | 2-4 min | C3DGS's colour VQ 128-139 s per process (D1 draft) |
| Per fold: OGC's Gram VQ | 25-110 s | E4q: 24.7-108.8 s per call |
| Per fold: renders, 2 rows x all training views | 0.5-2 min | 123-272 views at about 0.1-0.2 s each |
| Per fold: D_f | seconds | sums over `[n, 136]` |
| **Per fold** | **about 4-9 min** | |

Per scene: 4 folds + scene load and data (5-15 min) = **about 0.5-0.9 h**. **Ten scenes: about 5-9 h** in two
sessions (R37e: counter alone, about 0.5-0.9 h plus setup; then nine), each with setup (C3DGS build about 10-15 min;
the gsplat wheel restored, or about 73 min to build); sequential (E5 ran its scenes one at a time for host RAM;
`PARALLEL = True` in the notebook splits them over the two T4s).

## Not settled by it

Whether either ratio predicts P1 on scenes nobody has looked at: that is E9's job (`E9_DL3DV_DRAFT.md`, v3, R38, whose
predictor is a model-unseen validation ratio; R33 stays the first look on INRIA's scenes).
