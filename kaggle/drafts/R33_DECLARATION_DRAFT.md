# DRAFT (R33, 2026-11-25) — declaration: a training-views-only transfer ratio against OGC's lead, on the ten scenes with an OGC row (exploratory)

**Status: draft for Dace's go. Not in PREREG, not committed as a declaration, nothing run.** If Dace says go, this text
is committed alone (as a FINDINGS declaration, as sections 27, 28 and 32 were), pushed, and only then is the notebook
run. Code: `kaggle/drafts/r33_cv_ratio.py`; notebook: `kaggle/drafts/gn_r33_cv_ratio.ipynb` (built by
`build_r33_notebook.py`, pinned to the commit that holds the code).

## Disclosure

- **Not blind, and exploratory.** The ten scenes' P1, P2 and P_FT are committed (FINDINGS section 29) and have been read;
  R29 (section 32) found GN-VQ's E2 test/train ratio ranked almost as P1 (Spearman +0.9524, exact p 0.0011, n = 8), and
  this question was chosen after that result. Nothing here can confirm a predictor; it measures whether a ratio that
  reads no test view behaves like E2's on the scenes already studied.
- **Selected scenes:** the gate scenes were kept and the development scenes chosen on GN-VQ's transfer outcomes
  (section 30).

## The ratio (fixed now)

- **Views:** the scene's training views only (INRIA's split, C3DGS's `Scene` with `--eval`, sorted by image name).
  **Folds:** k = 4, `numpy.random.default_rng(0).permutation`, the view at permuted position p in fold p % 4. k = 4
  keeps each fold's metric on 75% of the views, close to the all-training-view metric E5-E7 used; k = 2 (E4q's
  even/odd) would halve the metric's views and cost half as much.
- **Per fold f:** in-fold = the other three folds, held-out = fold f. C3DGS's sensitivity pass on the in-fold views
  only; C3DGS's colour and geometry VQ at E5's middle rate point (j = 0, C3DGS's default colour threshold, K = 4,096),
  seed 0, with E3r's recording hooks; this project's 16 x 16 GN metric (`gn_metric.compute_gn`, `with_dc`, 16 probes,
  probe seed 0) on the in-fold views; OGC's Gram VQ by this project's derived code (`ogc_derived.gram_kmeans_ours`,
  metric "gram", 15 iterations, lam 1e-3, chunk 25,000) on the same quantizer input; both codebooks through C3DGS's int8
  table quantizer, substituted into the uncompressed model's colours for the colour-quantized splats (geometry and the
  other splats' colours unchanged, as E3r's and E4q's training-view scoring); each row rendered by C3DGS's rasterizer on
  every in-fold and held-out view, quantized to 8 bits, PSNR against the view's image as `Scene` loads it.
- **gain_in(f), gain_out(f):** the mean over the in-fold and the held-out views of PSNR(ogc_gram) - PSNR(c3dgs).
- **The scene's ratio:** the mean over folds of gain_out(f) / gain_in(f), over the folds with gain_in(f) > 0; a fold
  with gain_in(f) <= 0 is reported and left out; a scene with no such fold has no ratio and is reported as such. The
  ratio of the fold means is reported beside it.
- **Why j = 0 and colour only:** E5's middle rate point is the one rate point every E5-E7 process shares. Scoring with
  the colours substituted (no pruning, geometry VQ or `.npz` round trip) isolates the colour VQ, which is what OGC's
  metric changes; it differs from P1, which is a BD over three rate points of C3DGS's whole pipeline on test views.

## What is computed (after the bundle is committed)

1. Per scene: the four folds' gains, the ratio, the ratio of means.
2. **Spearman's rho of the R33 ratio with P1, P2 and P_FT** (section 29's registered values) over the scenes with a
   ratio (n <= 10), exact two-sided p over all n! orderings (10! = 3,628,800, enumerated); PCHIP's values beside.
3. Descriptive: Spearman's rho of the R33 ratio with E2's ratio on the eight scenes with both (section 32's).
4. Report only: no criterion, no verdict, no correction for multiplicity (three primary correlations, one descriptive).

## The guard (no test view)

`kaggle/drafts/r33_cv_ratio.py`: `install_guards` (`:87-95`) first wraps C3DGS's scene readers (`drop_test_infos`,
`:74-84`), so the test camera infos are dropped, and counted, before `Scene` decodes any image (C3DGS's reader opens
every image lazily for its size before it splits the views: headers, not pixels); then it makes `Scene.getTestCameras`,
C3DGS's `render_and_eval`, both `psnr` bindings and the LPIPS entry point raise `Forbidden`. `guard_scene` (`:98-101`)
replaces the loaded `Scene`'s `test_cameras` with `RaisingCameras` (`:62-68`), which raises on any access. `main` installs
the guards before any scene is loaded (`:182`) and wraps the `Scene` as it is built (`:203`). Tested by
`test_r33_guards_raise_on_any_test_view_access`, which also checks that `main`'s source names no test camera.

## Code identity

Every R33 session runs one pinned commit; no file the job imports (`kaggle/drafts/r33_cv_ratio.py`,
`kaggle/e3r_hooks.py`, `kaggle/e3q_c3dgs_run.py`, `kaggle/e3p_inria.py`, `kaggle/e5_data.py`, `kaggle/e6_scenes.py`,
`bench/gn/gn_metric.py`, `bench/gn/ogc_derived.py`, `bench/gn/e3r.py`, `gsplat/`) changes between its sessions.

## Cost (estimate; nothing measured for this job)

| Step, per fold | Estimate | Source |
|---|---|---|
| C3DGS sensitivity on 75% of the training views | 1-2 min | E5's processes (not separately timed) |
| C3DGS colour and geometry VQ | 2-4 min | C3DGS's colour VQ 128-139 s per process (D1 draft) |
| GN pass on 75% of the views | 10-30 s | E5/E6 full passes 13.8-39.9 s |
| OGC's Gram VQ | 25-110 s | E4q: 24.7-108.8 s per call |
| Renders: 2 rows x all training views | 0.5-2 min | 123-272 views at about 0.1-0.2 s each |
| **Per fold** | **about 5-9 min** | |

Per scene: 4 folds + scene load and data (5-15 min) = **about 0.5-0.9 h**. **Ten scenes: about 5-9 h** in one queue,
plus setup (C3DGS build about 10-15 min; the gsplat wheel restored, or about 73 min to build): **one Kaggle session**,
sequential (E5 ran its scenes one at a time for host RAM; `PARALLEL = True` in the notebook splits them over the two T4s).

## Not settled by it

Whether either ratio predicts P1 on scenes nobody has looked at: that is E9's job (`E9_DL3DV_DRAFT.md`, R36).
