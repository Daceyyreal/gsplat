# DRAFT (R34, 2026-11-25) — OGC's VQ with a cross-validated isotropic floor on its per-splat matrices (exploratory, training-free rows)

**Status: draft only; nothing in PREREG, no code, nothing run.** It depends on R33 (`R33_DECLARATION_DRAFT.md`): the same
folds, and R33's ratio as the moderator.

## Why

FINDINGS section 30 ties the development scenes to GN-VQ's poor train-to-test transfer, and section 32 found that
transfer ranked almost as OGC's P1 (Spearman +0.95, n = 8, post hoc, set membership not separated). A metric built from
training views costs nothing in directions no training view sees; the floor `M + rho tr(M) / 16 I` makes every direction
cost at least rho times the mean eigenvalue. It was tried before only with GN-VQ (FINDINGS sections 11, 12, 16, 17): it
helped fidelity on E2b's scenes but not PSNR, and in E4p it added +0.0007 dB on train. **It was never tried with OGC's
VQ** (R30): E4q's `lad_ridge_mean` removed it, and E5-E7 gave OGC the unfloored metric.

## Rows (per process; E5's fork, no fine-tuning)

| Row | Colour VQ | Metric |
|---|---|---|
| `c3dgs` | C3DGS's own | - |
| `ogc_gram` | OGC's `gram_kmeans` (from the private dataset or URL, Amendment 18 note 3's chain; else `ogc_derived`) | the 16 x 16 metric G_i, unfloored (as E5) |
| `ogc_gram_cvfloor` | the same call | `G_i + rho_cv tr(G_i) / 16 I` (`bench/gn/e2b.floored_metric` on the packed 16 x 16 rows, Amendment 14 b's 16 x 16 form) |

- The floor is applied to the input matrices, wherever OGC's code uses them (its assignment distances and its codeword
  update); OGC's code is not edited.
- **Licence:** no new OGC-derived file is needed: the floor is this project's code (`e2b.py`, Apache), applied to the
  input; OGC's `gram_kmeans` runs unedited, or `bench/gn/ogc_derived.py` (already PolyForm Noncommercial with both
  Required Notice lines) when OGC's sources are unavailable. A new file that adapted OGC's code would carry the same header.

## Choosing rho per scene from training views only

- **Grid:** rho in {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3} (E4q's grid for GN-VQ, which found 1e-2 on train and 1e-1 on
  treehill).
- **Score:** for each rho and each of R33's four folds, OGC's VQ with the floored in-fold metric, scored on the held-out
  fold by the gain in PSNR over C3DGS's row (R33's measurement); the scene's score for rho is the mean over folds.
  **rho_cv** = the best score; ties (within 1e-4 dB) to the smaller rho; rho = 0 is a candidate, so the floor can be
  switched off. Clamped dMSE against the uncompressed render (Amendment 10 a's fidelity measure) is reported beside it.
- No test view is read in the choice (R33's guards).

## Hypothesis, stated now

**H34:** the floor helps more where training-view weighting transfers worst: across scenes, Spearman's rho between R33's
ratio and BD-PSNR(`ogc_gram_cvfloor` vs `ogc_gram`) is negative (one-sided, exact permutation p, level 0.05).

## What counts as a positive result

E2b's lesson (FINDINGS section 11): a floor can improve fidelity without improving PSNR, and only PSNR counts here.
- **Positive:** both (a) the mean over scenes of BD-PSNR(`ogc_gram_cvfloor` vs `ogc_gram`) is above 2 SE (E5's
  criterion-3 form, `SD_pool` / sqrt(2n) from the j = 0 seed pair) and (b) H34's Spearman is negative at p < 0.05.
- **Fidelity only** (lower held-out clamped dMSE without (a)): reported as E2b's pattern repeated, not as a positive.
- **Null:** rho_cv = 0 on most scenes, or (a) fails: the metric's null space does not limit OGC here.
- All of it exploratory: ten selected scenes whose P values are known (the R29 and R33 disclosures carry over).

## Scenes

The ten scenes with an OGC row (seven gate, three development), so H34 has a spread of ratios; this is not a held-out
test and says so.

## Cost (estimates)

| Part | Per scene | Ten scenes | Source |
|---|---|---|---|
| rho_cv: 7 rho x 4 folds of OGC's VQ + scoring renders, reusing R33's per-fold sensitivity, VQ and metric | 0.5-1.4 h | 5-14 h | OGC call 25-110 s (E4q); renders as R33 |
| Test rows: 4 processes (j = -1, 0, +1 seed 0; j = 0 seed 1), training-free, rows c3dgs, ogc_gram, ogc_gram_cvfloor | 1.5-3 h | 15-30 h | E5's processes without fine-tuning |
| **Total** | **2-4.4 h** | **20-44 h, 2-4 sessions** | |

If R33 runs first and its per-fold records are kept, the rho_cv part reuses them and costs the lower end.
