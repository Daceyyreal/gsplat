# DRAFT — two candidate methods from E7's finding (2026-11-08; no code)

**Status: sketch for Dace. Nothing here is registered, coded or run.** File:line references are to C3DGS at `2a234af5`
(read through GitHub's raw files on 2026-11-08), MesonGS++ at its default branch on that date, and this repository at
`259cbd2e`.

## What the sketches start from

- **Before fine-tuning OGC's metric is worth a lot:** P1 0.3558 dB, P2 0.1874 dB (FINDINGS s19), and the sign holds in
  SSIM and LPIPS (s27).
- **C3DGS's fine-tuning erases most of it:** P_FT 0.0425 dB (s21); E7: neither the grids' re-fit nor codebook
  training explains the shrink (H1, H2); training only the codewords keeps most of the lead, training everything but
  the codewords loses all of it (H5, report-only and post hoc).
- **What fine-tuning cannot change:** the indices. `_feature_indices` is a `Parameter` with `requires_grad=False`
  (`scene/gaussian_model.py:727-730`, `:750-752`, `:778`), and the loop (`finetune.py:31-73`) steps only the Adam groups
  (s24 item 4). So whatever the metric decided **through the assignment** survives fine-tuning untouched, and the rest of
  the model (opacity, scaling, rotation, kept colours) re-fits the photometric loss (`finetune.py:52-56`) around it.
  The lead shrinking means the other method's model compensates for its worse assignment.

## (a) Observation-weighted fine-tuning: re-assign under the metric while fine-tuning

**Hypothesis.** If, every M iterations of fine-tuning, each quantized splat is re-assigned to its nearest codeword
under its own observation metric G_i, against a per-splat target colour, the assignment keeps tracking the moving
codebook and the compensation stops favouring the worse starting assignment, so **OGC's lead after fine-tuning
(P_FT) grows toward its lead before (P1)**. The fair comparison gives C3DGS the same re-assignment schedule under its
own criterion (squared error weighted by its sensitivity), so the test is the metric, not the re-assignment.

**What to implement, where.**
- *Target colours:* save the pre-VQ colour table before `compress_gaussians` (`compress.py:178-188`), in the fork's
  colour-call hook (`kaggle/e5_hooks.py`, as the row copy is made today).
- *The metric:* already computed per splat for OGC's call (this project's 16-probe estimate, `bench/gn/metric_store.py`,
  `bench/gn/gn_metric.py`); kept on the host, sliced per chunk as E4p/E5 do.
- *Re-assignment:* a hook after the optimizer step (`finetune.py:71-73`) every M iterations (M = 500 and 1,000 in the
  pilot): read the current codebook (`_features_dc`, `_features_rest` rows 0..K-1, `set_color_indexed`
  `gaussian_model.py:796-799`), run one exact assignment under G_i (`bench/gn/diagnostics.py` `lifted_argmin`, the
  same arithmetic E0-E2 validated; or one assignment step of OGC's `gram_kmeans`), write `_feature_indices` in place.
  The optimizer state of the codebook rows is kept; nothing else changes.
- *Arms:* two new rows per method in E7's `ARMS` form (`kaggle/e7_scenes.py:60-68`): `*_ft_re500`, `*_ft_re1000`.

**Cheap pilot (development scenes only: train, treehill, flowers; no gate or held-out scene).** E6 part (a)'s three
processes per scene (j = -1, 0, +1, seed 0) with `c3dgs_ft`, `ogc_gram_ft` and the two re-assignment rows per method;
readings: P_FT and P_FT,re per scene, and P_FT,re - P_FT. **About two sessions** (E6 part (a) took two for the same
scenes with fewer rows). Caveat from E6 part (a): train's fine-tuned rows moved by up to 0.19 dB between runs (G3), so
the pilot reads direction and size, not significance.

**Pre-registered test, if the pilot points that way.** On held-out scenes (E9's): primary P_FT,re = BD of
`ogc_gram_ft_re` against `c3dgs_ft_re`, with E5's three criteria and `SD_pool` / sqrt(n); second primary, within
process, Δ_re = P_FT,re - P_FT (does re-assignment widen the lead), E7's superiority test; M fixed from the pilot before
registration; a second seed at j = 0 for noise, as E5-E7.

## (b) A training-free pipeline built around the metric

**Hypothesis.** Without fine-tuning the metric's lead is large (P1), so a codec that never fine-tunes, built so that
every lossy step that touches colour uses G_i, reaches a **higher BD-PSNR than the same codec's own training-free
path, and closes part of the gap to fine-tuned codecs at a small fraction of their compute** (one GN pass and a VQ,
against 5,000-8,000 fine-tuning iterations).

**What to implement, where.**
- *Host:* MesonGS++ (MIT; training-free; its scheduler registers no fine-tuning after `vq_fe`,
  `splatwizard/model_zoo/mesongs_plus/model.py:1749-1756`), or MesonGS with `--iterations 0`
  (`kaggle/drafts/SECOND_HOST_FEASIBILITY.md`).
- *Colour VQ:* replace `vq_features` at `model.py:926` with Gram-metric Lloyd (OGC's `gram_kmeans`, or this project's
  own `bench/gn/gn_vq.py` frozen as `gn_vq_cvfloor`, E2c), on the 15 x 15 AC block.
- *Kept rows:* MesonGS++ keeps the most important SH rows unquantized by its importance (`meson_utils.py:1741-1760`,
  `sh_keep_threshold` / `sh_keep_topk`); choose them instead by the metric-weighted VQ error,
  (c_i - q_k(i))^T G_i (c_i - q_k(i)), the rows the codebook serves worst in the training views.
- *Already in this project:* the same idea in gsplat's PNG codec passed E2 (G2a, 9 of 9, mean BD-rate -5.37%) and E2c
  (G2c) with this project's own metric and VQ.

**Cheap pilot (train, treehill, flowers).** Three rate points per scene; rows: the host's own path, Gram VQ, Gram VQ
plus metric-chosen kept rows; BD against the host's own. **One session or less** (no fine-tuning; the GN pass
dominates).

**Pre-registered test.** On held-out scenes: P = BD of the full metric pipeline against the host's own training-free
path, E5's three criteria; secondary: against the host's fine-tuned path (report only).

## Which first

**(a) first.** It targets the paper's central finding directly: the metric's value is erased by fine-tuning, and the
only thing fine-tuning cannot touch is the assignment. If re-assignment under the metric keeps the lead, the paper
becomes a method paper with a mechanism (E7) behind it. It reuses E7's fork, hooks and harness, and its pilot costs
about two sessions on development scenes. (b) is lower-risk (the gain before fine-tuning is known), but it is largely
"OGC in another codec", which is the second-host result (MesonGS with `--iterations 0`) and so comes with that work
anyway. Run (b)'s pilot as part of the second-host pilot, not as a method of its own.

## Later (2026-11-08, second session)

- **(a)'s pilot is drafted as a declaration:** `kaggle/drafts/METHOD_A_PILOT_DECLARATION_DRAFT.md` (go rule from E7's
  noise, smallest detectable recovery 0.078 dB on three scenes, the C3DGS fairness arm, licences, code identity).
- **The development scenes (Dace's R7):** train, treehill and flowers are not gate scenes (`kaggle/e5_scenes.py`). Their
  measured process peaks in E6 part (a) were 4.92-7.00 GB allocated and 5.38-8.25 GB reserved, so (a)'s extra state
  (target colours, and the metric if kept on the GPU) fits a T4. But their lead to recover is small (P1 - P_FT mean
  0.092 dB), so a three-scene pilot can only detect a near-complete recovery: see the declaration draft's flag.
- **(b)'s object:** in MesonGS and MesonGS++ the metric would act on the AC block only (`SECOND_HOST_FEASIBILITY.md`,
  R6c); `SH_ONLY_CHECK_DECLARATION_DRAFT.md` measures that lead inside C3DGS first.
