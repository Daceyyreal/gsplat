# DRAFT — declaration for method (a)'s pilot: re-assignment under the metric during fine-tuning (non-confirmatory)

**Status: draft for Dace, 2026-11-08 (second session). Not registered, no code, nothing run.** If adopted it is committed
and pushed alone before any of its code. The method is sketched in `kaggle/drafts/METHOD_SKETCH.md`; numbers quoted are
FINDINGS' (sections 23, 25, 26, 28) or computed from them as stated.

## What it is

**Non-confirmatory.** A pilot on the development scenes **train, treehill and flowers**, fixed now. None of them is a
gate scene (the gate is bonsai, counter, kitchen, room, truck, drjohnson, playroom: `kaggle/e5_scenes.py`), and none
is held out (E3p-E6 part (a) used them). Its only output is "go" or "no go" for a pre-registered test on held-out scenes
(E9's), with the rule below fixed before any data.

## Design

- **Host and processes:** E7's fork, unchanged except for the re-assignment hook; per scene the three seed-0 processes
  (j = -1, 0, +1) and the j = 0 seed-1 process (for the pilot's own noise check).
- **Rows, both methods in every process:** `*_ft` (C3DGS's fine-tuning as in E6/E7) and `*_ft_re` (the same with
  re-assignment every 500 iterations). **Fairness:** C3DGS's rows re-assign under C3DGS's own criterion (nearest codeword
  in squared error, the assignment its `vq_features` makes), OGC's under its own metric (the 16 x 16 Gram estimate E5-E7
  give OGC's call); same schedule, same target (each splat's pre-VQ colour, saved at the colour call).
- **The reading:** per scene the **recovery** Rec_s = P_FT,re,s - P_FT,s, where P_FT,re is the BD of `ogc_gram_ft_re`
  against `c3dgs_ft_re` and P_FT of `ogc_gram_ft` against `c3dgs_ft`, both over the seed-0 curve in the same processes.
  Positive: re-assignment under each method's own criterion keeps more of OGC's lead.

## The go rule, from E7's noise (fixed before any data)

- **Per-scene noise of a within-process difference of two BDs:** σ = 0.034 dB. Two measurements agree on it: E7's
  replicate pair (the same arm run twice in one process) had an across-scene SD of 0.033988 (CLAIMS H10), and E7's
  BD-level noise for P_FT was `SD_BD` 0.024222 (H9), so a difference of two independent BDs has sqrt(2) x 0.024222 =
  0.034.
- **Mean over the three scenes:** `SE` = 0.034 / sqrt(3) = 0.0196 dB.
- **Go** if the mean recovery is at least t(0.95, 2) x `SE` = **0.057 dB** and Rec_s > 0 on **3 of 3** scenes; and the
  pilot's own seed-pair noise (the j = 0 replicate of each `*_ft_re` paired difference) is no more than 1.5 x what E7
  measured (`SD_pool` 0.0326 for E6's P_FT, section 25); otherwise **no decision** (the noise assumption failed).
- **Smallest detectable recovery** (80% power, one-sided 0.05, t with 2 degrees of freedom):
  (2.920 + 1.061) x 0.0196 = **0.078 dB**. With σ taken as known (normal quantiles) it would be 0.049 dB; the t
  version is the declared one, since development scenes may be noisier than the gate.

## A problem with the development scenes, flagged

The development scenes have little lead to recover. Their shrink, P1 - P_FT, is 0.090 dB on flowers, 0.051 on treehill
and 0.136 on train (registered BD values, section 28's table), **mean 0.092 dB**, against 0.307 dB on E7's gate scenes
(H4). **So the pilot can say "go" only for a recovery of about 85% of all there is (0.078 of 0.092 dB),** and a method
that recovers half the lead would read "no go" there. Two alternatives, for Dace:
1. **Run the pilot on gate scenes, labelled exploratory** (they are not held out; the confirmatory test would be on
   E9's held-out scenes). With the gate's shrink, 0.078 dB is about 25% of it. Seven scenes would also lower the
   threshold: `SE` 0.034 / sqrt(7) = 0.0129, go at 0.025 dB (t(0.95, 6) = 1.943), detectable at 80% power 0.037 dB.
2. Keep the development scenes and accept that only a near-complete recovery can pass.

## Rules that apply

- **Code identity:** every pilot session runs one pinned commit, and no run-path file changes between its sessions
  (the rule E5-E7 followed; retired after E7 for the closed experiments, re-applied here).
- **Licences:** the re-assignment under OGC's metric uses OGC's `gram_kmeans` arithmetic or this project's
  `bench/gn/ogc_derived.py`, both under PolyForm Noncommercial (research only), never committed from OGC's repository,
  pushed or bundled; C3DGS is under Inria's Gaussian-Splatting licence (research only); a later MesonGS host is under
  the same Inria licence. No commercial use of the pilot's code.
- **Memory (Dace's R7, from measurement rather than a model):** in E6 part (a) the fork's process peaked at 7.00 GB
  allocated / 8.02 GB reserved on flowers, 6.74 / 8.25 on treehill and 4.92 / 5.38 on train, all with images on the
  GPU (`kaggle/gn_e6/S4/gn6/`, `S5/gn6/` metas). The pilot adds each splat's target colour (48 floats, 192 bytes:
  0.70 GB on flowers, 0.73 GB on treehill) and, if kept on the GPU, the metric (544 bytes per splat packed: 1.98 and
  2.06 GB): at most about 9.7 GB allocated, 11.0 GB reserved, under the T4's 15.64 GB. Streaming the metric from host
  memory in chunks, as E5 does, keeps it near 8.4 GB.

## Cost

Per scene E6 part (a)'s processes plus one fine-tuned row pair per process; about two sessions for the three
development scenes, about four for the seven gate scenes (E7's per-row fine-tuning time, 449-956 s in E6's model).
