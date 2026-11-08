# DRAFT D3 — declaration: method (a)'s pilot on the gate scenes (exploratory)

**Status: ready to commit on Dace's go, alone and pushed, before any of its code. Not in PREREG; nothing run.**
Supersedes `kaggle/drafts/METHOD_A_PILOT_DECLARATION_DRAFT.md` (development scenes), adopting its own proposal. The
method: `kaggle/drafts/METHOD_SKETCH.md` (a).

## What it is

**Exploratory, non-confirmatory, on the seven gate scenes** (bonsai, counter, kitchen, room, truck, drjohnson,
playroom), which are not held out (E5-E7 used them). Its only output is "go" or "no go" for a later test. **Any
confirmatory test of the method must use held-out scenes, never the gate scenes** (E9's, or others fixed before their
data are read).

## Design

E7's fork and processes (j = -1, 0, +1 at seed 0; j = 0 at seed 1). Rows, both methods: `*_ft` (C3DGS's fine-tuning)
and `*_ft_re` (the same with re-assignment every 500 iterations; each splat's target is its pre-VQ colour). **Fairness:**
C3DGS's rows re-assign under C3DGS's own criterion (nearest codeword in squared error), OGC's under its metric.
**Recovery** per scene: Rec_s = P_FT,re,s - P_FT,s (both BDs over the seed-0 curve, same processes).

## The go rule (fixed now), and its derivation

- **Per-scene noise of a within-process difference of two BDs: σ = 0.034 dB.** E7's replicate pair (the same arm twice
  in one process) had an across-scene SD of 0.033988 (CLAIMS H10), and sqrt(2) x E7's BD-level `SD_BD` 0.024222 (H9) =
  0.034.
- **Mean over n = 7 scenes:** `SE` = 0.034 / sqrt(7) = **0.0129 dB**.
- **Go** if mean recovery >= t(0.95, 6) x `SE` = 1.943 x 0.0129 = **0.025 dB**, Rec_s > 0 on at least **5 of 7** (E5's
  count rule), and the pilot's own j = 0 seed-pair noise for the `*_ft_re` rows is no more than 1.5 x E6's (`SD_pool`
  0.0326, section 25); otherwise "no decision".
- **Smallest detectable recovery** (80% power, one-sided 0.05): (t(0.95, 6) + t(0.80, 6)) x `SE` = (1.943 + 0.906) x 0.0129
  = **0.037 dB**, about 12% of E7's shrink of 0.307 dB (H4). (t with 6 degrees of freedom is the conservative choice; σ
  is taken from E7, not estimated by the pilot.)
- **Note on Dace's "0.078 dB":** 0.078 dB was the three-development-scene figure ((2.920 + 1.061) x 0.034 / sqrt(3)); on
  seven gate scenes the smallest detectable recovery is 0.037 dB and the go threshold 0.025 dB.

## Rules that apply

- **Code identity:** every pilot session runs one pinned commit; no run-path file changes between its sessions.
- **Licences:** re-assignment under OGC's metric uses OGC's `gram_kmeans` arithmetic or `bench/gn/ogc_derived.py`, both
  PolyForm Noncommercial (research only; never committed from OGC's repository, pushed or bundled); C3DGS and MesonGS are
  under Inria's Gaussian-Splatting licence (research only).
- **Memory:** the gate scenes' measured process peaks were 4.69-9.76 GB allocated (E6 part (b), E7); the pilot adds each
  splat's target colour (192 bytes) and, if kept on the GPU, the metric (544 bytes): at most +2.5 GB on drjohnson (3.4 M
  splats, images on the CPU), +1.2 GB on room; under the T4.

## Cost

E6 part (b)'s four processes per scene (j = -1, 0, +1 at seed 0, j = 0 at seed 1) with the re-assignment pair added to
each (two more fine-tuned rows per process, at E6's per-row time): **43.0 h of jobs at the high end, 29.4 h calibrated by
E7's measured ratios** for the seven scenes, about five sessions with the cached wheel.
