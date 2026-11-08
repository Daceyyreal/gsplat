# DRAFT D2 — Amendment 21 (E8): one directional primary in a 2 x 2 within-process design

**Status: ready to commit on Dace's go, as a PREREG amendment committed and pushed alone before any E8 code. Not in
PREREG now; nothing run.** Supersedes `kaggle/drafts/E8_AMENDMENT21_DRAFT.md` (two primaries), on Dace's redesign.
Runtime from `kaggle/drafts/e8_estimate_draft.py`; numbers quoted are FINDINGS section 26's or CLAIMS group H's.

## a. What E8 tests, and what it does not

E7 found, report-only and post hoc, that training only the codewords keeps most of OGC's lead while training everything
but the codewords loses all of it (CLAIMS H5). From that reading comes one directional hypothesis about **mechanism**:
the lead is lost through opacity and geometry training, not through colour training. **E8 tests it on new runs of the
same seven gate scenes.** It is a mechanism test inside the scenes that suggested it, **not a generalisation**: no
held-out scene, no claim beyond C3DGS at `2a234af5` on INRIA's checkpoints of these seven scenes.

## b. Design (as E7 unless stated)

Scenes, checkpoints, host, OGC source chain, K = 4,096, j = -1, 0, +1, six processes per scene, the split into halves,
protocol ii: E7's (Amendment 20 b-c, notes 1-6). The cu130 wheel: `kaggle/drafts/WHEEL_CACHE_CU130.md`.

**Arms, both methods in every process:**

| Arm | Fine-tuning (5,000 iterations) | Colour grids | Colours (codewords and kept rows) | Opacity, scaling, rotation, scaling factor |
|---|---|---|---|---|
| **Z** | none (the unfine-tuned rows every process has) | shared pre-fine-tuning grid | - | - |
| **C** | colour only | frozen, pre | train | frozen |
| **N** | opacity and geometry only | frozen, pre | frozen | train |
| **G** | both (E7's G) | frozen, pre | train | train |

`_xyz` trains in no arm. j = -1 and +1 at seed 1 run G only (the robustness reading). Optional: **O** (opacity only)
and a **4th rate point** (j = +2 at seed 0, all arms), both secondary.

## c. Definitions

- **BD_X** = the BD-PSNR of OGC's row against C3DGS's row in arm X over the seed-0 curve (E5's registered fit; three
  points, a parabola through them). It is OGC's lead after arm X.
- **The primary,** per scene: **Δ_s = BD_C,s - BD_N,s**: OGC's lead when only the colours train, minus its lead when only
  opacity and geometry train.

## d. The primary hypothesis and its test

**H_E8: Δ > 0** (OGC keeps more of its lead under colour-only fine-tuning than under opacity-and-geometry-only
fine-tuning). **One-sided superiority, E7's superiority test without Holm** (one hypothesis, no family), on r(Δ_bar)
and r(SE) (r = round to 9 decimals, strict comparisons), with n = 7 and n degrees of freedom:
1. r(Δ_bar) > 0;
2. #{s : r(Δ_s) > 0} >= (7 n + 9) // 10 (5 of 7);
3. r(Δ_bar) > t(1 - 0.05, n) x r(SE), t(0.95, 7) = 1.895.

**H_E8 holds** if all three hold. Noise in E7's form: `d_sp` = (OGC - C3DGS in arm C) - (OGC - C3DGS in arm N) in
protocol ii's PSNR at j = 0, seed p; `SD_pool` = sqrt(mean `v_s`); **`SE` = `SD_pool` / sqrt(n)**. A missing BD enters
as 0 and counts as not positive; fewer than 5 scenes after drops: incomplete. Robustness reading: the same test with
`SE_BD` = sqrt(2) x `SD_BD` / sqrt(n) from G's seed-0 and seed-1 BD; a different outcome is "fragile" and the
registered `SE`'s outcome stands.

**Power, from E7's measured noise.** E7's within-process contrasts of two arms had `SE` 0.005561656 (Δ_refit) and
0.011134979 (Δ_cb) at n = 7 (H1, H2). Criterion 3's threshold is then 1.895 x `SE` = **0.0105-0.0211 dB**, and with 80%
power the test detects a true mean of (1.895 + 0.896) x `SE` = **0.0155-0.0311 dB**. E7's report-only contrast of the
codewords-only arm against the all-but-codewords arm was 0.266211750 dB in the mean, positive on 7 of 7 (H5); if Δ is
anywhere near that, the test is far from its threshold. The count criterion (5 of 7) is the tighter one if the effect is
uneven across scenes (as E7's kitchen was, H7).

## e. Outcomes, written before any data

| Result | Reading |
|---|---|
| H_E8 holds, not fragile | in these scenes, OGC's lead is lost mainly through opacity and geometry training: colour-only fine-tuning keeps more of it |
| H_E8 holds, fragile | the same, flagged: BD-level noise could carry it |
| criterion 1 or 2 fails, or Δ_bar <= 0 | E7's post-hoc reading is not supported by new runs |
| incomplete | E7's rules |

## f. Secondaries (report only, declared here, no bars)

1. **Factorial main effects** on the lead, ME_col = ½ [(BD_C - BD_Z) + (BD_G - BD_N)] and ME_geo = ½ [(BD_N - BD_Z) +
   (BD_G - BD_C)], with the identity **ME_col + ME_geo = BD_G - BD_Z** (the whole change from no fine-tuning to fine-tuning
   both, under the frozen grid); each factor's **share of the shrink**, ME / (BD_G - BD_Z), with a 95% percentile interval
   from 10,000 scene-bootstrap resamples (seed 0).
2. **Interaction** I = (BD_G - BD_N) - (BD_C - BD_Z).
3. **All four simple effects:** BD_C - BD_Z (colour, geometry frozen), BD_G - BD_N (colour, geometry training),
   BD_N - BD_Z (geometry, colour frozen), BD_G - BD_C (geometry, colour training).
4. **G against E7's G:** per scene BD_G(E8) - BD_G(E7) and a TOST at ±0.07 dB with `SE` = their across-scene SD / sqrt(7)
   (a cross-run comparison: seeded runs are not reproduced on the GPU, CLAIMS E2, so this checks replication within noise,
   not identity).
5. Perceptual BDs (section 27's method), PCHIP BDs and the fits' shape (section 28's), bytes, times, memory, guards.
6. **Optional O (opacity only):** BD_N - BD_O and BD_O - BD_Z. **Optional 4th rate point (j = +2):** every arm's BD over
   four points as a secondary only (E2's cubic is not used: the primary stays on E7's three points, j = -1, 0, +1).

## g. Cost (high / calibrated by E7's measured ratios; 420 s setup per session with the cached wheel)

| Design | Jobs (h) | Sessions | Session-hours | Weeks at 30 h | Largest half (s) |
|---|---|---|---|---|---|
| **4 arms (Z, C, N, G)** | 69.0 / 46.8 | 10 | **70.2 / 48.0** | 2.3 / 1.6 | 23,239 |
| + O | 83.0 / 56.4 | 11 | 84.3 / 57.6 | 2.8 / 1.9 | 28,973 |
| + 4th rate point | 81.3 / 55.5 | 10 | 82.5 / 56.6 | 2.8 / 1.9 | 30,182 |
| + O + 4th rate point | 98.8 / 67.5 | 11 | 100.0 / 68.8 | 3.3 / 2.3 | 37,827 |

Every half fits a session's 39,180 s at the high end. Add 1.24 h if the wheel must be built (`WHEEL_CACHE_CU130.md`).
The 4th point is costed as one more seed-0 process per scene at j = +1's cost, which a higher threshold (fewer kept
rows) would not exceed. 30 h a week is an assumption about the account's quota.
