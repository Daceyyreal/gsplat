# DRAFT — Amendment 21 (E8): a 2 x 2 factorial on what fine-tuning's shrink of OGC's lead comes from

**Status: draft for Dace, 2026-11-08 (second session). Not in `kaggle/PREREG_GN.md`, not registered, no code.**
Supersedes this file's first version (`96ade074`: arms R, G, N, C, K, R2), on Dace's redesign. Every number quoted from
E7 is FINDINGS section 26's or CLAIMS group H's; the runtime comes from `kaggle/drafts/e8_estimate_draft.py` (E7's
model, `bench/gn/e7_estimate.py`, by import).

## a. Why

E7 (section 26) found neither the colour grids' re-fit nor codebook training explains the shrink of OGC's lead (H1,
H2), and, report-only, that training only the codewords keeps most of the lead while training everything but the
codewords loses all of it (H5). It left open which of the parameters that train carry the shrink: the colours (the
codewords and the kept splats' colours) or opacity and geometry (`_opacity`, `_scaling`, `_rotation`,
`_scaling_factor`). E8 crosses the two in one process. **Not a held-out test** (the seven gate scenes, as E7).

## b. Scenes, checkpoints, processes, host

As E7 (Amendment 20 b-c, notes 1-6), unchanged: seven gate scenes, INRIA's checkpoints, C3DGS at `2a234af5` with its
recorded deviations, OGC at `49ccae72`, K = 4,096, j = -1, 0, +1, six processes per scene (four full: `j-1_p0`, `j0_p0`,
`j+1_p0`, `j0_p1`; `j-1_p1` and `j+1_p1` run the anchor arm only, for the robustness reading), every scene split into
its seed-0 and seed-1 halves, protocol ii. The cu130 image: `kaggle/drafts/WHEEL_CACHE_CU130.md`.

## c. The four arms (both methods in each, same process)

| Arm | Row names | Fine-tuning | Colour grids | Colours (codewords + kept rows) | Opacity, scaling, rotation, scaling factor |
|---|---|---|---|---|---|
| **Z** | `c3dgs`, `ogc_gram` (the unfine-tuned rows every process already has) | none | the shared pre-fine-tuning grid | not trained | not trained |
| **C** | `*_ft_c` | 5,000 iterations | frozen, pre | train | frozen |
| **N** | `*_ft_n` | 5,000 iterations | frozen, pre | frozen | train |
| **G** (anchor) | `*_ft_g` | 5,000 iterations | frozen, pre | train | train (E7's G: also a replication of E7's BD_G) |

- `_xyz` trains in no arm (learning rate 0 for a loaded checkpoint; section 24 item 4 and its erratum).
- **Dropped:** R and K (E7 measured both; neither enters a primary below), and R2 (G's seed pair and the seed-1
  curve give the noise and the robustness reading).
- As code, E7's `ARMS` form (`kaggle/e7_scenes.py:60-68`): C = `grid "pre"`, `no_grad NO_GRAD_K`; N = `grid "pre"`,
  `no_grad ("_features_dc", "_features_rest")`; G as E7. E7's guard that each frozen tensor is unchanged after
  `finetune` covers both.

## d. The quantities, in plain words

For an arm X, **BD_X** is OGC's lead after arm X: the BD-PSNR of OGC's row against C3DGS's row over the seed-0 curve,
exactly as E7's (for Z, before any fine-tuning; this is P1 in E8's own processes).

- **The shrink of an arm,** BD_Z - BD_X: how much of the lead that arm's fine-tuning removes.
- **Main effect of colour training,** M_col = ½ [(BD_Z - BD_C) + (BD_N - BD_G)]: the lead removed by switching colour
  training on, averaged over geometry-and-opacity training off (Z to C) and on (N to G). **Positive: colour training
  removes lead.**
- **Main effect of opacity-and-geometry training,** M_geo = ½ [(BD_Z - BD_N) + (BD_C - BD_G)]: the same for opacity and
  geometry. **Positive: their training removes lead.**
- **Interaction (report only),** I = (BD_Z - BD_C) - (BD_N - BD_G): how much more lead colour training removes when
  geometry is frozen than when it trains. Positive: the two compensate for the same thing (each does less when the
  other is on).
- **An identity to keep in mind:** M_col + M_geo = BD_Z - BD_G, the whole shrink under the frozen grid (E7 measured
  BD_R near 0.047 and BD_G 0.027 against P1 0.354), so the two main effects cannot both be near 0 if G replicates.

## e. Tests (E7's f, unchanged in form)

The two primaries, M_col and M_geo, per scene and mean (n = 7). Noise in E5's form on the j = 0 seed pair of each
primary's contrast of paired differences (`d_sp` = the same combination of the four arms' OGC-minus-C3DGS differences
at j = 0, seed p), `SD_pool` = sqrt(mean `v_s`), **`SE` = `SD_pool` / sqrt(n)**. Each test type one Holm family of the
two primaries at a family-wise 0.05, ordered by its own statistic: **superiority** (two-sided, E5's three criteria,
t(1 - 0.0125, n) then t(1 - 0.025, n)) and **equivalence** (TOST at δ = 0.07 dB, t(1 - 0.025, n) then t(1 - 0.05, n)).
Robustness reading: the same tests with `SE_BD` from G's seed-0 and seed-1 BD (E7 used R's). A missing BD enters as 0.

**Power from E7's measured noise.** E7's within-process contrasts of two arms had `SD_pool` 0.0147 (Δ_refit) and 0.0295
(Δ_cb) (H1, H2: `SE` 0.005561656 and 0.011134979 at n = 7). A main effect averages two such contrasts. Arms of one
process are strongly correlated (E6's P_FT seed-pair `SD_pool` per arm is 0.0326, section 25, more than either
contrast), so the conservative reading takes no reduction from the averaging: **`SE` 0.0056-0.0111.** Then:
- equivalence in Holm's first position (2.365) passes a true effect of 0 whenever `SE` < 0.0296, with 80% power for
  δ >= (2.365 + 1.415) x `SE` = 0.021-0.042 dB, so δ = 0.07 dB is powered;
- superiority in the first position (2.841) detects a mean main effect above 2.841 x `SE` = 0.016-0.032 dB;
- E7's whole shrink is 0.307 dB (H4) and its report-only contrasts were 0.25-0.27 dB (H5), so effects of the size
  that matters are far above both thresholds. The stated risk is E7's: scene spread (H8, `SE` 0.0227 for Δ_refit,
  driven by kitchen).

## f. Outcomes, written before any data

| M_col | M_geo | The claim it supports |
|---|---|---|
| superior, positive | equivalent | the shrink comes from colour training (codewords and kept colours), not from opacity and geometry |
| equivalent | superior, positive | the shrink comes from opacity and geometry training, not from colour training |
| superior, positive | superior, positive | both carry part of it; the interaction (report only) says whether they overlap or add |
| equivalent | equivalent | cannot both hold if G replicates E7 (the identity in d); if it happens, reported as a failed replication of BD_G, not interpreted |
| superior, negative (either) | | that training raises OGC's lead: reported, not interpreted |
| neither, or incomplete | | E7's h rules |

## g. Secondaries (report only, declared here)

1. I, per scene and mean. 2. BD_G against E7's BD_G per scene (a cross-run comparison, labelled so: seeded C3DGS runs
are not reproduced on the GPU, CLAIMS E2). 3. P1 in E8's processes (BD_Z). 4. Perceptual BDs of every arm and both main
effects (section 27's method). 5. The fits' shape (sections 25 and 28) and a PCHIP reading of each BD (section 28's
method). 6. Bytes per arm and point, times, memory, the frozen-tensor guards.

## h. Optional arm O (opacity only) — Dace's choice

O trains `_opacity` alone (`no_grad` the colours and `_scaling`, `_rotation`, `_scaling_factor`). It splits M_geo's
geometry from opacity; as a 2 x 2 x 2 it would need a sixth arm, so with O alone it is **report-only**:
BD_N - BD_O (scaling and rotation's share given opacity trains) and BD_Z - BD_O (opacity alone).

## i. Runtime and quota (`kaggle/drafts/e8_estimate_draft.py`, `variant`)

| Design | Jobs, high / calibrated (h) | Sessions | Session-hours, high / calibrated | Weeks at 30 h | Largest half (s) |
|---|---|---|---|---|---|
| **4-arm (Z, C, N, G)** | 69.0 / 46.8 | 10 + wheel | **71.5 / 49.2** | 2.4 / 1.6 | 23,239 |
| 5-arm (+ O) | 83.0 / 56.4 | 11 + wheel | 85.5 / 58.9 | 2.8 / 2.0 | 28,973 |
| first draft (R, G, N, C, K, R2) | 100.4 / 68.4 | 11 + wheel | 102.9 / 70.9 | 3.4 / 2.4 | 36,618 |

"Calibrated" scales each half by E7's measured / high-estimate ratio for that scene and half (0.55-0.86); "wheel" is a
1.24 h build session, not needed if E7 S11's output already holds the cu130 wheel (`WHEEL_CACHE_CU130.md`, step 1). The
4-arm design's largest half, 23,239 s at the high end, leaves 15,941 s of a session's 39,180 s budget with a restored
wheel and 11,897 s of 35,136 s with the wheel built in the session, so the cache no longer decides whether E8 fits; it
only saves about 1 h per session that would otherwise build. 30 h a week is an assumption about the account's quota.

## j. Open for Dace

- 4-arm or 5-arm; δ kept at 0.07 dB (E7's) or set lower (powered to about 0.042 dB at the larger `SE`).
- Whether the robustness reading uses G's seed-1 curve (proposed) or adds Z's.
