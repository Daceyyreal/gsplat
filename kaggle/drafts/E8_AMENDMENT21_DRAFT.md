# DRAFT — Amendment 21 (E8): which parameters carry the shrink of OGC's lead under fine-tuning

**Status: draft for Dace, 2026-11-08. Not in `kaggle/PREREG_GN.md`, not registered, no code.** Written after FINDINGS
sections 26 (E7) and 27 (perceptual secondaries). Every number quoted from E7 is FINDINGS section 26's or CLAIMS group
H's; the runtime figures come from `kaggle/drafts/e8_estimate_draft.py` (E7's model, `bench/gn/e7_estimate.py`, by
import).

## a. Why

E7 left one question open by design (CLAIMS H5, FINDINGS section 26 "What section 26 settles"): with both primaries
equivalent to 0, the shrink of OGC's lead (P1 0.3540 minus P_FT 0.0467 = 0.3073 dB in E7's own rows, H4) is carried by
what neither primary held fixed, `_opacity`, `_scaling`, `_rotation`, `_scaling_factor` and the kept splats' colours,
and **no E7 arm separates them**. Report-only, E7's arm K (only the K codewords train) kept most of the lead
(BD_K - BD_R mean 0.254563023, positive on 7 of 7), while CG (everything but the codewords trains) lost all of it
(post hoc: 103.8% of the shrink). E8 separates geometry and opacity from colour, inside one process, on the same seven
gate scenes. **E8 is not a held-out test** (as E7, Amendment 20's disclosure).

## b. Scenes, checkpoints, processes, host

As E7 (Amendment 20 b-c, notes 1-6), unchanged: the seven gate scenes, INRIA's checkpoints, C3DGS at `2a234af5` with the
recorded deviations, OGC at `49ccae72` (sources per Amendment 18 note 3), K = 4,096, colour thresholds j = -1, 0, +1, six
processes per scene (four full: `j-1_p0`, `j0_p0`, `j+1_p0`, `j0_p1`; arm R only in `j-1_p1` and `j+1_p1`), the
split of every scene into its seed-0 and seed-1 halves (20 j), protocol ii. **One change of image:** Kaggle's current
image is cu130 (E7 S11); see h for the wheel.

## c. Rows and arms

Unfine-tuned rows: `c3dgs` and `ogc_gram` with the shared pre-fine-tuning grid (as E5-E7). Each fine-tuned arm is
5,000 iterations of C3DGS's own `finetune`, from the copy with the row's table installed, one per method:

| Arm | Row names | Colour grids | K codewords (DC, AC) | Kept splats' colours | `_opacity`, `_scaling`, `_rotation`, `_scaling_factor` | As code (`kaggle/e7_scenes.py` `ARMS` form) |
|---|---|---|---|---|---|---|
| **R** | `c3dgs_ft`, `ogc_gram_ft` | moving | train | train | train | E7's R |
| **G** (anchor) | `*_ft_g` | frozen, pre | train | train | train | E7's G |
| **N** (no colour) | `*_ft_n` | frozen, pre | frozen | frozen | train | `grid: "pre"`, `no_grad: ("_features_dc", "_features_rest")` |
| **C** (colour only) | `*_ft_c` | frozen, pre | train | train | frozen | `grid: "pre"`, `no_grad: NO_GRAD_K` |
| **K** (codewords only) | `*_ft_k` | frozen, pre | train | frozen | frozen | E7's K (`zero_rows: "kept"`, `NO_GRAD_K`) |
| **R, replicate** (`j0_p0` only) | `*_ft_r2` | moving | as R | as R | as R | E7's R2 |

- `_xyz` trains in no arm (learning rate 0 for a loaded checkpoint, FINDINGS section 24 item 4 and its erratum).
- **Why G is the anchor and not R:** N, C and K freeze the grids; comparing them with G keeps the grids' re-fit out
  of the primaries (E7 found it equivalent to 0, H1, but its sign rests on kitchen, H7).
- **N and C are the two new arms;** K is repeated so that C - K (the kept colours) is a within-process comparison
  (seeded C3DGS runs are not reproduced on the GPU, CLAIMS E2, so no E7 row can be reused).
- New code: two `ARMS` entries; N's `no_grad` names two colour tensors where E7's named geometry. The guard that each
  frozen tensor is unchanged after `finetune` (E7's logs, 20 e) covers them.

## d. The primaries and their tests (E7's f, unchanged in form)

For an arm X, BD_X is the BD-PSNR of `ogc_gram_ft_X` against `c3dgs_ft_X` over the seed-0 curve, exactly as E7's.

1. **Δ_geo,s = BD_N,s - BD_G,s:** what removing all colour training, leaving geometry and opacity, does to OGC's lead.
   **Positive:** colour training is part of what reduces the lead. **Equivalent to 0:** geometry and opacity training
   alone reduce it as much as training everything does (under the fixed grid).
2. **Δ_col,s = BD_C,s - BD_G,s:** what removing geometry and opacity training, leaving the colours, does to the lead.
   **Positive:** geometry and opacity training is part of what reduces it. **Equivalent to 0:** colour training alone
   reduces it as much.

- **Noise:** E5's form on the j = 0 seed pair of each primary's paired difference, `SE` = `SD_pool` / sqrt(n), n
  degrees of freedom; the robustness reading with `SD_BD` from arm R's seed-1 curve; a missing BD enters as 0 (all as
  E7's f and Note 2 C2).
- **Tests:** each type one Holm family of the two primaries at a family-wise 0.05, ordered by the test's own statistic:
  superiority (two-sided, E5's three criteria with t(1 - 0.0125, n) then t(1 - 0.025, n)) and equivalence (TOST,
  t(1 - 0.025, n) then t(1 - 0.05, n)), on r(`Δ_bar`) and r(`SE`).
- **Margin δ = 0.07 dB,** E7's, so the two experiments' equivalence statements read alike. **Power, from E7's measured
  noise** (H1, H2: `SE` 0.005561656 and 0.011134979 at n = 7, so `SD_pool` 0.0147 and 0.0295): the TOST in Holm's first
  position (2.365) is passed by a true Δ of 0 whenever `SE` < 0.0296, so at either measured `SE` with a wide margin; at
  `SE` 0.0111 a true |Δ| up to about 0.044 dB still passes. Superiority at 2.841 x 0.0111 detects |Δ_bar| above about
  0.032 dB. E7's scene spread (H8, `SE` 0.0227 for Δ_refit, driven by kitchen) is the stated risk, as in E7.
- **What E7 suggests, not a hypothesis of E8:** BD_K kept the lead and CG lost it, so if geometry and opacity carry the
  shrink, Δ_geo is near 0 and Δ_col is large and positive; if the kept colours carry it, the reverse.

## e. Outcomes, written before any data (one line per combination)

| Δ_geo | Δ_col | Reading |
|---|---|---|
| equivalent | superior, positive | geometry and opacity training carries the shrink; colour training alone does not |
| superior, positive | equivalent | colour training (codewords and kept colours) carries the shrink |
| superior, positive | superior, positive | each part reduces the lead and neither alone produces all of it |
| equivalent | equivalent | either alone produces the shrink: the two compensate redundantly |
| superior, negative (either) | | freezing that part shrinks the lead further: reported as found, not interpreted |
| neither / incomplete | | E7's h rules |

## f. Secondaries (report only, no bars; declared here, before data)

1. **Δ_kept = BD_C - BD_K:** the kept colours' share with geometry frozen.
2. **BD_K - BD_G:** E7's H5 contrast within E8's processes.
3. **Δ_refit replicated, BD_G - BD_R,** described with E7's three outcomes' arithmetic but no verdict.
4. **P1 and P_FT replicated** (E7's g.4 form).
5. **Perceptual:** every BD and Δ above in BD-SSIM and BD-LPIPS (section 27's method), with sign agreement.
6. **The fits' shape:** section 25's bending count for every BD.
7. Bytes per arm and point, BD-rate per arm pair (E7's g.8 form), times, memory, the frozen-tensor guards.

## g. Optional part (c): the boundary question (Dace's choice)

**K = 512, OGC's equal-size row** (OGC's Table 1: +0.32 dB before fine-tuning, +0.01 dB after, quoted in the paper):
per scene E6 part (b)'s four processes (unfine-tuned pair and arm R; j = -1, 0, +1 at seed 0, j = 0 at seed 1).
Proposed as **descriptive** (P1_512, P_FT_512, and P_FT_512 - P_FT at K = 4,096 per scene), with no verdict, so it adds
no family. Note: in E3r the codebook size moved bytes by only 1.02x against the colour threshold's 2.49x (FINDINGS
section 15), so K = 512 is a **capacity** point, not a lower rate; a lower rate is j = +2, +3 (a higher threshold).
If the 2-bit-regime limitation is the target, K = 512 fits it; if rate is, j = +2, +3. Not both.

## h. Runtime and quota (`kaggle/drafts/e8_estimate_draft.py`)

Per scene (s), E7's model with E8's rows (8 more fine-tuned rows per full process instead of E7's 8 plus 2 `*_own` rows),
and "calibrated" = the high estimate times E7's own measured / high-estimate ratio for that scene and half
(0.55-0.86):

| Scene | Seed-0 half, low-high | calibrated | Seed-1 half, low-high | calibrated | Part (c), high |
|---|---|---|---|---|---|
| bonsai | 22,310-35,184 | 22,324 | 11,788-18,273 | 10,111 | 14,783 |
| counter | 20,792-33,228 | 25,262 | 10,985-17,093 | 10,919 | 13,645 |
| kitchen | 22,159-36,225 | 30,973 | 11,736-19,393 | 13,862 | 17,092 |
| room | 22,874-36,618 | 25,335 | 12,100-19,416 | 12,680 | 16,571 |
| truck | 15,960-28,147 | 15,958 | 7,882-13,826 | 8,308 | 9,852 |
| drjohnson | 20,190-35,802 | 25,176 | 10,510-19,633 | 13,175 | 18,578 |
| playroom | 18,346-31,875 | 22,121 | 9,433-16,648 | 9,914 | 14,124 |
| **all** | **60.3-100.4 h** | **68.4 h** | | | **29.1 h (19.9 calibrated)** |

- **The wheel.** On the cu130 image E7 S11 built the gsplat wheel in 4,034.8 s (setup 4,464.0 s). The budget from setup's
  end is then 35,136 s instead of 39,180 s, and **at the high end six of the eleven sessions below would not fit**
  (room, kitchen, drjohnson and bonsai seed-0 halves; room + playroom and bonsai + counter seed-1 pairs). So: **one short
  wheel-build session first** (about 1.24 h), its output saved as a Kaggle dataset and restored by every E8 session (as
  E7 S1-S10 restored theirs), setup 420 s.
- **Sessions** (one lane each, E7's packing): S1-S7 the seed-0 halves (room, kitchen, drjohnson, bonsai, counter,
  playroom, truck; 2,562-11,033 s spare at the high end), S8 kitchen + truck seed-1, S9 room + playroom, S10 bonsai +
  counter, S11 drjohnson (3,116-19,547 s spare).
- **Quota:** 102.9 session-hours at the high end, 70.9 calibrated, with the wheel session: **about 3.4 weeks at the high
  end and 2.4 weeks calibrated, at 30 GPU-hours a week (an assumption; check the account).** Part (c) adds 29.1 h at the
  high end (about 3 sessions, one more week).

## i. Open for Dace

- G as the anchor (above) or R; keeping K (needed for Δ_kept); part (c) and which form (K = 512 or j = +2, +3).
- δ = 0.07 dB kept from E7, or tightened (the measured `SE` would power about 0.05 dB in the first position).
- Whether the robustness reading keeps arm R's seed-1 curve (E7's) or adds G's (two more processes per scene).
