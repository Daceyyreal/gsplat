# Claims ledger (paper preparation)

**What this is:** every claim the paper could make, each tied to its evidence in `kaggle/FINDINGS.md`, the script that
re-checks it, the committed data and the limits that must travel with it. No paper prose. Nothing here is new: every
claim restates FINDINGS no more strongly than FINDINGS does, and every number is FINDINGS'. Written 2026-10-13, after
FINDINGS section 19 (`de8690ca`, `check_s19.py` `63fe6de0`).

**Status labels:**
- **confirmatory:** a pre-registered criterion and its verdict;
- **exploratory:** planned but without a bar, or chosen after the data (FINDINGS says "post hoc");
- **descriptive:** a pre-registered secondary with no bar, or a reported measurement;
- **negative:** a failed criterion, a withdrawal or a retirement.

**Check column:** the committed script that asserts the number. "**none**" is a flag: FINDINGS sections 1-10 have no
committed checker (sections 9 and 10 say a script re-checked them, but it is not in the repository).

**Correspondence column:** "**yes**" marks a claim that relies on the OGC author's email. It is paraphrased only and
never quoted.

**Limits common to every E5 row** (group A, and B1-B3): seven gate scenes (bonsai, counter, kitchen and room from
Mip-NeRF 360, truck from Tanks & Temples, drjohnson and playroom from Deep Blending); INRIA's published 30k
checkpoints; C3DGS at `2a234af5` with the recorded deviations (`<cstdint>` retry, chunked `eigh` / `det`, the CPU
evaluation fix); OGC's code at `49ccae72`; K = 4,096; three colour thresholds (j = -1, 0, +1), one process (seed 0) per
point; `SE_noise` from the two seeds at j = 0 only; protocol ii (this project's evaluation, not C3DGS's or the
paper's); 2x T4 and one Kaggle image. **Disclosure** (Amendment 15 d, kept by 17 d): E2 and E2c used bonsai, counter,
kitchen, room and truck with this project's own checkpoints, and OGC's paper reports bonsai, counter, kitchen and
room in its C3DGS host.

## (A) The OGC replication (E5) and fine-tuning against the paper

| ID | Claim | Status | Evidence | Check | Data | Scope limits that travel with it | Corr. |
|---|---|---|---|---|---|---|---|
| A1 | E5 passed as pre-registered: both primaries pass all three criteria, with n = 7, no scene dropped, no primary row or P missing, and no process deferred, deadline-cut, rerun or stopped. | confirmatory: Amendment 17 d (P1 and P2, criteria 1-3, n ≥ 5, `incomplete` > `fail` > `pass`) with Note 2 C2, C5, C7; verdict **pass** | s19 "The verdict" (combiner output, per-scene table) | `check_s19.py` | `kaggle/gn_e5/S1/gn5/`, `S2/gn5/`, `S3/gn5/` | the common E5 limits; the disclosure | no |
| A2 | Inside C3DGS, at equal rate over three colour thresholds, OGC's Gram-metric VQ beats C3DGS's own colour VQ on all seven gate scenes: BD-PSNR `P_bar` 0.3558 dB, against 2 x `SE_noise` 0.0065 dB. | confirmatory: Amendment 17 d P1; **pass** | s19 per-scene table (0.148587944 on truck to 0.484060073 on bonsai) | `check_s19.py`; the independent `numpy.polyfit` recompute inside it | as A1 | the common E5 limits; before fine-tuning; BD over three points is an exact parabola, so each P rests on six measurements; `SE_noise` is seed noise, not scene spread | no |
| A3 | Across scenes P1 is large against its spread: mean 0.3558 dB, SD 0.1175 dB, t = 8.01 over the seven scenes (P2: 0.1874, 0.0805, 6.16). | exploratory: labelled "not pre-registered" in s19 | s19 "Exploratory" | `check_s19.py` | as A1 | not a pre-registered test; seven scenes | no |
| A4 | Before C3DGS's fine-tuning, `ogc_gram` - `c3dgs` at j = 0 has mean +0.5239 dB (95% interval +0.3798 to +0.6681), reported next to the paper's +0.49 dB. | descriptive: Amendment 17 d secondary, no bar | s19 "Secondaries", fine-tuning bullet | `check_s19.py` | as A1 | **not like for like with the paper:** theirs is 9 Mip-NeRF 360 scenes, their evaluation, one run each; ours is 7 scenes (4 shared), protocol ii; j = 0 point only, so unequal bytes (E5p on train: +3.48%, s18) | no |
| A5 | After C3DGS's 5,000-iteration fine-tuning, the lead shrinks to a mean of +0.0604 dB (95% interval +0.0258 to +0.0949), whose interval holds the paper's +0.09 dB; drjohnson is the only scene below 0. | descriptive: Amendment 17 d secondary, no bar | s19 "Secondaries", fine-tuning bullet; settles bullet | `check_s19.py` | as A1 | as A4 (cross-protocol, cross-scene); **bytes after fine-tuning not compared in s19** (E5p on train: fine-tuning shrinks OGC's `features_rest` by 32.5%, s18); two processes per scene | no |
| A6 | Our derived implementation of OGC's VQ (`ogc_gram_ours`) reproduces OGC's labels, float codebook and `.npz` bytes exactly in every process: 28 in E5 and 4 in E5p. | descriptive: Amendment 18 d, report only | s19 "Secondaries" (`ogc_gram_ours` bullet); s18 table | `check_s19.py`, `check_s18.py` | as A1; `kaggle/gn_e5p/attempt2/gn5p/` | identity stops at labels, codebook and bytes; array contents not hashed (s18); GPU; `"plain"` and `"scalar"` paths rest on a small-case test (s18) | no |
| A7 | On train, a development scene, the pilot already showed the direction: OGC's VQ leads C3DGS's by +0.1292 dB BD-PSNR (P1; P2 +0.0641 dB). | exploratory: E5p, Amendment 17 c, no verdict | s18 "Rate and distortion" | `check_s18.py` | `kaggle/gn_e5p/attempt2/gn5p/` | one development scene; no verdict; process 1's images forced onto the CPU | no |

## (B) The dissection: plain, scalar and Gram metrics in OGC's Lloyd

| ID | Claim | Status | Evidence | Check | Data | Scope limits that travel with it | Corr. |
|---|---|---|---|---|---|---|---|
| B1 | In OGC's own Lloyd, the 16 x 16 Gram metric beats its trace (`tr(G_i) / 16 * I`) at equal rate on all seven gate scenes: BD-PSNR `P_bar` 0.1874 dB, against 2 x `SE_noise` 0.0003 dB. | confirmatory: Amendment 17 d P2; **pass** | s19 per-scene table (0.100030562 on truck to 0.320522033 on bonsai) | `check_s19.py` | as A1 | the common E5 limits; before fine-tuning | no |
| B2 | Mean BD-PSNR over the seven scenes splits as `ogc_plain` against `c3dgs` 0.1185 dB, `ogc_scalar` against `ogc_plain` 0.0520 dB and `ogc_gram` against `ogc_scalar` 0.1874 dB; they sum to 0.3579 dB against P1's 0.3558 dB, and the Gram metric gives about half of P1 (52.7%). | exploratory: labelled "not pre-registered" in s19 | s19 "Exploratory" | `check_s19.py` | as A1 | BD differences are not additive (each pair has its own byte range); a mean over scenes; see Gaps (conflicting shares) | no |
| B3 | The two other steps are positive on every scene in BD-PSNR and at j = 0: `ogc_scalar` - `ogc_plain` (paired mean +0.0613 dB, `SE_noise` 0.000125) and `ogc_plain` - `c3dgs` (+0.2167 dB, `SE_noise` 0.003220). | descriptive: Amendment 17 d secondaries, no bars | s19 "Secondaries" table and paired bullet | `check_s19.py` | as A1 | the common E5 limits | no |
| B4 | On train at j = 0, OGC's Lloyd with the identity metric gives 47.9% of its lead over C3DGS's VQ, the trace weight 6.2% and the 16 x 16 metric 45.9%. | exploratory: E5p, no verdict | s18 "The decomposition at j = 0" | `check_s18.py` | `kaggle/gn_e5p/attempt2/gn5p/` | one development scene; one point (j = 0); additive by construction there; not comparable with B6 (s18 says so) | no |
| B5 | OGC's lead over GN-VQ is codebook use, not regularization: with OGC's choices in GN-VQ's code (`lad_all`), GN-VQ comes within 0.0013 dB (train) and 0.0009 dB (treehill) of OGC's VQ; the large single steps fill empty entries (reseeding +0.0394 dB on train, OGC's start +0.0179 dB on treehill), and each regularization choice is 0.0097 dB or less. | exploratory: E4q, Amendment 16 b, no verdict | s17 "The ladder"; s17 settles | `check_s17.py` | `kaggle/gn_e4q/gn4q/` | two development scenes; default point; GN-VQ's code with OGC's arithmetic reproduces OGC bit for bit (`bench/gn/test_gn.py`) | no |
| B6 | Post hoc on train, the 16 x 16 metric against its trace is 65.4% of the chain from C3DGS's VQ to OGC's, and codebook choices 22.4%. | exploratory, post hoc | s17 settles ("mostly the metric") | `check_s17.py` | `kaggle/gn_e4q/gn4q/` (with E4p's rows, `kaggle/gn_e4p/gn4p/`) | post hoc; one scene; a chain through GN-VQ's code, not OGC's Lloyd (see Gaps) | no |
| B7 | OGC's VQ ridge `lam` chosen by cross-validation is OGC's code default, 1e-3, on both development scenes, on shallow minima; the paper's 1e-6 scores far worse. | exploratory: E4q | s17 "`rho_cv` and `lam_cv`"; s17 settles | `check_s17.py` | `kaggle/gn_e4q/gn4q/` | train and treehill only; supports E5's fixed `lam` 1e-3 (Amendment 17 b) | no |

## (C) Earlier findings: the gsplat PNG codec, the GN metric, GN-VQ and the cross-validated floor

These still stand after E5 (Amendment 17 a: GN-VQ's results stand as reported). **But every C row is GN-VQ or k-means
inside gsplat's `PngCompression`, against gsplat k-means comparators (`lloyd_wopa_area`, `lloyd_trace`), never against
OGC.** Inside C3DGS, OGC is ahead of GN-VQ on development scenes (D5, D6).

| ID | Claim | Status | Evidence | Check | Data | Scope limits that travel with it | Corr. |
|---|---|---|---|---|---|---|---|
| C1 | Opacity x footprint-area weighted Lloyd k-means (`lloyd_wopa_area`) for gsplat's shN codebook gains +0.111 dB mean PSNR at -0.10% mean size on all 9 MipNeRF360 scenes of `mcmc.sh`, better on every scene. | confirmatory: the rule fixed before run 4 (`run4/tilequant/run4_decision.json`, not PREREG_GN.md); both candidates pass, `pr_candidate` = `lloyd_wopa_area` | s5 "Verdict"; summary "Run 4" | **none** | `kaggle/run4/tilequant/` | k-means seed 0, PLAS seed 0; this project's checkpoints (`mcmc.sh`); gsplat `PngCompression` | no |
| C2 | The library code (upstream PR #1063) reproduces the benchmark rows exactly and passes the same rule on Tanks & Temples: +0.052 dB mean PSNR on train and truck at +0.08% mean size, at higher k-means time and peak GPU memory (3.37-3.62 GB against 1.03-1.26 GB). | confirmatory (run 4's rule, reused) and descriptive (cost) | s6; summary "Run 5" | **none** | `kaggle/run5/tilequant/` (two files regenerated locally after a label fix, s6 Sources) | two T&T scenes; one seed | no |
| C3 | A per-splat Gauss-Newton metric on shN ranks codebooks by their rendering error: 34 non-tied pairs of 36, none misordered, on both view sets and at all three codebook sizes; it is calibrated within 0.5-2x everywhere. | confirmatory: G0 (PREREG_GN.md G0, Amendments 1-4); **pass**. Calibration: descriptive, not part of the verdict | s8; s8 settles | **none** | `kaggle/gn_e0/gn/` | garden and bicycle, this project's checkpoints; three codebook sizes | no |
| C4 | E1's pre-registered rate-distortion secondary favours GN-VQ over `lloyd_wopa_area` on both scenes: bicycle BD-rate -9.84%, and on garden every GN-VQ point lies above every `lloyd_wopa_area` point in PSNR. | descriptive: Amendment 5 d secondary | s9 settles | **none** | `kaggle/gn_e1/gn1/` | seed 0, two scenes GN-VQ was designed and tuned on, three points per curve; the gate G1 failed (D1) | no |
| C5 | On 9 held-out scenes GN-VQ (eps 1e-2) beats `lloyd_wopa_area` in rate-distortion on all 9, mean BD-rate -5.37% against the pre-registered -5%. | confirmatory: G2a (Amendment 7 f, 8 a); **pass**, **after a declared deviation** (Amendment 7 b: the project continued after G1 failed) | s10; s10 settles | **none** | `kaggle/gn_e2/gn2/` | thin margin (0.37 pp); treehill's win flips under PCHIP (post hoc, 8 of 9, -5.66%); k-means seed 0, four points per curve | no |
| C6 | The per-splat matrix beats the scalar `tr(M)` weighting in GN-VQ on 8 of 9 held-out scenes (H2b, reported); post hoc, the gain is in the codebook stream (about -31% BD-rate on `shN.npz` alone). | descriptive: H2b (Amendment 7 g, reported, not gating); the stream figure post hoc | s10; s10 settles | **none** (`bench/gn/bd_sensitivity.py` recomputes the post-hoc BD) | `kaggle/gn_e2/gn2/`, `kaggle/gn_e2/bd_sensitivity.json` | treehill's computed win is a cubic-fit artifact; seed 0 | no |
| C7 | An isotropic floor on the GN metric, its strength chosen by train-view cross-validation, makes GN-VQ's codebook more faithful on test views in all 6 cells (treehill at K = 65,536: R 1.1253 to 0.6602), with the garden control held. | exploratory with a pre-stated criterion: E2b fidelity criterion and garden control (Amendments 9, 10, 10 f); **met** | s11; s11 settles | `check_s11.py` | `kaggle/gn_e2b/gn2b/` (with `kaggle/gn_e2/gn2/`) | four development scenes, seed 0, two K per curve (no BD); `rho_cv` at the top of the grid in all six cells; fails the PSNR criterion (D3) | no |
| C8 | GN-VQ with the cross-validated floor (`gn_vq_cvfloor`, frozen) beats `lloyd_trace` on all 5 of the last unused scenes, with mean BD-rate -6.17% against `lloyd_wopa_area` and BD-PSNR -0.0013 to +0.0141 dB against E2's GN-VQ (no harm by G2c's bound). | confirmatory: G2c (Amendment 11); **pass** | s12; s12 settles | `check_s12.py` | `kaggle/gn_e2c/gn2c/` | k-means seed 0, four points per curve, one session; no seed spread for the small differences from `gn_vq` (s12 post hoc); these five scenes are five of E5's seven (other checkpoints) | no |

## (D) Negative, withdrawn or retired results

| ID | Claim | Status | Evidence | Check | Data | Scope limits that travel with it | Corr. |
|---|---|---|---|---|---|---|---|
| D1 | GN-VQ failed its first gate on the size rule: it beat `lloyd_wopa_area` in test PSNR on every seed (+0.195 to +0.201 dB garden, +0.087 to +0.091 dB bicycle) but was +2.37% and +0.98% to +1.01% larger, beyond the pre-registered 0.5%. | negative: G1 (Amendment 5); **fail**, not amended (Amendment 7 a) | s9; s9 settles | **none** | `kaggle/gn_e1/gn1/` | equal K is not equal bytes: GN-VQ's codes compress worse | no |
| D2 | No tile-wise or smooth-range PNG quantization config and no shN codebook quantization change beats gsplat's library defaults; run 3's strict rule (`pr_worthy`) failed only on two garden SSIM cells inside the baseline's own seed spread. | negative, descriptive | s1, s2, s4; summary "Runs 1-2", "Run 3" | **none** | `kaggle/run2/tilequant/`, `kaggle/run3/tilequant/` | garden and bicycle; this project's checkpoints | no |
| D3 | The floor does not work by E2b's PSNR criterion: on treehill it closed most of the test-PSNR gap to `lloyd_trace` but not all of it (-0.0016 and -0.0120 dB). | negative: Amendment 9's PSNR criterion; **not met** | s11 settles | `check_s11.py` | `kaggle/gn_e2b/gn2b/` | as C7 | no |
| D4 | GN-VQ injected into C3DGS read +0.127 dB at +1.485% bytes against the probe run, but this cannot be read as GN-VQ's effect: seeded C3DGS runs are not reproduced on the GPU. | negative (inconclusive), exploratory: E3r | s15 summary and settles | `check_s15.py` | `kaggle/gn_e3r/gn3r/` | train; geometry and C3DGS's codebook differ between the runs | no |
| D5 | E4 was withdrawn before any gate-scene data: on train D1 (`rho_cv` against rho = 0) was +0.0007 dB (`SE_noise` 0.0020), and D2 (GN-VQ against OGC, -0.0403 dB in all three processes) compared unequal effective codebooks (OGC 4,096 used, GN-VQ 2,310-2,592) at unequal bytes (+275,459 for OGC). | negative: Amendment 16 a (withdrawal) | s16; s16 settles; PREREG Amendment 16 a | `check_s16.py` | `kaggle/gn_e4p/gn4p/` | one development scene; E4 has no verdict | no |
| D6 | GN-VQ was retired from any gate: at equal bytes on train OGC needs 14.42% fewer bytes than GN-VQ (BD-PSNR +0.0228 dB), with every sign holding under the post-hoc checks; GN-VQ's added floor is worth 0.0007 dB. | negative: Amendment 17 a (retirement), on exploratory evidence | s17 settles; PREREG Amendment 17 a | `check_s17.py`, `check_s16.py` | `kaggle/gn_e4q/gn4q/`, `kaggle/gn_e4p/gn4p/` | train only for the rate comparison (treehill had no sweep); development scenes | no |

## (E) Method and infrastructure facts

| ID | Claim | Status | Evidence | Check | Data | Scope limits that travel with it | Corr. |
|---|---|---|---|---|---|---|---|
| E1 | C3DGS's own evaluation fails when the training images are kept on the CPU (`compress.py:105` at `2a234af5`); a wrapper that moves the ground truth to the render's device, and only then, lets it run (E5p: 798 calls moved in the CPU process, 0 in the GPU processes). | descriptive | s17 "C3DGS's own evaluation fails"; s18 "C3DGS's evaluation with the images on the CPU" | `check_s17.py`, `check_s18.py` | `kaggle/gn_e4q/gn4q/`, `kaggle/gn_e5p/attempt2/gn5p/` | C3DGS `2a234af5`; recorded as a deviation (Amendment 17 e.2) | no |
| E2 | Seeded C3DGS runs are not reproduced on the GPU: between seeded runs the geometry and C3DGS's own colour codebook differ. | descriptive: E3r | s15 summary and settles | `check_s15.py` | `kaggle/gn_e3r/gn3r/` | train; T4 | no |
| E3 | The C3DGS host reproduces C3DGS's published train row within 0.020 dB in PSNR and 0.134% in size; under protocol ii the compressed models read 0.410-0.523 dB lower than in C3DGS's own evaluation, for reasons the files do not show. | descriptive: E3q | s14 settles | `check_s14.py` | `kaggle/gn_e3q/attempt2/gn3q/` | train only; sizes are MiB under C3DGS's "MB" label (s14); the protocol gap is unexplained | no |
| E4 | OGC's released code reproduces its paper's Table 19 train row to two decimals, and this project's 16-probe metric has 0.919 of the exact Gram's total trace. | descriptive: E4p | s16 summary and settles | `check_s16.py` | `kaggle/gn_e4p/gn4p/` | train; the trace shortfall's cause is not settled (s16) | no |
| E5 | C3DGS's fine-tuning moves the two VQs' bytes apart: on train it shrinks OGC's `features_rest` by 32.5% and grows C3DGS's by 2.4% / 2.3%. The mechanism (OGC's AC codebook outside C3DGS's int8 grid, re-fitted during fine-tuning) is inferred, not recorded. | descriptive, post hoc (mechanism inferred) | s18 "Fine-tuning" | `check_s18.py` | `kaggle/gn_e5p/attempt2/gn5p/` (E4p's rows agree, `kaggle/gn_e4p/gn4p/`) | train; the record that would show it (Amendment 18 note 2) exists in E5's bundles but is not analysed in FINDINGS (Gaps) | no |
| E6 | GN-VQ's codebook values beyond C3DGS's int8 AC grid are clamped at the save, and GN-VQ's final assignment is made against the clamped table; FINDINGS calls this the frozen method's design, not a bug. | descriptive | s16 table-range bullets | `check_s16.py` | `kaggle/gn_e4p/gn4p/` | one scene | no |
| E7 | C3DGS's run failed in a cuSOLVER batched eigendecomposition; with `eigh` and `det` chunked (Amendment 13 g) both runs completed, with no refusal at 8,192. | descriptive: E3q attempt 1 and attempt 2 (Amendment 13 g) | s14 summary | `check_s14.py` | `kaggle/gn_e3q/attempt1/gn3q/`, `kaggle/gn_e3q/attempt2/gn3q/` | T4, torch 2.10; E0's first crash (s8, no check) is the same failure class | no |
| E8 | A per-scene host-memory model checked against measurement under-predicted twice in E5: the C3DGS step peaked 0.87 GB (bonsai) and 0.20 GB (kitchen) above Note 3's figures, the other five below; Note 1 had left out five job steps (npz2ply up to 16.39 GB on treehill), which Note 3 corrected before E5. | descriptive | s19 "How the sessions ran" table; PREREG Amendment 17 Notes 1 and 3 | `check_s19.py`; the Note 3 table by `test_e5_feasibility_reproduces_amendment_17_note_3` | `kaggle/gn_e5/S*/gn5/`; `kaggle/gn_e5p/attempt2/gn5p/`, `kaggle/gn_e4q/gn4q/` | the model's margin is its own calibration's largest gap; no consequence, since no two scenes ran at once | no |
| E9 | E4p's reserved-memory peaks were recorded per row, not per process (a recording bug), so E4p has no process-level reserved peak. | descriptive | s16 "A recording bug" | `check_s16.py` | `kaggle/gn_e4p/gn4p/` | fixed for E4q on (Amendment 16 c) | no |
| E10 | The full pipeline runs at INRIA scale on one T4 (bicycle, 6,131,954 splats, K = 65,536) with no step out of memory, at 11.51 GB allocated and 15.18 of 15.64 GB reserved; the colour threshold, not K, is C3DGS's rate knob (2.49x against 1.02x in bytes on train). | descriptive: E3p, E3r | s13 settles; s15 settles | `check_s13.py`, `check_s15.py` | `kaggle/gn_e3p/gn3p/`, `kaggle/gn_e3r/gn3r/` | small headroom; train for the rate knob | no |
| E11 | E5's three sessions took 14.59 h of setup and jobs against an estimate of 16.0-24.9 h; every scene's job but truck's came in below the estimate's low end. | descriptive | s19 "How the sessions ran" | `check_s19.py` | `kaggle/gn_e5/S*/gn5/` | S1 ran its scenes in sequence (RAM 8,192 bytes below Note 1's figure) | no |
| E12 | OGC's code was used for noncommercial research under its PolyForm Noncommercial 1.0.0 licence. The author confirmed by email that it is available for research purposes, and states that the method is patented for commercial use (paraphrased; never quoted). No OGC file is in the repository or any bundle. | provenance, not a finding | PREREG Amendment 18 c; `kaggle/RELATED_WORK_OGC.md` §1 | **none** (the bundle guard refuses OGC's files; `check_s19.py` does not test this) | - | the private dataset copy stays private | **yes** |

## Gaps

**Wanted for a paper, with no backing in FINDINGS:**
1. **Pre-registered E5 secondaries missing from section 19.** Amendment 17 d lists codewords used, index entropy and
   per-array bytes per row, and note ii's fidelity per angle (mean PSNR and pooled-MSE PSNR). The combiner computes
   them (`per_row_measures`, `fidelity_per_angle_j0`), but FINDINGS does not report them.
2. **Bytes after fine-tuning in E5.** Section 19 compares PSNR after fine-tuning but not bytes. E5p on train showed
   them moving apart (E5), so A5 is not known to be at equal rate.
3. **Amendment 18 note 2's quantizer records** (`qa_at_save`, `codebook_int8`) are in all 28 of E5's fine-tuned rows
   and are not analysed, so the mechanism in E5 stays inferred.
4. **A like-for-like comparison with the paper.** Its +0.49 / +0.09 dB are 9 Mip-NeRF 360 scenes, its evaluation, one
   run each. E5 shares four scenes, uses protocol ii, and runs two processes at j = 0. No per-scene comparison with
   the paper's Table 12 on the four shared scenes is in FINDINGS (only a tabulation of their numbers in
   `RELATED_WORK_OGC.md`).
5. **The protocol gap.** E3q found protocol ii 0.410-0.523 dB below C3DGS's own evaluation, unexplained. E5's rows
   record C3DGS's evaluation (`c3dgs_PSNR`), but section 19 reports protocol ii only.
6. **SSIM and LPIPS for E5's rows** are recorded, not reported.
7. **A rate-matched comparison** (the paper's 512-entry codebook row) is not in FINDINGS.
8. **BD-level noise.** The j = -1 and +1 points have one seed each; section 19 states this as a limit.
9. **Cost of OGC's VQ against C3DGS's on the gate scenes:** section 19 gives job times only; per-row VQ times exist for
   train only (s18).
10. **GN-VQ against OGC on any gate scene:** none (E4 withdrawn). GN-VQ's held-out wins (C5, C8) are against gsplat
    k-means only.
11. **Committed checkers for FINDINGS sections 1-10** (C1-C6, D1, D2, E7's E0 part): none.
12. **The paper's open questions** (`RELATED_WORK_OGC.md`, "Open questions"): which VQ ridge produced their tables
    (10^-6 in the text, 1e-3 in the code), which C3DGS commit, and their run-to-run spread. None is answered, and a
    discussion of any discrepancy with the paper would need them.

**Claims in FINDINGS that conflict with each other or with E5, or must not be quoted together without their measure:**
1. **The Gram metric's share differs with the measure:**
   - 65.4% of the chain (E4q post hoc, train, through GN-VQ's code; B6);
   - 45.9% of the j = 0 difference (E5p, train, OGC's Lloyd; B4);
   - 52.7% of P1 as a ratio of mean BD-PSNRs (E5, seven scenes, exploratory, non-additive; B2).

   Section 18 already says the first two are not comparable. A paper must not quote one as "the" share.
2. **The share of OGC's Lloyd with the identity metric:** 47.9% of the j = 0 difference on train (B4), against a mean
   BD-PSNR of 0.1185 dB of P1's 0.3558 dB over E5's scenes (B2). These are different measures on different scenes.
3. **The lead after fine-tuning:** E5p's two train processes read +0.0211 and +0.2592 dB (seed and image device
   confounded, `SE_noise` 0.1190), against E5's mean +0.0604 dB over seven scenes. Different scenes and designs; E5p's
   value must not be quoted as a replication of the paper's +0.09.
4. **Gate-scene reuse:** E2c's closing remark (s12) says any later held-out claim needs "new scenes or checkpoints".
   E5 reuses five of E2c's scenes with INRIA's checkpoints. That is consistent with "or checkpoints" but must travel
   as a disclosure (Amendment 17 d).
5. **GN-VQ's standing:**
   - positive held-out gates in gsplat's PNG codec against k-means (C5, C8);
   - behind OGC inside C3DGS on development scenes (D5, D6).

   Not a contradiction (different host and comparator), but GN-VQ must not be presented as competitive with OGC.
6. **E2 and E2c rest on a declared deviation:** G1 failed, and the project continued after the results were known
   (Amendment 7 b). Any claim from C5-C8 carries it.

## Candidate headlines

No pick.

1. **"A pre-registered, independent replication of OGC inside C3DGS: on seven gate scenes OGC's Gram-metric VQ beats
   C3DGS's own colour VQ at equal rate, and the Gram metric itself beats its trace."**
   Rows: A1, A2, B1, A3, A6, A7, B3; limits from the common E5 block, E1, E2 and E3.
2. **"OGC's gain is real before fine-tuning and mostly gone after it":** +0.5239 dB before against +0.0604 dB after,
   bracketing the paper's +0.49 / +0.09 dB.
   Rows: A4, A5, E5; it needs Gaps 2-4 closed or stated as limits.
3. **"What makes a matrix-weighted VQ work: codebook use and the metric."** A dissection across C3DGS's VQ, OGC's
   Lloyd under three metrics and GN-VQ.
   Rows: B1, B2, B3, B4, B5, B6, B7, D5, D6, A6; with Gaps conflicts 1-2 stated.
