# DRAFT E9 (R27, 2026-11-24) — a held-out test of P1 on DL3DV benchmark scenes drawn by a pre-registered rule

**Status: draft for Dace. Not in PREREG; nothing run; no data fetched.** Supersedes `D4_MEMORY_PROBE_AND_E9_INRIA_DRAFT.md`
sections 2-5 and `E9_FEASIBILITY.md` section 4 as the E9 design (both stay as records). Power from
`kaggle/drafts/e9_dl3dv_power_draft.py`; every effect and SD it uses comes from **selected scenes** (FINDINGS sections 30-31):
the gate scenes were kept on GN-VQ outcomes and the development scenes chosen on them, so they are planning numbers, not
estimates for DL3DV.

## 1. Scenes: a seeded, stratified draw from DL3DV's benchmark list, fixed before any training

- **The list:** the scene directories of the Hugging Face dataset `DL3DV/DL3DV-Benchmark` at a pinned revision. Read on
  2026-11-24 through the public API (no access request): revision `9684e8382278c5e18173c1e72bd246daf2874539` (last modified
  2025-09-12) lists **141** 64-hex scene directories (the paper and README say 140; the extra one is to be explained before
  registration) and a top-level `benchmark-meta.csv` (25,573 bytes).
- **The labels:** DL3DV labels every benchmark scene. The paper (arXiv 2312.16256v2, p. 5): DL3DV-140 "are categorized as
  indoor (bounded) scenes vs. outdoor (unbounded) scenes", and the environment annotation "distinguishes between indoor and
  outdoor settings"; the README: "All videos are annotated by scene environment (indoor vs. outdoor)". **Where the
  per-scene label is stored is not verified:** `benchmark-meta.csv` is the likely file, but the repository is gated and
  reading it means requesting access, which accepts DL3DV's terms (section 6): that is Dace's step, not mine. The public
  `cache/DL3DV-valid.csv` on GitHub (10,320 rows: hash, a sensitivity label, batch, duration) has no environment column.
- **If the labels cannot be read, a labelling rule that uses no rendering:** DL3DV itself equates indoor with bounded and
  outdoor with unbounded, so label from the released camera poses and COLMAP sparse points alone: a scene is **outdoor**
  if more than 10% of its sparse points lie farther from the camera path's centroid than 3 x the path's radius (the
  largest camera distance from that centroid), else indoor. Thresholds fixed in the registration; checked, before any
  draw, on the ten scenes with OGC rows (their sparse points only, no P value read), where it must reproduce section 31's
  labels on all ten or the rule is not used.
- **The draw:** sort the scene hashes; split them by label; draw n/2 from each with `numpy.random.default_rng(seed)`,
  `seed` = the integer of the first 16 hex digits of sha256 of the registration commit's tree id (so the seed is fixed by
  the registration and nobody can choose it); replace a drawn scene only by the registered fit rule (T4 memory, section 5)
  and then by the next draw in the same stratum.
- **Nothing on a drawn scene is read before the registration** except its label, frame count and image size.

## 2. Hypotheses

- **Primary: P1 > 0** (E5's P1, OGC's Gram VQ against C3DGS's own at equal rate before fine-tuning), one-sided paired t
  over the n drawn scenes at 0.05 (each scene's BD is already a within-scene paired difference); E5's count and criterion 3
  beside it, as in E5.
- **Declared secondary: indoor P1 > outdoor P1**, one-sided at 0.05, by **Welch's t** (recommended) with the exact
  rank-sum reported beside it. Welch because the strata are small and their SDs differ (indoor 0.0809, outdoor 0.0555 on
  the ten scenes), and because the exact rank-sum is coarse here: at 3+3 its smallest p is exactly 0.0500 (only "every
  indoor above every outdoor" rejects), at 4+4 its exact level is 0.0286; it is the check against non-normality, not the
  test.
- **P_FT: estimation only,** a mean with a 95% CI (t, n - 1 df), no test.
- **INRIA's bicycle, garden and stump: secondary, descriptive,** flagged as GN-VQ development scenes (Amendments 7, 10;
  FINDINGS section 30), not pooled with the draw.
- **The bridging control:** one gate scene (counter, the smallest INRIA model, 1,222,956 splats) trained by us with the
  E9 recipe, its P1 compared descriptively with its INRIA-checkpoint value (+0.3192 dB, section 29). It says whether our
  checkpoints change P1 by more than E5's seed noise; it is not pooled and not a test.

## 3. Power (`e9_dl3dv_power_draft.py`; registered-fit values)

Primary, P1 > 0, one-sided paired t at 0.05:

| n | (a) development-like, 0.0929 / SD 0.0588 | (a') development mean, ten-scene SD 0.1616 | (b) the four outdoor scenes, 0.1068 / 0.0555 | (c) half indoor, half outdoor (mean 0.2486) |
|---|---|---|---|---|
| 6 | 0.950 | 0.335 | 0.990 | 1.000 |
| 8 | 0.991 | 0.430 | 0.999 | 1.000 |
| 10 | 0.998 | 0.514 | 1.000 | 1.000 |

- (c) draws each stratum from its own ten-scene mean and SD (indoor 0.3904 / 0.0809, outdoor 0.1068 / 0.0555; Monte Carlo,
  200,000 draws, normal within stratum).
- **Read:** the design is powered unless DL3DV's scenes look like the development scenes with the whole ten scenes' spread
  (a'): 0.34-0.51. That case is the selection's worst case (section 30), and n = 10 does not cover it.

Declared secondary, indoor P1 > outdoor P1, one-sided at 0.05 (ten scenes' strata; "half the gap" moves the indoor mean
halfway to the outdoor one):

| indoor + outdoor | Welch t | rank-sum: exact level / smallest p | rank-sum | Welch, half the gap | rank-sum, half the gap |
|---|---|---|---|---|---|
| 3 + 3 | 0.972 | 0.0500 / 0.0500 | 0.985 | 0.588 | 0.649 |
| 4 + 4 | 0.999 | 0.0286 / 0.0143 | 0.995 | 0.782 | 0.681 |
| 5 + 5 | 1.000 | 0.0476 / 0.0040 | 1.000 | 0.888 | 0.870 |

- The ten-scene gap (0.28 dB) is itself confounded with the selection, the datasets, the data factor and the camera
  (section 31), so the "half the gap" columns are the planning case: 5 + 5 for about 0.89.

P_FT, 95% CI half-width at the ten scenes' SD (0.0394 dB): **+-0.041 dB at n = 6, +-0.033 at 8, +-0.028 at 10**, so even
n = 10 cannot tell the gate's 0.0425 dB from 0.

## 4. The GN-VQ ratio, or a training-view version, as a pre-registered predictor of per-scene P1

**What R29 found (FINDINGS section 32, post hoc, declared at `98d1aba8` before computing):** on the eight scenes with
both, GN-VQ's E2 test/train gain ratio and P1 rank almost alike (Spearman +0.9524, exact p 0.0011; +0.9286, p 0.0067
without treehill); P2 similarly (+0.9286); P_FT not (+0.5238, p 0.1966). **But the ratio chose the development scenes**
(section 30), so these eight scenes cannot tell a ratio that predicts P1 from a gate/development difference; no within-set
correlation was declared.

**Could it serve as a pre-registered predictor of per-scene P1 on the DL3DV draw?** Yes as a declared secondary, not as
the primary and never to choose scenes:

- **E2's ratio as defined** needs, per scene, this project's gsplat MCMC checkpoint (a second model, not the 3DGS
  checkpoint E9 tests), the GN pass, GN-VQ and `lloyd_trace` at K = 65,536 and PSNR on the test views. It reads the test
  views of a different model, so it must be computed and committed before any E9 P1 value. **Cost (estimate):** gsplat
  training 1,932-3,398 s per scene (run 4's measured range on Mip-NeRF 360) plus the E2 job's GN pass and two
  codebooks, about 1-2 session-hours per scene; for n = 8, about 8-16 more session-hours.
- **A training-view version (recommended):** split each scene's training views into halves (even and odd, as E4q's
  cross-validation), accumulate the 16 x 16 metric on the even half, quantize with OGC's `gram_kmeans` and with its
  trace metric (`ogc_scalar`) on the E9 checkpoint itself, and take the ratio of the Gram-over-trace gain in PSNR (or in
  clamped dMSE, Amendment 10 a) on the odd half to that on the even half. **It reads no test view**, so it can be computed
  before the P1 processes without touching the held-out evaluation, and it uses E9's own model and OGC's own VQ.
  **Cost (estimate):** one GN pass over the training views plus four `gram_kmeans` calls and two half-view renders per
  scene; E4q's calls took 24.7-108.8 s each (train, treehill), so about 0.3-1 session-hour per scene, 3-8 h for n = 8.
- **Its validity is not known:** it has never been computed on any scene here (E4q's odd-view tables score `rho` and
  `lam`, not this ratio). Checking it first on the ten scenes with OGC rows would cost about 3-10 session-hours and would
  validate on selected scenes, so the registration should state it as untested.
- **How to declare it:** "Spearman's rho of the training-view ratio with P1 over the drawn scenes, one-sided (positive),
  exact permutation p", computed from values committed before any P1 value exists. At n = 8 the smallest exact one-sided p
  is 1/40,320; the test has power only for a strong monotone relation, as R29's +0.95 would be.

## 5. Cost (labelled; nothing measured on DL3DV)

- **Training (estimate, not measured for this code):** INRIA's `gaussian-splatting` at a pinned release-period commit,
  30k iterations, 1-2 h per scene on a T4 (E9_FEASIBILITY section 3: this project's gsplat MCMC runs took 1,932-3,398 s
  per Mip-NeRF 360 scene; INRIA's densification is slower), two scenes at a time on 2x T4.
- **Confirmation:** E6 part (b)'s four processes per scene (P1 and P_FT, the replicate pair), 2.9-4.1 session-hours per
  scene at the gate scenes' size (20-29 h for seven, `e8_estimate_draft.py` calibrated / high); DL3DV's 960P frames and
  smaller models would cost less, which the draw settles.
- **T4 memory from the fitted model** (E9_FEASIBILITY section 2: allocated = 2.00 + 1.12 x image GB + 0.70 x M splats;
  reserved = 1.80 + 1.40 x image GB + 0.94 x M splats; largest residual 0.14 / 0.54 GB): assuming 200-400 training views
  at 960 x 540 (1.24-2.49 GB as float32 on the GPU) and 1-4 M splats, **4.1-7.6 GB allocated, 4.5-9.0 GB reserved**,
  under the T4's 15.64 GB. Both the view count and the splat count are assumptions until the draw.

| n (+ bridge) | Scenes trained | Training, session-h (2 at a time) | Confirmation, session-h | Total, session-h | Sessions (<= 12 h) | Weeks at 30 h |
|---|---|---|---|---|---|---|
| 6 + 1 | 7 | 3.5-7 | 20.3-28.7 | 24-36 | 3-4 | 0.8-1.2 |
| 8 + 1 | 9 | 4.5-9 | 26.1-36.9 | 31-46 | 4-5 | 1.0-1.5 |
| 10 + 1 | 11 | 5.5-11 | 31.9-45.1 | 37-56 | 4-5 | 1.2-1.9 |

Plus one wheel session if the cached cu130 wheel does not match the image, and bicycle, garden and stump's processes
(3 x 2.9-4.1 h) if the INRIA secondary is run.

## 6. DL3DV's licence and terms (read in full on 2026-11-24: `License.md` and `DL3DV-10K_term_of_use.pdf` at github.com/DL3DV-10K/Dataset, and the README)

Every constraint they state:
1. **Licence:** "DL3DV-10K Term of use and Creative Commons Attribution-NonCommercial 4.0 International License".
2. **Use:** "only for non-commercial research and educational purposes" (terms 1; CC BY-NC's NonCommercial).
3. **No warranty** of non-infringement or fitness (terms 2).
4. **Indemnity:** the researcher "accepts full responsibility" and "shall defend and indemnify" the DL3DV organization,
   including for "copies of copyrighted 3D models" created from the dataset (terms 3).
5. **Sharing:** access may be given to associates and colleagues only after they have also agreed to the terms;
   "Sharing the data otherwise is strictly prohibited" (terms 4). This is stricter than CC BY-NC's sharing permission.
6. **GDPR:** the researcher must be able to delete all person-specific data on request (terms 5); and (disclaimer 2) must
   delete or obscure data that may violate privacy or ethical standards and keep monitoring the guidelines.
7. **Termination:** access can be terminated at any time (terms 6).
8. **Employer:** a for-profit employer is also bound (terms 7).
9. **Law:** Indiana (terms 8).
10. **Changes:** the organization may modify the licence and terms at any time; "Continued use of this dataset means
    you're accepting these modifications" (disclaimer 1).
11. **Access:** requesting access on Hugging Face "automatically" signs the terms and licence (README); the repository is
    gated ("auto").
12. **CC BY-NC 4.0, if material is shared:** attribution, a licence notice, and an indication of modifications; Adapted
    Material only non-commercially.

What this means for E9 (my reading, for Dace to confirm):
- **Use:** research use is allowed; this project is noncommercial (it already runs OGC under PolyForm Noncommercial).
- **Checkpoints:** trained 3DGS checkpoints are plausibly "Adapted Material" or "the data"; under terms 4 they should
  **not** be published or put in a public Kaggle dataset. Keep them in private Kaggle datasets and bundles; commit only
  derived numbers (rows, BD values), as for INRIA's scenes.
- **Bundles:** E9's bundles must hold no DL3DV image or frame; renders saved for checks stay out of the repository.
- **Publication:** nothing restricts publishing results; cite DL3DV (Ling et al., CVPR 2024) and state the licence.
- **Kaggle:** a private dataset holding DL3DV frames is sharing with Kaggle as a host, not with a person; whether that is
  "sharing the data" under terms 4 is not settled by the text. Ask DL3DV, or fetch from Hugging Face inside each session.

## 7. What Dace decides

1. Request access to `DL3DV/DL3DV-Benchmark` (which accepts the terms), so that `benchmark-meta.csv`'s labels can be read
   and the 141-vs-140 count explained; or choose the no-rendering labelling rule.
2. n (6, 8 or 10) and whether the INRIA secondary runs.
3. Whether to ask DL3DV about private Kaggle datasets and checkpoints.
