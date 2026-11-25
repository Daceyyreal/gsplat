# DRAFT E9 v3 (R27 2026-11-24, R36 2026-11-25, R38 2026-11-25) — a held-out test on DL3DV benchmark scenes drawn by a pre-registered rule: does a model-unseen validation ratio predict OGC's lead, and does the lead hold

**Status: draft for Dace. Not in PREREG; nothing run; no data fetched.** v3 (R38): **the predictor is now a validation
ratio from E9's own training** (section 1a): a fixed, interleaved block of training views (no seed: it mirrors the test split's rule) is held out of model
training, and the transfer ratio is measured on it, model-unseen and metric-unseen, before any test-view row; R33 stays
the first look on INRIA's ten scenes and is not run on DL3DV; the bridge scene is trained with and without the block;
the cost is recomputed (`e9_v3_cost_draft.py`). v2 (R36): the primary became a training-views-only ratio against P1
(Spearman), with P1 > 0 as a co-primary under Holm; the revision is pinned and the 141-vs-140 rule fixed. Supersedes `D4_MEMORY_PROBE_AND_E9_INRIA_DRAFT.md`
sections 2-5 and `E9_FEASIBILITY.md` section 4 as the E9 design (both stay as records). Power from
`kaggle/drafts/e9_dl3dv_power_draft.py`; every effect and SD it uses comes from **selected scenes** (FINDINGS sections 30-31):
the gate scenes were kept on GN-VQ outcomes and the development scenes chosen on them, so they are planning numbers, not
estimates for DL3DV.

## 1. Scenes: a seeded, stratified draw from DL3DV's benchmark list, fixed before any training

- **The list:** the scene directories of the Hugging Face dataset `DL3DV/DL3DV-Benchmark` at a pinned revision. Read on
  2026-11-24 through the public API (no access request): **pinned revision `9684e8382278c5e18173c1e72bd246daf2874539`**
  (last modified 2025-09-12) lists **141** 64-hex scene directories (the paper and README say 140) and a top-level
  `benchmark-meta.csv` (25,573 bytes). The draw reads the list and the labels at this revision only.
- **141 against 140, resolved by rule:** a directory is eligible only if its hash has exactly one row in
  `benchmark-meta.csv` at the pinned revision with an indoor or outdoor label; a directory without a label row, or with
  more than one, is excluded and listed in the registration with the reason. If all 141 have one labelled row, all 141 are
  eligible and the registration says the paper's 140 is out of date. This is decided before the seed is known (the seed
  comes from the registration commit, below), so the rule cannot be tuned to the draw.
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
- **Order on each drawn scene (v3):** train its checkpoint on the training views minus the validation block (section
  1a); compute the validation ratio and commit its bundle; only then run any OGC or C3DGS row on its test views. So the
  ratio exists, committed, before any value it is meant to predict.

## 1a. The validation block and the validation ratio (v3, R38)

- **Why:** R33's held-out views are held out from the metric, not from the model (INRIA's checkpoints saw every
  training view), so R33 cannot see how the model does on views it never saw, which plausibly drives much of E2's
  test/train ratio (R33 declaration draft v2, R37c). A block held out of E9's own training gives views that are unseen
  by both, like the test views, without reading a test view.
- **The block (fixed in the registration):** with the views sorted by image name and indexed i = 0, 1, ... over all of
  the scene's images, the test views are i % 8 == 0 (INRIA's `--eval` split, unchanged); the **validation block is
  i % 16 == 4**, 1/16 of the images (1/14 of the training views); the model trains on the rest (13/16 of the images,
  92.9% of the training views it would otherwise see). **Why interleaved, and why there:** DL3DV's frames are a video
  in time order, so i % 16 == 4 puts each validation view halfway between two test views, with its two neighbouring
  frames (i - 1, i + 1) in training, exactly as each test view has; a contiguous block would test extrapolation along
  the path, which the test views do not. **Why 1/16, not 1/8:** 1/8 (i % 8 == 4) would remove twice the training views
  (14.3%) and shift P1 more, for a ratio whose noise is dominated by between-view variation already at 1/16. **Floor:**
  if i % 16 == 4 gives fewer than 12 views (a scene with fewer than 181 images), the block is i % 8 == 4 for that
  scene, decided from the frame count, which may be read before the registration (section 1).
- **Implementation (to build):** a wrapper on INRIA's `gaussian-splatting` reader that moves the block's camera infos
  out of `train_cameras` before training (as R33's `drop_test_infos` does for test infos), recorded in the run's
  `cfg_args`-side metadata; the test split is untouched.
- **The validation ratio (the E9 predictor):** on E9's model, R33's computation with one fixed split in place of the
  folds: in-set = every training view the model trained on, held-out = the validation block. C3DGS's sensitivity, the
  GN metric and both colour VQs (C3DGS's and OGC's Gram, j = 0, K = 4,096, seed 0) on the in-set views; both codebooks
  through C3DGS's int8 quantizer, colours substituted; PSNR on every in-set and validation view;
  **v_s = gain_val / gain_in** (the same gain, PSNR(ogc_gram) - PSNR(c3dgs), averaged over the views), defined if
  gain_in > 0. Beside it: the ratio in the MSE domain (the mean reduction in MSE on validation views over that on
  in-set views), which removes the dB arithmetic by which a larger model error shrinks a gain (R37c); and Gram
  stability D between the in-set and validation metrics, per-view normalised as R37b, as a secondary.
- **Its precision and the usefulness rule:** SE of v_s by the delta method from the per-view gains (the validation
  views are the noisier term); E9's analysis applies R37a's rule (uninformative if 1.4826 x MAD of the v_s < 2 x their
  median SE, or if fewer than n - 2 scenes have a ratio) before Primary 1 is computed, and if v_s is uninformative,
  Primary 1 is reported as not tested and Holm's family is Primary 2 alone.
- **Bridge (v3):** counter, trained with the E9 recipe **twice**: with the validation block (its v_s computed, and its
  P1) and without it (P1 only). Reported: the P1 shift (with block minus without), against E5's seed noise, and each
  P1 against counter's INRIA-checkpoint value (+0.3192 dB, section 29). Descriptive, not pooled, not a test. A shift
  larger than the seed noise means E9's P1 is measured on models weaker than the standard protocol's, which the
  registration must then state beside Primary 2.

## 2. Hypotheses

- **Primary 1 (v3): Spearman's rho between the validation ratio v_s (section 1a) and P1 across the drawn scenes,
  one-sided (rho > 0),** exact permutation p (n! orderings, enumerated up to n = 10; for n = 12, 12! = 4.8e8 orderings, by
  2,000,000 seeded random orderings, stated as Monte Carlo). R29 (FINDINGS section 32) is the reason to ask: E2's ratio
  ranked almost as P1 on eight selected scenes, post hoc, with set membership not separated; the drawn scenes are unseen.
  R33 is not computed on the drawn scenes; its INRIA result informs the registration (section 3) but is not a predictor
  here.
- **Primary 2 (co-primary): P1 > 0** (E5's P1), one-sided paired t over the n drawn scenes; E5's count and criterion 3
  beside it.
- **Multiplicity: Holm over the two primaries at family level 0.05** (the smaller p against 0.025, then the other
  against 0.05). **Recommended over making P1 > 0 a secondary:** P1 > 0 held on 10 of 10 scenes so far, so its p is
  expected to be the smaller one, in which case Holm tests the Spearman correlation at the full 0.05 and the co-primary
  costs it almost nothing; and keeping P1 > 0 primary makes the replication on unseen scenes a confirmatory claim rather
  than a description. If P1 > 0 is not the smaller p, the Spearman test is run at 0.025: the price is in the table
  below.
- **Declared secondary: indoor P1 > outdoor P1**, one-sided at 0.05, by **Welch's t** (recommended) with the exact
  rank-sum reported beside it. Welch because the strata are small and their SDs differ (indoor 0.0809, outdoor 0.0555 on
  the ten scenes), and because the exact rank-sum is coarse here: at 3+3 its smallest p is exactly 0.0500 (only "every
  indoor above every outdoor" rejects), at 4+4 its exact level is 0.0286; it is the check against non-normality, not the
  test.
- **P_FT: estimation only,** a mean with a 95% CI (t, n - 1 df), no test.
- **INRIA's bicycle, garden and stump: secondary, descriptive,** flagged as GN-VQ development scenes (Amendments 7, 10;
  FINDINGS section 30), not pooled with the draw.
- **The bridging control (v3: two models, section 1a):** one gate scene (counter, the smallest INRIA model, 1,222,956
  splats) trained by us with the E9 recipe with and without the validation block; the P1 shift between the two, and each
  P1 against its INRIA-checkpoint value (+0.3192 dB, section 29). It says whether our checkpoints, and the block, change
  P1 by more than E5's seed noise; it is not pooled and not a test.

## 3. Power (`e9_dl3dv_power_draft.py`, `e9_spearman_power_draft.py`; registered-fit values)

Primary 1, Spearman rho > 0, one-sided at 0.05 (`e9_spearman_power_draft.py`, seeded `default_rng(20261125)`): each
scene's (ratio, P1) bivariate normal with Pearson r = 2 sin(pi rho_s / 6), the normal model's link to Spearman's rho_s;
the exact permutation critical value (n = 8: rho >= 0.6429, level 0.0481; n = 10: 0.5636, 0.0481, both enumerated;
n = 12: 0.5035, 0.0498, from 2,000,000 random orderings); 50,000 simulated samples per cell:

| n | rho_s = 0.5 | rho_s = 0.7 | rho_s = 0.9 |
|---|---|---|---|
| 8 | 0.325 | 0.591 | 0.923 |
| 10 | 0.411 | 0.721 | 0.976 |
| 12 | 0.492 | 0.814 | 0.993 |

- **Read:** R29's +0.95 was on selected scenes with set membership mixed in; a planning value of 0.7 needs n = 12 for
  about 0.8, and 0.5 is under-powered at every n here. **v3: rho_s = 0.7 stays the sizing case.** R33's rho on INRIA's
  ten scenes, whatever it is, will be optimistic as a planning value for v_s: the scenes are selected (section 30) and
  were chosen after R29 (not blind), and R33 measures a different quantity (metric transfer at training-level fit; v_s
  adds the model's fit to unseen views). It can lower the planning value (if R33 is flat or uninformative) but should
  not raise it. If the Spearman p is the smaller of the two, Holm tests it at
  0.025, where the critical values rise and these powers fall.

Primary 2, P1 > 0, one-sided paired t, at 0.05 and at Holm's first step (0.025):

| n | (a) 0.0929 / SD 0.0588 | (a') 0.0929 / SD 0.1616 | (b) 0.1068 / SD 0.0555 |
|---|---|---|---|
| 8 | 0.991 / 0.967 | 0.430 / 0.290 | 0.999 / 0.996 |
| 10 | 0.998 / 0.993 | 0.514 / 0.369 | 1.000 / 1.000 |
| 12 | 1.000 / 0.999 | 0.588 / 0.443 | 1.000 / 1.000 |

The tables below are v1's (n = 6, 8, 10), kept for the secondary and the mixture case.

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

## 4. The GN-VQ ratio, or a training-view version, as a pre-registered predictor of per-scene P1 (v2: R33's version became Primary 1; v3: the validation ratio, section 1a, replaces it)

v3 note: this section is v1's and v2's reasoning, kept as the record. Its "training-view version" (and R33) holds views
out of the metric only; v3's predictor holds them out of the model as well (section 1a).

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

**v3 (R38), `kaggle/drafts/e9_v3_cost_draft.py`:** n drawn scenes plus the bridge's two counter models; the
validation ratio on the n + 1 models with a block (one R33 fold with every training view in-set, plus scene load:
0.2-0.4 session-h); the confirmation on all n + 2 models (the bridge's second model at full cost, an upper bound: its
P1 alone would take about half); sessions = ceil(training / 12) + ceil(ratio / 12) + ceil(confirmation / 12), each step
type in its own sessions because the ratios are committed before any test-view row.

| n (+ bridge models) | Models trained | Training, session-h (2 at a time) | Validation ratio, session-h | Confirmation, session-h | Total, session-h | Sessions (<= 12 h) | Weeks at 30 h |
|---|---|---|---|---|---|---|---|
| 8 + 2 | 10 | 5.0-10.0 | 1.8-3.6 | 29.0-41.0 | 36-55 | 5-6 | 1.2-1.8 |
| 10 + 2 | 12 | 6.0-12.0 | 2.2-4.4 | 34.8-49.2 | 43-66 | 5-7 | 1.4-2.2 |
| 12 + 2 | 14 | 7.0-14.0 | 2.6-5.2 | 40.6-57.4 | 50-77 | 6-8 | 1.7-2.6 |

- **Against v2:** the bridge's second model adds a training and a confirmation (3.4-5.1 h); dropping R33 on the drawn
  scenes saves 0.3-0.5 h per scene; the totals are within an hour or two of v2's. Training on 13/16 of the images is
  slightly faster, not counted. **R33 on INRIA's ten scenes** (5-9 h, two sessions, its own declaration) is not in
  these totals.
- **Option B, not the design (listed for Dace):** train every drawn scene twice, with the block (for v_s) and without
  (for P1), so P1 is measured on standard-protocol models and no P1 shift needs reporting; the bridge is then counter once,
  without the block, against INRIA's checkpoint. n = 8 + 1: 17 models, 36-57 h, 5-7 sessions; 10 + 1: 21, 44-70 h, 5-7;
  12 + 1: 25, 53-83 h, 7-9. Its cost: v_s and P1 come from different models of the same scene (as E2's ratio and P1 do,
  R31).

v2's table (R33 per drawn scene; superseded):

| n (+ bridge) | Scenes trained | Training, session-h (2 at a time) | R33, session-h (0.5-0.9 h per scene) | Confirmation, session-h | Total, session-h | Sessions (<= 12 h) | Weeks at 30 h |
|---|---|---|---|---|---|---|---|
| 8 + 1 | 9 | 4.5-9 | 4.5-8.1 | 26.1-36.9 | 35-54 | 5-6 | 1.2-1.8 |
| 10 + 1 | 11 | 5.5-11 | 5.5-9.9 | 31.9-45.1 | 43-66 | 5-6 | 1.4-2.2 |
| 12 + 1 | 13 | 6.5-13 | 6.5-11.7 | 37.7-53.3 | 51-78 | 6-8 | 1.7-2.6 |

- **R33 per new scene (v2)** (`R33_DECLARATION_DRAFT.md`'s estimate: four folds of C3DGS's sensitivity and VQ, the GN pass, OGC's
  VQ and the training-view renders, plus loading): 0.5-0.9 session-hours, run in its own session before any test-view row,
  its bundle committed first. v1's n = 6 row is dropped: at n = 6 the exact Spearman test's critical rho is 0.8286 (level
  0.0292, coarse) and its power is 0.165, 0.323 and 0.678 at rho_s = 0.5, 0.7 and 0.9 (same simulation).

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

## 7. What Dace decides (v2)

1. Request access to `DL3DV/DL3DV-Benchmark` (which accepts the terms), so that `benchmark-meta.csv`'s labels can be read
   and the 141-vs-140 count explained; or choose the no-rendering labelling rule.
2. Whether the INRIA secondary runs (n: item 4).
3. Whether to ask DL3DV about private Kaggle datasets and checkpoints.
4. (v2) Whether R33 runs first on the ten scenes with OGC rows (its own declaration draft), and n = 8, 10 or 12.
5. (v3) The validation block as drafted (i % 16 == 4, floor i % 8 == 4 below 12 views) or another fraction; the design
   (block in every E9 model, bridge trained twice) or option B (each drawn scene trained twice).
