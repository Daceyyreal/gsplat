# DRAFT D4 — a held-out memory probe (bicycle, garden, stump) and the E9-INRIA options

**Status: prepared, not run. Not in PREREG.** Files: `kaggle/drafts/memory_probe.py` (the probe),
`kaggle/drafts/build_memory_probe.py` and `kaggle/drafts/gn_memory_probe.ipynb` (one Kaggle session). The probe has not
been run anywhere, not even on a CPU stand-in.

## 1. The probe

**Why:** the first E9 draft's "does not fit" rested on a prediction; a fit to measured process peaks (E6 parts (a) and
(b)) puts bicycle at about 9.0 GB allocated / 10.9 GB reserved with images on the GPU (`kaggle/drafts/E9_FEASIBILITY.md`
section 2), an extrapolation to 1.6 x the largest model measured. One session measures it.

**What it runs,** in C3DGS at `2a234af5` per scene (stump, then garden, then bicycle), images on the GPU first and on the
CPU only if that fails: the chunked `eigh`/`det` deviation (Amendment 13 g); loading the checkpoint and scene; C3DGS's
sensitivity pass (training views); C3DGS's VQ at j = 0; OGC's Gram k-means arithmetic (`bench/gn/ogc_derived.py`, K =
4,096, chunk 25,000, 15 iterations) with a **synthetic** packed 16 x 16 metric of the real size on the host (memory
depends on its shape, not its values); 200 iterations of C3DGS's fine-tuning (training views).

**What it never does, asserted in code:** `Scene.getTestCameras`, C3DGS's `render_and_eval`, `utils.image_utils.psnr` and
the LPIPS entry point raise if called, so **no test view is rendered and no image metric computed**. The fine-tuning
loop's training loss (L1 and D-SSIM on training views) is internal and never logged; `--ft_iters 0` removes it.
The data: INRIA's members are read from the archive's live central directory (no pins exist for garden and stump;
each member's size and crc32 are checked on fetch, its sha1 recorded); the images come from the Mip-NeRF 360 zip by the
same range reader as runs 4-5. Everything is deleted after each scene.

**What it logs** (one JSON per scene and image device, zipped as `memory_probe_bundle.zip`): per step the peak allocated
and reserved GPU bytes, seconds and host RSS peak; the splat count, the colour-quantized count before pruning, the
host bytes of the synthetic metric; the image bytes per device, the image size, the numbers of training and test
cameras (counted, not rendered); the checkpoint's sha1; the T4's total memory; the chunked-linalg call counts.

**Cost:** about one session (three scenes; the downloads and the sensitivity pass dominate).

**Instructions:** import the notebook pinned at the commit that holds these files (the builder writes that commit into
it); GPU T4 x2, Internet on; run all; save the version; download `memory_probe_bundle.zip` and ingest it as data only.
Licence: `ogc_derived.py` is PolyForm Noncommercial (research use); no OGC file is used.

## 2. E9 with INRIA's three held-out scenes only

Three scenes allow no sign test (one-sided p for 3 of 3 positive is 0.125), and E5's count rule would need 3 of 3. A
declared, **descriptive** criterion could be: P1 > 0 on all three scenes and mean P1 > 2 x `SE` (E5's `SE` form from each
scene's j = 0 seed pair), with P_FT and E7's two primaries reported against their gate-scene values (inside or outside
the gate's 95% range), and no verdict on them. It adds three unseen scenes to the evidence; it cannot confirm.

## 3. A hybrid: 3 INRIA + k DL3DV scenes

One-sided sign test at 0.05 on the count of positive scenes, power at a per-scene positive rate of 6/7 (E6's P_FT, 6 of 7)
and of 3/4 (computed with `scipy.stats.binom`):

| k | n | Positive scenes needed | Its exact level | Power at 6/7 | Power at 3/4 | E5's count rule |
|---|---|---|---|---|---|---|
| 4 | 7 | 7 | 0.0078 | 0.34 | 0.13 | 5 |
| 5 | 8 | 7 | 0.0352 | 0.68 | 0.37 | 6 |
| 6 | 9 | 8 | 0.0195 | 0.62 | 0.30 | 7 |
| 7 | 10 | 9 | 0.0107 | 0.57 | 0.24 | 7 |
| 8 | 11 | 9 | 0.0327 | 0.80 | 0.46 | 8 |

For P1 (positive on 7 of 7 gate scenes) any of these is well powered; for P_FT, **3 INRIA + 8 DL3DV (n = 11) is the
smallest design with 80% power at 6/7**, and E5's three criteria (mean, count, `SE`) can sit beside the sign test as in
E5-E7. Training eight DL3DV checkpoints is about one more session than E9's seven (`E9_FEASIBILITY.md`).

## 4. A mean-based criterion beside the sign test (R15, added 2026-11-08, fourth session)

Scenes needed at 80% power, one-sided 0.05, for a **paired mean criterion**: each scene's BD value is already a paired,
within-scene difference (both rows on the same scene and processes), and the test is a one-sample t-test of the
per-scene values against 0, with the effect and the **across-scene SD** taken from the measured per-scene values
(`kaggle/drafts/e9_power_draft.py`, from `bench/gn/check_s28.py`'s values; the search starts at n = 3, so "3" means 3 or
fewer). Beside it, E5's criterion 3 (`P_bar` > 2 x `SE_noise`, `SE_noise` = `SD_pool` / sqrt(2n), E5's `SD_pool` for P1,
E6 part (b)'s for P_FT), with the scene mean's spread from the same across-scene SD.

| Criterion | Target | Effect and SD assumed (source) | Scenes needed | Mix |
|---|---|---|---|---|
| sign test (section 3) | P1 | every scene positive (7 of 7 gate, 3 of 3 development) | 5 (5 of 5 positive: p 0.031) | 3 INRIA + 2 DL3DV |
| sign test (section 3) | P_FT | positive rate 6/7 (E6 part (b)) | 11 | 3 INRIA + 8 DL3DV |
| paired t, mean | P1 | 0.3558, SD 0.1175 (E5, gate 7) | 3 (power 0.93 at 3), **fragile** | 3 INRIA alone |
| paired t, mean | P1 | 0.2770, SD 0.1616 (gate 7 + development 3) | 4 | 3 INRIA + 1 DL3DV |
| paired t, mean | P1 | 0.0929, SD 0.0588 (development 3: flowers, treehill, train) | 5 | 3 INRIA + 2 DL3DV |
| paired t, mean | P_FT | 0.0425, SD 0.0373 (E6 part (b), gate 7) | 7 (power 0.84) | 3 INRIA + 4 DL3DV |
| paired t, mean | P_FT | 0.0467, SD 0.0309 (E7's BD_R, seed 0, gate 7) | 5 | 3 INRIA + 2 DL3DV |
| paired t, mean | P_FT | 0.0299, SD 0.0394 (gate 7 + development 3) | 13 | 3 INRIA + 10 DL3DV |
| paired t, mean | P_FT | 0.0005, SD 0.0308 (development 3) | none (no effect to detect) | - |
| E5's criterion 3 (seed `SE`) | P1 | 0.3558, SD 0.1175 | 3 | 3 INRIA alone |
| E5's criterion 3 (seed `SE`) | P_FT | 0.0425, SD 0.0373 / gate + development 0.0299, 0.0394 | 4 / 8 | 3 + 1 / 3 + 5 DL3DV |

- **The mean criterion needs fewer scenes than the sign test for P_FT at the gate's effect (7 against 11),** because
  the sign test discards each value's size.
- **The planning effect is the open question, not the test.** INRIA's three scenes are Mip-NeRF 360 outdoor scenes,
  like flowers and treehill, where P1 was 0.124 and 0.025 dB and P_FT +0.034 and -0.026 dB (E6 part (a)); the
  development scenes' P_FT mean is 0.0005 dB. On the gate's effect P_FT needs 7 scenes, on the pooled ten 13, and on
  the development scenes alone no n suffices. A design sized on the gate scenes alone is optimistic for outdoor scenes.
- **E5's criterion 3 is not a test across scenes:** its `SE_noise` is seed noise; it is shown because E5-E7 used it, not
  as the basis for a held-out claim.
- **The 3-scene paired-t row is fragile:** its power treats the gate scenes' SD (0.1175 dB, from seven scenes) as known,
  and with n = 3 the test has 2 degrees of freedom, so a held-out SD only somewhat larger, or a smaller effect, drops
  the power fast (at the pooled ten scenes' mean and SD, n = 3 gives 0.61). It is not a design to fund on its own.

## 5. The recommended E9 design (R19, added 2026-11-09)

- **Primary: P1, one-sided paired t-test of the per-scene BD values against 0 at 0.05, sized on the pooled gate and
  development scenes** (mean 0.2770 dB, SD 0.1616 dB, n = 10; FINDINGS section 29 gives the per-scene values): **4
  scenes for 80% power (0.82), 3 INRIA + 1 DL3DV.** E5's count and criterion 3 can sit beside it as in E5.
- **P_FT: estimation only, a mean with a 95% confidence interval and no test,** unless 13 scenes are funded (3 INRIA +
  10 DL3DV, 80% power at the pooled mean 0.0299 dB and SD 0.0394 dB). With the pooled SD the interval's half-width is
  about 0.063 dB at 4 scenes, 0.036 at 7, 0.027 at 11 and 0.024 at 13, so at 4 scenes it cannot tell the gate's
  0.0425 dB from 0.
- **Expected held-out range (an expectation, not a prediction to test):** P1 between the development scenes' 0.0929 dB
  and the gate scenes' 0.3558 dB, and P_FT between 0.0005 dB and 0.0425 dB (FINDINGS section 29). **INRIA's three
  held-out scenes, bicycle, garden and stump, are outdoor Mip-NeRF 360 scenes**, like flowers and treehill, so the lower
  end is as plausible as the upper. They are held out for OGC rows but not unseen: garden and bicycle are development
  scenes for GN-VQ (Amendment 7) and stump since Amendment 10, and E3p and E3r used bicycle's INRIA checkpoint
  (`kaggle/drafts/E9_FEASIBILITY.md` section 1).
- **How the 4-scene design fails:** if held-out P1 is near the development scenes' mean while the spread is the pooled
  SD (0.0929 dB, 0.1616 dB), 4 scenes give 0.23 power and 21 are needed; with the development scenes' own SD (0.0588 dB)
  5 are needed. Funding 5-7 scenes (3 INRIA + 2-4 DL3DV) covers the development-like case at the development SD
  (0.89 at 5, 0.98 at 7) and keeps the pooled case above 0.93.
