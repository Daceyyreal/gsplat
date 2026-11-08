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
