# DRAFT — declaration for an SH-only feasibility check (exploratory; not registered, nothing run)

**Status: draft for Dace, 2026-11-08 (second session).** If adopted it would be committed and pushed alone, as a
PREREG note or an exploratory declaration, before its code. Numbers quoted are FINDINGS' (sections 19, 21, 22, 26) or
come from `bench/gn/e5_estimate.py`.

## Purpose

A MesonGS or MesonGS++ host (`kaggle/drafts/SECOND_HOST_FEASIBILITY.md`) clusters only the SH of degree >= 1; the DC
goes through another quantizer. So OGC's metric there would act on its 15 x 15 AC block only. **Before any engineering
on a second host, measure inside C3DGS, with everything else held fixed, how large OGC's lead is when only the AC part
is clustered** (before fine-tuning, where the lead is largest).

## Design

- **Host, scenes, processes:** E5's fork and protocol (C3DGS at `2a234af5`, OGC at `49ccae72`, K = 4,096, j = -1, 0, +1
  at seed 0 and j = 0 at seed 1, protocol ii). **No fine-tuning.**
- **Rows, per process:** the existing `c3dgs` and `ogc_gram` (for P1 in the same process) and two new rows:
  - `c3dgs_ac`: C3DGS's own `vq_features` on the AC coefficients only (its importance, its K), the DC kept per splat;
  - `ogc_gram_ac`: OGC's `gram_kmeans` with `metric="gram"` on the AC coefficients only, given the 15 x 15 AC block of
    the same 16-probe metric (the DC row and column dropped, which is exact when the DC is not quantized), DC per splat.
- **One implementation caveat found while drafting:** C3DGS's own switch for this, `color_compress_non_dir=False`,
  VQs the 15 AC coefficients (`compression/vq.py:164-169`) but then hands them to `set_color_indexed`, which still splits
  `features[:, :1]` off as the DC (`compression/vq.py:193`, `scene/gaussian_model.py:796-799`). That path looks
  inconsistent at `2a234af5`, so the two rows need the fork to install a per-splat DC beside the AC codebook and to
  save it in the `.npz` the same way for both rows (C3DGS's DC grid, `int8`); the rest of the file is the process's.
  Both rows pay the same DC bytes, so their BD compares the AC clustering only. Its bytes are not comparable with
  `c3dgs`'s.
- **The readout (exploratory, no verdict):** **P_AC** = BD-PSNR of `ogc_gram_ac` against `c3dgs_ac` over the seed-0
  curve (the registered fit, and section 28's PCHIP beside it), per scene and mean; P1 in the same processes; the ratio
  P_AC / P1 per scene; the j = 0 seed pair for noise (`SD_pool`, `SE` = `SD_pool` / sqrt(n)).

## Scenes: gate scenes, labelled exploratory (recommended)

- **Recommended: the seven gate scenes, labelled exploratory.** The threshold below compares P_AC with P1 in the same
  processes, and the second host itself would run on INRIA's checkpoints of these scenes, so the check measures the
  same object it predicts. They are no longer held out (E5-E7 used them), so nothing is spent. n = 7 also gives
  `SD_pool` the same form as E5-E7.
- Alternative: the development scenes (train, treehill, flowers), about a third of the cost, but n = 3 and P1 there is
  smaller and noisier (E6 part (a), section 23), so a threshold on the mean would be read through more noise.

## Cost (`bench/gn/e5_estimate.py`: each process like E5's j = +-1 processes, four rows and no fine-tuning)

Per scene 3,370-4,435 s (truck) to 5,480-8,014 s (room); **all seven 9.1-12.9 h of jobs, two sessions** with the
cached wheel (`WHEEL_CACHE_CU130.md`), about 9.3-13.1 session-hours with two 420 s setups. The development scenes would take
about one session.

## What result would make the second host not worth doing (fixed before the data)

- **Training-free rows (MesonGS with `--iterations 0`, or MesonGS++):** worth building only if **mean P_AC >= 0.10 dB
  and P_AC > 0 on at least 5 of 7 scenes** (E5's count rule). Below that, the AC-only lead is under a third of P1
  (0.3558 dB, section 19) and a second host would at best show a small lead on a part of the colour.
- **Fine-tuned rows (MesonGS with fine-tuning):** informative only if **mean P_AC >= 0.25 dB.** After C3DGS's
  fine-tuning the lead kept 0.1195 of P1 replicated, as a ratio of means (section 22 b); at that ratio a 0.25 dB AC-only lead
  becomes about 0.03 dB, the size E7's within-process noise can just detect (`SE` 0.0056-0.0111, superiority threshold
  0.016-0.032 dB, section 26). Below 0.25 dB a fine-tuned second host would be expected to read near 0 whatever the
  metric does, so a null there would not be informative.
- Above both: build MesonGS for both regimes. Between them: training-free rows only.

## Licence and standing rules

OGC's code is used under its PolyForm Noncommercial licence (research use), from the private dataset or its URL as in
E5; no OGC file is committed, pushed or bundled. MesonGS and MesonGS++ code would not be vendored either (MesonGS:
Inria's Gaussian-Splatting licence, research only; MesonGS++: MIT).
