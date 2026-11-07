# DRAFT — E9 feasibility: a held-out confirmation of P1, P_FT and E7's primaries

**Status: draft for Dace, 2026-11-08. No amendment, no code, no run.** Sources: `kaggle/HANDOFF.md`, FINDINGS sections
4-26, `kaggle/E3_SCOUTING.md` (splat counts), Amendment 14 g (`kaggle/PREREG_GN.md`, memory), E6's note 1 (memory on
flowers and treehill), the E7/E8 runtime model (`bench/gn/e7_estimate.py`, `kaggle/drafts/e8_estimate_draft.py`), and the
dataset and code pages read on 2026-11-08 (listed at the end).

## 1. Every INRIA-checkpoint scene and every experiment that touched it

"INRIA" = INRIA's published 30k checkpoints; "own" = this project's gsplat MCMC checkpoints (runs 1-5), a different model
of the same images.

| Scene | Splats (INRIA, 30k) | Experiments on INRIA's checkpoint | OGC rows? | Experiments on the same scene, own checkpoints |
|---|---|---|---|---|
| bicycle | 6,131,954 | E3p (pipeline at scale), E3r (GN passes only; C3DGS restricted to train, Am14 g) | **never** | runs 1-5, E0, E1 (development) |
| garden | 5,834,784 | none | **never** | runs 1-5, E0, E1 (development), E2b (control) |
| stump | 4,961,797 | none | **never** | run 4-5, E2 (held out), E2b |
| treehill | 3,783,761 | E4q, E6 part (a) | yes | run 4-5, E2, E2b |
| flowers | 3,636,448 | E6 part (a) | yes | run 4-5, E2, E2b |
| train | 1,026,508 | E3p, E3q, E3r, E4p, E4q, E5p (E5s: source check, no scene data) , E6 part (a) | yes | run 5, E2 |
| bonsai, counter, kitchen, room | 1.22-1.85 M | E5, E6 part (b), E7 | yes | run 4-5, E2, E2c |
| truck | 2,541,226 | E5, E6 part (b), E7 | yes | run 5, E2, E2c |
| drjohnson, playroom | 3,405,153; 2,546,116 | E5, E6 part (b), E7 | yes | none |

**Never any OGC row: bicycle, garden and stump**, the three largest Mip-NeRF 360 models. All three were read with own
checkpoints (runs 1-5, E0-E2b), and all three are among the nine scenes OGC's own paper reports in its C3DGS host, so a
confirmation there would be held out for this project's OGC rows but not unseen (the same disclosure as Amendment 15 d).

## 2. Can bicycle and garden run on a T4?

**No, not with the host as it is, and stump is doubtful.**
- **Amendment 14 g** (from the code, checked against E3q on train to 89.4%): C3DGS alone on bicycle needs 14.16-17.05 GB
  with the images on the GPU and **11.80-14.68 GB with them on the CPU**, before the CUDA context and fragmentation; the
  T4 has 15.64 GB, and "no method-neutral mitigation brings C3DGS itself under about 14 GB on bicycle". E3p's bicycle job
  already reserved 15.18 GB with 11.51 GB allocated (a lighter pipeline).
- **OGC's row adds to that:** its 16 x 16 metric per colour-quantized splat (544 bytes packed, Am14 g) is up to 3.34 GB
  on bicycle; E4p's fork peaked at 9.43 GB on train (1.03 M splats) with OGC's chunk.
- **Measured on smaller scenes:** E6's flowers and treehill (3.6-3.8 M splats) read 14.93 GB of 15.64 GB at the start,
  and note 1's gap check would have read 16.43 and 16.49 GB (a stated risk that held).
- garden is 95% of bicycle's splats and stump 81%; scaled from Am14 g, stump's C3DGS alone would be about 9.6-11.9 GB
  with the images on the CPU, before OGC's metric (about 2.7 GB) and E7's arm copies. **Feasible only with a memory
  change to the host**, which would be a new deviation on held-out scenes.
- **And n = 3 is too few** for E5's or E7's criteria (the count criterion would need 3 of 3; t with 3 degrees of
  freedom: 3.18 at 0.975 against 2.36 at n = 7).

## 3. The alternative: our own 3DGS checkpoints on new public scenes

- **Training code:** C3DGS has no `train.py`; INRIA's checkpoints come from `graphdeco-inria/gaussian-splatting`
  (same licence family as C3DGS), whose output (`point_cloud/iteration_30000/point_cloud.ply`, `cfg_args`,
  `cameras.json`) is exactly what the host loads today. Pin a commit from INRIA's release period so densification
  matches the gate's checkpoints; record the pin before any training.
- **Datasets (read 2026-11-08; licences to be read in full before use):**
  - **DL3DV-10K's benchmark split** (140 scenes; Hugging Face, gated; the repository says it provides 3DGS-format scenes
    with camera poses, so no COLMAP run; released under its own "DL3DV-10K Terms of Use", not checked here).
  - **Tanks and Temples, intermediate set** (Family, Francis, Horse, Lighthouse, M60, Panther, Playground; Train is
    already used): standard in 3DGS papers, but the images need a COLMAP run (CPU hours per scene on Kaggle, not
    measured); the licence page names CC BY 4.0 and also grants only non-commercial research use (it reads as
    research-only; to confirm).
  - Deep Blending's other scenes come from its authors' release, not INRIA's archive; not checked.
- **Training cost (estimate, not measured for this code):** this project's run 4 trained gsplat MCMC models (1 M cap) in
  1,932-3,398 s per Mip-NeRF 360 scene on a T4; INRIA's densification grows 1-3 M splats and is slower, so **1-2 h per
  scene**, two scenes at a time on 2x T4: **one session of about 5-8 h for seven scenes** (plus the one-off wheel and
  rasterizer build).
- **Confirmation cost:** P1, P_FT, Δ_refit and Δ_cb need arms R, G and CG (E7's rows without K, O and `*_own`: four
  fine-tuned rows beyond R per full process, plus the replicate pair), so **for seven scenes the size of E7's, 72.5 h at
  the high end and 49.3 h calibrated** (`kaggle/drafts/e8_estimate_draft.py`'s `parts` with `ft_rows=4`; DL3DV's 960p
  frames and smaller models would cost less, which the scene choice settles). A P1 + P_FT-only confirmation (E6 part (b)'s four processes) is about 29 h high, 20 h
  calibrated.
- **Held-out by construction:** scenes chosen by a rule fixed before download (for example a seeded draw of seven from
  the benchmark list, stratified indoor/outdoor), with hypotheses copied from E5, E6 and E7 unchanged; nothing on them
  read before the registration.

## 4. Recommendation

**Own checkpoints on seven DL3DV benchmark scenes, chosen by a pre-registered rule, confirming P1, P_FT, Δ_refit and
Δ_cb with E5's, E6's and E7's rules unchanged:** about one training session (5-8 h) plus 49-73 GPU-hours of
confirmation, **about 2-2.7 weeks of quota** at 30 h a week, after a wheel session. It gives n = 7 genuinely unseen scenes and
avoids both the T4 memory wall and a COLMAP step. If DL3DV's terms do not allow it, Tanks and Temples' intermediate set
with COLMAP is the fallback (plus the COLMAP cost, to measure on one scene first). The INRIA-held-out route (bicycle,
garden, stump) is not recommended: at most one scene fits, after a host change.

## Sources read 2026-11-08

- DL3DV benchmark: https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark and https://github.com/DL3DV-10K/Dataset
- Tanks and Temples licence and download pages: https://tanksandtemples.org/license and https://tanksandtemples.org/download/
- C3DGS tree at `2a234af5` (no `train.py`): https://github.com/KeKsBoTer/c3dgs
