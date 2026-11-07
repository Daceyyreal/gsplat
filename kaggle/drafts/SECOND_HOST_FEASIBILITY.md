# DRAFT — a second host for OGC's metric: feasibility (2026-11-08)

**Status: draft for Dace. No code, no run.** Repositories, licences and source lines were read on 2026-11-08 through
GitHub's API and raw files (no clone; nothing vendored). Line numbers are at each repository's default branch on that
date unless a commit is given. OGC's code stays outside the repository in every option (standing rule).

## What a host must have

The question E5-E7 leave open is whether the result carries over to another host (paper, Limitations "One host"). The
cheapest second host is one with **(1) a post-training VQ of the SH colour, with kept rows or a codebook we can replace
at one call, (2) optionally a fine-tuning stage after it, and (3) an INRIA-format checkpoint as input**, so the gate's
checkpoints, datasets, protocol ii and our harness carry over.

## Fine-tuned VQ pipelines

| Pipeline | Code, licence | SH-colour VQ | Fine-tuning after VQ | Fit |
|---|---|---|---|---|
| **MesonGS** (ECCV 2024) | `ShuzhaoXie/MesonGS`, Inria Gaussian-Splatting licence (research, as C3DGS's) | `vq_fe` calls C3DGS-style `vq_features` on the SH (`scene/gaussian_model.py:1299-1301`), from `training` (`mesongs.py:738`), after importance pruning, octree and RAHT | **yes**: the loop at `mesongs.py:767-842`, `--iterations 8000` in its script; `--iterations 0` for none | **best**: one call to replace, both regimes, INRIA `.ply` input |
| **LightGaussian** (NeurIPS 2024) | `VITA-Group/LightGaussian`, Inria licence | VecTree: importance-weighted EMA k-means, 8,192 entries, the least important 60% quantized (`vectree/vectree.py:166-207`) | **no** in the released code (the VecTree step is a standalone script after prune and distill; its README lists no fine-tuning after it) | a training-free host, not a fine-tuned one; needs its own prune + distill stages first |
| **CompGS** (ECCV 2024) | `UCDvision/compact3d`, MIT (on Inria's 3DGS) | k-means of colour and covariance **during** a full 30k-iteration training (every `kmeans_freq` iterations) | quantization-aware training from scratch | poor: the metric would be recomputed inside training, and every row is a full training |
| **EAGLES** (ECCV 2024) | `Sharath-girish/efficientgaussian`, Inria licence | quantized latents with small decoders, no SH codebook | trained end to end | none: no colour VQ to replace |
| **NSVQ-GS** (SCIA 2025) | `AaltoML/NSVQGS`, MIT | separate codebooks trained with noise substitution | trained from scratch | poor: the metric would enter a training loss, not a drop-in |

## Training-free codecs with public code

| Codec | Code, licence | Where colour is clustered | Fit |
|---|---|---|---|
| **MesonGS++** (2026) | `mmlab-sigs/mesongs_plus`, MIT (pushed 2026-05-07) | `vq_features` "borrowed from c3dgs" with kept SH rows (`splatwizard/model_zoo/mesongs_plus/meson_utils.py:1714-1740`), called at `splatwizard/model_zoo/mesongs_plus/model.py:926`; its scheduler registers no fine-tuning task after `vq_fe` (`model.py:1749-1756`, the fine-tuning tasks commented out) | **best training-free**: the same call shape as C3DGS's; INRIA `.ply` input (`compress_single_scene.sh`); its configs cover the same 13 scenes |
| **GSICO** (2026) | no repository found (v1 said after acceptance) | images of parameters, JPEG XL | not available |
| **3DTurboQuant** (2026) | `JaeLee18/3DTurboQuant` (code under an Inria licence file in `gaussian-splatting/`) | a data-independent rotation and Lloyd-Max scalar quantizer; no clustering | none: nothing for the metric to replace |
| **KISS-GS** (ECCV 2026) | `fraunhoferhhi/KISS-GS` holds the project page only; its code link is `w-m/ffsplat` (Apache-2.0) | sorted attribute images coded with image codecs; no VQ | none without redesign |

## Effort and Kaggle cost

**One of each, as asked:**

1. **Fine-tuned: MesonGS.**
   - *Engineering:* a hook at MesonGS's colour VQ (`gaussian_model.py:1301`) that calls OGC's `gram_kmeans` (or
     `gram_kmeans_ours`) the way `kaggle/e5_hooks.py` does at C3DGS's colour call; the metric is this project's 16-probe
     Gram estimate restricted to its 15 x 15 AC block, because MesonGS quantizes `_features_rest` only
     (`gaussian_model.py:1300`, from `vq import vq_features`, `:32`) and fine-tunes the codebook (`:1310`,
     `requires_grad=True`), where C3DGS's `color_compress_non_dir` puts DC in the codeword (16 x 16); the row harness (copy and
     restore of the model, protocol ii, bytes) ported from E5's; MesonGS's own indexed rasterizer built on the cu130
     image. **About E5's build: 1-2 weeks, and a pilot session on train.**
   - *Rate knob:* MesonGS's pruning percent or codebook size, to be scanned in the pilot (C3DGS's was the colour
     threshold, E3r).
   - *Cost (estimate):* per scene four processes (three rate points at seed 0, one more seed at the middle) with
     `c3dgs`-style and OGC rows, unfine-tuned and fine-tuned (8,000 iterations, about 1.6 x C3DGS's 5,000; E6's model
     gives 449-956 s per C3DGS fine-tuned row, so about 0.7-1.5 k s here): **about 20-26 k s per scene, 40-50 h for the
     seven gate scenes at the high end**, plus the pilot (about 6 h).
2. **Training-free: MesonGS++.** The same hook at `model.py:926`, no fine-tuning; its GPCC and entropy coders to build
   (not checked). **About 1 week, and 5-10 h of Kaggle** for seven scenes (the metric's GN pass dominates; compression
   takes minutes).

## Recommendation

**MesonGS first.** It is the only candidate that hosts both regimes in one codebase (`--iterations 0` and 8,000), its
colour VQ is C3DGS's function, so our injection ports with little change, and it reads INRIA's checkpoints, so the gate
scenes, datasets and protocol ii carry over. It answers two questions at once: whether OGC's lead before fine-tuning
carries over to another codec (training-free rows) and whether another codec's fine-tuning also erases it (fine-tuned
rows). MesonGS++ is the permissive (MIT) training-free alternative if the Inria licence is a concern for the
training-free part. LightGaussian is a second training-free option but needs its prune and distill stages first.
CompGS, EAGLES and NSVQ-GS are not drop-ins.
