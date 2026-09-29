# E3's C3DGS arm: integration design (2026-09-29, not pre-registered)

This is the design input for the C3DGS arm's pre-registration (Amendment 12 c: "The C3DGS comparison. Primary: no
fine-tuning [...] Secondary: with C3DGS's 5,000-iteration quantization-aware fine-tuning, for both"). **Nothing
here is decided.** The choices below are recommendations for that amendment, which must be written before any of
the arm's code or data.

Sources:
- C3DGS (`KeKsBoTer/c3dgs`) at `2a234af55fbe8b90c8829c1436ce80088c4b622b`, read from a scratch clone (file:line
  below);
- this repository (`bench/gn/`, `kaggle/`), PREREG_GN.md Amendments 11-13;
- the E3p bundle (`kaggle/gn_e3p/gn3p/`, FINDINGS section 13) and the E3q bundles (`kaggle/gn_e3q/attempt{1,2}/gn3q/`,
  FINDINGS section 14).

Every claim carries one of four tags:

| Tag | Meaning |
|---|---|
| **[code]** | read from source and not run |
| **[measured]** | from a committed bundle |
| **[estimate]** | derived arithmetic, labelled as such |
| **[unverified]** | not checked; the pre-registration must either check it or state it |

## 1. What C3DGS quantizes, and the knobs that set the rate

`compress.py`'s `run_vq` runs these steps in this order. The defaults are `arguments/__init__.py:69-96`,
`CompressionParams`.

1. **Sensitivity** (`compress.py:40-84`, `calc_importance`) **[code]**. One pass over the train views backpropagates
   `rendering.sum()`. Hooks take the absolute gradient of each SH coefficient per channel (`[N, 48]`, DC included)
   and of each normalized-covariance entry (`[N, 6]`), summed over views and divided by the total pixels. This is a
   first-order sensitivity, not a second moment.
2. **Per-splat scalars** (`compress.py:154,156`) **[code]**: `color_importance_n = amax` over the 48 values, and
   `gaussian_importance_n = amax` over the 6.
3. **Pruning** (`compression/vq.py`, `compress_gaussians`) **[code]**. With `prune_threshold = 0.0`, every splat with
   `color_importance_n <= 0` is removed (`mask_splats`).
   - **[measured]** On train this removed 115,907 of 1,026,508 splats (11.29%): both E3q runs kept 910,601.
4. **Colour** (`compression/vq.py:146-192`, `compress_color`) **[code]**.
   - **Who is kept.** Splats with `color_importance_n > color_importance_include` (0.6e-6) are **kept**: they
     store their own 48 SH values. The rest are **vector-quantized**.
   - **What is quantized.** The quantized vector is `gaussians.get_features.flatten(-2)`: all 16 coefficients x 3
     channels, **DC included**, because `color_compress_non_dir = True`. The order is coefficient-major, and the
     values are already through the int8 fake quantizer (`get_features` applies `features_dc_qa` /
     `features_rest_qa`).
   - **The clustering** is `vq_features` (`vq.py:77-107`) with K = `color_codebook_size` = 4,096:
     - it initializes uniformly in [min, max] (`rand_like`, the CUDA generator);
     - it runs `color_cluster_iterations` = 100 steps. Each step draws `color_batch_size` = 2^18 random splats with
       replacement (`torch.randint`, the global CPU generator), assigns them by plain squared L2 (the
       `weighted_distance` kernel is unweighted), and updates the codebook by an EMA (decay 0.8) of
       importance-weighted means, with the importance divided by its maximum;
     - it ends with a plain-L2 assignment of every quantized splat.
   - **The table.** `join_features` stacks the codebook and the kept rows into one table (codebook rows first). Each
     splat's index is its codeword, or K + its kept row.
   - **Freezing.** `set_color_indexed` makes the table rows the model's `_features_dc` / `_features_rest`
     parameters and fixes the indices.
   - **[unverified]** How many train splats kept their own colour: the files do not record it.
5. **Geometry** (`compress_covariance`) **[code]**. The same procedure runs on the trace-normalized covariance (6
   values), with `gaussian_importance_include` = 3e-6, a codebook of 4,096, 800 steps of 2^20 and
   `scale_normalize`. `extract_rot_scale` then turns the table into rotations and scales (`eigh`, then `det`:
   Amendment 13 g).
   - **[measured]** On train, 277,141 splats kept their own geometry; with the 4,096 codewords that is the
     281,237 matrices of FINDINGS section 14.
6. **Optional fine-tuning** (section 5 below).
7. **Encoding** (`scene/gaussian_model.py:442-576`, `save_npz`) **[code]**. The model is Morton-sorted, then
   written with `np.savez_compressed` (deflate). Per stream:

   | Stream | Stored as |
   |---|---|
   | `xyz` | float16 |
   | opacity | int8 (per-tensor scale) |
   | per-splat `scaling_factor` | int8 |
   | scale and rotation tables | int8 |
   | `features_dc` and `features_rest` tables | int8, separate per-tensor scales |
   | `feature_indices` and `gaussian_indices` | int32 |

**The rate knobs** (all `CompressionParams`) **[code]**:

| Knob | Default | What it changes |
|---|---|---|
| `color_codebook_size` | 4,096 | codebook rows (48 int8 values each) and the entropy of the colour indices |
| `color_importance_include` | 0.6e-6 | how many splats keep 48 own colour values: the colour stream's main lever |
| `color_cluster_iterations`, `color_batch_size`, `color_decay` | 100, 2^18, 0.8 | codebook quality, not bytes |
| `gaussian_codebook_size`, `gaussian_importance_include` | 4,096, 3e-6 | the geometry table and indices, as above |
| `gaussian_cluster_iterations`, `gaussian_batch_size`, `gaussian_decay` | 800, 2^20, 0.8 | geometry codebook quality |
| `prune_threshold` | 0.0 | the splat count (a negative value skips pruning) |
| `not_compress_color`, `not_compress_gaussians` | off | skip a VQ |
| `not_sort_morton` | off | stream order, hence deflate |
| `finetune_iterations` | 5,000 | values only; +89,088 bytes on train **[measured]** (int8 scales and deflate) |

- **Two options have no effect.** `color_weights_per_param` (`arguments/__init__.py:80`) and
  `color_importance_prune` (`:76`) are defined but never used **[code]**.
- **Where the bytes are.** The per-splat streams dominate. Before deflate, `xyz` (6 bytes), the two int32 indices
  (8) and opacity plus `scaling_factor` (2) come to 16 bytes per splat. The 4,096 x 48-byte colour codebook is
  196,608 bytes **[code, arithmetic]**. The whole `.npz` is 15.18 bytes per splat on train (13,822,087 / 910,601)
  **[measured]**. The per-stream split was not recorded **[unverified]**.

## 2. Where GN-VQ with the floor goes in

**The injection point** is the colour call of `vq_features` inside `compress_color`. It must return what
`vq_features` returns: a codebook `[K, 48]` and labels `[N_vq]` for the quantized splats, in their order.

**The recommended mechanism is Amendment 13 g's:** the wrapper patches functions in its own process, and C3DGS's
source stays unedited **[not implemented]**.

1. Wrap `compression.vq.compress_gaussians` to record `non_prune_mask` (`color_importance_n > prune_threshold`), so
   pruned rows map back to the checkpoint's splat ids. Wrap `compress_color` to form the same `vq_mask`.
   - `compress.py` imports `compress_gaussians` by name, and `compress_gaussians` calls `compress_color` through
     the module's globals, so patching both before `runpy` reaches both call sites **[code]**.
2. **Run C3DGS's own `vq_features` first, unchanged,** for two reasons:
   - it consumes the global CPU and CUDA random streams exactly as the baseline does, so the geometry VQ that
     follows draws the same batches in both arms;
   - its codebook and labels are the baseline arm's colour codebook, which is the natural warm start.
3. Run GN-VQ with the floored metric on the quantized splats from that warm start, with its own generators (no
   global torch or Python `random` draws: fine-tuning picks cameras with Python's `random.randint`). Return its
   codebook and labels.

**Warm start.** The frozen method warm-starts from `lloyd_wopa_area` (Amendment 11 b). Warm-starting from C3DGS's
own codebook is a change to it, so the amendment must state it.
- **Alternative:** run `lloyd_wopa_area` in the 48-dimensional space. That costs a clustering: 686.0 s on train at
  K = 65,536 in E3p **[measured]**.

**`M_i`: which splats and which coefficients.**
- **Splats:** only the quantized ones: after pruning, and with `color_importance_n` in (0, 0.6e-6]. Their number is
  not recorded **[unverified]**.
- **Coefficients: the default quantizes DC with the 15 AC coefficients,** while the frozen metric is 15x15 over AC
  only, shared across channels (Amendment 11 b; `tr(M_i) / 15`). Three ways to reconcile them:
  - **A (recommended). Extend the GN pass to 16 coefficients.** `M_i` becomes 16x16, 136 packed values against
    120, and the floor uses `tr(M_i) / 16`. The DC column of the Jacobian is the same render derivative times the
    constant basis value. This is an extension of the frozen method and must be pre-registered as one. **[estimate]**
    It costs 544 bytes per splat against E3p's measured 480 (+13%) in the GN pass and GN-VQ, with no run yet
    **[unverified]**.
  - **B. Keep DC out of GN-VQ** by quantizing it separately. C3DGS's table cannot express that without a source
    change. Not recommended.
  - **C. `color_compress_non_dir = False`.** This looks **broken at `2a234af5`** **[code, not run]**:
    - `compress_color` flattens only `get_features[:, 1:]` (45 values) and reshapes the table to `(-1, 15, 3)`;
    - `set_color_indexed` then takes row 0 of those 15 as DC.

    So the real DC is dropped and the first AC coefficient becomes DC. It is also not the published setting. Not
    usable without a source edit.
- **Input values.** The features C3DGS quantizes are already int8-fake-quantized (`get_features`), so GN-VQ must
  take the same tensor.
- **Where `M_i` comes from.** From `bench/gn`'s gsplat GN pass on the INRIA checkpoint, as in E3p. C3DGS renders
  with INRIA's rasterizer; parity between the two on these checkpoints was never checked **[unverified]**. The 0.410-0.523
  dB gap between C3DGS's evaluation and protocol ii (FINDINGS section 14) is unexplained and may include rasterizer
  differences.
- **The colour clamp.** Both renderers clamp colours as `max(SH + 0.5, 0)`: C3DGS's rasterizer in
  `computeColorFromSH` (`submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:63-70`, when
  `clamp_color` is on, the default), gsplat in its fused path **[code]**. C3DGS's own sensitivity pass renders with
  `clamp_color=False` (`compress.py:68`); the GN pass ignores the clamp too and reports how often it bites (E0's
  `clamp_fraction`).

**The codec's quantizer** (the frozen variant's clip and final quantized assignment).
- **Here the quantizer is C3DGS's.** It is a per-tensor int8 `FakeQuantize` over the whole colour table (codebook
  rows and kept rows together, DC and AC with separate scales), and its scale comes from an observer.
  - The GN-VQ codebook's range therefore also sets the int8 step of the **kept** rows **[code]**.
  - The frozen clip to the warm-start range keeps the codebook inside C3DGS's own codebook range, but the table's
    range also includes the kept rows.
- **What the amendment must fix:**
  - "the codec's quantizer" = this int8 quantizer, with the table's observed range at write time **[unverified in
    detail: the observer's update rule during `save_npz`]**;
  - the final assignment against the dequantized codebook.
- The geometry VQ, the pruning and everything else stay C3DGS's own.

## 3. `rho`'s cross-validation inside this codec

The frozen selection (Amendment 11 b):
- `M_even` from the even-indexed train views;
- for each of the 7 `rho`, GN-VQ;
- each `rho` scored by the render-vs-render dMSE on the odd-indexed train views, **with only the SH swapped**;
- `rho_cv` = the lowest score.

Two ways to carry it over:

| | (a) SH-only, as frozen | (b) the full C3DGS pipeline |
|---|---|---|
| What is scored | the quantized splats' colours replaced by the dequantized GN-VQ codebook, in the pruned checkpoint, rendered by gsplat against the same model without the swap | one `compress.py` run per `rho` (geometry VQ, quantization, the `.npz` round trip), scored on the odd views |
| Per `rho` | one GN-VQ run on `M_even` plus one scoring row: 4.9-7.8 s per million splats in E3p **[measured]** | one `c3dgs_ft0` run: 351.6 s on train **[measured]**, unknown elsewhere |
| Per scene, 4 K x 7 `rho` | 141-224 s of scoring on train, 843-1,340 s on bicycle **[estimate]** | 9,845 s on train; 9,845-58,810 s on bicycle, flat to linear in splats **[estimate]** |
| Consistency | the frozen rule; the same renderer as `M` | a new rule (a deviation); another renderer, and the geometry VQ mixed into the score |

- **Recommendation: (a).** It is the frozen rule, it isolates the colour codebook, and the geometry VQ is identical
  across `rho` anyway, because the random streams are preserved (section 2).
- **Its limit:** interactions with the geometry VQ do not enter the score **[unverified effect]**.
- (b) adds 88,605-206,819 s over the nine scenes **[estimate, section 7]**.

## 4. The rate-distortion curve and BD-rate

- **Arms per scene:**
  - C3DGS's own colour VQ;
  - `gn_vq_cvfloor` injected into it, with everything else identical (the same random streams, section 2).
- **Recommended knob:** `color_codebook_size` K in {1,024, 4,096, 16,384, 65,536}. That is E2's and E2c's grid, it
  includes C3DGS's default 4,096, and it gives **4 points per arm**, which is enough for the cubic BD fit.
  - The BD measures would use the domain-scaled fit from E2b on (Amendment 9 a; `g2.bd_rate_scaled` /
    `bd_psnr_scaled`), with bytes = the `.npz`.
  - The indices stay int32 at every K.
- **The risk:** the colour codebook is a small part of the bytes (section 1). K moves the rate mainly through the
  indices' deflated entropy and the 48-byte rows, so the four points may span a narrow rate range, and then the
  overlap BD-rate needs is short and ill-conditioned **[unverified]**.
- **The alternative knob,** `color_importance_include`, moves bytes strongly. But each point then quantizes a
  different set of splats, so `M_i`'s subset and GN-VQ's share change along the curve.
- **Recommendation:** a pilot on the development scenes (train, bicycle) that measures the baseline arm's `.npz`
  bytes and PSNR at the four K, **before** the arm's pre-registration fixes the knob. It is cheap: 8 `c3dgs_ft0`
  runs.
- **Also unmeasured:** C3DGS's own clustering at K other than 4,096. 100 steps of 2^18 L2 distances to K codewords
  grow with K, so 16,384 and 65,536 may cost many times the 257.7 s measured at 4,096. That time includes the
  geometry VQ **[unverified]**.
- **Units:** sizes in bytes, MiB and MB, as in E3q.

## 5. Fine-tuning and an injected codebook

**What fine-tuning does** (`finetune.py`, `scene/gaussian_model.py:223-272`) **[code]**:
- It runs Adam (lr 0 base, eps 1e-15) over `_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`,
  `_scaling_factor` and `_rotation`, for 5,000 iterations.
- The loss is L1 + 0.2 D-SSIM on one random train view per step.
- The feature learning rates are 0.0025 (DC) and 0.0025 / 20 (AC); the position rate is scheduled from iteration
  30,000 on.
- The fake quantizers stay active, and their observers keep updating.

**After `set_color_indexed`, `_features_dc` / `_features_rest` are the table.** The index tensors are parameters with
`requires_grad=False`, and nothing else writes them. So:
- **the codebook values are re-optimized, jointly with the kept rows and the geometry;**
- **the labels are never changed.**

**Consequences:**
- The secondary (fine-tuned) comparison measures GN-VQ's labels plus a retrained table, not GN-VQ's codebook.
- **[measured]** On train, C3DGS's fine-tuning alone moved protocol ii by +0.448 dB (FINDINGS section 14). That is
  ten times the 0.043 dB GN-VQ gained over `lloyd_wopa_area` on the same checkpoint in E3p's different codec.
  That comparison is context only, but it suggests the secondary may not resolve the colour codebook's
  contribution. Amendment 12 c already makes the no-fine-tuning comparison primary.
- **Random streams:** fine-tuning draws cameras from Python's `random` and trains with CUDA non-determinism, so
  the two arms' fine-tuned rows are not paired draw for draw **[code]**. Their run-to-run spread is unmeasured
  **[unverified]**.

## 6. Evaluation protocol

- **Cross-paper comparisons use protocol ii** (Amendment 12 c). Each row should carry both C3DGS's own evaluation
  and protocol ii, as E3q's did.
- **The gap between the two** (FINDINGS section 14) is unexplained, and E3q never ran C3DGS's evaluation on the
  uncompressed model.
- **Recommendation:** a same-session C3DGS-side evaluation of the uncompressed checkpoint (C3DGS's `metrics.py` or
  `render.py`, **[unverified]** whether either takes a `.ply` directly). The arm's pre-registration should state
  which protocol its rules use. BD measures in protocol ii would keep the arm comparable with E3p.

## 7. Compute per scene **[estimate]**

**The model.** It is built from these measurements:

| Input | Source | Value |
|---|---|---|
| per-million-splat rates for the two GN passes, a GN-VQ run at K = 65,536 and an SH-only scoring row | E3p, via `bench/gn/e3p_estimate.py` | as measured |
| GN-VQ at 1,024-16,384 together, over K = 65,536 | E2c | 1.885-2.202 |
| `c3dgs_ft0` / `c3dgs_ft5000` wall time on train | E3q attempt 2 | 351.6 / 666.4 s |
| decoding and protocol ii per row on train | E3q attempt 2 | 20.1 / 20.3 s |

**What it counts, per scene:**
- the two GN passes;
- 8 GN-VQ runs per K over 4 K: 7 `rho` plus the final one, on all splats, an upper bound for the quantized subset;
- 28 SH-only scoring rows;
- 2 arms x 4 K x (both C3DGS runs + decoding + protocol ii).

**How C3DGS's runs scale:** they are taken as flat (train's times) to linear in the splat count. Their clustering
uses fixed batch sizes, and their fine-tuning and evaluation render all splats.

**What is not counted:**
- downloads: the dataset took 352.0 s and 66.6 s for train in the two E3q attempts, and INRIA's members 28.7-141.2 s
  in E3p;
- the C3DGS build (204.3 s per session);
- warm starts other than C3DGS's own;
- the 16-coefficient extension's extra cost (section 2);
- C3DGS's clustering at K other than 4,096 (section 4).

**Per scene** (seconds):

| Scene | Splats | GN passes | GN-VQ, 32 runs | SH-only CV scoring | C3DGS runs, 2 arms x 4 K | **Total** | Full-pipeline CV instead of SH-only, extra |
|---|---|---|---|---|---|---|---|
| train | 1,026,508 | 7-27 | 3,393-6,823 | 141-224 | 8,467-8,467 | **12,009-15,541** | 9,845-9,845 |
| bicycle | 6,131,954 | 43-159 | 20,269-40,755 | 843-1,340 | 8,467-50,580 | **29,622-92,833** | 9,845-58,810 |
| bonsai | 1,244,819 | 9-32 | 4,115-8,274 | 171-272 | 8,467-10,268 | **12,762-18,846** | 9,845-11,939 |
| counter | 1,222,956 | 9-32 | 4,042-8,128 | 168-267 | 8,467-10,088 | **12,686-18,515** | 9,845-11,729 |
| kitchen | 1,852,335 | 13-48 | 6,123-12,311 | 255-405 | 8,467-15,279 | **14,858-28,043** | 9,845-17,765 |
| room | 1,593,376 | 11-41 | 5,267-10,590 | 219-348 | 8,467-13,143 | **13,964-24,123** | 9,845-15,282 |
| truck | 2,541,226 | 18-66 | 8,400-16,890 | 349-555 | 8,467-20,961 | **17,234-38,472** | 9,845-24,372 |
| drjohnson | 3,405,153 | 24-88 | 11,256-22,632 | 468-744 | 8,467-28,087 | **20,215-51,552** | 9,845-32,658 |
| playroom | 2,546,116 | 18-66 | 8,416-16,922 | 350-556 | 8,467-21,002 | **17,251-38,546** | 9,845-24,419 |

**The nine scenes together:**
- 150,601-326,470 s, i.e. 41.8-90.7 GPU-hours;
- on two T4s, no schedule shorter than 20.9-45.3 h. That is a floor, `max(total / 2, longest scene)`, as in
  FINDINGS section 13;
- (b)'s full-pipeline cross-validation would add 88,605-206,819 s (24.6-57.4 h).

**What drives it.** GN-VQ dominates everywhere except train, where the C3DGS runs do. Bicycle's upper end makes it
the longest scene, and it may not fit one 9.5 h session: its GN-VQ alone is 20,269-40,755 s, so it would need
splitting per K, as E3p's resume allows. The splat counts are E3_SCOUTING.md a's.

## 8. Scene sets

| Set | Scenes | Checkpoints | Status |
|---|---|---|---|
| development | train, bicycle | INRIA 30k, pinned (Amendment 12 a) | used by E3p (both) and E3q (train). Development scenes since Amendments 7 and 9 |
| gate | bonsai, counter, kitchen, room, truck | INRIA 30k | E2c's gate scenes, on this project's MCMC checkpoints. Untouched as INRIA checkpoints (Amendment 12 a) |
| gate | drjohnson, playroom (Deep Blending) | INRIA 30k | never used by this project |

**Notes for the pre-registration:**
- **Held-out means the checkpoints, not the scenes' images.** bonsai, counter, kitchen, room and truck informed
  G2c (FINDINGS section 12) through their images and this project's own checkpoints. The INRIA checkpoints are
  new; the images are not. Deep Blending is new in both.
- **Pins.** Only bicycle's and train's archive members are pinned (Amendment 12 a). The seven gate scenes' three
  members each (offset, sizes, CRC32) must be read from the archive's zip directory and pinned in the amendment,
  before any fetch **[not done]**.
- **Deep Blending data.** INRIA's `tandt_db.zip` holds it; E3q's downloader fetched only train's files from it
  **[measured: the job log's "305 files, 0.22 GB"]**. Its layout, and whether gsplat's COLMAP parser and E3p's
  camera-frame and split checks work on it, are **[unverified]**. Its `cfg_args` use full-resolution images
  (E3_SCOUTING.md a).
- **The frame check.** E3p's check against `cameras.json`, which stops a scene on failure, is required on every
  new scene.
- **Scale.** drjohnson (3.41M splats) and truck, playroom and kitchen are all above train's 1.03M. C3DGS has run
  only on train **[unverified elsewhere]**. Its peak was 4.55 GB **[measured]**.

## 9. Unverified or open, collected

1. The number of splats C3DGS colour-quantizes on each scene: the pruned count and `color_importance_n <= 0.6e-6`.
   The wrapper could record it.
2. Parity between gsplat's and INRIA's rasterizers on these checkpoints; the source of section 14's protocol gap.
3. The 16-coefficient GN pass (DC in `M_i`): its correctness check (the E0-style toy and end-to-end exactness
   checks would need re-running) and its cost.
4. `color_compress_non_dir = False` being broken, read from the code and not run.
5. The int8 table quantizer's exact behaviour at write time (observer updates), which the port's clip and final
   assignment depend on.
6. The rate range the four K span in C3DGS's `.npz`, and C3DGS's clustering time at K other than 4,096. Both
   are for the pilot.
7. The per-stream split of the `.npz` bytes.
8. Run-to-run spread of C3DGS's own pipeline (the scatter-add EMA, fine-tuning's camera draws), which no
   comparison here can be judged against yet.
9. Deep Blending's layout in the harness; the pins of the seven gate scenes.
10. A C3DGS-side evaluation of the uncompressed checkpoint in the same session.
