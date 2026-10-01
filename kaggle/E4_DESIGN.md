# E4 design: the OGC reproduction and the floor test inside C3DGS

**A design document, not a pre-registration.** Nothing here is a rule. It collects what E4's pre-registration has to
decide, with the facts and estimates behind each choice; the amendment that pre-registers E4 comes later, before any
E4 code or data. No code, Kaggle run or gate-scene data went into this document (2026-10-01). Inputs:
`kaggle/RELATED_WORK_OGC.md` (`04577841`), `kaggle/E3_C3DGS_DESIGN.md`, FINDINGS sections 13-15,
`kaggle/gn_e3r_memory/e3r_memory.json`, and the OGC code at `49ccae72` in a scratch clone (PolyForm Noncommercial:
E4 would call it, never copy it into this repository).

Labels as in `E3_C3DGS_DESIGN.md`: **[measured]** from a committed file, **[code]** read from code, **[estimate]**
computed here from measured inputs, **[source]** from a paper, **[open]** for Dace.

## 0. Three facts from the sources

1. **Index entropy (arXiv 2609.28997, Table 13, p. 19) [source].**
   - Their minibatch EMA k-means, C3DGS's own colour VQ, uses the 4,096-entry codebook "unevenly (about 10 bit of
     zero-order entropy per index)". The generalised Lloyd assignment uses it "almost uniformly (about 12 bit)".
   - K = 4,096; the same table lists rows at 2,048 and 1,024 entries. §4.1 (p. 6) puts it as about two bits less per
     Gaussian for their k-means "in their zlib container".
   - **What C3DGS stores per index:** int32 (`feature_indices ... .int()`, C3DGS `scene/gaussian_model.py:515-516`
     at `2a234af5`), in `np.savez_compressed` (`:575`), so deflate [code].
   - **Against FINDINGS' G1 result (section 9): same direction, different stream.** In E1 GN-VQ was larger at equal
     K, but "almost all of the difference is `centroids.npy`", the 6-bit codebook codes. `labels.npy` changed by +13
     to +27 bytes (garden) and -453 to -754 bytes (bicycle).
   - In OGC's C3DGS host the extra bytes are in the index stream (Table 13 caption: "only the colour-index stream
     differs").
   - E3r's +206,079 bytes (+1.485%, FINDINGS section 15) were not broken down by array: E3r kept no per-array sizes.
     Which stream carries them is not known.
2. **OGC's VQ entry point, callable on our tensors without their renderer [code].**
   - `vq.gram_kmeans(X, G, K, metric="gram", iters=20, device="mps", chunk=150000, seed=0, verbose=False, lam=1e-3)`
     (`vq.py:17`). `vq.py` imports only `time`, `numpy` and `torch` (`vq.py:13`).
   - **Inputs:** `X [n, 3, q]` and `G [n, q, q]`, CPU float32 (`vq.py:18`).
   - **Outputs:** codebook `C [K, 3, q]` and assignment `[n]` on the CPU (`vq.py:18, 88`).
   - **Our tensors map directly:**
     - C3DGS's colour input `[n, 48]` (index `k * 3 + channel`) becomes `X = x.reshape(-1, 16, 3).permute(0, 2, 1)`,
       as their host does (`hosts/c3dgs_run.py:50`);
     - the packed 16 x 16 metric becomes `G = gn_metric.unpack(M16)`;
     - their codebook goes back with `C.permute(0, 2, 1).reshape(K, 48)` (`hosts/c3dgs_run.py:62`).
   - **The ridge** is the `lam` argument: ρ = lam · tr(Σ G_i)/q, toward the cluster mean (`vq.py:63-73`).
   - **`plugin.gram_vq`** (`plugin.py:92-98`) wraps the same function but cannot set `lam`. It also imports their
     renderer modules at load time (`plugin.py:20`), though it calls none of them; calling `vq` directly avoids both.
3. **Arc-held-out checkpoints: not found [code].** The repository has the scripts:
   - `train/ogc_train.py:68` (`install_arc_split`, `--split arc --arc_deg`);
   - `arc_eval.py`;
   - `train/ogc_eval.py:69`.

   No `.ply` or `.pt` file is in the repository, and the README's data section (`README.md:50-56`) points only to
   INRIA's `models.zip` and the datasets. Using the arc protocol would mean training those models ourselves.

## 1. Question

- **Primary:** inside C3DGS, at its default operating point, does GN-VQ with the cross-validated floor (`rho_cv`)
  give better test-view quality than:
  - the same GN-VQ without the floor (full Gram, ρ = 0);
  - OGC's released Gram VQ (ridge toward the cluster mean)?

  And is any such difference larger than C3DGS's own run-to-run spread?
- **Secondary:**
  - the full Gram against the scalar tr(M_i) weighting;
  - whether the differences survive C3DGS's 5,000-iteration fine-tuning;
  - rate-distortion (BD) over the colour-importance threshold.
- **Why now:**
  - E3r showed the injected row cannot be read as GN-VQ's effect, because seeded C3DGS runs differ on the GPU in
    geometry and in C3DGS's own warm start (FINDINGS section 15);
  - OGC reports the matrix-against-scalar gain without a floor, without repeats and with a ridge whose size differs
    between paper and code (RELATED_WORK_OGC b, g; open questions 1, 3).

  E4 holds geometry, warm start and keep mask fixed across rows by construction (section 3), and measures the spread
  with repeats (section 4).

## 2. Rows per point (one C3DGS process)

| Row | Colour codebook | Warm start | Settings |
|---|---|---|---|
| R1 `c3dgs` | C3DGS's own `vq_features` | - | C3DGS defaults (K = 4,096 at the default point) |
| R2 `ogc` | `vq.gram_kmeans(X, G, K, metric="gram", iters=15, lam=1e-3, seed=0, chunk=100000)` on our M16 | **its own** (tr-weighted sample, `vq.py:31-33`): it takes no initial codebook | as their C3DGS host runs it (below) |
| R3 `gnvq_rho0` | GN-VQ (E2's variant, eps = 1e-2, at most 20 iterations, the 1e-3 rule, clip, final assignment against C3DGS's int8 table) on M16 | R1's codebook and labels | ρ = 0 |
| R4 `scalar` | the same GN-VQ with M_i replaced by tr(M_i)/16 · I | R1's | the scalar weighting (OGC's `ours_scalar` analogue) |
| R5 `gnvq_cv` | the same GN-VQ on M16 floored at `rho_cv` | R1's | Amendment 14 b's SH-only cross-validation |
| context: R2b `ogc_512` | R2 at K = 512 | its own | OGC's rate-matched row (their Table 1) |
| context: R2c `ogc_lam1e-6` | R2 with `lam=1e-6` | its own | the paper's stated ridge |

- **Same for every row, by construction:** the keep mask (C3DGS's colour importance and threshold, computed once),
  the quantized set, the geometry VQ (computed once, after the colour step; section 3) and the container.
- **Same warm start:** R3-R5 start from R1's codebook. R2 cannot, because `gram_kmeans` has no initial-codebook
  argument and E4 does not modify their code. So R2 differs from R3 in init and stopping as well as in the
  regulariser, and R2 against R3 is "their released VQ against ours", not a clean ridge-against-floor contrast.
  - The clean contrast of the regulariser is R5 against R3.
  - A ridge-to-mean variant inside our GN-VQ would isolate it, at the cost of a new variant: **[open]**, see 9.
- **R2's ridge is the code default, `lam=1e-3`, because it is what their released host runs:**
  - `hosts/c3dgs_run.py:61` passes no `lam`;
  - the code comment (`vq.py:63-68`) says the ridge keeps codewords from saturating C3DGS's int8 quantiser.

  The paper's 1e-6 (p. 14) goes in as context row R2c. Iterations: 15 (`hosts/c3dgs_run.py:76`). K: C3DGS's
  `codebook_size`. Chunk: 100,000 (`hosts/c3dgs_run.py:61`). Seed: 0 (`vq.py:17`). Device: CUDA.
- **R2's metric is ours, not theirs.** R2 runs on our M16, a 16-probe estimate equal in expectation to their S2
  Gram (RELATED_WORK_OGC section 3). This reproduces their VQ, not their whole statistic pipeline (their PyTorch
  renderer, exact weights, centred principal point). Computing their A_i with `plugin.observation_gram` is
  possible but costs their renderer's forward pass per scene: **[open]**, see 9.
- **Rates:** every row is written by C3DGS's own `save_npz`, so bytes compare directly. E4 also records the
  per-array deflate size and the index entropy of every row (OGC's Table 13 measure, `hosts/c3dgs_size_breakdown.py`'s
  quantities, recomputed by our own code). That shows which stream carries a size difference (fact 1).

## 3. Fork mechanics in the hooks

E4 extends `kaggle/e3r_hooks.py` and `kaggle/e3q_c3dgs_run.py`; C3DGS's source stays unedited, as in E3q/E3r.

1. **Interception, as E3r** [code]:
   - `Hooks.install` (`e3r_hooks.py:70-90`) patches `compress_gaussians`, `compress_color`, `vq_features`,
     `compress_covariance` and `GaussianModel.save_npz`;
   - `compress_color` (`:115-134`) records the keep mask and the quantized ids;
   - `vq_features` (`:136-148`) runs C3DGS's own VQ first and holds its input `x`, `C0`, `L0` and the quantizer
     state.
2. **The five colour tables, at the colour hook.** Where E3r's `_inject` (`:195-239`) replaced `(C0, L0)` with one
   GN-VQ result, E4 computes R2-R5 (and the context rows) from the same `x`, M16 and `(C0, L0)`, stores each
   `(C_r, L_r)`, and **returns R1's `(C0, L0)`** to C3DGS. C3DGS's own state therefore proceeds exactly as R1.
   - The metric is loaded once and floored per row, as `_inject` does (`metric_store`, one device copy).
   - R2 reads the unpacked metric from host memory (`[n, 16, 16]` float32, 1,024 bytes per quantized splat; about
     1.9 GB on kitchen).
3. **One geometry VQ.** `compress_covariance` runs once (`:150-161`), after the colour step. It reads the
   covariances and the gaussian importance, not the colour table, so every row shares the same geometry codebook,
   indices and SHA-1. That removes E3r's confound by construction. E4 records the SHA-1 once and asserts it at
   every row's save.
4. **The snapshot,** taken after `compress_gaussians` returns (`:105-113`), before C3DGS's fine-tuning or save.
   Deep-copied to host memory:
   - the model's parameters (`_xyz`, `_opacity`, `_scaling`, `_scaling_factor`, `_rotation`, `_features_dc`,
     `_features_rest`, `_feature_indices`, `_gaussian_indices`);
   - the state dicts of every fake quantizer, observers included (`features_dc_qa`, `features_rest_qa`, and the
     others `get_*` use, since evaluation updates them);
   - the colour input `x`, the keep mask, and each `(C_r, L_r)`;
   - the global RNG states (Python, numpy, torch CPU and CUDA).

   **Why a deep copy rather than a re-index:** `save_npz(sort_morton=True)` reorders the model in place (C3DGS
   `scene/gaussian_model.py:450-451`, `_sort_morton` at `:735-752`), so a second row's labels would no longer line up with the in-memory order.
5. **Five encodes and evaluations.** C3DGS's own run then saves and evaluates R1 as usual, through the `save_npz`
   hook (`:85-89`, `:163-179`). After `runpy.run_path` returns (`e3q_c3dgs_run.py:248-252`), the wrapper loops over
   R2-R5. For each row it:
   1. restores the snapshot;
   2. installs the row's table with C3DGS's own `join_features` and `set_color_indexed`, as `compress_color` does;
   3. saves with C3DGS's `save_npz` to the row's path;
   4. evaluates with C3DGS's `render_and_eval`.

   It needs `scene` and the parsed parameters:
   - `scene` through a wrapper on `scene.Scene.__init__` installed with the hooks;
   - the parameters by re-parsing the same argv with C3DGS's `arguments` module;
   - `render_and_eval` and `finetune` from the globals that `runpy.run_path` returns.

   This must be shown first on the CPU stand-in (`bench/gn/dryrun/fake_c3dgs/`).
6. **Protocol ii** then decodes each row's `.npz` one at a time with `npz2ply.py`, as E3r did, in the harness
   process.
7. **Fine-tuning per row (secondary).**
   - **Procedure:** restore the snapshot, install the row's table, reset the RNG states to the snapshot's, then run
     C3DGS's `finetune` (5,000 iterations), `save_npz` and `render_and_eval`.
   - **Memory:** one fine-tune at a time; E3q and E3r measured no rise in peak with fine-tuning on train (4.55 / 4.71
     GB) [measured]. The snapshot lives in host memory (about 0.5 GB at kitchen's size) [estimate].
   - **Time:** 314.2 s of fine-tuning per row on train [measured], plus an evaluation.
   - **Pairing:** resetting the RNG pairs the camera order across rows, but GPU atomics still separate the rows
     (FINDINGS section 15). The fine-tuned differences therefore carry run-to-run noise that the primary's do not.
8. **rho_cv:** from a separate probe run per scene (C3DGS's own run with `--record`, as E3r's probe), then E3r's
   SH-only cross-validation in the harness. Every forked process then uses that one `rho_cv`. **[open]**, see 9:
   per-process CV costs about 3 x 720 s per scene on train's scale.

## 4. Repeats and spread

- **Three independent processes per scene at the default point**, seeded 0, 1 and 2 (seeding pairs nothing on the
  GPU, FINDINGS section 15, so distinct seeds make the independence explicit). Each process runs all rows.
- **Per row:** mean, standard deviation (n = 3) and range over the three processes, for every protocol-ii metric,
  C3DGS's own metrics and bytes. R1's spread is C3DGS's own run-to-run spread, which section 15 left unmeasured.
- **Per difference** (R5 − R3, R5 − R2, R3 − R4, R2 − R1, ...): computed within each process, where geometry, keep
  mask and, for R3-R5, warm start are shared. Then mean, sd and range over the three processes.
  - Each process's difference is paired.
  - The spread across processes is the variability from C3DGS's warm start and geometry, which is what "beyond
    run-to-run spread" has to be measured against.
- **Pooled over scenes:** the per-scene mean differences and their signs. The bar is **[open]**, see 9.

## 5. Knobs

- **Primary:** C3DGS's default point, K = 4,096 and `color_importance_include` = 0.6e-6, without fine-tuning.
- **Secondary BD:** the threshold sweep 0.6e-6 x 3^j, j = -2..2, one forked process per non-default point. On train
  this knob spans 2.49x in bytes against K's 1.02x (FINDINGS section 15). The default point's `rho_cv` would be
  reused at the other points; re-running the CV per point is **[open]**.
- **Context:** R2b, OGC's 512-entry codebook, their rate-matched point (Table 1: 100% of their bytes, +0.32 dB before
  fine-tuning, +0.01 dB after) [source].

## 6. Scenes and feasibility

**The scenes:** the INRIA 30k checkpoints of bonsai, counter, kitchen, room, truck, drjohnson and playroom, the gate
list of `E3_C3DGS_DESIGN.md` section 8. Every scene used in E2 or E3 is development.

**A conflict to resolve first [open].**
- E2's nine held-out scenes included bonsai, counter, kitchen, room and truck, and E2c gated on them, with this
  project's own gsplat MCMC checkpoints and the same test views.
- So under a literal "used in E2 is development", only drjohnson and playroom are untouched.
- HANDOFF's position so far is that the INRIA checkpoints are new data, which keeps the seven. The INRIA checkpoints
  of all seven have not been fetched, loaded or rendered by this project.

**Feasibility from committed metadata only.**
- **Inputs:**
  - splat counts: `kaggle/E3_SCOUTING.md`, from the archive's zip directory;
  - full-resolution image size and image count: `kaggle/tilequant_run4_analysis.py` `SCENE_META` (bonsai, counter,
    kitchen, room) and `kaggle/tilequant_run5_analysis.py` `TANDT_META` (truck), from dataset zip directories and
    `cameras.bin` headers read in runs 4 and 5.
- **What C3DGS loads:** INRIA's protocol, images_2 for the indoor Mip-NeRF 360 scenes and half resolution for
  Tanks&Temples. Train's 1959 x 1090 loaded as 980 x 545 in E3p [measured]; ceil(full / 2) for the others
  [estimate].
- **Nothing was read for this document:** no image, `.ply`, `cameras.json`, render or metric of any gate scene.
- **Deep Blending has no committed metadata.** Its camera count and sizes would need a read of `cameras.json` (and a
  `.ply` header for an exact splat count).
  - Whether such a read, of counts and sizes only, counts as touching is Dace's decision.
  - The recommendation: it does not, because it carries no pixel, splat value or metric. The pre-registration should
    record it either way. It was not done here.

**The model** is Amendment 14 g's (`bench/gn/e3r_memory.py`), C3DGS's sensitivity pass:
- 1,850 bytes per splat;
- the images, all views, float32 RGB, when on CUDA;
- 28 bytes per pixel of image state;
- 36 bytes per tile instance;
- the upper end adds train's unexplained 470 bytes per splat.

Tile instances in the worst view need the geometry, which this document did not read. They are bracketed from the
two measured scenes instead [estimate]:
- per pixel, 11.6 (bicycle) to 11.7 (train);
- per splat, up to 6.09 (train).

The colour-step bound adds the larger of our metric copy plus GN-VQ's update chunk (544 bytes per splat + 0.92 GB)
and R2's assignment chunk (100,000 x 4,096 float32 scores and their inputs, 1.76 GB). It is conservative: on train,
E3r's injected run peaked 1.4 MB above the probe run (4,714,224,640 against 4,712,849,408 bytes) [measured].

| Scene | Splats | Loaded size, views | Images on CUDA (GB) | Sensitivity peak, images on CUDA (GB) | images on CPU (GB) | + colour step, CUDA / CPU (GB) | Verdict |
|---|---|---|---|---|---|---|---|
| bonsai | 1,244,819 | 1559 x 1039, 292 | 5.68 | 8.70-9.29 | 3.02-3.62 | 11.06 / 5.38 | CUDA likely |
| counter | 1,222,956 | 1558 x 1038, 240 | 4.66 | 7.64-8.22 | 2.98-3.56 | 9.99 / 5.33 | CUDA likely |
| kitchen | 1,852,335 | 1558 x 1039, 279 | 5.42 | 9.57-10.44 | 4.15-5.03 | 12.38 / 6.96 | CUDA tight; CPU sure |
| room | 1,593,376 | 1557 x 1038, 311 | 6.03 | 9.70-10.45 | 3.67-4.42 | 12.24 / 6.21 | CUDA tight; CPU sure |
| truck | 2,541,226 | 979 x 546, 251 | 1.61 | 6.55-8.08 | 4.94-6.47 | 10.38 / 8.77 | CUDA likely |
| drjohnson | 3,405,153 | not in the repo | - | per-splat part alone 6.30-7.90 | - | - | unknown; needs the read |
| playroom | 2,546,116 | not in the repo | - | per-splat part alone 4.71-5.91 | - | - | unknown; needs the read |

- **"Tight":**
  - E3r's C3DGS runs reserved 1.14 times what they allocated (5.38 / 4.71 GB) [measured];
  - kitchen's and room's colour-step bounds times 1.14 come to about 14.0-14.1 GB of the T4's 15.64 GB.
- **The fallback:** E3r's pre-registered one, a run out of memory retried with `--data_device cpu` (Amendment 14 d).
  It would apply unchanged, and every scene fits with images on the CPU by this model.
- **Checking the model against the anchors** [measured]:
  - it explains train's E3q peak at 89.4% without the residual;
  - E3r's hooked runs peaked 4% above E3q's (4.737 against 4.551 GB).

## 7. Cost

**The method** [estimate]:
- The base is E3r's measured train times:

  | Part | Seconds |
  |---|---|
  | sensitivity | 15.8 |
  | C3DGS's clustering at K = 4,096 | 257.7, of which the geometry VQ is 121 by E3q's progress bars |
  | encode | 1.6 |
  | loading and C3DGS's own evaluation | 70.9 |
  | one injection | 101.3 (GN-VQ 98.0) |
  | fine-tuning | 314.2 |
  | decode and protocol ii per row | 19.6 |
  | the 16 x 16 GN passes and cache writes | 28.5 |
  | the CV with its scoring and calibration | 719.9 |
  | runner | 46.1 |
  | download | 219.9 |

- **Scaling:** flat (lower) to linear in splat count (upper); C3DGS's own evaluation by test-view pixels.
- **R2's time is an assumption:** 15 of their iterations at our per-iteration rate.
- **The two Deep Blending scenes are not estimated** (no metadata).

| Scene | Probe run | One forked process (5 rows) | Five separate C3DGS runs instead | Primary: setup, GN, CV, probe, 3 forked processes, decode and protocol ii | Fine-tuning secondary, one process, 5 rows | BD secondary, 4 points x 1 process |
|---|---|---|---|---|---|---|
| bonsai | 471-531 s | 1,622-1,762 s | 2,731-3,111 s | 7,245-7,894 s | 2,513-2,847 s | 7,647-8,208 s |
| counter | 435-489 | 1,439-1,565 | 2,551-2,894 | 6,488-7,073 | 2,333-2,634 | 6,693-7,198 |
| kitchen | 460-687 | 1,570-2,100 | 2,679-4,118 | 7,028-9,485 | 2,461-3,725 | 7,374-9,495 |
| room | 480-636 | 1,672-2,036 | 2,779-3,767 | 7,450-9,137 | 2,562-3,429 | 7,906-9,363 |
| truck | 336-752 | 934-1,907 | 2,058-4,697 | 4,407-8,914 | 1,840-4,158 | 4,067-7,958 |
| **5 scenes** | | | | **32,618-42,503 s (9.1-11.8 GPU-h)** | **11,709-16,793 s** | **33,687-42,221 s** |

- **The fork's saving:** one forked process costs 41-60% of five separate C3DGS runs (40-59% saved). Each extra
  row needs only its colour VQ, an encode and an evaluation; the load, the sensitivity pass, C3DGS's own colour VQ
  and the geometry VQ run once.
- **Sessions on two T4s:**
  - the primary is about 4.5-5.9 h of wall time plus setup (396 s in E3r), one Kaggle session;
  - with the fine-tuning secondary, about 6.2-8.2 h, still one session;
  - the BD secondary adds about the primary's cost again, a second session.

  The largest single job is kitchen's primary, up to 9,485 s (2.6 h).

## 8. What carries over from E3r unchanged

The wrapper and its chunked `eigh` / `det` (Amendment 13 g), seeding (14 f), the out-of-memory retry and deadline
(14 d), one device copy of the metric (14 g), the 16 x 16 metric with the floor `tr / 16` (14 b), SH-only CV over the
seven `rho`, protocol ii on the decoded `.ply`, and E3r's per-row records (quantizer states, labels, the lifted check).

## 9. Open decisions for Dace

Each lists options, then a recommended pick with the reason.

1. **Primary metric and bar.**
   - Options:
     - (a) protocol-ii test PSNR of R5 − R3, paired within process, mean over 3 processes per scene;
     - (b) the same on the SH-only test dMSE (E2b's fidelity measure), more sensitive to the codebook alone;
     - (c) BD-rate over the threshold sweep.
   - Bar options for (a):
     - the mean difference over scenes > 0 with ≥ 5 of 7 (or ≥ 4 of 5) scenes positive;
     - each scene's mean difference beyond twice its between-process sd;
     - the pooled mean beyond the pooled sd of R1 (C3DGS's own spread).
   - **Recommended: (a) with "pooled mean > 0, ≥ 5 of 7 scenes positive, and the pooled mean above C3DGS's own
     between-process sd", with (b) reported.** (a) is the quality question, the pairing keeps it cheap, and the last
     clause is the "beyond the spread" requirement written as a test.
2. **Number of repeats.**
   - Options: 2, 3 or 5 processes per scene.
   - **Recommended: 3.** It is the smallest number that gives an sd. 5 adds two processes per scene (each about
     900-2,100 s by section 7), about 45-55% more on the primary, and still fits one session.
3. **Fine-tuning.**
   - Options: in the primary; a secondary on one process per scene; out.
   - **Recommended: a secondary on one process per scene.** OGC's own gain shrank from +0.49 to +0.09 dB with it
     (+0.32 to +0.01 at matched rate) [source]. Keeping it out of the primary keeps the primary a test of the
     codebook. One process per scene costs 11.7-16.8k s in all.
4. **The arc protocol.**
   - Options:
     - include it on the gate scenes;
     - include it on one or two development scenes as exploratory;
     - leave it out.
   - **Recommended: leave it out of E4.** It needs models retrained with an arc held out: OGC released scripts, not
     checkpoints (fact 3). INRIA training on a T4 has not been measured here. It also tests extrapolation, not the
     host comparison. Revisit as its own pilot.
5. **Scenes that do not fit.**
   - Options:
     - E3r's pre-registered CPU-images fallback;
     - exclude them and report;
     - read Deep Blending's camera counts and sizes first and decide before the pre-registration.
   - **Recommended: read Deep Blending's counts and sizes before the pre-registration (if 6 says it is not
     touching), and keep the CPU fallback.** By the model every scene with known metadata fits with images on the
     CPU.
6. **Which scenes are held out** (section 6's conflict).
   - Options:
     - the seven, treating INRIA checkpoints as new data (HANDOFF's position);
     - only drjohnson and playroom, under a literal "used in E2/E3".
   - **Recommended: the seven.** The checkpoints, and so every metric E4 would measure, are new. State in the
     pre-registration that the test views were seen in E2/E2c with other checkpoints.
7. **rho_cv once per scene or once per process.**
   - **Recommended: once per scene, from a separate probe run, as E3r.** Per process costs about 2 x 720 s more per
     scene at train's scale, and ρ is a scene-level setting in E2c and E3r.
8. **R2's statistic.**
   - Options:
     - our M16;
     - their A_i from `plugin.observation_gram` (their renderer, exact weights).
   - **Recommended: our M16, with the difference stated.** It isolates their VQ from their renderer and is equal in
     expectation. Their A_i would add their renderer's forward pass over every training view per scene, not
     measured on a T4.
9. **A ridge-to-mean variant inside our GN-VQ.**
   - Options: add it as a sixth row; leave it out.
   - **Recommended: leave it out of the primary and decide after E4.** R5 against R3 already tests the floor, and
     R2 shows their released method. A sixth row costs about one more GN-VQ (about 100 s per process at train's
     scale).
10. **Calling PolyForm Noncommercial code.**
    - **Recommended:** fetch `ogc-3dgs` at `49ccae72` in the Kaggle session and import `vq.py` at run time; never
      vendor it. This project is non-commercial research, but whether that holds for Dace's use is Dace's call.
11. **Bytes per stream.**
    - **Recommended: record, for every row,**
      - each array's deflate size;
      - the colour-index entropy;
      - the number of codes used.

      These answer fact 1's open point for C3DGS and are report only.
