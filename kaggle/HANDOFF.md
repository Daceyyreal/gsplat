# Handoff: gsplat `PngCompression` benchmark (`bench/tilequant`, `bench/gn-vq`)

For a fresh session. Results live in `kaggle/FINDINGS.md`; this file covers how the work is organized.
Runs 1-5 are on `bench/tilequant`. E0 (a Gauss-Newton metric for shN), E1 (GN-VQ, G1 failed), E2
(GN-VQ on held-out scenes, G2a passed) and E2b (an isotropic floor on the metric, exploratory; done:
works in fidelity terms, not by the PSNR criterion) are on `bench/gn-vq`: see the E0, E1, E2 and E2b
notebook sections below and `kaggle/PREREG_GN.md`. E2c (GN-VQ with the cross-validated floor, gated by
G2c on bonsai, counter, kitchen, room and truck; Amendment 11) is done: **G2c passed**, and the method is
frozen (FINDINGS section 12, "The method, as frozen"). E3's C3DGS arm has an integration design
(`kaggle/E3_C3DGS_DESIGN.md`, not pre-registered), and its pilot E3r is built (below). **E3p**, an exploratory
engineering pilot on INRIA's bicycle and train 30k checkpoints (Amendment 12), is done: no step ran out of
memory, and its C3DGS build check failed before compiling (FINDINGS section 13). **E3q**, a smoke test of the
C3DGS host on train (Amendment 13): attempt 1 built C3DGS but both runs failed on a cuSOLVER batched-eigen
refusal (`kaggle/gn_e3q/attempt1/`). With the fix (Amendment 13 g: the wrapper chunks `torch.linalg.eigh` and
`torch.Tensor.det`), **attempt 2 ran both C3DGS runs, and E3q is closed** (`kaggle/gn_e3q/attempt2/`, FINDINGS
section 14). **E3r**, a pilot of the C3DGS host on train and bicycle before the C3DGS arm's pre-registration
(Amendment 14, with its notes f, g and h), **ran on Kaggle on 2026-09-30** ("E3r C3DGS pilot", imported at tip
`b27e1421`) **and is closed**: its bundle is committed (`kaggle/gn_e3r/gn3r/`, `9c28cc89`) and FINDINGS section 15
quotes it (`2eb091f9`, checked by `bench/gn/check_s15.py`, `5fda2ea7`). Every step ran; the colour threshold spans
2.49x in bytes, K 1.02x; seeded C3DGS runs are not reproduced on the GPU. Two overlapping preprints were found and read
(`kaggle/RELATED_WORK_OGC.md`), and the C3DGS arm became **E4** (`kaggle/E4_DESIGN.md`), with Dace's final decisions in
"Related work, paper direction and E4 decisions (2026-10-01)" below. **Amendment 15** (E4p and E4) is committed
(`573142ac`); its decisions are in the same entry. Note i (`85b43cff`: the header read, the seven scenes' archive pins,
Deep Blending's feasibility; its scripts `15f263a7`) and note ii (`fcd1914f`: report-only measures) are committed, and
**E4p ran** on Kaggle ("E4p C3DGS fork pilot"); its bundle is committed data only (`kaggle/gn_e4p/gn4p/`, `60c4839b`) and
was checked against Dace's readings in chat, not yet written up. **Dace decided to withdraw E4 before any E4 data**
(Amendment 16, not drafted) **and to run E4q instead**, an exploratory dissection on development scenes only; see "E4p
results check and the E4q pivot (2026-10-02)" below. **Next: draft FINDINGS section 16 (E4p) and `check_s16.py`, then
draft Amendment 16** (E4 withdrawn, E4q registered); both are shown to Dace before commit. No E4 or E4q data exists.

## Context and rules

Dace (ICCA 2026 paper "Beyond Naive INT8") is testing whether gsplat's `PngCompression` can be improved
upstream (nerfstudio-project/gsplat). There is no local NVIDIA GPU, so all GPU work runs on Kaggle
(2x T4).

Standing rules from Dace:

- **Numbers:** every number must come from result files or actual runs. Never estimate or fill in values.
  Tables of numbers Dace reported must be labeled as such.
- **History:** never rewrite pushed history (no rebase, amend or force-push on pushed branches). Push
  fast-forward only; check with `git merge-base --is-ancestor fork/<branch> <branch>` first. Ask before
  any destructive git operation.
- **Scope:** commit only to feature branches on the fork. Don't open PRs or post comments unless asked.
  Work only on the branch the current task names.
- **Git access:** there is no `gh` CLI and no Kaggle credentials. The unauthenticated GitHub API
  rate-limits quickly, so read issues and PRs on github.com pages.

## Repository and branches

Clone: `F:\gsplat`. Remotes: `origin` = nerfstudio-project/gsplat, `fork` = github.com/Daceyyreal/gsplat.
Upstream main at the time of this work: `28e794c`.

| Branch | Purpose | Status |
|---|---|---|
| `fix/png-empty-tensor` (`d1632e8`) | empty-tensor / `sh_degree=0` fix in `png_compression.py` + tests | upstream PR #1061; leave alone unless asked |
| `feat/png-tile-quantization` (`3ff67e8`) | `PngCompression(tile_size=, bits=)` | benchmark said no PR; leave alone |
| `feat/png-weighted-kmeans` (`61cd1baf`) | `gsplat/compression/kmeans.py` + `kmeans_backend` / `kmeans_weighting` / `kmeans_chunk_size`, off upstream main `28e794c`; `tests/test_kmeans.py`. Commits `9348e32` (backend + options), `a4c31082` (`kmeans_chunk_size`), `61cd1baf` (default flip, droppable) | **upstream PR [#1063](https://github.com/nerfstudio-project/gsplat/pull/1063), open** (opened 2026-09-18), measured by run 5. Keep this branch clean: library only (3 files). Upstream main had not moved on 2026-09-18. |
| `bench/tilequant` | both feat branches merged + benchmark code and results; never goes upstream | runs 1-5 done; head = `git log -1 fork/bench/tilequant`. The blog post links its FINDINGS, so keep those numbers stable. |
| `bench/gn-vq` | off `bench/tilequant` (`12f912eb`): the E0 / E1 / E2 / E2b / E2c pre-registration (Amendments 1-11), `bench/gn/` (GN metric, diagnostics, G0 / G1 / G2 rules, GN-VQ, E2b's and E2c's rules, the BD sensitivity script, smoke tests, scene fixtures), the E0, E1, E2, E2b and E2c jobs and notebooks, and E0's, E1's, E2's and E2b's results; never goes upstream | **E0 done: G0 passed** (2026-09-20, `kaggle/gn_e0/gn/`, FINDINGS section 8). **E1 done: G1 failed** on the size rule (`kaggle/gn_e1/gn1/`, FINDINGS section 9). **E2 done: G2a passed** (`kaggle/gn_e2/gn2/`, FINDINGS section 10). **E2b done** (Amendments 9-10, exploratory): fidelity criterion `works`, garden control held, PSNR criterion `does not work` (`kaggle/gn_e2b/gn2b/`, FINDINGS section 11). **E2c done: G2c passed** (Amendment 11; `kaggle/gn_e2c/gn2c/`, FINDINGS section 12); the method `gn_vq_cvfloor` is frozen. **E3p done** (Amendment 12, exploratory pilot; `kaggle/gn_e3p/gn3p/`, FINDINGS section 13). **E3q done** (Amendment 13, C3DGS smoke test): attempt 1 failed in cuSOLVER (`kaggle/gn_e3q/attempt1/gn3q/`, `59303486`), and attempt 2, with Amendment 13 g's fix, ran both runs (`kaggle/gn_e3q/attempt2/gn3q/`, `a6c6f725`; FINDINGS section 14). **E3r done** (Amendment 14 with notes f, g and h, C3DGS-host pilot; C3DGS steps on train only, bicycle's 16 x 16 GN passes; ran on Kaggle 2026-09-30 from `b27e1421`; `kaggle/gn_e3r/gn3r/`, `9c28cc89`; FINDINGS section 15, `2eb091f9`). E3: the C3DGS arm's integration design is in `kaggle/E3_C3DGS_DESIGN.md`; it became E4 (`kaggle/E4_DESIGN.md`), pre-registered with its pilot E4p by Amendment 15 (`573142ac`, notes i `85b43cff` and ii `fcd1914f`); **E4p built** (`kaggle/gn_e4p_scene.py`, notebook "E4p C3DGS fork pilot"), not run; no E4p or E4 data. `gsplat/` and `setup.py` are identical to the run-5 commit, so the run-5 wheel's key matches. Do not modify `feat/png-weighted-kmeans` (PR #1063) from here. |

Untracked local drafts (excluded in `.git/info/exclude`, never commit or post): `PR_DRAFT_weighted_kmeans.md`,
`PR_DRAFT_empty_tensor.md`, `ISSUE_566_COMMENT.md`, `ISSUE_787_COMMENT.md`.

## File map (`kaggle/`)

| File | Role |
|---|---|
| `build_tilequant_bench.py` | generates `tilequant_bench.ipynb`. **Edit the builder, never the JSON**; the committed notebook must equal the builder output (`python kaggle/build_tilequant_bench.py`). |
| `tilequant_bench.ipynb` | Kaggle notebook (build output) |
| `tilequant_sweep.py` / `tilequant_analysis.py` | run 1: tile / global / smooth-range PNG params; PLAS sort cache; `build_runner`, `evaluate`, `dir_sizes`, sanity gate, repo-row reader, plot style |
| `tilequant_shn.py` / `tilequant_shn_analysis.py` | run 2: shN codebook (`ShnBenchCompression`, `precomputed_kmeans`, decomposition P/S/F) |
| `tilequant_run3.py` / `tilequant_run3_analysis.py` | run 3: clustering levers (`lloyd`, `torchpq_kmeans`, `cluster_weights`, `decide_run3`) |
| `tilequant_run4.py` | run 4: one job per new MipNeRF360 scene (download, `mcmc.sh` training, uncompressed / baseline / gate / candidates), resumable per step |
| `tilequant_run5.py` | run 5: the same rows produced by **library** code (`PngCompression(kmeans_backend=...)`), Tanks & Temples download / training, clustering timing, peak GPU memory, `--cpu_smoke_json` |
| `tilequant_run5_analysis.py` | run 5: `parse_benchmark_sh` (also `mcmc_tt.sh`), `TANDT_META`, parity gate, `decide_run5` (the run-4 rule), tables (`run5_table` needs `repo_csv=`, the dataset's own results CSV, which also labels the reference row), cost table, estimate, plot |
| `tilequant_run4_analysis.py` | run 4: `parse_mcmc_sh`, `SCENE_META` (zip sizes, image sizes), `scene_plan`, `sanity_gate`, **pre-registered** `decide_run4`, `run4_table`, runtime estimate / session split, `plot_run4` |
| `FINDINGS.md` | results write-up, every number from the committed bundles |
| `E3_SCOUTING.md` | E3 scouting (2026-09-27, not pre-registered): the INRIA pretrained models (13 scenes, splat counts derived from the archive), the candidate hosts (C3DGS, LightGaussian, MesonGS++, POTR: where the SH VQ is, what GN-VQ needs, builds, protocols, published numbers and their MB / MiB units), compute estimates at INRIA scale, cost options, and the recommendation (C3DGS first; E3a feasible) |
| `gn_e3p_scene.py` | E3p, one scene per process (Amendment 12; exploratory, no verdict): INRIA's pinned members fetched and checked, a runner holding the INRIA model (world space not normalized; `cameras.json` frame check; render parity), the uncompressed model under protocols i and ii, the GN passes (moved to host memory), the PLAS sort, `upstream_l1` and `lloyd_wopa_area` at K = 65,536, `gn_vq_cvfloor` at K = 65,536 as E2c ran it with `M` in one GPU copy; every step timed with its peak GPU memory (`Steps`), an out-of-memory step recorded with where it failed and the rest still run; `summarize` writes the notebook's summary |
| `e3p_inria.py` | INRIA's archive for E3p: the pinned members (Amendment 12 a's table), the zip64 directory reader and range-request fetch with size / CRC32 checks and SHA-1s, the `.ply` loader (INRIA's channel-major `f_rest` to gsplat's `shN`), `cfg_args`, the camera-frame and split checks, INRIA's image size rule, 8-bit quantization and protocol ii's evaluation; the published PSNRs with their table |
| `build_gn_e3p_bench.py` / `gn_e3p_bench.ipynb` | E3p notebook, Kaggle title "E3p INRIA pilot" (build output; edit the builder, never the JSON). E0's-E2c's notebooks are left exactly as they ran |
| `gn_e3p/gn3p/` | the E3p results bundle, unpacked as downloaded (27 files, `25f2133a`); FINDINGS section 13 quotes it |
| `gn_e3q_scene.py` | E3q, one process (Amendment 13; no verdicts): E3p's pinned train members laid out as INRIA's model directory, the dataset, C3DGS's build into the session's Python (`e3q_c3dgs.build`), its `compress.py` with 0 and 5,000 fine-tuning iterations and its `npz2ply.py`, then this harness's protocol ii of the uncompressed and decoded models; E3p's `Steps` with every error caught (Amendment 13 f); `summarize` writes the notebook's summary |
| `e3q_c3dgs.py` / `e3q_c3dgs_run.py` | the C3DGS host: the build (pinned commit, `--no-deps`, README deviations, Amendment 13 b's two fallbacks), `layout_model`, `run_compress` (through the wrapper, which runs `compress.py` unchanged and records the process's peak GPU memory), `npz_to_ply`; C3DGS's published train numbers (arXiv 2401.02436v2, Table 9). From attempt 2 on, the wrapper's `ChunkedLinalg` replaces `torch.linalg.eigh` and `torch.Tensor.det` in its own process (Amendment 13 g: `batched.py`'s chunking, 8,192 per call, halving on refusal; the op's own memory layout kept; every call, reduction and float64 check recorded under `linalg_patch`) |
| `build_gn_e3q_bench.py` / `gn_e3q_bench.ipynb` | E3q notebook, Kaggle title "E3q C3DGS smoke" (build output; edit the builder, never the JSON), now attempt 2's (bundle `E3q_bundle_2.zip`). E0's-E3p's notebooks are left exactly as they ran |
| `gn_e3q/attempt1/gn3q/` | E3q attempt 1's bundle (`gn3q_bundle.zip`), unpacked as downloaded (6 files, `59303486`): C3DGS built, both runs failed in cuSOLVER; the uncompressed row is its only result |
| `gn_e3q/attempt2/gn3q/` | E3q attempt 2's bundle (`E3q_bundle_2.zip`), unpacked as downloaded (6 files, `a6c6f725`): all three rows `ok`; FINDINGS section 14 quotes both attempts |
| `gn_e3r_scene.py` | E3r, one scene per process (Amendment 14): `--build_only` builds C3DGS once per session (E3q's build); a scene job runs the probe run (C3DGS at K = 4,096 with its colour VQ recorded), the harness phase (runner, protocol ii of the uncompressed model, the 16 x 16 GN passes, the colour-quantized trace share, the SH-only cross-validation over 7 `rho`, the calibration), the injected run(s) at `rho_cv`, the other K and (train) the thresholds, then protocol ii of every decoded row one `.ply` at a time; an out-of-memory C3DGS run retried once on the CPU data device; a deadline; any error caught (E3q's `Steps`); `summarize` writes the notebook's summary |
| `gn_e3r/gn3r/` | the E3r bundle (`E3r_bundle.zip`, 61,230 bytes, SHA-1 `e108184f480479a475e31feca10220cf5a2e8c57`), unpacked as downloaded (17 files, `9c28cc89`); FINDINGS section 15 quotes it |
| `gn_e3r_memory/e3r_memory.json` | E3r's GPU-memory check (Amendment 14 g), computed locally by `bench/gn/e3r_memory.py` from the pinned train and bicycle checkpoints (SHA-1s equal to E3p's): tile instances per train view by the rasterizer's formulas, the tie to E3q's train peak, the bicycle projections, E3r's step budgets |
| `e3r_hooks.py` | E3r's hooks inside C3DGS's process, installed by E3q's wrapper (`--observe`, `--record`, `--inject`): counts, quantizer state, the geometry SHA-1; the probe record; GN-VQ injected in place of C3DGS's colour codebook after C3DGS's own `vq_features` ran; the save check (labels survive, table drift). Never calls `get_features` (its observers) |
| `build_gn_e3r_bench.py` / `gn_e3r_bench.ipynb` | E3r notebook, Kaggle title "E3r C3DGS pilot" (build output; edit the builder, never the JSON); bundle `E3r_bundle.zip`. E0's-E3q's notebooks are left exactly as they ran |
| `E3_C3DGS_DESIGN.md` | the C3DGS arm's integration design (2026-09-29, docs, **not pre-registered**): what C3DGS quantizes and its rate knobs, where GN-VQ goes in (the wrapper patches, the random streams, `M_i` with DC), `rho`'s cross-validation in the codec, the RD knob and BD-rate, fine-tuning's effect on an injected codebook, the protocol, a per-scene compute estimate, the scene sets, and the unverified items. Each claim is tagged [code] / [measured] / [estimate] / [unverified] |
| `gn_e4p_scene.py` | E4p (Amendment 15 b-f, notes i and ii), train only, two jobs. **`--job fork`**: INRIA's members, the dataset, the OGC clone (HEAD checked); the probe run (seeded 0, `--record`, its evaluation deferred); E3r's harness phase unchanged (`gn_e3r_scene.harness_phase`: runner, 16 x 16 GN passes, the SH-only CV giving `rho_cv`; its rows in `gn4p_cv_<scene>.csv` with E3r's columns); note ii's geometry, orbit reference renders and coverage; three processes (seeds 0, 1, 2) through the wrapper's `--fork`, each row's `.npz` measured (bytes, per-array sizes, index entropy), decoded and evaluated (protocol ii per view, note ii a's orbit fidelity), Amendment 15 d's attempt rules (`e4p.next_action`); the probe evaluated from its `.npz` after the first process. **`--job ogc`**: OGC's dependencies in an isolated `--target` only, their Table 19 rows, their exact Gram against our `M` (recomputed in the job). Every step with time, peak GPU memory and host RSS (process and children). `summarize` writes the notebook's summary (no verdict) |
| `e4p_hooks.py` | E4p's fork inside C3DGS's process (extends `e3r_hooks.Hooks`), installed by the wrapper's `--defer_eval` (the probe), `--fork CONFIG` or `--eval_npz NPZ`: rows 2, 2b, 3, 4, 5 at the colour call; `join_features`' inputs captured; the model, quantizers and RNG copied when `compress_gaussians` returns; at the save, `run_vq`'s frame gives `scene` and the parameters and its global `render_and_eval` is replaced by the fork's phase (every row restored from the copy, installed, checked, hashed and saved; then every row evaluated; then rows 1, 2, 5 fine-tuned from the copy); per-row time, GPU peak and host RSS; the report written after every row |
| `e4p_ogc.py` | OGC at run time from a pinned clone (`49ccae72`; never copied or bundled): `ensure_clone` (HEAD check), `load_vq` (as `ogc_vq`), `ogc_codebook` (rows 2 / 2b, their host's call), `plan_deps` (pip `--dry-run --report` against the session; a replacement skips Table 19, else `--no-deps --target`), `prepare_data`, `run_table19` (their `gram.py`, `shfit.py`, `run_exps.py --stage core`), the `exact_gram` subcommand (their `plugin.observation_gram`, its own process), `compare_gram` (Amendment 15 f's measures) |
| `build_gn_e4p_bench.py` / `gn_e4p_bench.ipynb` | E4p notebook, Kaggle title "E4p C3DGS fork pilot" (build output; edit the builder, never the JSON); bundle `E4p_bundle.zip`, arcname `gn4p/`. E0's-E3r's notebooks are left exactly as they ran |
| `gn_e4_note_i/` | Amendment 15 note i's scripts (`header_read.py`, `feas7.py`, `cost7.py`, `15f263a7`), the recorded header read (`header_read.json`) and their outputs; `feas7.py` and `cost7.py` regenerate their JSONs byte-identical, `header_read.py` re-reads the archive (network) |
| `HANDOFF.md` | this file |
| `run2/tilequant/` ... `run5/tilequant/` | results bundles (`results_bundle.zip` contents); each session restores the previous one, so files repeat (git stores them once). In `run5/`, `run5_tt_table.csv` and `rd_run5.png` were regenerated locally after the label fix `1b4d40f8` (see FINDINGS sources); the downloaded original is `~/Downloads/results_bundle (3).zip` |
| `PREREG_GN.md` | E0 / E1 pre-registration (`bench/gn-vq`): G0 rule, validity checks, exploratory scope, G1 for E1. Amendment 1: the toy check. Amendment 2: G0 over 9 codebooks per scene with tie-exempt pairs, and exact-assignment refines instead of the shortlist one. Amendment 3: the G0 verdict is the ranking alone (the ratio is reported as calibration), the end-to-end exactness check, the lifted-check criterion v2, and a proximal rise invalidating that variant instead of stopping. Amendment 4: the toy and end-to-end scenes are committed fixtures with pinned hashes (a correction: the CUDA toy check would have drawn a different scene from the simulated one), end-to-end preconditions read from gsplat's render, and a report-only probe-noise diagnostic. Amendment 5 (after G0 passed, before any E1 code): E1's GN-VQ variant, the one-sided size matching that G1's last sentence delegates, the reported secondaries and the exploratory ablations. Amendment 6 (2026-09-21, before any E1 run): two more exploratory rows, GN-VQ at ridge `eps` = 1e-3 and 1e-2, seed 0 at K = 65,536 on both scenes, not judged by anything; and the final codebook's own quantizer range logged beside the warm start's. Amendment 7 (2026-09-21, after E1's results, before any E2 code): G1 failed as pre-registered and is not amended; the project's deviation, stated as one (it continues on Amendment 5 d's rate-distortion evidence; garden and bicycle become development scenes); E2's variant (eps = 1e-2, 20 iterations), 9 held-out scenes, 4 configs x 4 K, G2a (gate) and H2b (reported). Amendment 8 (2026-09-21, before any E2 data): G2a's mean is over all 9 held-out scenes, a scene without a defined BD-rate entering with a substitute (a: `gn_vq` reaches the baseline's best PSNR for fewer bytes; b: the reverse; c: 0%), so it is always defined; exploratory `gn_vq_eps1e4` rows on garden and bicycle, not judged. Amendment 9 (2026-09-22, after E2's results, before any E2b code): from E2b on, BD measures are computed with the domain-scaled fit (`g2.bd_rate_scaled` / `bd_psnr_scaled`; the pre-registered quantity is unchanged, E2's values stand); E2b, exploratory with a pre-stated criterion: E2's GN-VQ with `M_i + rho tr(M_i)/15 I` in assignment and update, rho chosen by train-view cross-validation, on treehill, flowers and train with garden as the control (those three become development scenes). Amendment 10 (2026-09-25, before any E2b data): E2b's scenes become treehill, flowers and stump (the largest train-to-test growth of the gn_vq / `lloyd_trace` dMSE ratio; Amendment 9's PSNR ratio mixed in the cross term with the model's own error), train dropped (it stays a development scene); E2b also runs its own rho = 0 full-`M` rows, checked against E2's `gn_vq` rows; a fidelity criterion on test dMSE (R below its own rho = 0 in >= 5 of 6 cells, treehill's R at K = 65,536 below 1; garden within 5%); Amendment 9's PSNR criteria reported alongside with the cross-term caveat; neither gates. Amendment 10 f (2026-09-26, before any E2b data): a claim that the floor works needs both the fidelity verdict and the garden control. Amendment 11 (2026-09-26, after E2b's results, before any E2c code): E2c, `gn_vq_cvfloor` (E2b's floor at `rho_cv` from the grid {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3}, final codebook on E2's full `M`) on bonsai, counter, kitchen, room and truck, all gate scenes, K = 1,024-65,536, E2's rows as comparators and E2's warm starts with no fallback; gate G2c (BD-rate < 0 against `lloyd_trace` on all 5, mean BD-rate <= -5% against `lloyd_wopa_area`, BD-PSNR >= -0.01 dB against E2's `gn_vq` on every scene; domain-scaled fit). Amendment 13 (2026-09-28, after E3p's results, before any E3q code): E3q, an engineering smoke test of the C3DGS host on train with no verdicts: C3DGS `2a234af5` into the session's Python (no venv: no `ensurepip`) with `--no-deps`, every README deviation recorded, two pre-registered fallbacks (a current `plyfile`, a `<cstdint>` retry); its own `compress.py` with and without its 5k fine-tuning; sizes in MiB and MB, its own metrics, protocol ii on the decoded model, time and memory; C3DGS's published Table 9 as a sanity check. Amendment 13 g (2026-09-28, a note after E3q's attempt 1, before any attempt-2 code): attempt 1 failed in C3DGS's `extract_rot_scale` (`torch.linalg.eigh`, 3x3 float32, one batch of up to 1,030,604 on train); for attempt 2 the wrapper replaces `torch.linalg.eigh` and `torch.Tensor.det` with `batched.py`'s chunking (8,192 per call, halving on refusal, every reduction recorded), C3DGS unedited, recorded as a deviation, with a report-only float64 CPU check on a sample of the actual inputs; `det` was added by Dace's decision after the diagnosis. Amendment 14 (2026-09-29, after E3q's results, before any E3r code): E3r, an engineering pilot of the C3DGS host on train and bicycle with no verdicts; the planned 16 x 16 metric extension (bands 0-3, floor tr / 16) and how GN-VQ sits in C3DGS (after C3DGS's own vq_features, its int8 table quantizer, SH-only cross-validation, the final codebook inside the injected run); C3DGS's baseline at K = 1,024-65,536, the colour threshold 0.6e-6 x 3^j on train only (by the pre-run estimate), the colour-quantized trace share, GN-VQ injected at K = 4,096, and on train with 5k fine-tuning; failures, an out-of-memory retry, a deadline. Amendment 14 f (note, before any E3r code was committed): every E3r run seeded with 0 as C3DGS's own safe_state seeds, because compress.py seeds nothing. Amendment 14 g (note, after E3r's code was pushed, before any E3r run): a GPU-memory check from the code puts C3DGS's own sensitivity pass on bicycle at 11.80-17.05 GB, so every C3DGS step runs on train only, bicycle measures its 16 x 16 GN passes, and E3r's own steps hold one device copy of the metric. Amendment 14 h (2026-09-30, note, before any E3r run): a report-only pruned-splat trace count on train (C3DGS's prune mask from the probe run against tr(M16) == 0 exactly), bearing on rasterizer visibility agreement, not render parity. Amendment 12 (2026-09-27, after E3's scouting, before any E3p code): E3p, an exploratory engineering pilot with no verdicts, on INRIA's 30k checkpoints of bicycle and train only (Deep Blending and bonsai, counter, kitchen, room, truck untouched), the archive members pinned (offset, sizes, CRC32); two evaluation protocols; one GPU copy of `M`, method-neutral; step costs; a C3DGS build check; the 7-`rho` grid kept for E3; planned with their own pre-registrations: the C3DGS comparison (primary without fine-tuning, secondary with C3DGS's 5k-iteration fine-tuning) and INRIA's protocol for E3's cross-paper comparisons. Amendment 15 (2026-10-01, after E3r's results and the OGC related-work notes, before any E4p or E4 code): E4, a gate inside C3DGS on INRIA's bonsai, counter, kitchen, room, truck, drjohnson and playroom, with one forked C3DGS process sharing warm start, keep mask and geometry VQ across its rows (C3DGS's own VQ, OGC's `gram_kmeans` on our 16 x 16 metric as their host calls it, GN-VQ at rho = 0, the scalar tr(M) weighting, GN-VQ at `rho_cv` from a per-scene probe run, and a report-only OGC row at `lam` 1e-6), 3 processes per scene; co-primary D1 (`rho_cv` minus rho = 0) and D2 (`rho_cv` minus OGC) on protocol-ii test PSNR at K = 4,096, each passing on mean > 0, >= ceil(0.7 n) scenes positive and mean > 2 SE_noise, n >= 5; descriptive secondaries (pre- and post-fine-tuning differences, Gram vs scalar, per-array bytes and index entropy, a later threshold-sweep BD); E4p, the same on train with OGC's Table 19 code check and an exact-Gram comparison; the header-read rule; the order and session rules. **Never edit a rule after results exist**; add a dated amendment instead. |
| `gn_e0/gn/` | the E0 results bundle, unpacked as downloaded (22 files); FINDINGS section 8 quotes it |
| `gn_e1/gn1/` | the E1 results bundle, unpacked as downloaded (28 files); FINDINGS section 9 quotes it |
| `gn_e2/gn2/` | the E2 results bundle, unpacked as downloaded (91 files); FINDINGS section 10 quotes it. E2b's analysis cell reads its comparator rows from here |
| `gn_e2c/gn2c/` | the E2c results bundle, unpacked as downloaded (179 files, `b1163f24`); FINDINGS section 12 quotes it |
| `gn_e2b/gn2b/` | the E2b results bundle, unpacked as downloaded (80 files, `a3c0099a`); FINDINGS section 11 quotes it |
| `gn_e2/bd_sensitivity.json` | post-hoc sensitivity of E2's BD measures (`bench/gn/bd_sensitivity.py`, from the bundle only): the reproduction within tolerance, the exact cubic, PCHIP, the shN stream alone, monotonicity and sign-disagreement flags |
| `gn_e2b_scene.py` | E2b, one scene per process (Amendments 9 b and 10): per K, `gn_vq_floor_cv` at the four rho (floored `M` from the even-indexed train views, scored by dMSE on the odd-indexed ones), then `gn_vq_floor` at the four rho (floored full `M`, evaluated like E2's rows; rho = 0 reproduces E2's `gn_vq`); warm starts and `M` from E2's caches with recorded fallbacks; imports E0-E2's modules unchanged; resumable per (config, K, rho) |
| `gn_e2c_scene.py` | E2c, one scene per process (Amendment 11): per K, `gn_vq_cvfloor_cv` at the 7 rho (floored `M_even`, scored by odd-view dMSE), then `gn_vq_cvfloor` on E2's full `M` at `rho_cv` (the CV rows' argmin, read back from the CSV), evaluated like E2's rows; E2's `M` and warm starts only, no fallback; imports E0-E2b's modules unchanged; resumable per (config, K, rho) |
| `build_gn_e2c_bench.py` / `gn_e2c_bench.ipynb` | E2c notebook (build output; edit the builder, never the JSON). E0's-E2b's notebooks are left exactly as they ran |
| `build_gn_e2b_bench.py` / `gn_e2b_bench.ipynb` | E2b notebook (build output; edit the builder, never the JSON). E0's, E1's and E2's notebooks are left exactly as they ran |
| `gn_e2_scene.py` | E2, one scene per process (Amendment 7): `upstream_l1`, `lloyd_wopa_area`, `lloyd_trace` and `gn_vq` (eps 1e-2, 20 iterations) at K = 1,024-65,536, seed 0, plus `uncompressed`; `--configs gn_vq_eps1e4` adds Amendment 8's exploratory rows (garden and bicycle only). Checks the checkpoint's pinned sha1 first, downloads and later deletes its own scene, reuses E0's / E1's modules unchanged; resumable per (config, K) |
| `build_gn_e2_bench.py` / `gn_e2_bench.ipynb` | E2 notebook (build output; edit the builder, never the JSON). E0's and E1's notebooks are left exactly as they ran |
| `gn_e1_scene.py` | E1, one scene per process: the G1 baseline and GN-VQ at K = 65,536 for seeds 0-2, the two secondary weightings, the seed-0 K grid, the two ablations and an `uncompressed` row. Reuses E0's GN cache; resumable per (config, K, seed) |
| `build_gn_e1_bench.py` / `gn_e1_bench.ipynb` | E1 notebook (build output; edit the builder, never the JSON). E0's notebook is left exactly as it ran |
| `gn_e0_scene.py` | E0, one scene per process: render parity, GN pass (`gn_cache/<scene>.pt`), spectrum, Spearman, the 9 G0 codebooks (predicted vs measured, test and train GT metrics, reproduction fields at K = 65,536), the lifted-assignment check (gates only the refines), the ridge / proximal refines (a proximal rise marks that row invalid). Resumable per (scene, config, K, seed). |
| `build_gn_bench.py` / `gn_bench.ipynb` | E0 notebook (build output; edit the builder, never the JSON) |
| `../bench/gn/` | `sh_basis.py`, `gn_metric.py`, `batched.py` (chunked batched linalg and the finite check; the fix for the first Kaggle crash), `diagnostics.py` (spectrum, Spearman, predicted / measured, lifted exact assignment and its check, refines, end-to-end exactness check; since E3r the coefficient count is read from the metric's packed width, 15 or 16, and nothing changed at 15), `e3r.py` (E3r: C3DGS's int8 colour-table quantizer, GN-VQ with the floored 16 x 16 metric, the colour dMSE, the trace shares), `e3r_estimate.py` (Amendment 14 e's pre-run estimate and g's), `e3r_memory.py` (Amendment 14 g's memory check), `dryrun/fake_c3dgs/` (a CPU stand-in with C3DGS's call structure, for E3r's tests and dry run), `g0.py` / `g1.py` / `g2.py` (the G0, G1 and G2a / H2b rules as code), `gn_vq.py` (E1's variant and the codec's quantizer; `report_metrics` adds reporting only), `e2b.py` (E2b's floored metric, selection, the fidelity and PSNR criteria, the rho = 0 reproduction check and the Spearman), `e2c.py` (E2c's grid, `rho_cv`, G2c and its reported items), `metric_store.py` (E3p's one-GPU-copy layout of `M`, Amendment 12 a: host metrics, one floored device buffer, `HostMetric` reads for `quad_form`; bit-identical to E2c's path on the CPU), `bd_sensitivity.py` (E2's BD measures reproduced, the 50-digit exact cubic, PCHIP, the shN stream), `check_s11.py` / `check_s12.py` / `check_s13.py` / `check_s14.py` / `check_s15.py` (re-check every number in FINDINGS sections 11-15 against the committed E2b, E2c, E2, E3p, E3q and E3r bundles; exit 0 = no failure; all run by pytest), `e3p_estimate.py` (FINDINGS section 13's post-hoc cost estimate for the frozen method at 4 K on the 13 INRIA scenes, from E3p's, E2c's and E2's files), `e4p.py` (E4p's pure pieces: the rows and OGC's call, the fork's state copy / restore / hash, the RNG states, row 4's isotropic metric, `.npz` bytes and index entropy, Amendment 15 c's components, note ii's geometry, coverage and power check, `HostRss`, the attempt rules), `selftest.py` (the notebook's smoke tests; `--device cpu` is the CPU stand-in), `fixtures/` (the committed toy and end-to-end scenes, `.npz` + `.json`, Amendment 4) and `make_fixtures.py` (wrote them), `toy_render.py` (CPU renderer for tests), `toy_noise.py` / `.json` (Amendment 1), `test_gn.py` (130 CPU tests), `dryrun/` (`fake_env.py` with the shared CPU stand-in, the E0, E1, E2, E2b, E2c, E3p, E3q and E3r dry runs, the writer-parity check and `check_chunked_distance.py`, E2c's path with the chunked `direct_distance`; not collected by pytest) |
| `.gitignore` | ignores only the 64 run-5 bundle files that were unpacked flat into `kaggle/` by hand (anchored names; nothing deleted; the committed copy is `run5/tilequant/`) |

CPU dry runs are **not in the repo**. They live in the scratchpad of session `51b5c32d`:
`C:\Users\Dace\AppData\Local\Temp\claude\F--\51b5c32d-122a-4650-8d8a-533b96ba5785\scratchpad\ref\dryrun_*.py`
(run with `F:\gsplat\.venv\Scripts\python.exe`, `PYTHONIOENCODING=utf-8`): `dryrun_sweep`, `dryrun_analysis`,
`dryrun_notebook`, `dryrun_shn`, `dryrun_shn_analysis`, `dryrun_notebook_shn`, `dryrun_run3`, `dryrun_notebook_run3`,
`dryrun_run4_analysis`, `dryrun_run4`, `dryrun_notebook_run4`, `dryrun_run5_analysis`, `dryrun_run5`,
`dryrun_notebook_run5` (helpers: `dryrun_run3_rows.py`; the analysis dry runs read `..\r4\zip_listing.json`
and `..\r5\tandt_listing.json`, written by the `zip_listing.py` / `tandt_listing.py` next to them).
One-off checks in `..\r5\`:

- `check_identical.py`: defaults byte-identical to upstream main. Set `GSPLAT_REPO` to a checkout of
  the feat branch; it prints which module it imported and refuses any other.
- `check_parity.py`: library `weighted_kmeans` == the bench `lloyd`.
- `regen_run5_tables.py`: rebuilds `run5_table.csv`, `run5_tt_table.csv`, `run5_costs.csv` and
  `rd_run5.png` from the run-5 bundle with the current code, and recomputes the decision against the
  bundle.
- `run5_numbers.py`, `run5_sidecheck.py`: every run-5 number in FINDINGS and the PR draft.

Library tests:
- bench: `.venv\Scripts\python.exe -m pytest tests/test_compression.py` (52 passed, 1 skipped on CPU).
- feat branch: `tests/test_kmeans.py tests/test_compression.py` in a worktree of
  `feat/png-weighted-kmeans` (16 passed, 1 skipped: the CUDA test). The venv imports gsplat from the
  current directory, so run it from the worktree.

The run-4 / run-5 dry runs were updated for `run5_table(..., repo_csv=)` and now assert the
reference-row labels. Logs: `..\r5\after_fix_*.log`, all OK.

## Notebook

- **Config cell:** `RUN_MODE`, `ALLOW_WHEEL_BUILD`, `INPUT_ROOT`, `NOTEBOOK_T0`, scenes, and the helpers
  `sh`, `record_timing`, `run_on_gpus` (job i of a batch on GPU i, one CSV per scene), `run_gpu_queue`
  (next job on the first free GPU, `can_start` guard, no new job after a failure) and `write_bundle`.
- **Discovery cell (before any install):** walks `/kaggle/input` recursively for the anchors
  `results/`, `tilequant/` and `wheels/`, and restores all of them into `/kaggle/working`, least
  complete first, so the most complete output wins. In a resume mode it raises before any install if
  required inputs (or required rows) are missing, and prints the input tree.
- **Install cell:** uses a restored gsplat wheel whose key matches (gsplat/ tree + setup.py + torch +
  GPU arch). In a resume mode it never builds without `ALLOW_WHEEL_BUILD` (the build takes about
  73 min). Any change under `gsplat/` changes the key.
- **Run cells:** training, current-main gate and the cells of earlier runs are gated by `RUN_MODE`
  (if/else, never `raise SystemExit`).
- **Final cell:** writes `/kaggle/working/results_bundle.zip` (top-level csv/json/png files of
  `tilequant/`, arcname `tilequant/`).
- **Run modes:**
  - `full`: runs everything missing.
  - `run3`: needs the run-2 output.
  - `run4`: needs the run-3 output or a partial run-4 output, with the garden / bicycle rows it reuses.
    It reruns nothing from runs 1–3 and downloads no garden / bicycle data. It runs one queued
    `tilequant_run4.py` job per unfinished new scene of the current session in `run4_plan.json`. The plan
    is computed once and then fixed.
  - `run5` (default): needs the run-4 output or a partial run-5 output. Parity gate first (it raises and
    stops the run if the library does not reproduce the run-3 rows), then MipNeRF360 and Tanks & Temples
    jobs. **It builds the gsplat wheel** (about 73 min): run 5 changes `gsplat/`, so the restored wheel's
    cache key no longer matches. `ALLOW_WHEEL_BUILD` only gates run3 / run4. **Done** on 2026-09-18 in
    one session; no further Kaggle run is planned.
- **Kaggle steps:**
  1. Import the notebook from
     `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/tilequant/kaggle/tilequant_bench.ipynb`.
  2. Attach the previous session's notebook output as input.
  3. Set GPU T4 x2 and Internet on.
  4. Save & Run All.
  5. Bring back `results_bundle.zip`. It has landed in `~/Downloads` (sometimes as
     `results_bundle (1).zip`), not the repo, so search there.

## E0 notebook (`bench/gn-vq`, `kaggle/gn_bench.ipynb`)

**Kaggle steps:**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_bench.ipynb`.
2. Attach the **run-5 notebook output** as input. It has the garden / bicycle checkpoints, the seed-0
   sort caches (`tilequant/sweep/<scene>/cache`), the run-3 clustering caches
   (`tilequant/run3/<scene>/kmeans/<name>_s<seed>.pt`), `run3_results.csv` and the gsplat wheel. To
   resume a partial E0 run, also attach that session's output (`gn/`, `gn_cache/`, `gn_work/`).
3. Set GPU T4 x2 and Internet on.
4. Save & Run All.
5. Bring back `/kaggle/working/gn_bundle.zip` (look in `~/Downloads`).

**What it does:**

- **Restore:** copies only the garden / bicycle checkpoints, sort caches, run-3 caches,
  `run3_results.csv` and the wheel. It raises before any install if a checkpoint or sort cache is
  missing, and warns if the run-3 caches are missing; the configs are then re-clustered, and TorchPQ
  reproduces run 3 only to about 0.002 dB.
- **Install:** clones `bench/gn-vq`. It reuses the wheel if the key matches (it should: `gsplat/` and
  `setup.py` are unchanged since run 5), and builds with `MAX_JOBS=2` otherwise
  (`ALLOW_WHEEL_BUILD = True`). It installs torchpq + cupy only in case a TorchPQ clustering must be
  recomputed. PLAS is not needed: E0 uses the cached order with `use_sort=False`.
- **CUDA smoke tests** (`bench/gn/selftest.py --device cuda`) go to `gn/gn_selftest.json`.
  - **First, before anything is rendered:** the committed scene fixtures must hash to their pinned
    values (Amendment 4): `bench/gn/fixtures/toy_scene_seed0` (`gn_metric.TOY_SCENE_SHA256`,
    `1bb442ee...`) and `e2e_scene_seed0` (`diagnostics.E2E_SCENE_SHA256`, `103c99e0...`). On a
    mismatch nothing else runs and the script exits with `FIXTURE HASH MISMATCH`.
  - The G0 validity checks: `sh_basis` (`sh_basis.cuda_check`, max abs error < 1e-5 against
    gsplat's `spherical_harmonics`); `toy_exactness` (`gn_metric.toy_exactness`, Amendment 1, on its
    fixture); `e2e_exactness` (`diagnostics.e2e_exactness`, Amendments 3-4: on the 48 non-overlapping
    fixture splats with exact `s_iv` and the fixture's +-0.1 shN perturbation, predicted = measured
    unclamped dMSE to 1e-4; the overlapping variants are reported only).
  - `e2e_exactness` first checks its preconditions on gsplat's own renders: one splat per pixel;
    each splat's rendered colour (SH-render value / identity weight at a pixel only it covers) in
    (0, 1); covered pixels in (0, 1), uncovered exactly 0. If one fails, its `status` is
    `preconditions_failed`, nothing is compared, and the run stops with "the exactness claim does not
    apply ... not a prediction mismatch". Otherwise `status` is `pass` or `mismatch`.
  - If anything fails, the script exits non-zero and the notebook stops before the data download
    and the scene jobs.
  - `linalg_scale`, an **engineering check** that also stops the run but is **not** a G0 validity
    entry: the job's own batched linalg at the job's sizes and dtypes (`eigen_stats` over 1,000,000
    packed PSD float64 matrices, `update_centroids` over 65,536 systems for both variants), the path
    that crashed the first run. It reports any batch reductions the helper had to make.
  - Report-only, outside every verdict: the toy check's `noise_diagnostic` (exact `sigma_rel` of the
    64-probe estimate of the sum and the normal-approximation false-fail probability of the 5% rule),
    logged before the check is judged; `lifted_random`, the lifted fp32 vs direct float64 assignment
    on random data (criterion v2).
- **Jobs:** garden on GPU 0 and bicycle on GPU 1 in parallel (`run_on_gpus`), each running
  `kaggle/gn_e0_scene.py`. Each job:
  - checks **render parity** first: the direct `rasterization` call against `Runner.rasterize_splats`
    with the eval's keyword arguments; it raises if the max abs difference is above 1e-6;
  - runs the GN pass and caches it;
  - writes the spectrum and Spearman files;
  - runs the **9 G0 codebooks** (Amendment 2): `upstream_l1`, `plain_l2` and `lloyd_wopa_area` at
    K = 4,096, 16,384 and 65,536. The run-3 caches cover K = 65,536; the other K are clustered here
    and cached in `gn_work/<scene>/clusters/<config>_k<K>_s<seed>.pt`;
  - checks the **lifted assignment** (Amendment 3, criterion version 2): 10,000 random splats, evaluated
    over those with `tr(M_i) > 0`; excess = direct distance at the lifted fp32 argmin minus the
    exhaustive float64 minimum; pass needs `sum excess / sum d_min <= 1e-4` and
    `excess_i <= 1e-4 * scale_i` per splat, `scale_i = |c_i - m|^2_{M_i} + tr(M_i) max_k |q_k - m|^2`
    (`m` = the codebook mean of the fp32 shift). If it fails, **only the refines are skipped**
    (`refines_skipped` in the meta); the job goes on to the `uncompressed` row, and the notebook to G0
    and the bundle. The G0 rows are already written by then;
  - runs the two **exact-assignment refines** of `lloyd_wopa_area` K = 65,536 (`gn_refine_ridge`,
    `gn_refine_prox`; excluded from G0). If the proximal objective rises by more than 1e-6 relative at
    any step, the rise is logged and that row gets `valid` = False (the variant still runs its 3
    iterations and is written); everything else continues;
  - adds an `uncompressed` reference row.
- **After the jobs,** in the same cell's `finally`: each job's exit code and the last 200 lines of its
  log go into `gn/gn_e0_<scene>_log_tail.json`, and then the bundle is written. This happens whether
  or not a job failed, so a crash can be read from the bundle without the working directory.
- **G0 cell:** `bench/gn/g0.py` (Amendment 3) on the rows, with the validity dict (`sh_basis`,
  `toy_exactness`, `e2e_exactness`, parity per scene, reproduction of the run-3 `lloyd_wopa_area`
  K = 65,536 seed-0 row when that row exists). It writes `gn/gn_g0.json`, prints the calibration
  summary and the refines' lifted-check / validity state, and plots `gn/gn_g0.png`.

**G0 rule (Amendment 3; the ratio was part of the verdict in Amendment 2 and no longer is):**

- Within each K, the 3 config pairs on train and test views of both scenes (36 pair checks). A pair
  whose clamped measured errors differ by less than 5% relative is a tie and exempt; every other pair
  must be ordered by P as by D (equal P counts as misordered).
- Verdicts: `incomplete` > `invalid` > `fail` (any misordered non-tied pair) > `inconclusive` (fewer
  than 6 non-tied pairs) > `pass`.
- Reported, not judged (`calibration` in `gn_g0.json`): per codebook `P / D` on train and test views,
  clamped and unclamped, the train ratios flagged `calibrated` within 0.5-2x (inclusive), and the
  cross/diagonal term ratio `D_train_unclamped / P - 1`. The ranking on unclamped measurements is
  reported too (`raw`).
- If `inconclusive`: G0 is re-judged in E1 over the non-GN rungs only (upstream L1, `lloyd_w1`,
  `lloyd_wopa_area`, C3DGS-style weights), same rule (`g0.judge_g0(..., configs=...)`).

**Outputs** (`gn_bundle.zip`, arcname `gn/`):

- `gn_results_<scene>.csv`: one row per (config, K, seed): the 9 G0 codebooks, `gn_refine_ridge`,
  `gn_refine_prox` and `uncompressed`. Columns:
  - `valid` / `invalid_reason` (False only for a proximal variant whose objective rose);
  - P and, for the refines, `objective_unquantized`;
  - D train / test (clamped and raw) and the ratios;
  - full-pipeline test metrics and train metrics (`train_PSNR` / `SSIM` / `LPIPS`), and shN-only GT
    metrics;
  - bytes per file (`file_bytes` JSON) and the `shN.npz` members (`shN_centroids_bytes`,
    `shN_labels_bytes`);
  - the clustering source, timings, and the run-3 reproduction fields (K = 65,536 only);
- `gn_meta_<scene>.json`:
  - settings, render parity, and the GN pass (views, pixels, clamp fraction, visible splats,
    zero-trace splats);
  - `render_range`: per view set and channel, the fraction of pixels of the original eval render
    below 0 / above 1 before clamping;
  - `finite_check`: the non-finite entry counts of `M` (the guard before any linalg);
  - `lifted_check`: `criterion_version` (2), `n_sample`, `n_zero_trace_in_sample`, `n` (evaluated),
    `sum_excess_over_sum_dmin`, `max_excess_over_scale` (the worst per-splat relative excess),
    `n_excess_over_tol_scale`, `n_dmin_below_1e-3_scale`, `n_dmin_zero`, `max_excess_over_dmin` (the
    Amendment-2 measure, for reference), `same_index_fraction`, `pass`, `time_s`;
  - `refines_skipped`, only when a failed lifted check skipped the refines;
  - `metric_parity`: the train-metric code run on the test views vs `Runner.eval`;
  - timings;
- `gn_spectrum_<scene>.csv`, `gn_spectrum_hist_<scene>.csv`, `gn_spectrum_<scene>.png`;
- `gn_spearman_<scene>.csv`;
- `gn_refine_ridge_<scene>.json`, `gn_refine_prox_<scene>.json` (absent if the refines were skipped):
  - the objective after every assignment and update step;
  - per assignment: labels changed, splats kept by the guard, and the top-64 L2 share (all splats and
    `tr(M) > 0`);
  - `valid`, `invalid_reason`, `objective_rises` (each rise: iteration, step, before, after) and
    `monotone_rtol`;
  - the final unquantized and quantized objective, and times;
- `gn_selftest.json` (`device`, `fixtures`, `sh_basis`, `toy_exactness` with `noise_diagnostic`,
  `linalg_scale`,
  `e2e_exactness` with `status` and its `non_overlapping` / `overlapping` /
  `overlapping_shared_delta` cases, `pass`, `lifted_random`),
  `gn_g0.json` (verdict, `clamped` and `raw` pairs, `calibration`), `gn_g0.png`, `timings.json`.

**Bundle contents** (`/kaggle/working/gn_bundle.zip`; the top-level csv / json / png files of `gn/`,
22 files for the two scenes when both jobs finish, 18 if both skip the refines):

- `gn/gn_g0.json`, `gn/gn_g0.png`, `gn/gn_selftest.json`, `gn/timings.json`
- per scene (`garden`, `bicycle`): `gn/gn_results_<scene>.csv`, `gn/gn_meta_<scene>.json`,
  `gn/gn_refine_ridge_<scene>.json`, `gn/gn_refine_prox_<scene>.json`, `gn/gn_spectrum_<scene>.csv`,
  `gn/gn_spectrum_hist_<scene>.csv`, `gn/gn_spectrum_<scene>.png`, `gn/gn_spearman_<scene>.csv`,
  `gn/gn_e0_<scene>_log_tail.json` (the job's exit code and the last 200 lines of its log, written in
  cell 6's `finally` whether or not the job failed, so a crash is readable from the bundle alone)

**Not bundled** (`gn_cache/` and `gn_work/` are siblings of `gn/`, and the bundle takes only files at
the top level of `gn/`): `gn_cache/<scene>.pt` (M is 1,000,000 x 120 fp32 = 480 MB per scene) and
`gn_work/` (runner stats, E0 clustering cache, job logs). Nothing deletes them (the final cell removes
only `/tmp/gn_runs`), so they stay in `/kaggle/working` and a later notebook can attach this output to
resume. The dry run checks that the bundle holds no `gn_cache/` entry and that the caches are still
there after the bundle cell.

**Local checks** (no GPU):

- `.venv\Scripts\python.exe -m pytest bench/gn/test_gn.py`: 41 CPU tests, among them the chunked
  batched linalg helper against the unchunked call (eigvalsh / solve / cholesky / inv / inv_ex, zero
  and rank-deficient matrices, a batch that never divides evenly), its halving retry, the finite
  check, the routed call sites; the lifted
  argmin on N = 2,000, K = 256 with rank-deficient and zero M_i against brute force; the v2 lifted
  criterion (passes fp32 near-tie rounding on rank-deficient M_i where d_min = 0, fails a wrong
  assignment, both branches, the version re-run rule); the end-to-end exactness check on the CPU
  renderer, its failure on a 0.1% render mismatch, and a failed precondition reported as such and
  not as a mismatch; the fixtures (pinned hashes, provenance against a fresh CPU draw, both
  provenance modes, a hash mismatch or a tampered file stopping everything before any render);
  `sigma_rel` against Monte Carlo; the CPU selftest and its non-zero exits; clusters with
  `tr(sum M) = 0` keeping `q_old`; a proximal rise recorded, not raised; every G0 verdict path and
  degenerate case under Amendment 3. The suite also passes with `ATEN_CPU_CAPABILITY=default`
  (the provenance test then compares within 1e-5 and warns with the capability difference).
- `PYTHONPATH=F:\gsplat .venv\Scripts\python.exe bench/gn/selftest.py --device cpu --out <file>`: the
  smoke tests on the CPU stand-in (the venv has no installed gsplat; the script never puts the source
  tree ahead of an installed wheel, which matters on Kaggle).
- In the repo, `bench/gn/dryrun/`. Run them with the venv python from any directory; they find the
  repo from their own location.
  - `python bench/gn/dryrun/dryrun_gn_e0.py`: the whole job on a 4,096-splat toy with a fake runner
    and the CPU renderer (K = 16 / 32 / 64, TorchPQ stand-in). Stages: (0) `selftest.py --device cpu`
    as a subprocess, including the end-to-end check; (1) both scenes, bicycle with an injected
    proximal rise (row marked invalid, job finishes); (2) resume; (3) a parity failure; (4) the
    notebook's G0 and bundle cells (smoke cell before the data and job cells, `e2e_exactness` in the
    validity dict, no `gn_cache/` in the bundle, the invalid variant bundled with its flag); (5) the
    lifted check: a wrong assignment skips only the refines, a failed or other-version record is
    re-run on resume, a current pass is reused. It reads `kaggle/gn_bench.ipynb`, so rebuild the
    notebook first after builder changes.
  - `python bench/gn/dryrun/check_writer_parity.py`: E0's writer produces byte-identical files to the
    run-3 writer for the same codebook.

  Both pass. Scratch files go to a fresh system temp directory that is deleted on exit;
  `GN_DRYRUN_KEEP=1` keeps the dry run's directory for debugging.

**After the run:**

1. Unpack `gn_bundle.zip` into `kaggle/gn_e0/` and check the files against the zip, CR-insensitively.
2. Read `gn_g0.json` first. Its verdict is `pass` / `fail` / `inconclusive` / `invalid` /
   `incomplete`, from the ranking alone; `calibration` holds the ratios and cross/diagonal ratios.
3. Check `gn_meta_<scene>.json` (`lifted_check`, `refines_skipped`) and the refine rows' `valid` before
   quoting any refine number.
4. Then fill FINDINGS section 8 from the files.

G1 is not judged in E0. **Done for E0:** the bundle is committed under `kaggle/gn_e0/gn/`, the
verdict was `pass`, and FINDINGS section 8 quotes the files. Nothing in E0 is left to run.

## E1 notebook (`bench/gn-vq`, `kaggle/gn_e1_bench.ipynb`)

**Done (run 2026-09-20, bundle committed 2026-09-21): G1 failed, on the size rule, on all six seeds.**
The bundle is unpacked unchanged in `kaggle/gn_e1/gn1/` (28 files) and FINDINGS section 9 quotes it.
Nothing in E1 is left to run; the notebook stays as it ran. What follows is how it was run.

E1 judges **G1**: does GN-VQ beat `lloyd_wopa_area` at equal size? The variant, the size matching, the
secondaries and the ablations were fixed in `PREREG_GN.md` **Amendment 5**, after G0 passed and before
any E1 code; **Amendment 6** (2026-09-21, before any E1 run) adds two more exploratory rows and one
logging field. E0's notebook, builder and job are untouched.

**Kaggle steps:**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_e1_bench.ipynb`.
2. Attach the inputs in the table below.
3. GPU T4 x2, Internet on. Save & Run All.
4. Bring back `/kaggle/working/gn1_bundle.zip` (look in `~/Downloads`).

**Inputs to attach, and what each is for.** Nothing else is needed, and nothing else is read: the
restore cell copies only these.

| Attach | Required | What E1 takes from it, and why |
|---|---|---|
| the **run-5 notebook output** | yes | the garden / bicycle checkpoints (`results/benchmark_mcmc_1M_png_compression/<scene>/ckpts/ckpt_29999_rank0.pt`), the seed-0 PLAS sort caches (`tilequant/sweep/<scene>/cache`), the run-3 `lloyd_wopa_area` clustering caches for seeds 0-2 at K = 65,536 (`tilequant/run3/<scene>/kmeans`) and the gsplat wheel (`wheels/`). The restore cell raises before any install if a checkpoint or sort cache is missing; a missing run-3 cache only warns, and those configs are re-clustered with the source recorded per row |
| the **E0 notebook output** | no, but attach it | **`gn_cache/<scene>.pt` only** - the GN metric, so E1 skips the GN pass (`gm.CACHE_VERSION` is unchanged, so the key matches). Without it each job recomputes the metric, about 12 s per scene. **No E0 result row is taken from it:** E0 writes its rows into `gn/`, and its work into `gn_work/`, neither of which E1 restores |
| **this notebook's own earlier output** | only to resume | `gn1/` (the rows already measured, the GN-VQ reports, the meta) and `gn1_work/` (the E1 clustering cache and the job logs). Every row resumes on its own |

**E0's rows cannot reach E1's results,** although `gn_cache/` is shared and E0's whole output is
attached. Five independent layers, each covered by the dry run's stage (4):

1. `discover` matches only the directory names `gn_cache`, `gn1` and `gn1_work`. E0's results are in
   `gn/` and its work in `gn_work/`, which match nothing, so they are never copied;
2. for the `gn1` and `gn1_work` slots it additionally refuses any directory holding E0 result files
   (`gn_g0.json`, `gn_selftest.json`, `gn_results_<scene>.csv`), and says so;
3. after restoring, the cell raises **before any install** if an E0 result file is sitting in `gn1/`;
4. `gn_e1_scene.assert_e1_csv` refuses a `gn1_results_<scene>.csv` whose header is not E1's own (E0's
   header has `refine_variant` and the `run3_*` columns), **before any GPU work**, so a foreign file
   can never reach the resume set or the CSV;
5. the G1 cell calls `g1.check_rows`, which refuses any scene or config that is not E1's, and
   `write_bundle` bundles only E1's own file names (`gn1_*`, `gn_e1_*_log_tail.json`,
   `timings.json`), warning about anything else rather than raising - it runs in the jobs cell's
   `finally` and must not hide a job failure.

None of these deletes anything. `gn_cache/` is the one deliberately shared input and holds the GN
metric, not rows.

**What it does:** the same smoke tests as E0 (`bench/gn/selftest.py`: fixture hashes, SH, toy,
end-to-end, and `linalg_scale`), then one job per scene in parallel (`kaggle/gn_e1_scene.py`), then the
G1 cell.

**Rows per scene** (`gn1_results_<scene>.csv`, 21):

- `lloyd_wopa_area` at K = 65,536, seeds 0-2: G1's baseline, from the run-3 caches (a missing seed is
  reclustered and the row records `recomputed`);
- `gn_vq` at K = 65,536, seeds 0-2: the variant G1 judges;
- `lloyd_trace`, `lloyd_c3dgs` at K = 65,536, seeds 0-2: the secondary weightings, reported only;
- `lloyd_wopa_area` and `gn_vq` at K = 4,096 and 16,384, seed 0: the rate-distortion grid;
- `gn_vq_noclip`, `gn_vq_noqassign`: the exploratory ablations, seed 0 (Amendment 5 e);
- `gn_vq_eps1e3`, `gn_vq_eps1e2`: GN-VQ at ridge `eps` = 1e-3 and 1e-2, seed 0 (Amendment 6). They
  differ from `gn_vq` in that one number only; `VQ_CONFIGS` names the ridge per config and the job
  passes it instead of `--eps`, so the pre-registered row keeps `vq.RIDGE_EPS` = 1e-4;
- `uncompressed`.

The four exploratory rows are reported only. `g1.py` picks G1's rows, the secondary comparisons and
the rate-distortion curves by config name, so none of them can enter a verdict; a test adds them with
absurd numbers and asserts the whole verdict object is unchanged.

Every row logs predicted vs measured error on train and test, the quantizer's range and step, the
fraction of centroid coordinates outside the warm-start range, the objective before and after
quantization, and `writer_codes_equal` (the job raises if the writer's codes differ from the ones
GN-VQ quantized). Train-view SSIM and LPIPS are **not** computed; train PSNR is.

Two quantizer ranges per GN-VQ row (Amendment 6): `quant_mins` / `quant_maxs` / `quant_step` are the
**final** codebook's own - the range the codec uses when that row is written - and `warm_quant_*` the
**warm start's**, so the two read side by side. With the clip on the first is inside the second by
construction, which the dry run asserts. `ridge_eps` records the ridge the row used, and
`gn1_meta_<scene>.json` carries `gn_vq.ridge_eps_by_config`.

**G1 cell** (`bench/gn/g1.py` -> `gn1_g1.json`): the verdict (`incomplete` > `fail` > `pass`), the
per-seed size rule and PSNR differences, the dominance flags, the two secondary comparisons and the
rate-distortion curves with BD-rate (`gn1_rd.png`). Only the GN-VQ vs `lloyd_wopa_area` comparison at
K = 65,536 over seeds 0-2 can pass or fail G1.

**Outputs** (`gn1_bundle.zip`, arcname `gn1/`): `gn1_results_<scene>.csv`, `gn1_meta_<scene>.json`,
one `gn1_<config>_k<K>_s<seed>_<scene>.json` per GN-VQ row (its history, the clip's rejections, the
quantizer range and both objectives), `gn1_g1.json`, `gn1_rd.png`, `gn1_selftest.json`,
`gn_e1_<scene>_log_tail.json` and `timings.json`. `gn_cache/` and `gn1_work/` stay in
`/kaggle/working`.

**Cost:** the secondary weightings are two fresh Lloyd clusterings per seed at K = 65,536, which in E0
took 400-640 s each, so they dominate the run; everything else is the 19 evaluations and the GN-VQ
iterations (E0's refines were about 10 s per iteration at K = 65,536).

**Local checks:** `pytest bench/gn/test_gn.py` (56 tests) and
`python bench/gn/dryrun/dryrun_gn_e1.py` (the whole job on the toy: all 21 rows, a reclustered seed,
the writer codes, the GN-VQ reports with each row's ridge and both quantizer ranges, resume, the
notebook's G1 and bundle cells, and stage (4), the E0/E1 output isolation on a fake `/kaggle/input`
holding both outputs). Rebuild the notebook with `python kaggle/build_gn_e1_bench.py` before the dry
run.

## E2 notebook (`bench/gn-vq`, `kaggle/gn_e2_bench.ipynb`)

**Done (rows timestamped 2026-09-21T23:03 to 2026-09-22T03:49; bundle committed 2026-09-22): G2a
passed**, 9 of 9 held-out wins, mean BD-rate -5.37%; H2b passed as computed and is counted 8 of 9
(treehill's win is a cubic-fit artifact). The bundle is unpacked unchanged in `kaggle/gn_e2/gn2/`
(91 files), FINDINGS section 10 quotes it, and the per-scene runtime is in the decision index below.
Nothing in E2 is left to run; the notebook stays as it ran. What follows is how it was run.

E2 is **pre-registered in `PREREG_GN.md` Amendment 7** (after E1's results, before any E2 code). G1
failed and is not amended; the project continues on the rate-distortion evidence Amendment 5 d
pre-registered for that case, which is a deviation and is stated as one. Garden and bicycle are now
development scenes; the gate is on **9 held-out scenes**. **Amendment 8** (before any E2 data) makes
G2a's mean always defined and adds exploratory eps = 1e-4 rows on the development scenes. E0's and
E1's notebooks, builders and jobs are untouched.

**Kaggle steps:**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_e2_bench.ipynb`.
2. Attach the inputs in the table below.
3. GPU T4 x2, Internet on. Save & Run All.
4. Bring back `/kaggle/working/gn2_bundle.zip` (look in `~/Downloads`).

**Inputs to attach, and what each is for.** The restore cell copies only these.

| Attach | Required | What E2 takes from it, and why |
|---|---|---|
| the **run-5 notebook output** | yes | the 11 MCMC-1M checkpoints: `results/benchmark_mcmc_1M_png_compression/<scene>/ckpts/ckpt_29999_rank0.pt` for the 9 MipNeRF360 scenes and `results/benchmark_tt_mcmc_1M_png_compression/<scene>/ckpts/ckpt_29999_rank0.pt` for train and truck; the 11 seed-0 PLAS sort caches (`tilequant/sweep/<scene>/cache`); the K = 65,536 clustering caches `tilequant/run3/<scene>/kmeans` (garden, bicycle) and `tilequant/run4/<scene>/kmeans` (the 7 other MipNeRF360 scenes), of which only `lloyd_wopa_area_s0.pt` and `manhattan_log_s0.pt` are copied; the gsplat wheel (`wheels/`). **Every checkpoint's sha1 is checked against Amendment 7's pin, and a missing checkpoint or sort cache or a wrong sha1 stops the notebook before any install** |
| the **E0 or E1 notebook output** | no | `gn_cache/garden.pt`, `gn_cache/bicycle.pt`: the GN metric, so those two jobs skip a GN pass of about 12 s each. No E0 or E1 result row is read: their rows are in `gn/` and `gn1/`, which match no restore slot, and an E2-named directory holding E0 / E1 result files is refused |
| **this notebook's own earlier output** | only to resume | `gn2/` (the rows measured so far, the GN-VQ reports) and `gn2_work/` (the E2 clustering cache, the job logs). Every (config, K) row resumes on its own. Only the first `gn_cache/` found is restored, so with several outputs attached a scene may redo its GN pass (about 12 s on garden); the cache key makes any mix safe |

**What it does:**

- **Restore** (above), then **install**: the restored run-5 wheel when its key matches, TorchPQ + cupy
  (for `upstream_l1`), the example dependencies.
- **CUDA smoke tests** (`bench/gn/selftest.py`, as in E0 and E1) -> `gn2_selftest.json`; a failure
  stops the notebook before any scene job.
- **Scene jobs** (`kaggle/gn_e2_scene.py`), each on the first free GPU in this order: train, truck
  (the longest jobs first), stump, bonsai, counter, kitchen, room, treehill, flowers, then garden and
  bicycle. Each job:
  - checks the checkpoint's sha1 and loads the cached sort order before any download or GPU work;
  - downloads its own scene (run 4's MipNeRF360 downloader, run 5's Tanks & Temples one; one download
    at a time) with the data factor of `mcmc.sh` / `mcmc_tt.sh` (4 outdoor, 2 indoor, 1 for T&T);
  - checks render parity, computes or reuses the GN metric, checks the lifted assignment once;
  - writes 17 rows, per K the G2a pair (`lloyd_wopa_area`, then `gn_vq` warm-started from it) first,
    then `lloyd_trace` and `upstream_l1`, then `uncompressed`;
  - deletes the scene's data at the end (garden and bicycle keep theirs for the exploratory phase).
  A failed job does not stop the others. No job starts after 9.5 h of notebook time, so a started
  Tanks & Temples job still ends inside Kaggle's 12 h. Each job's exit code and last 200 log lines are
  bundled.
- **Exploratory phase** (Amendment 8 b), a second queue that starts only after every scene job has
  finished, under the same start cutoff: `gn_vq_eps1e4` (E2's variant with eps = 1e-4) at the four K
  on garden and bicycle, 8 rows, then those two scenes' data is deleted. No verdict reads them.
- **G2 cell** (`bench/gn/g2.py` -> `gn2_g2.json`, `gn2_rd.png`): G2a (the gate), H2b (reported), and the
  reported extras, including the eps 1e-4 curves. Then the **bundle cell**, which raises last if a
  scene job failed.

**The verdicts (Amendment 7 f-g, `g2.py`):** per held-out scene, the BD-rate of `gn_vq` against the
comparator over the four points (degree-3 Bjontegaard, defined only if the curves share a PSNR
range): a win if below 0; if undefined, the BD-PSNR over the shared byte range decides (a win if above
0); if neither is defined, a loss; a missing row makes the scene incomplete. **G2a** (against
`lloyd_wopa_area`) passes with at least 8 of 9 wins and a **mean of at most -5% over all 9 held-out
scenes** (Amendment 8 a): a scene enters with its BD-rate, or, where that is undefined, with a
substitute: (a) the cheapest `gn_vq` point reaching the baseline's best PSNR at fewer bytes than the
baseline's best point, `-(1 - bytes_gn / bytes_base) x 100%`; (b) the cheapest baseline point
reaching `gn_vq`'s best PSNR at fewer bytes than `gn_vq`'s best point, `+(bytes_gn / bytes_base - 1)
x 100%`; (c) otherwise 0%. The substitutes feed only the mean, never the win count. On E1's garden
points (a) gives -9.77%. **H2b** (against `lloyd_trace`) holds with at least 7 of 9 wins; its reported
mean uses the same terms. Verdict order `incomplete` > `fail` > `pass`. Garden and bicycle are reported
and never read. `gn2_g2.json` records each scene's `mean_term` and `mean_term_source` (`bd_rate` or
`substitute_a` / `_b` / `_c`) and G2a's `n_substituted`.

**Outputs** (`gn2_bundle.zip`, arcname `gn2/`): `gn2_results_<scene>.csv` (E1's columns plus `dataset`,
`scene_set`, `data_factor`, `vq_max_iters`), `gn2_meta_<scene>.json` (`missing_rows` always counts the
full pre-registered set; the development scenes also have `missing_exploratory`),
`gn2_gn_vq_k<K>_s0_<scene>.json` and `gn2_gn_vq_eps1e4_k<K>_s0_<scene>.json`, `gn2_g2.json`, `gn2_rd.png`,
`gn2_selftest.json`, `gn_e2_<scene>_log_tail.json` and `gn_e2_<scene>_eps1e4_log_tail.json`,
`timings.json`. Not
bundled, left in `/kaggle/working` for a resume: the copied checkpoints and caches, `gn_cache/` (480 MB
per scene, 11 scenes) and `gn2_work/`.

**Cost, from measured parts:** per scene, 12 clusterings, 4 GN-VQ runs and 17 evaluations. Measured
so far: TorchPQ `upstream_l1` 99.0-105.4 s at K = 4,096 and 16,384 and 419.1-432.1 s at K = 65,536
(E0, FINDINGS section 8); library Lloyd 37.7-39.8 s at K = 4,096, 158.4-165.2 s at K = 16,384 and
266.6-655.7 s at K = 65,536 (E1); GN-VQ 61.8-62.7 s at K = 4,096, 78.8-80.8 s at K = 16,384 and
138.9-156.7 s at K = 65,536 for 9-10 iterations (E1); downloads 40.5-141.1 s per MipNeRF360 scene and
234.8 / 508.4 s for truck / train (run 5). The `lloyd_wopa_area` K = 65,536 clustering is cached for
the 9 MipNeRF360 scenes. By those numbers a MipNeRF360 scene takes on the order of an hour and a Tanks
& Temples scene more, so the queue should fit one session on two GPUs; **that is an estimate, not a
measurement**: nothing has run at K = 1,024, at data factor 1 for the GN pass and train PSNR, or with
eps = 1e-2 below K = 65,536. The start cutoff and per-row resume cover an overrun. The exploratory
phase adds 4 GN-VQ runs and 4 evaluations per development scene and no clustering: their warm starts
are the `lloyd_wopa_area` codebooks the main phase cached.

**Local checks:** `pytest bench/gn/test_gn.py` (71 tests; the G2 rules on every verdict path, every
Amendment 8 substitute branch, and E1's garden example) and `python bench/gn/dryrun/dryrun_gn_e2.py`
(8 stages: notebook structure, pins and the two-phase jobs cell; three scenes with all 17 rows; the
exploratory rows (refused on a held-out scene, after garden's 17, an exploratory-only run never
marking a scene done); resume; four refusals before GPU work; the G2 and bundle cells; the queue; the
restore cell on a fake `/kaggle/input`). Rebuild the notebook with `python kaggle/build_gn_e2_bench.py` first.

**After the run:** unpack `gn2_bundle.zip` into `kaggle/gn_e2/` (arcname `gn2/`) and check the files
against the zip; read `gn2_g2.json` first (`g2a.verdict`, then `h2b.verdict`); check `writer_codes_equal`,
`valid` and each meta's `missing_rows` before quoting a row; then write FINDINGS section 10 from the
files.

## E2b notebook (`bench/gn-vq`, `kaggle/gn_e2b_bench.ipynb`)

**Done (rows timestamped 2026-09-25T19:02 to 20:28; bundle committed 2026-09-26): fidelity criterion
`works` and garden control held, so under Amendment 10 f the floor may be described as working in
fidelity terms; PSNR criterion `does not work`** (treehill stays below `lloyd_trace` at both K). All 8
`rho = 0` cells reproduce E2's `gn_vq` rows `identical`. The bundle is unpacked unchanged in
`kaggle/gn_e2b/gn2b/` (80 files, `a3c0099a`), FINDINGS section 11 quotes it, and the measured runtime is
below. Nothing in E2b is left to run; the notebook stays as it ran. What follows is how it was run.

E2b is **pre-registered in `PREREG_GN.md` Amendments 9 and 10** (after E2's results, before any E2b code
or data). It is exploratory: its criteria are stated in advance but neither is a gate, and nothing about
E2's verdicts changes.

- **Variant:** E2's GN-VQ (ridge eps 1e-2, at most 20 iterations) with `M_i` replaced everywhere inside
  GN-VQ by `M_i + rho * tr(M_i) / 15 * I`, rho in {0, 1e-3, 1e-2, 1e-1}.
- **Selection:** `M_even` from the even-indexed train views; each codebook scored by its render-vs-render
  dMSE on the odd-indexed train views; `rho_cv` is the minimizer (ties to the smaller rho). No test view.
- **Full-`M` rows:** every rho, 0 included (Amendment 10 c), run with E2's full `M` and evaluated on the
  test views like E2's rows. The rho = 0 row is checked against E2's `gn_vq` row: `identical`,
  `within_tolerance` (|dPSNR| <= 1e-3 dB and relative test-dMSE difference <= 1e-3), `not_reproduced`
  (**flagged**), or `inputs_differ` when that row did not use E2's `M` and warm start.
- **Scenes (Amendment 10 b):** treehill, flowers and stump, the three held-out scenes whose dMSE ratio of
  GN-VQ to `lloyd_trace` grows most from train to test views in E2, and garden, the control; K = 4,096
  and 65,536; seed 0. Train is not used (Amendment 9 had chosen it by a PSNR ratio).
- **Fidelity criterion (Amendment 10 d), on test dMSE:** R = test dMSE of the full-`M` codebook at
  `rho_cv` / test dMSE of E2's `lloyd_trace` row. The floor works if R is below E2b's own rho = 0 R in
  at least 5 of the 6 cells of treehill, flowers and stump, and treehill's R at K = 65,536 is below 1.
  Garden control, reported with it: test dMSE at `rho_cv` at most 5% above garden's own rho = 0.
- **PSNR criteria (Amendment 9 b.e), reported alongside with the cross-term caveat:** treehill beats
  E2's `lloyd_trace` at both K; garden within 0.02 dB of E2's `gn_vq`. Computed as Amendment 9 wrote them,
  with E2's `gn_vq` row as the rho = 0 codebook; Amendment 10's items use E2b's own rho = 0 row.

**Kaggle steps:**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_e2b_bench.ipynb`.
2. Attach **both** inputs in the table below.
3. GPU T4 x2, Internet on. Save & Run All.
4. Bring back `/kaggle/working/gn2b_bundle.zip` (look in `~/Downloads`).

**Inputs to attach. Both are required.**

| Attach | Required | What E2b takes from it, and why |
|---|---|---|
| **E2's notebook output** (the session that produced `gn2_bundle.zip`) | yes | `gn_cache/<scene>.pt` for treehill, flowers, stump and garden: E2's full `M`, so the full-`M` rows use the same `M` as E2's rows and the rho = 0 reproduction check applies. E2's warm starts: `gn2_work/<scene>/clusters/lloyd_wopa_area_k4096_s0.pt` and `tilequant/e2_kmeans/<scene>/lloyd_wopa_area_s0.pt` (K = 65,536, the run-3 / run-4 cache E2 copied), for all four scenes. It also holds copies of the checkpoints, sort caches and the wheel. **Nothing in its `gn2/` is read:** E2's comparator rows come from the committed `kaggle/gn_e2/gn2/` in the cloned repo |
| the **run-5 notebook output** | yes | the four checkpoints (`results/benchmark_mcmc_1M_png_compression/<scene>/ckpts/`), their seed-0 sort caches (`tilequant/sweep/<scene>/cache`), the run-3 / run-4 K = 65,536 `lloyd_wopa_area` caches (`tilequant/run4/<scene>/kmeans` for treehill, flowers, stump; `tilequant/run3/garden/kmeans`; the warm-start fallback) and the gsplat wheel |
| this notebook's own earlier output | only to resume | `gn2b/` (rows, reports, metas), `gn2b_work/` (logs, any reclustered warm start) and `gn_cache_even/` (`M_even`); every (config, K, rho) row resumes on its own |

The restore cell raises **before any install** if a checkpoint or a sort cache is missing, a checkpoint's
sha1 is not Amendment 7's pin, E2's `M` or warm starts are missing, or no run-5 cache is found. The last
two have override flags in the config cell, `ALLOW_WITHOUT_E2_OUTPUT` and `ALLOW_WITHOUT_RUN5_OUTPUT`,
both False; set one only to run without that input on purpose. Without E2's output the full `M` is
recomputed (a new realization of the probe noise, not E2's `M`) and the warm starts come from the
run-5 caches or a reclustering; every row records its `m_source` and `warm_start_source`, and the
reproduction check then reports `inputs_differ`. Whether Kaggle still keeps E2's notebook output could
not be checked from here (no Kaggle credentials); E2's working directory held about 11 checkpoints,
11 GN caches of 480 MB and the clustering caches, so it should be within Kaggle's output limit.

**What it does:** restore, install (the restored run-5 wheel; no TorchPQ, E2b clusters nothing with it),
the CUDA smoke tests, one job per scene on the first free GPU in the order treehill, garden, flowers,
stump, with the 9.5 h start cutoff; then the E2b cell and the bundle cell. Each job
(`kaggle/gn_e2b_scene.py`):

- refuses a scene that is not E2b's, a wrong sha1, a missing sort cache or a results CSV that is not
  its own, before any download or GPU work;
- downloads its scene and deletes it at the end; checks render parity;
- loads E2's full `M` (key-checked, and compared with the key in E2's committed meta) or recomputes it;
  computes `M_even` with E0's GN pass over the even-indexed train views (cached in `gn_cache_even/`);
- runs the lifted-assignment check once per metric the pending rows use: `M_even` at the four rho and the
  full `M` at the four rho, 8 per scene;
- writes per K the four CV rows, then the four full-`M` rows (`gn2b_results_<scene>.csv`, 16 rows). CV
  rows carry the odd-view dMSE (`measured_odd_*`, the selection), the test-view dMSE (reported), `P` on
  `M_even` and the bytes; full-`M` rows carry E2's whole measurement (`P`, `D` on train and test views,
  test PSNR / SSIM / LPIPS, train PSNR, bytes). Every row also has the floored objective GN-VQ minimized
  and the unfloored one (`objective_M_*`), and its warm start's and `M`'s sources.

**E2b cell** (`bench/gn/e2b.py` -> `gn2b_e2b.json`, `gn2b_rho.png`): `rho_cv` per scene and K;
`verdicts` = `{fidelity, garden_control, psnr}` (`fidelity` and `psnr` are `works` / `does not work` /
`incomplete`); `criterion_fidelity` (R per cell, the count, treehill's R); `criterion_psnr` (with the
cross-term caveat); `reproduction_rho0` (the status per scene and K, and the `flagged` cells); the
Spearman correlations. The plot shows R and test PSNR against rho. `e2b.check_rows` and `g2.check_rows`
refuse any row that is not E2b's or E2's own.

**Outputs** (`gn2b_bundle.zip`, arcname `gn2b/`): `gn2b_results_<scene>.csv`, `gn2b_meta_<scene>.json`
(sources, lifted checks, `gn_full` / `gn_even`, `missing_rows`, timings), one
`gn2b_<config>_rho<rho>_k<K>_s0_<scene>.json` per row (64), `gn2b_e2b.json`, `gn2b_rho.png`,
`gn2b_selftest.json`, `gn_e2b_<scene>_log_tail.json`, `timings.json`.

**Cost, an estimate from E2's measured parts, not a measurement:** per scene, 16 GN-VQ runs (E2 measured
112.6-120.7 s at K = 4,096 and 134.6-155.2 s at K = 65,536 on these four scenes, about 1,980-2,210 s for
8 of each; the floor may change the iteration counts), 8 lifted checks (27.0-28.2 s each in E2), 16
evaluations (E2's job time not spent on clustering, GN-VQ, the lifted check, the download or the GN pass
came to 37.2-51.2 s per row on these scenes; CV rows skip the full-pipeline evaluation), a GN pass over
half the train views (E2's full pass took 8.4-12.8 s on the three scenes that ran one) and the download
(38.4-73.5 s). That is roughly 2,800-3,300 s per scene, so the four jobs, two rounds of two, should take
about 1.6-1.9 h on two GPUs.

**Measured E2b runtime per scene** (`kaggle/gn_e2b/gn2b/timings.json`, `gn2b_meta_<scene>.json`; the same
numbers are tabulated in FINDINGS section 11):

| Scene | Queue wall time `gn_e2b_<scene>_s` | `timings_s.job` (meta) | Download | GN pass, even views |
|---|---|---|---|---|
| treehill | 2,760.5 | 2,750.0 | 85.5 | 4.8 |
| garden | 3,090.5 | 3,073.7 | 164.8 | 5.9 |
| flowers | 2,790.5 | 2,777.4 | 90.6 | 6.8 |
| stump | 2,730.5 | 2,714.5 | 72.6 | 4.3 |

Seconds; the queue time is measured at the queue's 15 s poll, so up to 15 s late. Per session:
`checkpoint_sha1_s` 9.9, `restore_s` 28.8, `install_s` 158.4 (a restored wheel; no
`gsplat_wheel_build_s`), `selftest_s` 15.3. The per-scene estimate above was 2,800-3,300 s; the jobs took
2,730.5-3,090.5 s, the longest being garden, whose download took 164.8 s against 73.5 s in E2. The full `M`
came from E2's cache on all four scenes, so no full GN pass ran. GN-VQ took 91.0-119.5 s per row at
K = 4,096 and 136.4-154.7 s at K = 65,536, and each of the 32 lifted checks 27.0-28.8 s. All 4 jobs
exited with code 0 and none was skipped by the start cutoff.

**Local checks:** `pytest bench/gn/test_gn.py` (82 tests) and `python bench/gn/dryrun/dryrun_gn_e2b.py`
(8 stages: the notebook's structure and pins; E2 on the toy for the four scenes; the restore cell on a
fake `/kaggle/input` with an E2 and a run-5 output, including both refusals and their flags; all 16 rows
on 4 scenes with the fallbacks on flowers; resume; rho = 0 reproducing E2's toy GN-VQ report exactly and
E2b's own rho = 0 rows equal to E2's toy rows; refusals before GPU work; the E2b and bundle cells).
Rebuild the notebook with `python kaggle/build_gn_e2b_bench.py` first.

**After the run:** unpack `gn2b_bundle.zip` into `kaggle/gn_e2b/` (arcname `gn2b/`), check every file
against its zip entry, and make a data-only commit. Read `gn2b_e2b.json` first: `reproduction_rho0`
(anything `flagged`, or `inputs_differ`, qualifies every comparison with E2), then `verdicts`, then
`criterion_fidelity` and `criterion_psnr`; then each meta's `missing_rows`, `lifted_checks`,
`gn_full.source` and `warm_starts`, every row's `writer_codes_equal`, and each log tail's `exit_code` /
`skipped_by_cutoff`, before quoting a row. Then FINDINGS section 11 from the files, checked
mechanically, post-hoc items labelled.

## E2c notebook (`bench/gn-vq`, `kaggle/gn_e2c_bench.ipynb`)

**Done (rows timestamped 2026-09-26T17:57 to 21:45; bundle committed 2026-09-27): G2c passed**, all three
conditions. The floor was selected in all 20 cells (`rho_cv` 1e-3 to 1e-1, never the top of the grid); the
reproduction check is `not_applicable` everywhere, because no `rho_cv` was 0. The bundle is unpacked unchanged
in `kaggle/gn_e2c/gn2c/` (179 files, `b1163f24`), FINDINGS section 12 quotes it, and the measured runtime is
below. Nothing in E2c is left to run; the notebook stays as it ran. What follows is how it was run.

**The frozen method, `gn_vq_cvfloor`,** is stated in one place in FINDINGS section 12 ("The method, as
frozen"), and defined in `PREREG_GN.md` Amendment 11 b. E3 starts from it.

E2c is **pre-registered in `PREREG_GN.md` Amendment 11** (`14145093`, after E2b's results, before any E2c
code). It is a **gated** run: G2c.

- **Method `gn_vq_cvfloor`:** E2's GN-VQ (ridge eps 1e-2, at most 20 iterations, clip, final quantized
  assignment) with `M_i + rho * tr(M_i) / 15 * I` in the assignment and the update (E2b's
  `e2b.floored_metric`).
- **Selection, per scene and K:** `M_even` (E0's GN pass over the even-indexed train views, its own probe
  draws); one cross-validation codebook (`gn_vq_cvfloor_cv`) per `rho` in {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3},
  scored by render-vs-render dMSE on the odd-indexed train views; `rho_cv` is the argmin, ties to the
  smaller `rho`. CV codebooks also get the test-view dMSE (reported), never the full evaluation.
- **Final codebook (`gn_vq_cvfloor`):** E2's full `M` at `rho_cv`, from E2's warm start, evaluated like
  E2's rows. Run at every `rho_cv`, 0 included; at 0 it is E2's `gn_vq` again and its reproduction is
  reported (Amendment 10 c's statuses).
- **Scenes:** bonsai, counter, kitchen, room (MipNeRF360, data factor 2) and truck (Tanks & Temples, data
  factor 1), all gate scenes; K = 1,024 / 4,096 / 16,384 / 65,536; seed 0. 32 GN-VQ rows per scene.
- **G2c** (`bench/gn/e2c.py`, domain-scaled BD fit, 9-decimal rounding): (1) BD-rate against E2's
  `lloyd_trace` below 0 on all 5 scenes by G2a's rule; (2) mean BD-rate against E2's `lloyd_wopa_area` at
  most -5% with Amendment 8's substitutes; (3) BD-PSNR against E2's `gn_vq` at least -0.01 dB on every
  scene, an undefined BD-PSNR failing. `incomplete` > `fail` > `pass`. **Read Amendment 11 f before quoting
  a pass:** E2's own `gn_vq` rows already meet (1) and (2), so a pass with few `rho_cv > 0` cells says little
  about the floor.

**Kaggle steps:**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_e2c_bench.ipynb`.
2. Attach **both** inputs in the table below.
3. GPU T4 x2, Internet on. Save & Run All.
4. Bring back `/kaggle/working/gn2c_bundle.zip` (look in `~/Downloads`).

**Inputs to attach. Both are required, and there is no override flag** (Amendment 11 c): the restore cell
raises before any install if either is missing.

| Attach | Required | What E2c takes from it, and why |
|---|---|---|
| **E2's notebook output** (the session that produced `gn2_bundle.zip`; E2b used it on 2026-09-25) | yes | `gn_cache/<scene>.pt` for the 5 scenes (E2's full `M`); E2's warm starts `gn2_work/<scene>/clusters/lloyd_wopa_area_k<K>_s0.pt` for K = 1,024, 4,096 and 16,384 on all 5 scenes and for K = 65,536 on truck (E2 reclustered it), and `tilequant/e2_kmeans/<scene>/lloyd_wopa_area_s0.pt` for K = 65,536 on the 4 MipNeRF360 scenes. **Nothing in its `gn2/` is read:** the comparators are the committed `kaggle/gn_e2/gn2/` |
| the **run-5 notebook output** | yes | the 5 checkpoints (sha1 pinned by Amendment 7; the first copy found is used, E2's output holds copies too), their seed-0 sort caches, the gsplat wheel, and `tilequant/run4/<scene>/kmeans/lloyd_wopa_area_s0.pt` for the 4 MipNeRF360 scenes: its presence is how the cell recognizes this output, and each job records whether E2's K = 65,536 copy equals it (`warm_start_equals_run5_cache`, reported) |
| this notebook's own earlier output | only to resume | `gn2c/` (rows, reports, metas), `gn2c_work/` (logs) and `gn2c_cache_even/` (`M_even`); every (config, K, rho) row resumes on its own, and a final row is rerun from the CV rows already written |

**What it does:** restore, install (the restored wheel; no TorchPQ), the CUDA smoke tests, one job per scene
on the first free GPU in the order room, bonsai, kitchen, counter, truck (longest first by E2's measured
per-row cost), with the 9.5 h start cutoff; then the G2c cell and the bundle cell. Each job
(`kaggle/gn_e2c_scene.py`):

- refuses a scene that is not E2c's, a wrong sha1, a missing sort cache, a results CSV that is not its own
  (E2's and E2b's headers differ), and a missing E2 `M` or warm-start file, before any download;
- downloads its scene and deletes it at the end; checks render parity;
- loads E2's full `M` only if its key matches this checkpoint's and E2's committed meta (otherwise it stops
  before any GN-VQ run: there is no recomputation) and E2's warm starts, key-checked (no fallback);
  computes `M_even` (cached in `gn2c_cache_even/`);
- per K, writes the 7 CV rows, reads their odd-view scores back from its CSV, selects `rho_cv`
  (`e2c.select_rho_cv`) and writes the final row with `rho_cv` and the 7 scores in it;
- runs the lifted check once per metric: `M_even` at the 7 `rho`, and the full `M` at each distinct `rho_cv`
  (8-11 per scene). A failed check skips the rows using that metric, which leaves G2c `incomplete`.

**G2c cell** (`bench/gn/e2c.py` -> `gn2c_g2c.json`, `gn2c_rd.png`): the verdict, the three conditions with
their per-scene values, `rho_cv` per cell, the counts of cells with `rho_cv > 0` and at the top of the grid,
the reproduction statuses, and per cell and scene everything Amendment 11 e reports (dMSE ratios against
`lloyd_trace` and `gn_vq`, LPIPS / SSIM / bytes against `gn_vq`, the CV Spearman, the BD measures against all
four E2 curves, the monotonicity and sign-disagreement flags). `e2c.check_rows` and `g2.check_rows` refuse
anything that is not E2c's or E2's own; a final row whose `rho` is not its CV rows' argmin counts as missing.

**Outputs** (`gn2c_bundle.zip`, arcname `gn2c/`): `gn2c_results_<scene>.csv`, `gn2c_meta_<scene>.json`
(sources, lifted checks, `gn_full` / `gn_even`, `rho_cv`, `missing_rows`, timings), one
`gn2c_<config>_rho<rho>_k<K>_s0_<scene>.json` per row (160), `gn2c_g2c.json`, `gn2c_rd.png`,
`gn2c_selftest.json`, `gn_e2c_<scene>_log_tail.json`, `timings.json`.

**Runtime, an estimate from E2's and E2b's measured parts, not a measurement:**

- **GN-VQ:** E2 measured `gn_vq` on these scenes at 109.9-112.5 s (K = 1,024), 92.9-115.1 s (4,096),
  94.3-107.0 s (16,384) and 135.2-161.6 s (65,536). Eight runs per K give 3,620-3,846 s per scene. The floor
  shortened E2b's runs at K = 4,096, and `rho` above 1e-1 has never run, so this may be high or low.
- **Evaluation:** E2's per-row time outside clustering, GN-VQ, the lifted check, the download and the GN pass
  (with each cached clustering's recorded time left out) was 94.1-114.3 s on the four MipNeRF360 scenes and
  34.2 s on truck. On E2b's scenes a CV row cost 0.36-0.54 of that (derived from E2b's job times), so 28 CV
  rows and 4 final rows per scene.
- **Other:** lifted checks 25.9-28.5 s each in E2 on these scenes (8-11 per scene), half a GN pass for
  `M_even` (E2's full pass: 15.8-32.8 s), and the download (53.5-89.3 s MipNeRF360, 233.2 s truck).
- **Total:** about 5,330-6,230 s per MipNeRF360 scene and 4,800-5,050 s for truck. With five jobs on two GPUs
  in queue order, about 15,500-16,950 s (4.3-4.7 h) plus the session steps (E2b's took 212 s). That is well
  inside the 9.5 h start cutoff. Truck's CV evaluation at data factor 1 has no measured E2b counterpart.

**Measured E2c runtime per scene** (`kaggle/gn_e2c/gn2c/timings.json`, `gn2c_meta_<scene>.json`; the same
numbers are tabulated in FINDINGS section 12):

| Scene | Queue wall time `gn_e2c_<scene>_s` | `timings_s.job` (meta) | Download | GN pass, even views | Estimate (this build's, from the parts above) |
|---|---|---|---|---|---|
| room | 5,281.0 | 5,265.2 | 458.2 | 15.1 | 5,586-6,234 |
| bonsai | 5,821.1 | 5,811.0 | 911.2 | 16.2 | 5,385-5,986 |
| kitchen | 5,266.1 | 5,246.4 | 434.9 | 16.8 | 5,334-5,911 |
| counter | 5,161.1 | 5,145.4 | 371.7 | 16.2 | 5,349-5,898 |
| truck | 4,290.5 | 4,283.8 | 36.9 | 8.0 | 4,795-5,051 |

Seconds; the queue time is measured at the queue's 15 s poll, so up to 15 s late. Per session:
`checkpoint_sha1_s` 7.9, `restore_s` 41.1, `install_s` 168.5 (a restored wheel; no `gsplat_wheel_build_s`),
`selftest_s` 15.6.

- **Against the estimate:** four jobs came in at or below its low end; bonsai was inside it.
- **Downloads:** the MipNeRF360 downloads took 371.7-911.2 s, against 53.5-89.3 s for the same scenes in E2.
  Without them the jobs would have been well below the estimate.
- **GN-VQ:** 75.9-167.4 s per row. The floor shortened the CV runs, as in E2b.
- **Lifted checks:** 43 in all, 8-10 per scene, 25.9-29.8 s each.

All 5 jobs exited with code 0 and none was skipped by the start cutoff.

**Local checks:** `pytest bench/gn/test_gn.py` (88 tests: G2c on every condition and boundary, incomplete
cases, a final row at the wrong `rho`, the reported items, the job's constants, pins against run 5 and
Amendment 11's table, the refusals, and `check_s11.py`) and `python bench/gn/dryrun/dryrun_gn_e2c.py` (8
stages, see its docstring; 549 s on this machine). The dry run replaces the real downloaders with a function that raises, so a job whose fake data marker is missing fails instead of fetching the scene. Rebuild the notebook with
`python kaggle/build_gn_e2c_bench.py` first.

**After the run:** unpack `gn2c_bundle.zip` into `kaggle/gn_e2c/` (arcname `gn2c/`), check every file
against its zip entry, and make a data-only commit. Read `gn2c_g2c.json` first: `missing`, then `verdict`
and `conditions`, then `n_cells_rho_cv_above_0` (Amendment 11 f) and `reproduction_rho0`; then each meta's
`missing_rows`, `lifted_checks`, `gn_full`, `warm_starts` and `warm_start_equals_run5_cache`, every row's
`writer_codes_equal`, and each log tail's `exit_code` / `skipped_by_cutoff`, before quoting a row. Then
FINDINGS section 12 from the files, checked mechanically as `check_s11.py` does, post-hoc items labelled.

## E3p notebook (`bench/gn-vq`, `kaggle/gn_e3p_bench.ipynb`, Kaggle title "E3p INRIA pilot")

**Done (rows timestamped 2026-09-27T21:11 to 2026-09-28T02:32; bundle committed 2026-09-28): no step ran out of
memory; no verdict (none exists).** The bundle is unpacked unchanged in `kaggle/gn_e3p/gn3p/` (27 files,
`25f2133a`), FINDINGS section 13 quotes it (`08902af2`, re-checked by `bench/gn/check_s13.py`), and the measured
runtime is under "After the run" below. Nothing in E3p is left to run; the notebook stays as it ran. What follows
is how it was run.

E3p is **pre-registered in `PREREG_GN.md` Amendment 12** (`aa67e21e`, docs only, before any
E3p code). It is an **exploratory engineering pilot and produces no verdicts**: it measures whether the frozen
pipeline runs at INRIA scale and what each step costs, so that E3 can be designed and pre-registered from
measured parts. Its numbers inform E3's design; no claim about the method rests on them.

- **Scenes:** bicycle and train, both already development scenes, with **INRIA's 30k checkpoints** (not the
  gsplat MCMC checkpoints of runs 4-5). Deep Blending and bonsai, counter, kitchen, room and truck are not
  touched; neither are garden, stump, treehill and flowers.
- **The method:** `gn_vq_cvfloor` exactly as E2c ran it (Amendment 11 b), at K = 65,536 only, with the 7-`rho`
  grid (Amendment 12 b keeps it for E3).
- **One method-neutral change:** `M` lives in one GPU copy (`bench/gn/metric_store.py`); pytest checks on the
  CPU that codebooks, labels, reports, `P` and the lifted check are bit-identical to E2c's code path.

**Kaggle steps:**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_e3p_bench.ipynb` and title it
   **E3p INRIA pilot**.
2. Attach the input in the table below (optional, but it saves the wheel build).
3. GPU T4 x2, Internet on. Save & Run All.
4. Bring back `/kaggle/working/gn3p_bundle.zip` (look in `~/Downloads`).

**Inputs.** E3p needs no checkpoint and no earlier result: its models come from INRIA's archive, fetched and
checked by each job, and its datasets from runs 4-5's downloaders.

| Attach | Required | What E3p takes from it, and why |
|---|---|---|
| **"R5 tilequant"** (the run-5 notebook output) | no, but attach it | **`wheels/` only**: the gsplat wheel is reused when its key matches (`gsplat/` and `setup.py` are unchanged since run 5; E2c reused it on 2026-09-26, install 168.5 s). Without it the install cell builds the wheel (4,404 s in run 5). Nothing else in it is read: no checkpoint, sort cache or clustering |
| this notebook's own earlier output | only to resume | `gn3p/` (rows, metas, reports), `gn3p_work/` (the sort orders, the clusterings, the job logs), `e3p_inria/` (the fetched members; each is re-checked by size and CRC32 before it is reused). Every row resumes on its own; the `M` caches live in `/tmp` and are recomputed |

**What it does:** restore (the wheel, a resume), install (the wheel, TorchPQ + cupy, PLAS from its GitHub head as
run 4 installed it, the example dependencies, `remotezip`), `gn3p_env.json` (torch, CUDA, cuDNN, the GPU, the
driver from `nvidia-smi`, `nvcc`, package versions), the CUDA smoke tests (`bench/gn/selftest.py`), the C3DGS
build check, then one job per scene (bicycle on the first GPU, train on the second, 9.5 h start cutoff), the
summary cell and the bundle cell. Each job (`kaggle/gn_e3p_scene.py`):

- refuses a scene that is not bicycle or train and a results CSV that is not its own, before anything else;
- **INRIA members:** reads the archive's zip directory (13,708 bytes) by range requests and compares it with
  Amendment 12 a's pins (archive size, entry count, directory offset and size; per member the local-header
  offset, sizes, CRC32 and method), **before anything is fetched**; then fetches `point_cloud.ply`,
  `cameras.json` and `cfg_args` (bicycle: 1,353,363,151 compressed bytes; train: 219,441,845), checks each
  local header's name and method, inflates, checks size and CRC32 before keeping the file, and records its
  SHA-1 (`meta["inria"]`). `cfg_args` must say what the pins say (bicycle `images_4`, train `images`, both
  `resolution=1`, `sh_degree=3`, `eval=True`); the `.ply` must hold the pinned vertex count (6,131,954 /
  1,026,508) and exactly INRIA's 62 properties;
- downloads the dataset (bicycle from `360_v2.zip` with `images/`, `images_4/`, `sparse/`; train from
  `tandt_db.zip`) and builds a runner holding the INRIA model with **`normalize_world_space` off** (INRIA's
  splats are in COLMAP's frame). The runner's cameras must equal `cameras.json` to 1e-4
  (`camera_frame_check`, a stop condition; checked here on bicycle's real COLMAP model against the fetched
  `cameras.json`: 194 of 194 matched, largest differences 8.9e-16 in position and 3.3e-16 in rotation). The
  test split is compared with INRIA's (reported; on bicycle it matched, 25 test views). Render parity as in
  E0-E2c (a stop condition);
- **`uncompressed`** under both protocols: (i) `Runner.eval` on gsplat's own downscaled images (bicycle's
  `images_4_png`, resized by gsplat from the full-resolution JPEGs), float renders clamped; (ii) INRIA's:
  `cfg_args`'s image folder loaded directly (bicycle's `images_4` JPEGs, train's `images`), the size INRIA's
  `loadCam` gives it, INRIA's camera (focal `fx * w / W` from `cameras.json`, principal point at the centre),
  the render quantized as `save_image` does, the runner's PSNR / SSIM / LPIPS-VGG modules. The summary puts
  protocol ii's PSNR next to INRIA's published number (bicycle 25.246, train 21.097: arXiv 2308.04079v1,
  Tables 5 and 8, row Ours-30k), a sanity check only: INRIA's README says the released models were made with
  the release codebase and differ from the paper's;
- the GN passes, `M_even` then `M` (E0's `compute_gn`), each moved to host memory as it finishes and cached in
  `/tmp/gn3p_cache` (the file size is recorded next to `M`'s 480 bytes per splat);
- the PLAS sort (run 4's `load_or_build_sort_order`, seed 0). The codec keeps the largest square number of
  splats (bicycle 2,476^2 = 6,130,576, dropping 1,378; train 1,013^2 = 1,026,169, dropping 339); in the
  shN-only renders a dropped splat keeps its own shN. If the sort runs out of memory the job continues with
  the crop alone (`order_source` says so);
- `upstream_l1` (TorchPQ) and `lloyd_wopa_area` (the library's weighted Lloyd) at K = 65,536 through E2's
  `get_codebook`, each written with `PngCompression`, decoded and evaluated under both protocols;
- `gn_vq_cvfloor` at K = 65,536: per `rho` the metric buffer filled, the lifted check, GN-VQ on `M_even`, the
  write, the odd- and test-view dMSE; then `rho_cv` (read back from the CSV, `e2c.select_rho_cv`) and the final
  codebook on `M`, measured as E2c's final rows (`P`, D train / test, protocol i, train PSNR, shN-only) plus
  protocol ii.

**Every step is a record in `meta["steps"]`** (`Steps`): status, wall time, the device memory held at its start,
free and total device memory, the peak allocated and reserved memory during the step (reset at its start) and
the host's peak RSS. **A step that runs out of memory** (CUDA's OOM, an allocation failure in cuBLAS / cuSOLVER
/ cuDNN, or host `MemoryError`) is recorded with its message, the innermost frames where it was raised and the
memory at that point; the device cache is emptied and the job continues. Steps that need its product are
recorded as `skipped` with the reason (no `lloyd_wopa_area` means no GN-VQ rows; no `M_even` means no CV rows).
An OOM leaves the job's exit code 0 and its rows missing (`missing_rows`, `oom_steps`, `skipped_steps` in the
meta); any other error stops the job and the notebook's bundle cell raises at the end.

**C3DGS build check** (`gn3p_c3dgs_build.json`, never stops the notebook, capped at 3,600 s): clone
`KeKsBoTer/c3dgs`, check out `2a234af55fbe8b90c8829c1436ce80088c4b622b` (the commit E3_SCOUTING.md read) with its
glm submodule; a venv with `--system-site-packages` over the session's torch; `plyfile==0.8.1` and `tqdm`
(environment.yml's pins), `torch-scatter` from the PyG wheel index for this torch and CUDA, then
`submodules/diff-gaussian-rasterization` and `submodules/weighted_distance` built with
`--no-build-isolation`; then only imports: `torch`, `torchvision`, `torch_scatter`, `plyfile`, `tqdm`,
`diff_gaussian_rasterization` (and `._C`), `weighted_distance._C`, and C3DGS's own `compression.vq` and
`gaussian_renderer` (the modules that load them). The README's own route is a conda environment (python 3.8,
pytorch-cuda 12.1, cuda-toolkit 12.1); the JSON lists the deviations (the session's python, torch and CUDA
toolkit) and records whether `conda` was on the path, each step's command, return code, time and last 60
lines, `build_time_s` (the three builds), `total_time_s` and the first failed step.

**Outputs** (`gn3p_bundle.zip`, arcname `gn3p/`): `gn3p_results_<scene>.csv` (11 rows: `uncompressed`,
`upstream_l1`, `lloyd_wopa_area`, 7 `gn_vq_cvfloor_cv`, `gn_vq_cvfloor`; E2c's columns plus `PSNR_ii` /
`SSIM_ii` / `LPIPS_ii`, both resolutions, the encode / decode / evaluation times, sizes in bytes, MB and MiB,
the crop), `gn3p_meta_<scene>.json` (the INRIA members with SHA-1s, `cfg_args`, the frame and split checks,
`env`, `steps`, `gn`, `order`, `lifted_checks`, `metric_store`, `rho_cv`, `phases`, `missing_rows`,
`oom_steps`, `skipped_steps`), one `gn3p_<config>_rho<rho>_k65536_s0_<scene>.json` per GN-VQ row (16),
`gn3p_env.json`, `gn3p_selftest.json`, `gn3p_c3dgs_build.json`, `gn3p_summary.json`,
`gn_e3p_<scene>_log_tail.json`, `timings.json`. Not bundled: `e3p_inria/` (1.77 GB of members) and
`gn3p_work/` stay in `/kaggle/working` for a resume; the `M` caches and run directories are in `/tmp`.

**Runtime, an estimate, not a measurement.** Every input is a number this repository already records: the
parts at 1,000,000 splats (E3_SCOUTING.md c; the E2 and E2c tables above), scaled linearly by the splat count
(6.13x for bicycle, 1.03x for train), which is itself unverified above 1M:

| Part | bicycle (s) | train (s) |
|---|---|---|
| GN-VQ, 8 runs at K = 65,536 | 6,817-7,618 (E3_SCOUTING.md c) | 1,141-1,275 (E2c: 1,112-1,242 at 1M) |
| TorchPQ `upstream_l1` (E2: 398.2-425.7 at 1M) | 2,442-2,610 | 409-437 |
| `lloyd_wopa_area` (E2 / run 5: 276.0-646.0 at 1M) | 1,692-3,961 | 283-663 |
| PLAS sort (run 4: 69-83 at 1M) | 423-509 | 71-85 |
| GN passes, 1.5 full passes (E2: 8.4-12.8 outdoor, 16.2 train) | 77-118 | 25 |
| Lifted checks, 8 x 25.9-29.8 (E2c) | 207-238 | 207-238 |
| Evaluations: bicycle 7 CV rows (E2b 13.0-20.6 at 1M) and 4 full ones (E2 36.2-44.4 at 1M); train 11 rows at E2c truck's 373 s / 32 rows | 1,446-1,973 | 132 |
| **Job, before downloads** | **13,105-17,028 (3.6-4.7 h)** | **2,268-2,855 (0.6-0.8 h)** |

Downloads come on top: the datasets took 63.0 s (bicycle) and 500.0 s (train) in E2, while E2c's MipNeRF360
downloads took 371.7-911.2 s; the INRIA fetches (1.35 GB and 0.22 GB compressed) are unmeasured. The two jobs run
in parallel, so the session is about bicycle's job plus the session steps: install 168.5 s with the restored
wheel (E2c; 4,404 s more without it), smoke tests about 16 s, and the C3DGS check, unmeasured and capped at
3,600 s. **About 4-6 h in all, inside the 9.5 h start cutoff.** Not covered by the estimate: protocol ii's cost
(assumed within the full evaluations' figure), the extra GN-VQ memory traffic from host to device (2.94 GB per
metric fill at bicycle), and any step that runs out of memory, which would shorten the run.

**Memory, what to expect (an estimate):** during GN-VQ on bicycle the device holds the runner's model (1.45 GB),
the sorted copy the writer needs (1.45 GB) and the one metric buffer (2.94 GB), 5.84 GB in all, plus GN-VQ's
transients, each now about 1 GB at most: `direct_distance` per chunk of 262,144 splats (chunked on 2026-09-28,
below; before, it built three `[N, 15, 3]` float64 tensors, 6.6 GB at once), `update_centroids` per chunk
(about 0.9 GB), `share_in_l2_topk`'s scores at iteration 1 (4,096 x 65,536 float32, 1.07 GB), the lifted
assignment's (0.5 GB). That puts GN-VQ near 7-8 GB of a T4's 15 GB, against about 12-13 GB before the change.
The final row's evaluation adds the decoded model (1.45 GB) and its full shN (1.1 GB). Unmeasured until the run.

**Also recorded, report only:** `meta["direct_distance_check"]`, `diagnostics.direct_distance` against the form
E0-E2c ran, on the GPU, on the first 787,432 sorted splats (four chunks). See "`direct_distance` chunked over
splats" under the session decisions.

**Local checks:** `pytest bench/gn/test_gn.py` (100 tests; the chunked `direct_distance` and `P`'s chunked
difference bit-identical to E0-E2c's forms, and E3p's: the metric layout bit-identical to E2c's path
for every `rho` and both metrics, `HostMetric`'s slice-only reads, the zip64 directory reader against
`zipfile`, the fetch's refusals (a changed pin before anything is fetched, a CRC32 mismatch, a wrong local
header), INRIA's `.ply` layout, `cfg_args`, `loadCam`'s sizes, `save_image`'s rounding, the frame and split
checks, the pins against Amendment 12's table, the job's constants and refusals, `Steps` on an OOM, and the timed
writer byte-identical to E0's) and `python bench/gn/dryrun/dryrun_gn_e3p.py` (6 stages, see its docstring: a
local archive laid out like INRIA's with zip64 records, 4,133-splat models, fake datasets with reduced JPEGs,
both scenes end to end with protocol ii recomputed independently, resume, two injected OOMs, six refusals, and
the notebook's restore, C3DGS (stubbed), summary and bundle cells; 95 s on this machine) and
`python bench/gn/dryrun/check_chunked_distance.py` (E2 and E2c on the toy with the old and the new
`direct_distance`: identical; 151-269 s on this machine). Rebuild the notebook with
`python kaggle/build_gn_e3p_bench.py` first.

**After the run:** unpack `gn3p_bundle.zip` into `kaggle/gn_e3p/` (arcname `gn3p/`), check every file against its
zip entry, and make a data-only commit. Read `gn3p_summary.json` first, then each meta's `oom_steps`,
`skipped_steps`, `missing_rows`, `camera_frame_check`, `split_check` and `inria.members`, then `steps` for the
times and peaks, and `gn3p_c3dgs_build.json`. No verdict exists to read. Then write E3's design (its own
amendment) from these measured parts; any number quoted from E3p comes from these files.

**Measured E3p runtime** (`kaggle/gn_e3p/gn3p/timings.json`, `gn3p_meta_<scene>.json`; FINDINGS section 13 has
the per-step table):

| Scene | Queue wall time `gn_e3p_<scene>_s` | `timings_s.job` (meta) | Dataset download | INRIA fetch | `gn_vq_cvfloor` phase |
|---|---|---|---|---|---|
| bicycle | 19,710.7 | 19,699.0 | 256.8 | 141.2 | 11,857.7 |
| train | 3,435.2 | 3,427.5 | 351.7 | 28.7 | 1,595.2 |

Seconds. Per session: restore 1.7 s, install 167.8 s (the restored wheel; no build), smoke tests 15.2 s, the C3DGS
check 7.9 s (it failed at the venv). Both jobs exited 0; neither was skipped by the cutoff.

- **Against the estimate above:** bicycle's job took 19,699.0 s against 13,105-17,028 s before downloads. GN-VQ
  took 11,287.0 s against 6,817-7,618 s (14-17 iterations per run, where E2c's runs at K = 65,536 took 8-10),
  and the PLAS sort 947.0 s against 423-509 s. The two clusterings fell inside their estimates.
- **Memory:** bicycle peaked at 11.51 GB allocated (the final row's protocol-i evaluation); its GN-VQ runs at
  8.49 GB, above the 7-8 GB estimate; the allocator reserved 15.18 GB of the T4's 15.64 GB. Train peaked at 3.30 GB.

## E3q notebook (`bench/gn-vq`, `kaggle/gn_e3q_bench.ipynb`, Kaggle title "E3q C3DGS smoke")

**Closed (2026-09-29): attempt 1 failed; attempt 2 ran every step and all three rows** (FINDINGS section 14;
the measured runtime is below). The notebook at the branch head is attempt 2's; nothing in E3q is left to run.
E3q is **pre-registered in `PREREG_GN.md`
Amendment 13** (`b1e8725b`, docs only, before any E3q code), with **note g** (`0b11f7ab`, docs only, before any
attempt-2 code). It is an **engineering smoke test of the C3DGS host and has no verdicts.** GN-VQ does not run
here; E3's C3DGS comparison (Amendment 12 c) still gets its own pre-registration.

- **Scene:** train only (a development scene), INRIA's 30k checkpoint through E3p's pins, the Tanks & Temples
  dataset through run 5's downloader. Nothing else is fetched.
- **Why now:** E3p's C3DGS check failed at `python -m venv` (the session's Python 3.12.13 has no `ensurepip`),
  before compiling anything (FINDINGS section 13).

**Attempt 1 (2026-09-28; `kaggle/gn_e3q/attempt1/gn3q/`, `59303486`).** Every figure below is from those files.

- **What worked:**
  - **The build.** C3DGS built and imported (all 10 imports ok, head `2a234af5`). It took 204.9 s: the two CUDA
    extensions 112.8 s and 70.4 s, and `torch-scatter` 2.2 s from a PyG wheel, not a source build. Neither fallback
    was needed; the deviations are the six fixed ones.
  - **The inputs.** The restored gsplat wheel matched (install 163.7 s), and the attached E3p output's members
    were reused.
  - **The uncompressed row.** Protocol ii gave 21.293174743652344 dB, the same value as E3p's train row. The frame
    and split checks passed.
- **What failed:** both `compress.py` runs, after 270.8 s (`c3dgs_ft0`) and 273.3 s (`c3dgs_ft5000`).
  - **Where:** in `compress_gaussians` -> `compress_covariance` (`compression/vq.py`, line 231) ->
    `extract_rot_scale` (`utils/splats.py`, line 29). There, `torch.linalg.eigh(cov + eye * 1e-8, UPLO="U")` on
    one float32 batch of 3x3 matrices was refused: `CUSOLVER_STATUS_INVALID_VALUE` from
    `cusolverDnXsyevBatched_bufferSize`.
  - **Memory:** the runs peaked at 3,418,784,768 and 3,419,284,992 bytes allocated.
  - **Output:** no `.npz` was written, so neither run has a row.
- **Session time:** the job took 885.0 s in the queue (`timings.json`), 871.3 s by its own clock.

**Kaggle steps (attempt 2):**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_e3q_bench.ipynb` and title it
   **E3q C3DGS smoke**. Its first cell has an "Attempt 2" paragraph; if it does not, the import is stale.
2. Attach "R5 tilequant" and the E3p output, as in attempt 1. **Do not attach attempt 1's output.** Attempt 2
   must be a fresh run, so that all three rows come from one session with the patched wrapper; the uncompressed
   row is re-measured by design.
3. GPU T4 x2 (one is used), Internet on. Save & Run All.
4. Bring back `/kaggle/working/E3q_bundle_2.zip` (look in `~/Downloads`). Its arcname is still `gn3q/`.

| Attach | Required | What E3q takes from it, and why |
|---|---|---|
| **"R5 tilequant"** (the run-5 notebook output) | no, but attach it | **`wheels/` only**: the gsplat wheel, reused when its key matches (E3p reused it on 2026-09-27: torch 2.10.0+cu128, install 167.8 s). Without it the wheel is built (4,404 s in run 5). The harness needs gsplat for protocol ii; C3DGS does not |
| the **E3p notebook output** | no | `e3p_inria/` only: train's three fetched members, so the job skips its 219 MB fetch (28.7 s in E3p). Each member is re-checked by size and CRC32 before it is reused. Nothing else in it is read |
| this notebook's own earlier output | only to resume attempt 2; **never attempt 1's** | `gn3q/` (rows with status `ok` are not rerun) and `gn3q_work/` |

**What it does:** restore, install (the gsplat wheel and the example dependencies; no TorchPQ, no PLAS; no
smoke tests, since nothing in E3q uses the GN code), `gn3q_env.json`, then one job (`kaggle/gn_e3q_scene.py`) on
the first GPU, then the summary and the bundle. The job:

- **fetches** train's three members through E3p's pins (or re-checks an attached copy) and lays them out as INRIA's
  model directory (`cfg_args`, `cameras.json`, `point_cloud/iteration_30000/point_cloud.ply`); downloads the dataset;
- **builds C3DGS** into the session's Python (`e3q_c3dgs.build`): clone at `2a234af55fbe8b90c8829c1436ce80088c4b622b`
  with the glm submodule; `plyfile==0.8.1`, `tqdm` if missing, `torch-scatter` from
  `https://data.pyg.org/whl/torch-<torch>+<cuda>.html`, the two CUDA extensions with `--no-build-isolation`, **all
  with `--no-deps`**; an import check (the packages, both extensions, and C3DGS's `compression.vq` and
  `gaussian_renderer`). Every deviation from the README is written to `meta["c3dgs_build"]["deviations"]`: the
  fixed ones of Amendment 13 b, and any of these when they happen: `plyfile` upgraded (0.8.1 failed to write or read
  a `.ply`), `torch-scatter` built from source (no wheel), an extension rebuilt with `<cstdint>` force-included
  (its log named a missing fixed-width integer type). The build stops at its first failed step;
- **runs C3DGS's `compress.py`** twice, `--finetune_iterations 0` and `5000`, through `e3q_c3dgs_run.py` (it runs
  `compress.py` unchanged as `__main__` and records wall time and the process's peak allocated and reserved GPU
  memory), with `--source_path` pointing at the downloaded dataset (the checkpoint's `cfg_args` names its authors'
  paths); then decodes each `point_cloud.npz` with C3DGS's `npz2ply.py`. The C3DGS runs come before the harness's
  runner, so C3DGS has the GPU to itself. **From attempt 2 on** (Amendment 13 g), the wrapper first installs
  `ChunkedLinalg`: in its process, `torch.linalg.eigh` and `torch.Tensor.det` on a batch `[N, n, n]` run through
  `batched.py`'s `batched_linalg` (8,192 per call, halved on a backend refusal), with the op's own memory layout
  kept; anything else passes through. Its record, `wrapper.linalg_patch` in each `c3dgs_runs` entry, lists every
  call (op, batch size `n`, start batch, reductions, time, and the report-only float64 CPU check on up to 4,096 of
  the call's own matrices), every reduction (`fallbacks`) and the deviation text;
- **evaluates with this project's harness:** a runner holding the INRIA model (E3p's `build_runner`, camera-frame
  and split checks), protocol ii of the uncompressed model and of each decoded `.ply` (read by E3p's loader).
  The `.ply` carries the decoded attributes but not C3DGS's render-time fake quantization;
- **records every failure and goes on** (Amendment 13 f): each step is an E3p `Steps` record, with any error caught
  (status `error`, message and where); rows that could not be produced are written with status `failed` and a
  `reason`. The job exits 0; the bundle cell raises only if the job itself crashed.

**Outputs** (`E3q_bundle_2.zip` from attempt 2; attempt 1's was `gn3q_bundle.zip`; arcname `gn3q/` in both):
- `gn3q_results_train.csv`: rows `uncompressed`, `c3dgs_ft0` and `c3dgs_ft5000`. Each has its status and reason,
  C3DGS's PSNR / SSIM / LPIPS and its own size figure, the `.npz` in bytes, MiB and MB, protocol ii, the `.ply`'s
  splat count, C3DGS's `times.json` parts, wall time and peak GPU memory. The columns are unchanged from attempt 1.
- `gn3q_meta_train.json`: the members; the build, with every command's return code, time and last 60 lines; the
  deviations and the imports; both runs, with the wrapper's `linalg_patch` from attempt 2 on; the frame and split
  checks; `env` and `steps`.
- `gn3q_env.json`.
- `gn3q_summary.json`: the rows next to C3DGS's published train numbers. From attempt 2 on, it also has
  `c3dgs_runs` (per run: the wrapper's status, error, deviations and `linalg_patch`) and `run_deviations`.
- `gn_e3q_train_log_tail.json` and `timings.json`.

Not bundled: the C3DGS checkout (`/tmp/c3dgs`), its outputs and the model directory (`gn3q_work/`).

**Measured runtime, attempt 2** (`kaggle/gn_e3q/attempt2/gn3q/`, rows timestamped 2026-09-28T20:08):
- **Session:** restore 20.1 s; install 160.7 s (restored wheel).
- **The job:** 1,695.0 s in the queue, 1,676.4 s by its own clock. Within it:
  - the INRIA members (E3p's copies re-checked) 4.3 s;
  - the dataset download 352.0 s (66.6 s in attempt 1);
  - the C3DGS build 204.3 s;
  - `c3dgs_ft0` 351.6 s in the wrapper: sensitivity 16.0, clustering 257.7, encode 1.7;
  - `c3dgs_ft5000` 666.4 s: sensitivity 16.6, clustering 257.1, fine-tuning 317.9, encode 1.7;
  - `npz2ply.py` 11.9 / 11.8 s;
  - the harness's runner 42.5 s;
  - protocol ii 8.4 / 7.9 / 8.1 s.
- **Peak memory of the C3DGS runs:** 4.55 GB allocated.
- **The chunked calls:** one `eigh` and one `det` per run, on 281,237 matrices, with no refusal at 8,192.
- **Attempt 1, for comparison:** install 163.7 s; the job 885.0 s in the queue, both runs failing at 270.8 s and
  273.3 s.
- The job still caps the build at 5,400 s and each run at 7,200 s.

**Local checks:**
- `pytest bench/gn/test_gn.py` (107 tests). The E3q tests cover:
  - the build, with a stubbed command runner: every pip install with `--no-deps`, the README deviations, both
    fallbacks used only on their triggers, and a failed step stopping the build;
  - the wrapper running a `compress.py` as `__main__` and recording a raised error;
  - the model layout, and a run's parsing, with sizes in MiB and MB;
  - the job's constants, Amendment 13's published numbers and commit, and `Steps` catching any error;
  - **Amendment 13 g:**
    - the chunked `eigh` and `det` equal the unpatched ops, bitwise, on random symmetric float32 and float64
      batches of 1,000 (chunk 64) and 20,037 (chunk 8,192), with the eigenvector layout and `R.det()` on it
      included;
    - non-batch shapes pass through, and the global RNG is untouched;
    - a refusing backend halves 16 -> 8 -> 4, records both reductions, and remembers 4;
    - the real wrapper chunks a stand-in `compress.py`'s `eigh` and `R.det()` and keeps its record when the script
      raises.
- `python bench/gn/dryrun/dryrun_gn_e3q.py` (6 stages; see its docstring). It uses a local INRIA-like archive and
  stubs every command except `compress.py`, which the **real wrapper** runs as a stand-in for C3DGS's `eigh` /
  `R.det()` on 20,037 matrices; the results equal the unpatched ops. Both runs are decoded and evaluated, with
  protocol ii recomputed independently. It also covers resume, three failures that do not stop the job (the
  failed run keeps its `linalg_patch`), refusals, and the notebook's restore, summary and bundle cells.

Rebuild the notebook with `python kaggle/build_gn_e3q_bench.py` first.

**After attempt 2 (done, 2026-09-29):**
- The bundle is unpacked unchanged into `kaggle/gn_e3q/attempt2/gn3q/` (`a6c6f725`), and FINDINGS section 14
  quotes it (`6568cf25`).
- No verdict exists. Anything quoted from E3q comes from the two bundles; the published numbers are a sanity
  check only (Amendment 13 e).

## E3r notebook (`bench/gn-vq`, `kaggle/gn_e3r_bench.ipynb`, Kaggle title "E3r C3DGS pilot")

**Done: ran on Kaggle on 2026-09-30** (imported at tip `b27e1421`) **and committed.** The bundle
`~/Downloads/E3r_bundle.zip` (61,230 bytes, SHA-1 `e108184f480479a475e31feca10220cf5a2e8c57`) is unpacked unchanged
in `kaggle/gn_e3r/gn3r/` (17 files, `9c28cc89`); FINDINGS section 15 quotes it (`2eb091f9`), re-checked by
`bench/gn/check_s15.py` (`5fda2ea7`). Nothing in E3r is left to run; the notebook stays as it ran.

**Measured runtime** (`timings.json`, the metas' `timings_s`):
- setup: restore 45.0 s, install 149.9 s (the restored wheel), C3DGS build 201.1 s;
- train job: 7,483.6 s by its own clock, 7,500.3 s in the queue, inside the estimate's 5,362-8,309 s;
- bicycle job: 594.7 s, 600.0 s in the queue, above the estimate's 492-498 s. Its download (290.9 s against E3p's
  256.8 s) and runner (208.0 s against 182.6 s) were slower, and the estimate counted neither the two GN cache writes
  (20.4 s) nor the member re-check (5.0 s);
- setup plus the train job: 7,896.3 s (2.2 h), inside the session estimate of 5,747-8,694 s.

What follows is how it was run. E3r is **pre-registered in `PREREG_GN.md` Amendment 14** (`693a6a4b`, docs only, before any E3r
code) with **note f** (`a850c4ea`: every run seeded as C3DGS's own `safe_state` seeds) and **note g** (`b06f9293`: a
memory check restricts every C3DGS step to train). It is an **engineering pilot of the C3DGS host with no verdicts**:
what the C3DGS arm's pre-registration needs to know. Train and bicycle (development scenes), INRIA's 30k checkpoints;
the gate scenes untouched.

**Kaggle steps:**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_e3r_bench.ipynb` and title it
   **E3r C3DGS pilot**.
2. Attach **"R5 tilequant"** (the run-5 output) and **"E3p INRIA pilot"** (the E3p notebook output).
3. GPU T4 x2 (both are used), Internet on. Save & Run All.
4. Bring back `/kaggle/working/E3r_bundle.zip` (look in `~/Downloads`); its arcname is `gn3r/`.

| Attach | Required | What E3r takes from it, and why |
|---|---|---|
| **"R5 tilequant"** | no, but attach it | **`wheels/` only**: the gsplat wheel, reused when its key matches (E3q attempt 2 reused it: install 160.7 s). Without it the wheel is built (4,404 s in run 5) |
| **"E3p INRIA pilot"** | no, but attach it | **`e3p_inria/` only**: bicycle's and train's three pinned members (1.52 GB and 0.25 GB `.ply`), re-checked by size and CRC32 before reuse; without it they are fetched (141.2 s and 28.7 s in E3p) |
| this notebook's own earlier output | only to resume | `gn3r/` (rows with status `ok` are not rerun) and `gn3r_work/` (the probe records and C3DGS's `.npz` files) |

**What it does:** restore; install (as E3q); **build C3DGS once** (`gn_e3r_scene.py --build_only`,
`gn3r_c3dgs_build.json`); then **two jobs in parallel**, bicycle on GPU 0 and train on GPU 1
(`kaggle/gn_e3r_scene.py`, Amendment 14 d's order). **Bicycle runs no C3DGS step** (Amendment 14 g): it measures the
runner, the uncompressed model's protocol ii and the two 16 x 16 GN passes, and ends. **Train** runs everything:

1. **Probe run:** C3DGS at K = 4,096 and the default threshold, no fine-tuning. The hooks record its colour VQ:
   input, masks, codebook, labels, quantizer.
2. **Harness phase:**
   - the runner, with the frame and split checks;
   - protocol ii of the uncompressed model;
   - the **16 x 16 GN passes** (even views and all train views, cached in `/tmp/gn3r_cache`);
   - the colour-quantized splats' **trace share**;
   - the **SH-only cross-validation**: 7 GN-VQ runs on the even-view metric, scored by odd-view dMSE, giving
     `rho_cv`;
   - the **calibration**.
3. **Injected runs:** GN-VQ at `rho_cv` inside C3DGS's own run, with the lifted check and the set against the
   probe's; then the same with 5,000 fine-tuning iterations, plus the label-survival check.
4. **The other K** (1,024, 16,384, 65,536) and the thresholds 0.6e-6 x 3^j.
5. **Protocol ii of every decoded row,** one `.ply` at a time.

The cross-validation and the injection hold **one device copy** of the floored metric (`metric_store`, as E3p;
bit-identical to the full-copy path, a CPU test), the reference colours stay in host memory, and the metric's buffer
is released during the scoring renders (Amendment 14 g).

Every C3DGS run is seeded with 0 (`--seed 0`) and goes through E3q's wrapper. A run out of GPU memory is retried once
with `--data_device cpu`, recorded as a deviation. No new C3DGS run starts after 11.5 h minus the job's 1,800 s
reserve. Everything else is recorded and continues.

**Outputs** (`E3r_bundle.zip`, arcname `gn3r/`):
- `gn3r_results_<scene>.csv`: `uncompressed`, the 7 `gnvq_cv_rho*` rows, `c3dgs_k<K>`, `gnvq_k4096`,
  `gnvq_k4096_ft5000` and `c3dgs_k4096_j<j>`. Each row has its status and reason, the counts, sizes, both
  evaluations, times, memory, the data device, the injection's checks, the quantizer states and the geometry SHA-1.
- `gn3r_meta_<scene>.json`: the steps, the runs with their wrapper records, the GN metrics, the colour share,
  `pruned_trace_check` (Amendment 14 h), `rho_cv`, the CV scores and the calibration.
  - `pruned_trace_check` is `{status, n_splats, n_pruned, n_pruned_tr0, n_tr0_all, mask_source: "probe"}`.
  - Its status is `ok`, `not_computed` with a reason, or on bicycle `not_applicable`.
  - It is also in `gn3r_summary.json`, and it is no CSV column.
- `gn3r_cv_rho<rho>_<scene>.json`: the GN-VQ reports.
- `gn3r_c3dgs_build.json`, `gn3r_env.json`, `gn3r_summary.json` (each knob's byte and PSNR ranges and the rest,
  no verdict), the log tails and `timings.json`.

Not bundled: `gn3r_work/` (the probe records, `.npz` files and model directories), `/tmp/gn3r_cache` and the C3DGS
checkout.

**Runtime, an estimate, not a measurement** (`bench/gn/e3r_estimate.py`, `estimate_g`, Amendment 14 g, from E3q's,
E3p's and E2c's files):

| Part | Estimate |
|---|---|
| train job | 5,362-8,309 s |
| bicycle job, GN passes only | 492-498 s |
| setup (restore, install, build) | 385 s |
| **session, both jobs in parallel** | **5,747-8,694 s (1.6-2.4 h)** |

The upper end assumes C3DGS's colour clustering grows linearly with K; nothing at K other than 4,096 has been measured.
Amendment 14 e's earlier table (bicycle with C3DGS: up to 13.0 h) is superseded by g.

**GPU memory, computed from the code** (`bench/gn/e3r_memory.py`; the numbers are in
`kaggle/gn_e3r_memory/e3r_memory.json`; GB = 10^9 bytes; the T4 has 15.64 GB):

- **The 16 x 16 metric** is 544 bytes per splat as stored (float32, 136 packed values). The GN pass accumulates it
  for every splat, at 628 bytes per splat with its sums. The splats that need the metric in C3DGS are the
  colour-quantized ones. Their count is known only after C3DGS runs, and the checkpoint bounds it only by the model:
  6,131,774 of bicycle's 6,131,954 splats touch a tile in some train view. So one device copy is at most 3.34 GB on
  bicycle and 0.56 GB on train.
- **C3DGS's own peak** is the backward pass of its sensitivity computation at its worst train view. The code gives:
  - 1,850 bytes per splat: its parameters and gradients, the fake-quantized features, the rasterizer's per-splat
    state and 75-float backward buffers, the hooks' temporaries;
  - the images, which are on the GPU with `--data_device cuda`;
  - 36 bytes per tile instance. The instances were counted with the rasterizer's own formulas on the pinned
    checkpoints: at most 6,254,053 per view on train and 11,798,558 on bicycle.

  | | From the code | With train's unexplained rest (470 bytes per splat) | Measured |
  |---|---|---|---|
  | train (1,026,508 splats, images on the GPU) | 4.068 GB | 4.551 GB | 4.551 GB (E3q), 89.4% explained |
  | bicycle, images on the GPU | 14.16 GB | 17.05 GB | - |
  | bicycle, images on the CPU | 11.80 GB | 14.68 GB | - |

- **E3r's steps, at their largest** (bicycle; train's are all below 5 GB):

  | Step | As coded at `290e62f5` | Now (one metric copy, host colours) |
  |---|---|---|
  | GN pass, 16 x 16 (E3p's measured 7.99 GB plus the width) | 8.41 GB | 8.41 GB |
  | cross-validation, GN-VQ | 12.57 GB, plus a 6.67 GB float64 transient in `floored_metric` | 8.06 GB |
  | cross-validation, scoring renders | 11.51 GB | 5.82 GB |
  | injected run's colour step (C3DGS's state included), images on GPU / CPU | 15.11 / 12.75 GB | 11.78 / 9.41 GB |

  C3DGS's sensitivity pass precedes the colour step in every C3DGS run. At 11.80-17.05 GB, before two CUDA
  contexts and fragmentation, it is the step that does not fit. Hence Amendment 14 g: the C3DGS steps run on
  train only, and bicycle measures the GN passes (8.41 GB).

**Local checks:**
- `pytest bench/gn/test_gn.py` (116 tests). E3r's cover:
  - the 16 x 16 metric's AC block being the frozen metric;
  - the generic floor, update and assignment at 16;
  - the quantizer against C3DGS's fake quantizers and `save_npz`;
  - GN-VQ at 16;
  - the hooks in the CPU stand-in: observe and record change nothing and never call `get_features`, the probe
    record, injection paired (the same geometry SHA-1), labels surviving a fine-tune, a missing metric an error;
  - the job's constants and both estimates against Amendment 14's text;
  - the one-device-copy layout at 16, bit for bit against the full-copy path;
  - the tile formula on a splat whose footprint is known by hand, and the committed memory check reproducing
    Amendment 14 g's numbers.
- `python bench/gn/dryrun/dryrun_gn_e3r.py` (8 stages, about 90 s). It covers:
  - the notebook and the build;
  - train end to end through the **real wrapper and hooks** on the stand-in, including an out-of-memory retry;
  - bicycle's GN passes with no C3DGS run;
  - resume, a failed build, a passed deadline and refusals;
  - the restore, summary and bundle cells.

Rebuild the notebook with `python kaggle/build_gn_e3r_bench.py` first.

**After the run:**
1. Unpack `E3r_bundle.zip` into `kaggle/gn_e3r/` (arcname `gn3r/`), check every file against its zip entry, and
   make a data-only commit.
2. Read `gn3r_summary.json`:
   - `build`;
   - train's `knob_K` and `knob_threshold` (bytes and PSNR ranges: which knob spans enough rate for BD-rate);
   - bicycle's `gn` (the 16 x 16 passes' time and size; their peaks are in the meta's steps);
   - `colour_share`;
   - `pruned_trace_check` (train; report-only, Amendment 14 h): `n_pruned_tr0` against `n_pruned` is how many of
     C3DGS's pruned splats gsplat also sees with zero weight, and `n_tr0_all` how many it sees so in all. This is
     visibility agreement between the two rasterizers, not render parity;
   - `rho_cv` and `cv_odd_scores`;
   - `calibration`;
   - `injected`: `set_same_as_probe`, the geometry SHA-1 against `probe_geometry_sha1`, `lifted_check_pass`,
     `labels_survived`;
   - `missing_or_failed`.
3. Then read the meta's steps and runs.

No verdict exists. The injected row against the probe row is an engineering number, not a comparison
(Amendment 14 a). Then write the C3DGS arm's pre-registration.

**Done (2026-09-30):** steps 1-3 above, then FINDINGS section 15. See "E3r results, section 15 and check_s15" under
Session decisions.

## E4p notebook (`bench/gn-vq`, `kaggle/gn_e4p_bench.ipynb`, Kaggle title "E4p C3DGS fork pilot")

**Ran on Kaggle; bundle committed data only** (`kaggle/gn_e4p/gn4p/`, 11 files, `60c4839b`; zip 298,965 bytes, SHA-1
`7641a3cdc980e08a2b80cfa726bad68805a9af5a`). Not written up yet (FINDINGS section 16 is next). What follows is how it
was built and run. E4p is **pre-registered in `PREREG_GN.md` Amendment 15** (`573142ac`) with
**note i** (`85b43cff`) and **note ii** (`fcd1914f`). It is **exploratory, with no verdict**: E4's rows, measurements and
fine-tuning on **train** (a development scene), before E4. It may only raise E4's process count or fix bugs.

**Kaggle steps:**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_e4p_bench.ipynb` and title it
   **E4p C3DGS fork pilot**.
2. Attach **"R5 tilequant"** (the run-5 output) and **"E3p INRIA pilot"** (the E3p notebook output).
3. GPU T4 x2 (both are used), Internet on (the OGC clone, the dataset). Save & Run All.
4. Bring back `/kaggle/working/E4p_bundle.zip` (look in `~/Downloads`); its arcname is `gn4p/`.

| Attach | Required | What E4p takes from it |
|---|---|---|
| **"R5 tilequant"** | no, but attach it | **`wheels/` only**: the gsplat wheel, reused when its key matches |
| **"E3p INRIA pilot"** | no, but attach it | **`e3p_inria/` only**: train's three pinned members, re-checked by size and CRC32 |
| this notebook's own earlier output | only to resume | `gn4p/` and `gn4p_work/` |

**What it does:** restore; install (as E3r, plus `psutil`); **build C3DGS once** (`gn_e4p_scene.py --build_only`,
`gn4p_c3dgs_build.json`); then **two jobs in parallel**, both with `--keep_data`:
- **GPU 0, `gn_e4p_fork_train`:** the OGC clone (`/tmp/ogc_fork`, HEAD `49ccae72` or the job stops with exit 3 before
  any row); the probe run; E3r's harness phase (`rho_cv`); note ii's geometry, the uncompressed model's renders at
  the orbit cameras and the coverage; processes seeded 0, 1 and 2, each one C3DGS run with the fork, then every row's
  `.npz` measured, decoded and evaluated; the probe evaluated from its `.npz` after process 0.
- **GPU 1, `gn_e4p_ogc_train`:** its own clone (`/tmp/ogc_ogc`); OGC's dependencies (only missing ones, only into
  `/tmp/ogc_deps`); their Table 19 commands; their exact Gram (`/tmp/gn4p_cache/train_ogc_A.pt`, about 1 GB); our
  16 x 16 `M` recomputed and compared; whether job fork's cached `M` is bit-identical to it.

Rules as Amendment 15 d: images on the GPU first; a C3DGS run out of GPU memory retried with `--data_device cpu`, and
the scene's later runs on the CPU; one whole rerun when a primary row (2, 3, 5) is lost, a failed check included;
secondary failures recorded only; no new C3DGS run after 11.5 h minus the 1,800 s reserve.

**Outputs** (`E4p_bundle.zip`, arcname `gn4p/`; nothing of OGC's, no `.npz` / `.ply` / `.pt`):
- `gn4p_results_train.csv`: `uncompressed`, `probe`, and `p<seed>_<row>` for rows `c3dgs`, `ogc`, `ogc_lam1e6`,
  `gnvq_rho0`, `scalar`, `gnvq_cv` and `c3dgs_ft`, `ogc_ft`, `gnvq_cv_ft` (status, reason, attempt, data device,
  C3DGS's and protocol ii's metrics, the per-view PSNR, bytes, per-array sizes, index entropy and distinct indices,
  fidelity per angle, GN-VQ iterations and stopping, per-row time / GPU peak / host RSS, checks, geometry SHA-1).
- `gn4p_cv_train.csv`: the 7 CV rows (E3r's columns).
- `gn4p_meta_fork_train.json` and `gn4p_meta_ogc_train.json`: steps (each with `host_rss`), runs with the wrappers'
  records (`e4p.fork`: rows, costs, checks, state hashes), processes and attempts, note ii, `session_ram`.
- `gn4p_ogc_train.json`: the clone, `deps`, `table19` (against the published row), `exact_gram`, `comparison`,
  `fork_job_cache`.
- `gn4p_summary.json`, `gn4p_c3dgs_build.json`, `gn4p_env.json`, `timings.json`, the two log tails.

**Estimate (before the run; from E3r's measured train parts, kaggle/E4_DESIGN.md section 7's method; an estimate):**
- **job fork, about 8,700 s (2.4 h) without note ii's renders:** download 219.9 s, runner 46.1 s, protocol ii 8.1 s,
  GN passes 28.5 s, CV 719.9 s, the probe about 366 s, then about 2,440 s per process (about 1,240 s for the six
  rows, about 1,200 s for the three fine-tunes). OGC's two rows are taken at our per-iteration rate (about 74 s
  each); their update runs on the CPU, so that is an assumption.
- **note ii's orbit renders:** at most six times protocol ii's per-row time (E3r: 7.7-7.8 s for 38 views) per row:
  up to about 1,260 s over the 27 rows, plus about 55 s for the reference renders. So job fork is about 8,700-10,000 s.
- **setup:** about 396 s (E3r: restore, install, build).
- **job ogc:** not estimable: their PyTorch renderer has not run on a T4 here (two statistics passes over 263 views,
  7 evaluations of 38 views). Our GN pass (18.2 s) and runner (46.1 s) are E3r's. The deadline caps it.
- **GPU memory:** C3DGS peaked at 4.71-4.74 GB on train in E3r; at the colour step about 4.27 GB of C3DGS's state
  (Amendment 14 g's check) plus OGC's assignment chunk (1.76 GB), so about 6.0 GB; job ogc: their accumulator
  (1.05 GB) plus their renderer.
- **host RAM: never measured** (that is what E4p's `host_rss` records are for); roughly 12-14 GB for both jobs, a guess
  from the arrays they hold.

**After the run:**
1. Commit the bundle unchanged, data only, in `kaggle/gn_e4p/gn4p/` (check each file against its zip entry; CSVs
   CR-insensitively, as before).
2. Check Dace's reading against the files before anything is written. Reading order in `gn4p_summary.json`:
   1. `build`, then `scenes.train`: `dropped`, `scene_device`, `processes` (each attempt: device, kind, oom,
      lost rows, checks), `deviations`, `missing_or_failed`, `failed_steps`;
   2. `ogc_clone` (HEAD), and every row's `status` / `reason` / `checks_ok` in `rows`;
   3. `primary_components` D1 and D2 (`D_sp`, `D_s`, `v_s`, `SD_pool`, `SE_noise`; no verdict), then `secondaries`
      and `per_row_over_processes`;
   4. `note_ii`: `geometry.conditioning`, `coverage`, `fidelity_per_angle`, `terciles`, `power_check` (a proposed
      process count is for a dated note before any E4 code), `cost_per_row`, `fidelity_render_time_total_s`;
   5. `gn_vq_runs` (iterations, stopping), `ogc_row_times`;
   6. `ogc_job`: `deps` (installed or skipped, why), `table19` (each row minus the published), `exact_gram`,
      `comparison`, `fork_job_cache`;
   7. `steps` and `ogc_job_steps` (time, GPU peak, `host_rss_peak_bytes`) and `session_ram`.
3. FINDINGS section 16 (E4p) and its checker; then **the E4 session-assignment note** (Amendment 15 g), from note i's
   estimates corrected by E4p's measured times and host RSS.

## Session decisions (E0, 2026-09-19)

These are the decisions from building E0 whose reasons `PREREG_GN.md` and the code comments don't
give. Where PREREG or the code already records *what* was decided, the entry says so and adds the
*why*. Every decision below was made before E0 ran, so none was made or changed after a result
existed.

### The seven choices flagged at the end of the build

1. **Verdict order: `incomplete` > `invalid` > `fail` > `inconclusive` > `pass`.** The order itself
   is in PREREG Amendments 2-3 and `g0.py`. A misordered non-tied pair means `fail` even when fewer
   than 6 pairs are non-tied. (Under Amendment 2 a ratio outside 0.5-2x was a `fail` too; since
   Amendment 3 the ratio is reported as calibration and never decides the verdict.)
   - `fail` beats `inconclusive` because each failure is evidence against the metric, and a shortage
     of informative pairs can't cancel it. `inconclusive` means "too little evidence either way", so
     it applies only when nothing disagrees.
   - `invalid` outranks `fail` because a failed validity check means the measurement can't be trusted.
   - `incomplete` outranks everything because a missing row makes every count meaningless.
2. **Equal predictions on a non-tied pair count as disagreeing.** This is in PREREG and `g0.py`. The
   rule asks whether P orders the pair the way D does. If P can't separate two codebooks whose
   measured errors differ by 5% or more, P has not ordered them. Counting that as agreement would
   reward a degenerate predictor such as a constant P. Exact float equality is not expected with real
   data; this is a guard.
3. **Assignment guard: a splat keeps its current centroid unless the new one is strictly closer by the
   direct formula.** What it does is in PREREG and in `assign_exact`'s docstring.
   - The lifted argmin is an fp32 matrix product. On near-ties it can pick a centroid that is
     marginally farther by the exact float64 direct distance.
   - Without the guard, an assignment step could raise the objective by rounding. That would mark the
     proximal variant invalid (Amendment 3; before it, the monotonicity assertion stopped the job) for
     a numerical reason rather than a real one.
   - With it, assignment never raises the objective. The number of splats it keeps is logged
     (`kept_current_by_guard`).
4. **Codebook-mean shift before the fp32 product.** What it does, and its precision reason, are in
   PREREG and `lifted_argmin`'s docstring.
   - `d = const + u . v` cancels a large per-splat constant. The fp32 error in `u . v` grows with the
     magnitudes of c and q, not with the distance itself.
   - Shifting both by the same vector leaves every distance unchanged but keeps c and q small, so the
     cancellation error stays small relative to the distances being compared.
   - I chose this over a float64 product because T4 float64 throughput is a small fraction of fp32.
     The 10k-real-splat check shows whether it is enough.
5. **Splats with `tr(M_i) = 0` take their L2-nearest centroid.** This is in PREREG and the code.
   - These splats are seen in no train view, so every centroid is at distance 0 and the exact argmin
     is undefined; an fp32 argmin over all-equal scores would return index 0 by accident.
   - L2-nearest is what plain k-means gives them, which keeps them reasonable in test views, where
     some of them may be visible. They add nothing to the GN objective either way.
   - The top-64 share is reported over all splats and over `tr(M) > 0`, so their effect on it is
     visible.
6. **Train-view metrics for every row, not only the refines.** PREREG Amendment 2 ("Also logged")
   states it.
   - The refines are fit on train-view `M`, so their train/test gap is the number of interest. That
     gap needs the same numbers for the G0 codebooks as a reference.
   - They are computed with the eval's render call and the runner's own PSNR / SSIM / LPIPS modules,
     averaged per image as `Runner.eval` does.
   - `metric_parity` in `gn_meta_<scene>.json` records that code on the test views against
     `Runner.eval`, once per scene. It is recorded, not gated, because a mismatch would affect only the
     new train columns, not G0.
   - It costs one extra pass over the train views per row, not timed yet.
7. **The random-data smoke check is informational.** `selftest.py`'s docstring says so; the reasons:
   - The lifted-vs-direct check on random data in the smoke cell catches a CUDA-specific break of the
     lifted assignment early.
   - It doesn't stop the run because only the exploratory refines use the lifted assignment. Stopping
     there would also lose G0, which doesn't depend on it.
   - The gating check is the one on 10,000 real splats inside each job, placed after the G0 rows, and
     it gates only the refines.

### Other decisions not written elsewhere

- **`plain_l2` is run 3's `lloyd_w1` (the library's Lloyd without weights), not run 3's `euclid`
  (TorchPQ euclidean).** It is the same algorithm and code as `lloyd_wopa_area` without the weights,
  so the `plain_l2` / `lloyd_wopa_area` pair isolates the weighting. PREREG has the mapping but not
  this reason.
- **Failed validity checks stop the run early instead of letting it finish as `invalid`.** The smoke
  cell raises if the SH, toy or end-to-end check fails, and each job raises on a render-parity failure
  before any GN work. G0 would be invalid anyway, and stopping saves the GPU session.
  - The reproduction check is the exception: it can only be judged from the rows, so the G0 cell
    evaluates it and it never stops anything.
  - **Superseded on 2026-09-19 (Amendment 3), for variant-local checks only.** A proximal-objective
    rise used to raise and stop the job (and with it the notebook, before G0). Now it is logged, that
    variant's row gets `valid` = False, and the job continues with the remaining rows and the notebook
    with G0 and the bundle. The 10k lifted check stops only the refines: the job used to raise there
    too, which also lost the `uncompressed` row and the in-session G0 cell; now it records the failure
    (`refines_skipped`) and continues. The pipeline-wide checks above (render parity, the SH, toy and
    end-to-end checks) still stop the run early.
- **Nothing later from `feat/png-weighted-kmeans` was merged into `bench/gn-vq`.** `bench/tilequant`
  already contains the weighted k-means backend (`9348e32`). The later commits add
  `kmeans_chunk_size`, which E0 doesn't need because it calls `weighted_kmeans` directly. They also
  flip `PngCompression`'s defaults, which would silently change any default-constructed
  `PngCompression` in the harness. E0 passes the backend explicitly, but some older harness code
  relies on the defaults.
- **The lifted check is not repeated on resume once it has passed under the current criterion
  version** (`lifted_check.pass` and `lifted_check.criterion_version` in `gn_meta_<scene>.json`;
  `diagnostics.lifted_check_needed`). It depends only on the GN cache and the warm-start codebook, and
  a resume reuses both. A failed record, or one without the current version (an Amendment-2 record has
  none), is re-run. The check only runs while a refine row is still missing.
- **The top-64 share at each assignment step uses the codebook of that assignment**, before that
  step's update. It asks whether an L2 shortlist of the current codebook would have contained the
  exact argmin.
- **An `uncompressed` reference row per scene** (test and train metrics of the checkpoint itself) gives
  the loss of each compressed codebook without depending on earlier bundles.
- **Degenerate cases in `g0.py`,** none expected with real data; re-derived under Amendment 3 (which
  now states them) and each tested in `test_g0_degenerate_cases` and
  `test_g0_inconclusive_invalid_incomplete_and_tie_boundary`:
  - an empty validity dict counts as `invalid`, because it means the selftest results were never
    recorded (`incomplete` still outranks it);
  - a pair with `min(D) = 0` is a tie only if both are 0, because the relative difference is
    undefined; a pair with one zero is non-tied and judged like any other. Zero measured errors
    everywhere therefore give 0 non-tied pairs and `inconclusive`, never `pass`;
  - `D_train = 0` gives a ratio of +inf (NaN when `P = 0` too), which is not calibrated. Under
    Amendment 2 that failed the range and made the verdict `fail`; now it does not affect the
    verdict. The cross/diagonal ratio is -1 when `D_unclamped = 0 < P`, +inf when `P = 0 < D`, and NaN
    when both are 0.

### Amendment 3 session (2026-09-19)

Choices made while implementing Amendment 3 whose reasons PREREG and the code don't give. As before,
no E0 result exists, so none was made after a result.

- **The lifted check samples 10,000 splats from the whole scene, then evaluates those with
  `tr(M_i) > 0`** (Amendment 2 drew the 10,000 from `tr(M_i) > 0` splats only). This is what makes the
  logged "count of `tr(M_i) = 0` splats in the sample" meaningful: with the old sampling it would
  always be 0. The cost is that fewer than 10,000 splats are evaluated, short by the number of unseen
  splats in the sample, which is logged.
- **The end-to-end check runs in the smoke-test cell**, the first cell that does E0 work. It can't run
  earlier because it needs the installed CUDA gsplat, and it runs before the data download and the
  scene jobs, as required. The cell's inline script moved into `bench/gn/selftest.py`, so the dry run
  and pytest execute the same code on the CPU stand-in; the notebook only calls it.
- **`e2e_exactness` is also in the G0 validity dict,** like the SH and toy checks: it gates the run the
  same way, and listing it makes `gn_g0.json` self-describing.
- **The e2e scene is drawn with a CPU generator and moved to the device.** The CPU test then checks
  the exact scene the CUDA check renders, preconditions included. 48 splats because gsplat renders an
  identity image in chunks of 32 channels, and the remainder (16) must be a compiled channel count.
- **An extra reported case, `overlapping_shared_delta`:** the overlapping scene with one perturbation
  for all splats. The requested overlapping case uses independent per-splat perturbations, whose cross
  terms average out; the shared one is the fully correlated case that Amendment 3's reason describes.
  Nothing asserts on either.
- **After a proximal rise the variant still runs its 3 iterations** and its row is written with the
  flag, so the output has the same shape either way and the rise is visible in context. Only the
  proximal variant is held to monotonicity; the ridge variant pulls toward 0 and may legitimately
  raise the GN objective.
- **`calibrated` flags only the train ratios** (clamped and unclamped). `P` is accumulated over the
  train views, so only there is it a prediction of the same pixels; the test ratios are reported
  without a flag.
- **`g0.judge_g0(..., configs=...)`** exists so that E1 can re-judge an inconclusive G0 over its
  non-GN rungs with the same code.
- **`selftest.py` never adds the repo to `sys.path`.** On Kaggle that would import the source tree
  instead of the installed wheel; locally the CPU stand-in is run with `PYTHONPATH`.
- **Commit split as for Amendment 2:** the builder with the code, the rebuilt notebook with the docs.

### Amendment 4 session (2026-09-20)

Dace decided the toy-scene RNG item: fix it, as a correction (Amendment 1 already says the simulated
scene is the scene the CUDA check uses). Re-deriving it showed that drawing on the CPU is not enough,
and Dace chose committed fixtures over a fresh draw on Kaggle (bitwise or with a tolerance). As
before, no E0 result exists.

- **Why fixtures:** torch's CPU `randn` (float32, 16 or more values) uses an AVX2 SIMD approximation
  of log/sin/cos under AVX2 dispatch and scalar libm calls under DEFAULT or AVX512 dispatch
  (`ATen/native/cpu/DistributionTemplates.h`). On this machine (torch 2.11.0+cpu, Windows, AVX2 by
  default), `ATEN_CPU_CAPABILITY=default` changes 5,037 of the toy scene's 15,104 values (max abs
  1.76e-6) and both scene hashes (toy `f01d6b50...`, e2e `68934fd2...`). Kaggle's CPU and torch build
  are not known in advance, so a bitwise hash of a draw made there would most likely have stopped the
  run.
- **What was verified:** `make_fixtures.py --verify-commit c69ba388` re-derives the toy scene with
  `c69ba388`'s own `gn_metric.py` in a separate process: same hash as the current code's draw
  (`1bb442ee...`). The scene-only fields of `toy_noise.json` (blending fraction 0.8680054926192928,
  256 visible splats) reproduce exactly, but also from the DEFAULT-dispatch draw, so they cannot pin
  the dispatch; the simulation ran with this machine's default (AVX2), and no override was recorded.
  The probe simulation was not re-run.
- **Format:** `.npz` of float32 little-endian arrays plus a `.json` (hash, hash definition, shapes,
  CPU capability, torch version, platform, provenance). `fixtures/.gitattributes` marks `*.npz`
  binary, so `autocrlf` never touches them; the committed blobs equal the files byte for byte.
- **Hash:** SHA-256 over the sorted keys, each as `"<key>:<shape>:float32-le;"` then its bytes, so
  the hash does not depend on the file format. `load_fixture` checks it (and the one in the `.json`)
  before the tensors are moved to the device.
- **The e2e fixture includes the perturbation** (`delta`). The perturbation comes from the CPU
  generator too: `torch.rand` is exact integer-to-float, but committing it removes the question. The
  overlapping variants are derived on the device from the fixture (scales x5), as before; they are
  report-only.
- **E2E colour precondition reads (0, 1), not only > 0.** Dace's list says `SH + 0.5 > 0`; Amendment 3
  already pre-registered (0, 1), and the stricter range keeps both true. It is read from gsplat's own
  render: in a pixel covered by one splat, SH render / identity render = that splat's
  `max(SH + 0.5, 0)`. On the CPU stand-in it matches the basis value to 1e-5.
- **"All rendered values in (0, 1)" is read as:** covered pixels in (0, 1), uncovered pixels exactly
  0. With a background of 0 an uncovered pixel can't be in (0, 1); it contributes nothing to either
  side, so clamping changes nothing. That is Amendment 3's wording, kept.
- **Report-only diagnostic:** `hutchinson_rel_std` and `false_fail_probability` in `gn_metric`,
  stored in `toy_exactness["noise_diagnostic"]` and logged before the check is judged. On the CPU
  stand-in (the same fixture scene) `sigma_rel` is 0.01481 and the false-fail probability 0.000733
  (the dry run's stage 0 prints them). The CUDA value is the one the run will log.
- **A pre-existing test was fixed:** `test_predicted_dmse_matches_loop` compared a float32 reference
  loop to 1e-9 absolute and failed under DEFAULT dispatch; the reference is now float64.
- **Risk:** if a future torch changes its CPU RNG algorithm, the provenance test (tolerance mode)
  would fail on that machine. The CUDA checks are unaffected, because they read the files.

### Batched-linalg session (2026-09-20, after E0's first run crashed)

The crash and the fix are in the open items below and in `bench/gn/batched.py`'s module docstring.
These are the choices around them that neither records.

- **The default batch is 32,768, not the 65,536 Dace asked for.** `eigen_stats` already looped in
  chunks of 65,536, so 65,536 *was* the batch cuSOLVER refused: a helper defaulting to it would have
  left the failing call unchanged and crashed again. 32,768 is half of it and below the 65,535 grid
  limit. E0's run then measured the rest (see `OP_MAX_BATCH` below).
- **The helper also halves on demand,** because the real limit could not be checked without a GPU. It
  retries only when the backend's message names a batch or memory problem (`_RETRY_MARKERS`); a
  matrix that fails to converge is raised unchanged, so a genuine numerical failure is never buried
  under a ladder of retries.
- **Every batched linalg call goes through it, including the small ones** (`sh_basis.cuda_check`'s
  qr / det / inv at batch 3, `gn_metric.viewmat_of`'s inv_ex at batch 1). Uniformity means no call
  site is forgotten later; a call below the batch limit is a pass-through that returns the op's own
  result object, so values are bitwise unchanged. That is why the GN cache stayed valid across the
  fix, and a test pins `viewmat_of` against the unrouted call.
- **`linalg_scale` is an engineering check, not a G0 validity entry.** It is not pre-registered, so it
  must not enter the validity dict that `gn_g0.json` records; but a failure means the job cannot run,
  so it still stops the notebook. `selftest.py` keeps it in `ENGINEERING`, separate from `VALIDITY`.
- **It runs the job's own functions at the job's sizes** (`eigen_stats` over 1,000,000 packed float64
  matrices, `update_centroids` over 65,536 systems, both refine variants) rather than synthetic
  `torch.linalg` calls, so it exercises the same call path, dtypes and allocation pattern that
  crashed. It reports only the reductions made during its own run, not the process's whole history.
- **A backend failure inside it is caught and recorded,** so `gn_selftest.json` is still written and
  the run stops with a readable message instead of a bare traceback.
- **The finite check runs right after `M` is loaded or computed,** before `trace_packed` and before
  any linalg, and the job writes the meta file (with the counts) before raising. cuSOLVER reports
  non-finite input as an opaque backend error, which would have looked like another batch problem.
- **The per-job log tails** are written in cell 6's `finally`, before the bundle, from exit codes that
  `run_on_gpus` records in `JOB_EXITS` as each job ends. 200 lines was enough to hold E0's whole
  refine phase; a job that never wrote a log gets `exit_code: null` and an empty tail rather than
  being skipped.

### E0 results and E1 build (2026-09-20)

E0 ran, G0 passed, and E1 was pre-registered and built in the same session. Amendment 5 carries the
rules; `gn_vq.py`, `g1.py` and `gn_e1_scene.py` carry the mechanics. These are the choices behind
them.

- **The bundle is committed exactly as downloaded** (`kaggle/gn_e0/gn/`), each file checked byte for
  byte against its zip entry before committing. The CSVs keep the writer's CRLF on disk and `autocrlf`
  stores them as LF, as for the run 2-5 bundles; the PNGs are stored binary. Nothing was regenerated.
- **FINDINGS section 8's numbers were checked mechanically:** a script regenerated every table row,
  range and timing from the bundle and asserted each string appears in the section (0 failures). That
  is how the "every number from result files" rule is enforced when a section has this many.
- **The quantizer hypothesis is labelled a hypothesis** because the bundle cannot test it: it holds no
  centroids, so the two supporting observations (the quantized objective worsening while the
  unquantized one improves, and `centroids.npy` shrinking 13-45%) are consistent with it but do not
  establish it. E1's clip is the test.
- **Amendment 5's one-sided size rule was Dace's call.** G1's parenthetical is two-sided and the rule
  Dace specified is not; the work stopped there and Dace chose the one-sided form, plus the dominance
  flag and the rate-distortion reporting. The reason is recorded in Amendment 5 b.
- **The 19 rows per scene** follow from Amendment 5 and nothing else: G1 needs the baseline and GN-VQ
  at K = 65,536 for seeds 0, 1, 2 (6 rows); the two secondary weightings are "same K, same seeds"
  (6); the rate-distortion grid is seed 0 at K = 4,096 and 16,384 for both the baseline and GN-VQ (4);
  the two ablations are seed 0 at K = 65,536 (2); plus an `uncompressed` reference (1). The grid and
  the ablations are seed 0 only because Amendment 5 says so, which also keeps the run affordable.
- **E1 reuses E0's GN cache.** `gm.CACHE_VERSION` is unchanged and nothing on the GN path changed, so
  the key matches and the GN pass is skipped; the restore cell warns if no `gn_cache/` was attached
  and each job recomputes it (about 12 s per scene). Attaching E0's output is the documented step.
- **Train-view SSIM and LPIPS are dropped, train PSNR is kept.** No rule uses SSIM or LPIPS on train
  views, and computing them meant LPIPS-VGG over every train view (161 on garden, 169 on bicycle) for
  every row, the most expensive part of E0's per-row cost. Train PSNR stays because the train/test gap is the number of interest for
  a codebook fitted on train-view `M`.
- **The top-64 shortlist diagnostic runs at iteration 1 only.** In E0 it cost 12.4-13.0 s per
  iteration against 9.4-9.5 s for the assignment it checks, and the question it answers - would a
  plain-L2 shortlist have contained the exact argmin - only matters at the warm start, where a
  shortlist implementation would begin. E0's answer was 0.43 (garden) and 0.51 (bicycle) at iteration
  1, falling after that.
- **`OP_MAX_BATCH["linalg_eigvalsh"] = 8192` and a working batch is remembered.** E0 measured it:
  32,768 refused, 16,384 wanted 9.49 GiB, 8,192 ran; and the helper rediscovered that ladder on every
  chunk, 15 times per job. `LINALG_MAX_BATCH` stays 32,768 for everything else, because the
  65,536-system centroid solve was never a problem and chunking it smaller would only cost time.
- **One `dryrun/fake_env.py` for both dry runs.** E1 needed the same toy checkpoint, sort cache,
  run-3 caches and fake runner; a second copy would have drifted from E0's. Extracting it changed the
  toy's random permutation (the order now comes from its own generator), which changes the numbers the
  E0 dry run prints but none of its assertions, since those are structural.
- **The writer round-trip test pins `gn_vq`'s quantizer against the real one.** GN-VQ has to quantize
  the codebook itself, to measure the objective after quantization and to run the final assignment
  against the dequantized codebook, so its arithmetic must be the codec's. The test writes a codebook
  through gsplat's own writer and asserts the stored codes equal `quantize_codebook`'s and the decoded
  values equal `dequantize_codebook`'s bit for bit. The job repeats the code comparison for every row
  (`writer_codes_equal`) and raises on a difference, which is Amendment 5's "assert that re-quantizing
  gives the same codes".
- **The quantizer mirrors the codec's dtypes, not just its formula.** A float64 version of
  `mins = min + 1e-6` differed from the codec's float32 sum by 4.6e-8, enough to move a code at a
  boundary, so `_codec_bounds` adds the epsilon in float32 and the decode uses the float32 span the
  metadata stores. The reported `quant_step` is therefore the codec's float32 arithmetic, which agrees
  with a float64 recomputation from the stored range only to float32 precision.
- **`g1.py`: the size rule lives in `compare_seed`,** as `size_ratio = bytes / baseline_bytes - 1`,
  `size_ok = size_ratio <= 0.005` and `negative = dPSNR < 0 or not size_ok`, so a seed that breaks the
  size rule is negative whatever its PSNR did. The per-scene mean is the mean of the **measured**
  PSNR differences even then: the verdict already fails through the negative seed, and a mean that
  silently rewrote one seed's number would be harder to read. `dominates` is `bytes <= baseline` and
  `PSNR >= baseline`, computed per seed, reported and never read by the verdict. `judge_pair` is
  shared by G1 and the two secondary comparisons, so they cannot drift apart; only the GN-VQ vs
  `lloyd_wopa_area` call is the verdict.
- **BD-rate returns NaN when the two curves do not overlap in PSNR,** rather than extrapolating. The
  E1 dry run asserts NaN exactly when there is no overlap, which is what the toy produces.
- **G1 has no validity dict.** G0's validity checks are pre-registered and G1's are not, so inventing
  an `invalid` verdict would be adding a rule after results. Instead the E1 job stops on a render-parity
  failure or a writer-code mismatch, and a failed lifted check skips the GN-VQ rows, which leaves G1
  `incomplete` rather than wrong.
- **E1 got its own builder and notebook, and imports E0's job module** for the writer, the renderers
  and the clustering cache. E0's notebook has to stay byte-identical to the version that ran, and
  sharing the writer keeps E1's rows comparable with E0's.

### Amendment 6 session (2026-09-21, before any E1 run)

Dace asked for two more exploratory GN-VQ rows and for a check that attaching E0's output cannot leak
E0 rows into E1. The amendment was committed first, docs only, before any code. E1 has not run, so
again nothing here was decided after a result.

- **Two rows, not a sweep.** Amendment 6 fixes `eps` = 1e-3 and 1e-2 at seed 0 and K = 65,536 on both
  scenes, one and two orders of magnitude above the pre-registered 1e-4. Dace named the values and the
  scope; the reason recorded in the amendment is that the ridge is the knob for two risks at once
  (overfitting to train-view directions, and centroid extrapolation widening the quantizer's global
  range), with E0's train/test gap and E0's reading of the codec's quantizer as the evidence for each.
  Both numbers in that rationale come from `kaggle/gn_e0/gn/gn_results_bicycle.csv`, recomputed from
  the file: 34.51% on train views, 23.16% on test.
- **The ridge lives in `VQ_CONFIGS`, not in a new flag.** `gn_vq()` already took `eps`; the job passed
  `--eps` to every row. Now a config may name its own `eps`, the loop pops it, and `--eps` is the
  default for the rows that do not. So the pre-registered row keeps `vq.RIDGE_EPS` and the two new
  rows differ from it in one dict entry, which is what "identical otherwise" has to mean in code. A
  test asserts each new config equals `gn_vq`'s once `eps` is removed.
- **`wanted()` was not touched.** It already gave every GN-VQ variant except `gn_vq` seed 0 at G1's K,
  which is exactly Amendment 6's scope, so the two rows needed no scheduling code. That is why E1 goes
  from 19 to 21 rows and no other row moved.
- **Exploratory means picked by nothing, and that is now pinned by a test.** `g1.py` selects rows by
  config name, so the four exploratory rows were already invisible to G1, to the secondaries and to
  the rate-distortion curves. The test adds all four on both scenes at every K with PSNR 99 and 1 byte
  and asserts the entire verdict object is unchanged - a stronger statement than checking the verdict
  string, and it would catch a future comparison that globbed configs instead of naming them.
- **Both quantizer ranges were already computed; only the reporting was missing.** `report["quantizer"]`
  is the final codebook's own range and `report["warm_start"]["quantizer"]` the warm start's, but the
  CSV carried only the first. Amendment 6's logging item is therefore four new columns
  (`warm_quant_mins` / `_maxs` / `_step`, `ridge_eps`), not new arithmetic, and the dry run asserts the
  CSV's warm columns equal the report's dict.
- **The isolation was already true, and is now checked.** E0 writes `gn/` and `gn_work/`; E1 writes
  `gn1/` and `gn1_work/`; `discover` matches exact directory names, so no E0 result was ever going to
  be restored. Nothing *verified* that, so a rename or a hand-copied file would have broken it
  silently. The five layers listed under "E1 notebook" make it a checked property at every stage the
  rows pass through - restore, resume, CSV, verdict, bundle - and stage (4) of the E1 dry run exercises
  each on a fake `/kaggle/input` holding both outputs.
  - **`write_bundle` warns instead of raising** because it runs in the jobs cell's `finally`; raising
    there would replace a real job failure with a packaging error. Every other layer raises.
  - **`assert_e1_csv` compares the whole header** rather than looking for a marker column: E0's header
    differs by several columns in both directions, and an exact comparison also catches a CSV from a
    future E1 whose columns changed, which is the case that would silently misalign `DictWriter`.
  - **Nothing deletes anything,** here or in the restore cell. A refused file is reported and left
    where it is.
- **The stray `gn_bundle (1).zip` is ignored, not deleted,** by an anchored pattern in the root
  `.gitignore` matching that one name. Its contents are committed unpacked in `kaggle/gn_e0/gn/`, and
  no bundle zip is ever committed.

### E1 results and E2 build (2026-09-21)

E1's bundle was committed, FINDINGS section 9 written, and E2 pre-registered (Amendment 7) and built
in one session. No E2 result exists, so every E2 choice below was made before one.

- **The bundle is committed exactly as downloaded,** 28 files, each checked byte for byte against the
  zip on disk and again as a staged blob (the PNG exactly, the text files CR-insensitively).
- **FINDINGS section 9 was checked mechanically,** as section 8 was: a script recomputed every number
  from the bundle, and every numeric token in the section was matched against its output (the only
  unmatched tokens were range dashes and commit hashes). That check caught one wrong sentence before
  commit: the secondary clusterings were "most of each job's time" on garden (60.9%) but not on bicycle
  (44.9%).
- **Post-hoc material in section 9 is labelled as such:** the scalar weightings against
  `lloyd_wopa_area` (computed with the same `judge_pair`, not pre-registered) and the cross-K
  comparisons. Nothing in section 9 re-judges G1.
- **A logging quirk worth knowing:** `quant_step` (the written codebook, computed on the CPU) and
  `warm_quant_step` (the warm start, on the GPU) can differ by a float32 ulp (up to 3.73e-09) at
  identical min and max. Compare the min and max, or `quant_step` against the `lloyd_wopa_area` row's.
- **Do the 11 checkpoints exist?** Dace asked for this before any E2 code, and to stop if one is
  missing. There are no Kaggle credentials here, so the run-5 output cannot be listed directly. What
  the committed bundles show, per scene:

  | Scenes | Trained | Sort cache built | Evidence it is in the run-5 output |
  |---|---|---|---|
  | garden, bicycle | training #2 (runs 1-3) | runs 1-2 | E0 and E1 restored both checkpoints and sort caches from the run-5 output; E1's meta records the same sha1s |
  | stump, bonsai, counter, kitchen, room, treehill, flowers | run 4 (`run4_train_<scene>_s`) | run 4 (`run4_sort_<scene>_s`) | run 5's discovery cell raises unless all 9 MipNeRF360 checkpoints are restored into `/kaggle/working/results/`; run 5 evaluated each (`run5_results.csv`, one sha1 per scene) and did not re-sort them (no `run5_sort_` timing), so their caches were restored too |
  | train, truck | run 5 (`run5_train_<scene>_s`), into `results/benchmark_tt_mcmc_1M_png_compression/` | run 5 (`run5_sort_<scene>_s`) | written into `/kaggle/working` by run 5 itself |

  Nothing in the tilequant notebook deletes a final checkpoint or a sort cache (its cleanups remove
  `/tmp` data, run directories and `renders/`). So no checkpoint is known to be missing, and none was
  designed around; but for 9 of the 11 the only direct confirmation will be E2's restore cell, which
  lists every missing checkpoint or sort cache and checks every sha1 before any install.
- **Checkpoints are pinned by sha1** (Amendment 7 d), in both the notebook and each job, because E2's
  scenes are only comparable with runs 4-5 and with each other if they are the same checkpoints.
- **Amendment 7's reasons were recorded from the files, not restated.** "eps = 1e-2 had the best dev
  RD" rests on single points at K = 65,536 (the ridge rows have no curve), so the amendment says that
  and gives the numbers; "seed spread <= 0.006 dB" is true of G1's paired difference (0.0058 / 0.0040
  dB) but not of every config's PSNR (up to 0.0087 dB for `lloyd_trace` on garden), so the amendment
  gives both. Also recorded as not measured: eps = 1e-2 below K = 65,536.
- **Choices Amendment 7 makes that Dace's outline left open:**
  - **degree 3 for both BD measures** (the classic Bjontegaard cubic, exact on four points; E1's
    reported BD-rate was degree 2 on three points, also exact). `g2.bd_rate` is `g1.bd_rate` at degree
    3, so E1's function is reused, not rewritten;
  - **BD-PSNR's definition** (PSNR as a cubic in log10 bytes, averaged over the common byte range);
  - **"mean BD-rate over the scenes where it is defined" with no scene defined is "not met"**, so G2a
    fails. This is the conservative reading of an unstated case. It matters: garden in E1 was exactly
    this case (GN-VQ entirely above), so if every held-out scene looked like garden, G2a would fail on
    magnitude with 9 wins. **Superseded by Amendment 8** (below), before any E2 data;
  - a non-finite BD value counts as undefined; a BD-rate of exactly 0 is not a win; a missing row
    makes the scene `incomplete`, and any incomplete held-out scene makes the verdict `incomplete`.
- **E2 imports E0's and E1's job modules and changes neither.** The per-row measurement in E1 is a
  closure inside `gn_e1_scene.main`, so E2 carries its own copy (`evaluate_row` in `gn_e2_scene.main`)
  rather than refactoring code that produced E1's results; the clustering caches, the writer, the
  renderers and `train_psnr` are called from E0 and E1 directly.
- **Row order is the G2a pair first at each K** (`lloyd_wopa_area`, then `gn_vq` warm-started from
  it), so a scene cut short still has its gate rows; the lifted check needs the K = 65,536 warm start,
  which comes from a cache on 9 scenes.
- **The queue is first-free-GPU, not E1's paired batches,** because 11 jobs of different lengths would
  leave a GPU idle behind every long one; the Tanks & Temples scenes go first because they are the
  longest (larger downloads, data factor 1). **A failed scene does not stop the queue** (E1 raised
  after the batch): scenes are independent and a missing one only makes the verdicts incomplete, so the
  G2 cell still runs and the bundle cell raises at the end instead. **The 9.5 h start cutoff** exists
  because a Kaggle run killed at 12 h may not keep its output.
- **Each job downloads and deletes its own scene,** as run 4 and run 5 did (with their downloaders and
  their lock), instead of E1's notebook-level download: 11 scenes of data would not fit `/tmp` at once.
- **Only the two usable cache files are copied** from the run-3 / run-4 clustering directories, not
  the whole directories, to keep `/kaggle/working` small next to 11 checkpoints and 11 GN caches.
- **`gn1_bundle.zip` is ignored, not deleted,** by an anchored line in the root `.gitignore` next to
  E0's, following the procedure the previous session recorded for E1's bundle.

### Amendment 8 session (2026-09-21, before any E2 data)

Dace fixed G2a's mean so that it always exists, and asked for exploratory eps = 1e-4 rows. No E2 row,
bundle or log existed; the amendment says so and was committed before the code.

- **The exploratory rows went into Amendment 8 too** (part b), not only into code, because the rule
  is to pre-register before results; E1's ridge rows were handled the same way (Amendment 6).
- **Formulas as given, with three details fixed in writing:** a curve's best point is its highest PSNR
  with a tie going to fewer bytes (not expected with measured PSNRs); "fewer bytes" is strict and
  "PSNR >=" inclusive; and a non-finite BD-rate takes a substitute like an undefined one. With no PSNR
  overlap only one of a and b can apply, so their order matters only in those degenerate cases.
- **The worked example was recomputed from the bundle,** not taken from the request: -9.7654% for
  garden (`gn_vq` K = 4,096, 14,803,113 B, against `lloyd_wopa_area` K = 65,536, 16,405,132 B); bicycle
  has a defined BD-rate and would enter with it.
- **"Same construction elsewhere":** H2b's output already carried a (reported) mean, so it now uses
  the substitutes; its verdict is still the win count. The `upstream_l1` comparison never had a mean,
  so none was added; its per-scene substitute is reported.
- **"Queued after everything else" is a second queue,** started only when the scene queue has
  returned, rather than extra rows at the end of garden's and bicycle's jobs. Inside the dev jobs they
  could have run while other scenes' pre-registered rows were still running, could not be stopped by
  the start cutoff, and a separate job for the same scene running concurrently would share its CSV and
  meta. The main dev jobs keep their data (`--keep_data`) so the phase does not download again; the
  exploratory jobs delete it.
- **An exploratory-only invocation cannot mark a scene done:** it skips the `uncompressed` row, and
  the meta's `done` / `missing_rows` are now counted over the full pre-registered set whatever
  `--configs` asked for. Before this, `missing_rows` covered only the requested configs.
- **`g2.check_rows` refuses an exploratory row on a held-out scene,** and the job refuses to run one
  there, before any work.
- **Two tests encoded Amendment 7's mean rule** and were replaced, not loosened; the all-undefined case
  is now tested both ways (it failed with 9 of 9 wins under Amendment 7; it passes under Amendment 8,
  and its mirror image fails).

### E2 build and Amendment 8: decision index (2026-09-22)

Every decision from the E2 build and the Amendment 8 session, each with its reason and where it is
recorded. Where the two subsections above already give the reason, the entry says so and adds only
what they leave out. All of these were made before any E2 data existed.

**Scheduling**

- **The exploratory rows run in a second queue** (`run_gpu_queue(explore_jobs, ...)`, called only
  after `run_gpu_queue(jobs, ...)` has returned). Recorded: PREREG Amendment 8 b ("queued after
  everything else ... they never delay a pre-registered row"), the builder's jobs-cell comment, and
  "Amendment 8 session" above (why not rows inside the dev jobs, why not a concurrent job).
- **Why exploratory work cannot delay a gate row:** the 9 held-out jobs are the first 9 entries of the
  first queue, which starts jobs strictly in list order on the first free GPU; the exploratory jobs are
  not in that queue at all; and the second queue is only called once the first has returned, which
  `run_gpu_queue` does only when every one of its jobs has exited or been skipped by the cutoff. So an
  exploratory job never holds a GPU while a pre-registered row of this session is still pending.
  Inside one invocation the job also writes exploratory rows after every pre-registered row
  (`todo_x` after `todo` and `uncompressed` in `gn_e2_scene.main`). What they can still do is extend
  the session's wall time after the gate work; the start cutoff bounds that, next item.
- **The 9.5 h start cutoff** (`START_CUTOFF_S`, builder config cell, with its comment; the reason, that
  a Kaggle run killed at 12 h may not keep its output, is in "E1 results and E2 build"). Why 9.5 h: it
  leaves 2.5 h (9,000 s) for a job that starts just before it, against E1's longest measured job,
  garden at 6,450.4 s (21 rows including six ~650 s clusterings, FINDINGS section 9). E2's Tanks &
  Temples jobs at data factor 1 are not measured, which is the residual risk; the cutoff applies to
  both queues, so the exploratory jobs (4 GN-VQ runs and 4 evaluations per scene; E1 measured single
  GN-VQ runs at 61.8-156.7 s) cannot start late either.
- **Garden and bicycle keep their data until the exploratory jobs finish** (`--keep_data` on their
  main jobs; the exploratory jobs run without it and delete it). Recorded: the builder's jobs-cell
  comment, "Amendment 8 session" above. Reasons it is right and safe: it avoids a second download of
  each (run 5 measured 141.1 s for garden and 69.0 s for bicycle); nothing else touches those
  directories, because the exploratory jobs start only after the main queue has returned; the extra
  `/tmp` use is at most the two dev scenes plus the one scene still running on the other GPU, and
  run 4's downloader checks for 1.5x the download's size in free space before every download
  (`r4.MIN_FREE_FACTOR`), so a shortage fails loudly rather than filling `/tmp`; and if the cutoff skips
  the exploratory phase, the kept data simply dies with the session.
- **A cutoff skip is not a failure; a crash is.** `JOB_SKIPPED` is printed with a resume hint and the
  notebook ends normally, because a skip is the planned way to split E2 across sessions; `JOB_FAILED`
  makes the bundle cell raise, last, after the bundle is written, so the failure is visible and nothing
  is lost. **This includes an exploratory job:** its failure also ends the notebook in an error, on
  purpose (a silent exploratory failure would hide a bug), and it cannot affect a gate row, which ran
  first. Recorded: the builder's bundle cell; not written elsewhere until now.
- **Each log tail records `skipped_by_cutoff`** (`write_log_tails`). Both a job the cutoff never
  started and a job still running when the notebook itself stopped have `exit_code: null`; the flag
  tells them apart. A job that crashed has its exit code.
- **A job whose rows all exist returns before downloading anything** (`gn_e2_scene.main`), so a
  resumed session does not re-download finished scenes (run 5 measured 508.4 s for train's download).

**Guards and completeness**

- **The job refuses exploratory configs on a held-out scene** (`ValueError`, before the checkpoint is
  hashed or anything is downloaded) and **`g2.check_rows` refuses an exploratory row on a held-out
  scene.** Recorded: the job's module docstring and `EXPLORATORY` comment, `g2.py`'s docstring,
  PREREG Amendment 8 b ("on garden and bicycle only"). Reasons: Amendment 8 b restricts them to the
  development scenes, and a row of a dev-tuned variant sitting next to a gate result invites exactly
  the post-hoc comparison the held-out design exists to prevent; refusing in the job means a wrong
  `--configs` costs no GPU time; refusing in `g2` means the verdict's input is checked however the rows
  got there (a hand-edited CSV, another notebook version), as `g1.check_rows` does for E1.
- **Completeness is counted over the pre-registered rows only, whatever `--configs` asked for**
  (`done`, `missing_rows` in `gn2_meta_<scene>.json`; `missing_exploratory` separately on garden and
  bicycle). Recorded: the code comment above that block, "Amendment 8 session" above. Reason: `g2`
  derives completeness from the rows, not from the meta, so this is about the reader. "After the run"
  tells the next session to check `missing_rows` before quoting a row, and an exploratory-only run that
  judged completeness by its own configs would have written `done: true` on a scene with gate rows
  missing.

**Comparison rules**

- **Degree 3 for BD-rate and BD-PSNR.** Recorded: PREREG Amendment 7 f, `g2.py`'s module docstring,
  and the reason in "E1 results and E2 build" (the classic Bjontegaard cubic, exact on four points;
  E1's degree 2 on three points was the same exact construction; `g2.bd_rate` is `g1.bd_rate` at
  degree 3, so E1's function is reused).
- **BD-PSNR's definition** (PSNR as a cubic in `log10(bytes)`, averaged over the common byte range).
  Recorded: PREREG Amendment 7 f, `g2.bd_psnr`'s docstring. Reason, not written before: it is BD-rate's
  Bjontegaard dual, the same construction with the axes swapped, so the fallback measures the same gap
  in the other direction. It suits the fallback because curves that share no PSNR range, one lying
  entirely above the other, usually still share a byte range when both come from the same K grid:
  garden's did in E1 (14,778,950-16,405,132 B against 14,803,113-16,793,118 B). Where they do not,
  the scene is a loss (Amendment 7 f). Degree 3 for symmetry with BD-rate.
- **The tie rule** (a curve's best point is the highest PSNR, a tie going to fewer bytes). Recorded:
  PREREG Amendment 8 a ("Definitions"), `g2._best`'s docstring. Reason: the substitute needs exactly one
  best point, and a deterministic one; ties are not expected with measured float PSNRs. It is **not**
  uniformly conservative, and should not be read as a conservatism argument: for substitute a it picks
  the cheaper baseline point (a smaller claimed saving); for b it picks the cheaper `gn_vq` point (a
  smaller penalty). One simple rule was preferred over two because the case should not occur.
- **"PSNR >=" inclusive, "fewer bytes" strict.** Recorded: PREREG Amendment 8 a (Dace's wording, made
  explicit), `g2.mean_substitute`. Reasons: the substitute's claim is "reaches at least the other
  curve's best quality", which equality satisfies; strictness on bytes never changes a substitute's
  value when the curves share no PSNR range, the case the substitute is for (an equal-bytes point would
  give 0%, as c does); it only makes a and b mean an actual byte difference and labels that case c.
- **A non-finite BD-rate takes a substitute, like an undefined one.** Recorded: PREREG Amendment 8 a.
  Reason: the mean has to exist for every complete scene, and a non-finite value cannot be averaged.
- **`g1.bd_rate`'s numerical noise is left alone.** It fits on raw PSNR (about 25 dB) without
  centring, which leaves about 1e-7 percentage points of rounding in a cubic fit (the G2 tests allow
  1e-6 for that, and say so). That is far below both thresholds (0 and -5%), and the function produced
  E1's reported BD-rate, so it is not changed; `g2` wraps it instead.
  **Correction (2026-09-22), measured on E2's curves:** "about 1e-7 percentage points" held for the
  tests' synthetic curves, not for real ones. On E2's 39 comparisons, against the exact interpolating
  cubic in 50-digit arithmetic, the bundle's BD-rates (computed on Kaggle) are off by up to 6.09e-4 pp
  and this machine's recomputation by up to 1.16e-3 pp, both at H2b treehill, whose two curves lie
  within a few hundredths of a dB of each other; BD-PSNR is off by up to 1.46e-7 dB (bundle) and
  1.13e-7 dB (here). Recomputing on another machine therefore differs from the bundle by up to
  1.77e-3 pp and 1.60e-7 dB: E2's values reproduce within a tolerance, not bit for bit
  (`bench/gn/bd_sensitivity.py`, `kaggle/gn_e2/bd_sensitivity.json`). No sign, win or verdict changes,
  and the bundle's values stand as the verdict of record. **From E2b on** (Amendment 9 a), BD measures
  use `g2.bd_rate_scaled` / `g2.bd_psnr_scaled` (`numpy.polynomial.Polynomial.fit`): the same
  degree-3 quantity, better conditioned. On all 148 ordered pairs of E2's curves they are within
  5.6e-9 pp and 8.6e-13 dB of the exact cubic (the test allows 1e-8).

**Inputs**

- **The checkpoint sha1 check runs before any install, and again in each job.** Recorded: PREREG
  Amendment 7 d (the pins, and the job's refusal), the builder's restore-cell comment, the job's
  docstring, and why checkpoints are pinned at all in "E1 results and E2 build". Why before install:
  the install step can include a wheel build (4,404 s in run 5) and precedes the smoke tests, so a
  wrong or missing checkpoint found afterwards wastes the session, while found before it costs only the
  hashing (recorded as `checkpoint_sha1_s` in `timings.json`). Why again in the job: a resumed session
  or a hand-launched job need not go through the notebook's restore cell. Both raise with the found and
  the pinned sha1; the job writes both into its meta first, so a mismatch is readable in the bundle.
- **Checkpoints, sort caches and the wheel are copied into `/kaggle/working`,** as in E1, rather than
  read in place from `/kaggle/input`, so that E2's own output carries them into a resumed session. The
  run-3 / run-4 cluster files are copied to `tilequant/e2_kmeans/<scene>/`, a path the restore cell does
  not search, so a resume should still attach the run-5 output (the inputs table says so); without it,
  anything that still needs a K = 65,536 cache would be reclustered, with its source recorded.

**Measured E2 runtime per scene** (`kaggle/gn_e2/gn2/timings.json`, `gn2_meta_<scene>.json`; the
same numbers are tabulated in FINDINGS section 10):

| Scene | Data factor | Queue wall time `gn_e2_<scene>_s` | `timings_s.job` (meta) | Download | GN pass |
|---|---|---|---|---|---|
| train | 1 | 2,985.3 | 2,975.0 | 500.0 | 16.2 |
| truck | 1 | 2,940.3 | 2,923.0 | 233.2 | 15.8 |
| stump | 4 | 2,475.3 | 2,458.4 | 38.4 | 8.4 |
| bonsai | 2 | 3,465.4 | 3,456.8 | 67.6 | 30.5 |
| counter | 2 | 3,525.3 | 3,506.7 | 53.5 | 31.6 |
| kitchen | 2 | 3,600.3 | 3,586.9 | 78.1 | 32.8 |
| room | 2 | 3,780.3 | 3,766.3 | 89.3 | 29.7 |
| treehill | 4 | 2,325.2 | 2,318.4 | 62.1 | 9.5 |
| flowers | 4 | 2,580.3 | 2,573.1 | 72.7 | 12.8 |
| garden | 4 | 2,640.3 (+ 690.1 exploratory) | 3,315.9 (both jobs) | 73.5 | reused cache |
| bicycle | 4 | 2,190.1 (+ 720.1 exploratory) | 2,877.8 (both jobs) | 63.0 | reused cache |

Seconds. The queue time is measured at the queue's 15 s poll, so up to 15 s late; the meta's job time
adds up over invocations, so for garden and bicycle it covers the main and the exploratory job. The
longest job was room; the Tanks & Temples jobs took 2,940.3 s (truck) and 2,985.3 s (train), below the
MipNeRF360 indoor scenes (data factor 2), although train's download alone took 500.0 s. Per session:
`checkpoint_sha1_s` 21.4, `restore_s` 24.5, `install_s` 169.3 (the restored run-5 wheel; no
`gsplat_wheel_build_s`), `selftest_s` 15.7. No job was skipped by the start cutoff, and all 13 exited
with code 0.

### E2 results and E2b build (2026-09-22)

E2's bundle was committed, its BD measures checked, FINDINGS section 10 written, and E2b pre-registered
(Amendment 9) and built in one session. No E2b result exists, so every E2b choice below was made before
one.

- **The bundle is committed exactly as downloaded** (91 files), each checked byte for byte against the zip
  on disk and again as a staged blob (the PNG exactly, the text files CR-insensitively; the 11 CSVs carry
  the writer's CRLF, the JSON files are LF). It was unpacked straight from `~/Downloads`, so nothing
  needed ignoring.
- **E2's BD values reproduce within a tolerance, not exactly.** Dace asked for exact reproduction; on
  this machine only 4 of `gn2_g2.json`'s 120 BD fields came out bit for bit, because `np.polyfit` on
  uncentred PSNR is badly conditioned and its last digits depend on LAPACK. The work stopped there and
  Dace chose the rule `bd_sensitivity.py` now enforces: every sign, win, deciding measure, mean-term
  source, count and verdict identical, values within 2e-3 pp and 1e-6 dB, and the 50-digit exact cubic
  reported beside both. The bundle's values are the verdict of record.
- **The exact cubic** is the Lagrange form in Python's `decimal` at 50 digits on the floats' exact binary
  values (`Decimal(float)`), with `Decimal.log10` for the bytes: the same pre-registered quantity, with
  no fitting at all. It is a reference, not a replacement.
- **"Disagree in sign"** means BD-rate and BD-PSNR name different curves as better, which is the raw
  signs being equal (BD-rate < 0 favours the new curve, BD-PSNR > 0 does). Only H2b treehill is flagged
  under the cubic, and only G2a treehill under PCHIP.
- **PCHIP** is `scipy`'s `PchipInterpolator` on the same axes as the cubic, integrated exactly, with the
  same per-scene rule and Amendment 8's substitute; the shN-stream BD-rate is the same computation with
  the `shN_bytes` column as the rate. Both are post hoc and labelled so in FINDINGS.
- **Dace's corrections to the first reading were adopted in FINDINGS section 10:** treehill's
  `lloyd_trace` is higher in PSNR at all four K but cheaper at only three; the shN-stream BD-rate is
  quoted by both methods (the "about 0 on treehill" holds only for PCHIP); the fewest-train-views pattern
  was dropped (stump's ratio is above train's), replaced by the three lowest ratios.
- **FINDINGS section 10 was checked mechanically,** as sections 8 and 9 were. The check caught one wrong
  figure before commit: `meta.json` differs by up to 2 bytes between a GN-VQ row and `lloyd_wopa_area`,
  not 1.
- **E2b's scenes (superseded by Amendment 10, below), and the four readings of Amendment 9, are Dace's** (treehill, flowers and train with
  garden as the control; `rho_cv`'s codebook is the full-`M` codebook; the Spearman pairs the CV
  codebooks' odd-view dMSE with the full-`M` codebooks' test dMSE; warm starts from E2's notebook output;
  the full `M` from E2's `gn_cache`). The caveat in Amendment 9 b.e (dMSE and PSNR can disagree; E2's
  treehill at K = 4,096 is the example) was found in E2's rows while writing it, and recorded before any
  E2b code.
- **The CV codebooks are also measured on the test views,** reported only (the second Spearman). It costs
  one render pass per row and shows whether the selection score tracks the test error for the same
  codebook, separately from whether it tracks the full-`M` codebooks'.
- **One lifted check per floored metric, 7 per scene.** The floor changes the metric the lifted fp32
  assignment works with, so E2's single check on the unfloored `M` does not cover it; each check costs
  about 27 s at K = 65,536. **(Superseded by Amendment 10 c: 8 per scene, the unfloored full `M`
  included.)**
- **`gn_vq` gained an optional `report_metrics` argument** (reporting only) rather than E2b recomputing
  the unfloored objective afterwards: the objective before quantization needs the labels from before the
  final quantized assignment, which `gn_vq` does not return. The default path is unchanged; a test and
  the dry run's stage 5 pin that (rho = 0 reproduces E2's toy report exactly, and `report_metrics`
  changes neither the codebook nor the labels).
- **The rho grid is fixed in the job** (`e2b.RHOS`), with no command-line override, because Amendment 9
  fixes it; the K grid stays an argument so the dry run can shrink it.
- **E2b has its own configs, CSV and `rho` column** (`gn_vq_floor_cv`, `gn_vq_floor`), so `g2`'s
  functions and E2's rows can never pick an E2b row up, and `e2b.check_rows` refuses a full-`M` row at
  rho = 0 (that codebook is E2's row). **(The rho = 0 refusal was removed with Amendment 10 c, which makes
  E2b run that row; the rest stands.)**
- **Both inputs are required, each behind a flag rather than silently optional:** without E2's output the
  rho > 0 rows would use a different realization of `M` than E2's rho = 0 row, which the comparison
  should not do by accident. Without the run-5 output nothing breaks if E2's output is complete, but it
  is the documented source of the fallback caches.
- **E2's comparator rows come from the cloned repo** (`kaggle/gn_e2/gn2/`), the committed verdict of
  record, not from the attached E2 output, whose `gn2/` the restore cell never reads.
- **Queue order train, treehill, garden, flowers:** train is the longest job (data factor 1, a 500 s
  download in E2), and the two criterion scenes come next so that a short session still has them.
  **(Superseded by Amendment 10: train left E2b; see "Amendment 10 session".)**
- **Commit split as for E2:** the builder with the code, the rebuilt notebook with the docs.

### Amendment 10 session (2026-09-25, before any E2b data)

Dace re-derived E2b's scene choice from fidelity (dMSE) instead of PSNR and added a fidelity criterion and
a reproduction check. No E2b row, bundle or log existed; the amendment was committed before the code.

- **Dace's numbers were checked against E2's rows before writing the amendment,** and all held. Two
  things came out that the amendment records rather than hides: the selection measure is the difference
  (test ratio minus train ratio), because with the quotient room (1.624) ranks above flowers (1.596) at
  K = 65,536; and "outdoor 360-degree scenes" is not a clean separation (room, indoor, grows 0.3150 ->
  0.5115 at K = 65,536, and garden, outdoor, barely moves), so the amendment names the scenes by the
  measure only.
- **Train stays a development scene** although E2b no longer uses it: Amendment 9 declared it one after
  E2's results were seen, and dropping it from E2b does not make it unseen. This is the conservative
  reading of a case the request left open.
- **E2b now runs its own rho = 0 full-`M` rows,** which Amendment 9 b.c had excluded ("not recomputed").
  Both the reproduction check and "E2b's own rho = 0 value" need them. Amendment 9's items keep E2's
  `gn_vq` row as their rho = 0 codebook (the PSNR criteria and the Spearman as written); Amendment 10's
  items use E2b's own row, which shares the run and the realization of `M` with the other E2b rows. The
  Spearman is reported both ways.
- **The reproduction tolerances (1e-3 dB, 1e-3 relative test dMSE) are this session's choice**, stated
  in Amendment 10 c before any data: GN-VQ's update uses float64 `index_add_` on the GPU, whose atomic
  order is not fixed, so a bit-for-bit match is not guaranteed; 1e-3 dB is well under every gain E2
  measured at these K and half of TorchPQ's cross-session spread. `identical` is reported separately, and
  `inputs_differ` keeps a comparison with a recomputed `M` or warm start from being read as a failure.
- **The garden control is reported with the fidelity verdict, not part of it** (Amendment 10 d's
  wording), so a missing garden row leaves the fidelity verdict alone; a test pins that.
- **"At rho_cv = 0 a cell is not below its own rho = 0 value"** follows from R being the same number; it
  is written into the amendment so the case is not argued later.
- **Queue order treehill, garden, flowers, stump:** all four are MipNeRF360 at data factor 4 with similar
  E2 job times, so no longest-first reason remains; treehill and garden come first because both criteria
  need them.
- **One more lifted check per scene** (8): the rho = 0 full-`M` rows use the unfloored full `M`, which
  E2 checked, but E2b checks every metric it uses in its own session rather than rely on E2's record,
  which would not cover a recomputed `M`.

### Amendments 9-10 and the E2b build: decision index (2026-09-26)

Every decision from the Amendment 9 session, the E2b build and the Amendment 10 session, each with its
reason and where it is recorded. Where "E2 results and E2b build" or "Amendment 10 session" above
already gives the reason, the entry says so and adds only what they leave out. All were made before any
E2b data existed.

**E2's numbers of record and the BD computation**

- **E2's BD values reproduce within a tolerance, not bit for bit** (every sign, win, deciding measure,
  mean-term source, count and verdict identical; values within 2e-3 pp and 1e-6 dB). Dace's rule.
  Recorded: "E2 results and E2b build", `bd_sensitivity.py`'s docstring, its `reproduction` block.
  Measured here: at most 1.77e-3 pp (H2b treehill) and 1.60e-7 dB.
- **The old cubic's rounding, measured:** against the 50-digit exact cubic, E2's bundle (Kaggle) is off by
  up to 6.09e-4 pp in BD-rate and 1.46e-7 dB in BD-PSNR, a recomputation on this machine by up to
  1.16e-3 pp and 1.13e-7 dB, the worst BD-rate at H2b treehill in both; only 4 of `gn2_g2.json`'s 120 BD
  fields came out bit for bit here. Cause: `np.polyfit` on uncentred PSNR (21.59-32.31 dB) and log10
  bytes (7.13-7.23). Recorded: `kaggle/gn_e2/bd_sensitivity.json` (`exact_cubic`), PREREG Amendment 9 a,
  and the correction under "Comparison rules" in the E2 decision index (which had said "about 1e-7 pp").
- **From E2b on, BD-rate and BD-PSNR use the domain-scaled fit** (`numpy.polynomial.Polynomial.fit`,
  which maps the abscissa onto [-1, 1]); the pre-registered quantity, the degree-3 polynomial through
  the four points, is unchanged, and E2's recorded values stand. Recorded: Amendment 9 a;
  `g2.bd_rate_scaled` / `g2.bd_psnr_scaled` docstrings; `test_bd_scaled_matches_the_exact_cubic_on_all_of_e2s_curves`
  (all 148 ordered pairs of E2's curves, measured within 5.6e-9 pp and 8.6e-13 dB, allowed 1e-8). Not
  recorded before: **they are new functions beside the old ones, not replacements,** so neither E2's code
  path nor any value E2 recorded can change, and the same degree rule (`min(3, n - 1)`) and NaN cases
  apply, so a later caller swaps one name for the other. E2b itself computes no BD measure (two K per
  curve).
- **The exact cubic is a reference, not a replacement** (Lagrange form in `decimal` at 50 digits on the
  floats' exact binary values). Recorded: "E2 results and E2b build".
- **`bd_sensitivity.json` sits next to the bundle, not inside `gn2/`** (not recorded before): `gn2/` must
  stay byte-identical to the zip it was unpacked from.
- **Its input hashes are of the CR-normalized files** (`_sha256_lf`, not recorded before in prose):
  `autocrlf` rewrites the CSVs' line endings on checkout, so a raw hash would depend on the machine.
- **`held_out_without_flagged_scenes_post_hoc`** (not recorded before): the post-hoc count that FINDINGS
  section 10 quotes for H2b (8 of 9) and its mean over the other 8 (-4.47%) are fields of a committed
  file, not arithmetic in the text, so the mechanical number check can match them.

**E2b's scenes**

- **Treehill, flowers and stump, with garden as the control; train dropped.** Recorded: Amendment 10 a-b
  (the dMSE-ratio table, the selection by the difference test ratio minus train ratio, and why
  Amendment 9's PSNR ratio was the wrong measure); "Amendment 10 session" (the quotient ranks room above
  flowers at K = 65,536; "outdoor" is not a clean separation).
- **Train stays a development scene although E2b dropped it.** Recorded: Amendment 10 b; reason in
  "Amendment 10 session": Amendment 9 declared it one after E2's results had been seen, and removing it
  from E2b does not make those results unseen, so treating it as held out again could let a later claim
  use a scene that already shaped a decision. The conservative reading of a case the request left open.
- **A label quirk to know when reading E2's rows** (not recorded before): E2's `source` column says
  `run3_cache` for the K = 65,536 `lloyd_wopa_area` codebook on the seven run-4 scenes (stump, treehill,
  flowers among them), because E2's notebook passes its copy of the run-3 or run-4 cache as
  `--run3_kmeans_dir` and `gn_e1_scene.get_lloyd` labels any hit there `run3_cache`. In E2 it means "a
  runs 3-4 cache", not "run 3". E2b's own labels (`e2_kmeans_cache`, `run5_kmeans_cache`) say where the
  file came from instead.

**E2b's rho = 0 rows and which criteria use them**

- **E2b reruns rho = 0 with E2's full `M`** (Amendment 10 c), which Amendment 9 b.c had excluded: the
  reproduction check needs a row to compare with E2's, and the fidelity criterion's "E2b's own rho = 0
  value" needs a codebook from the same run and the same realization of `M` as the rho > 0 rows.
  Recorded: Amendment 10 c; "Amendment 10 session".
- **Which row each item uses** (Amendment 10 c; `e2b.py`'s docstring; `judge_scene_k`):
  - Amendment 9's PSNR criteria: **E2's `gn_vq` row** is the rho = 0 codebook (it is `rho_cv`'s codebook
    when `rho_cv = 0`) and garden's reference;
  - Amendment 9's Spearman (odd-view dMSE against full-`M` test dMSE): **E2's row** at rho = 0, as
    written; also reported with **E2b's own row** (`spearman_odd_vs_full_test_own_rho0`);
  - Amendment 10's fidelity criterion (R at `rho_cv = 0`, and each cell's own rho = 0 R): **E2b's own row**;
  - the garden control's denominator: **E2b's own row**;
  - the reproduction check: E2b's own row against E2's.
  Reason for the split: Amendment 9's items were pre-registered with E2's row and are kept exactly as
  written; Amendment 10's items compare codebooks that share one run and one `M`, so a difference between
  E2's run and E2b's cannot masquerade as an effect of the floor.

**The reproduction check**

- **Tolerances 1e-3 dB in test PSNR and 1e-3 relative in test dMSE**, with `identical` reported
  separately and only `not_reproduced` flagged. This session's choice, in Amendment 10 c before any data;
  reason in "Amendment 10 session" (GN-VQ's float64 `index_add_` on the GPU has no fixed atomic order;
  1e-3 dB is below every gain E2 measured at these K and half of TorchPQ's cross-session spread).
- **`inputs_differ`** (Amendment 10 c; `e2b.reproduction`): the row did not use E2's `M` (`m_source` is not
  `restored_cache`) or E2's own warm start (`warm_start_source` is not `e2_work_cache` / `e2_kmeans_cache`).
  The differences are still reported; the status only stops a comparison with different inputs from being
  read as a failure to reproduce. Not recorded before:
  - `run5_kmeans_cache` counts as differing although on these scenes it holds the same clustering E2 used:
    the status keeps to what the row records rather than inferring provenance; read `warm_start_source`.
  - The status uses `m_source`, not `m_key_equals_e2`: the cache key encodes the checkpoint, the render
    settings, the number of views and the probe seed (`gm.cache_key`), so a recomputed `M` has the same key
    as E2's without being bitwise E2's `M` (the GPU backward accumulates in no fixed order).
  - `m_source` is `restored_cache` when the cache loaded and the scene's meta has no record of E2b computing
    it; once E2b recomputes it, the meta says `recomputed` and a resume keeps that (`gn_e2b_scene.py`).

**The two criteria**

- **The garden control is reported beside the fidelity verdict, not inside it** (Amendment 10 d's wording,
  "reported with it"; `e2b._fidelity_criterion` keeps a missing garden row out of the verdict's `missing`,
  and `test_e2b_fidelity_criterion_every_path` pins that). The verdict answers Amendment 10 d's two
  conditions as written; **Amendment 10 f then governs what may be claimed:** that the floor works only if
  both the fidelity verdict and the garden control hold, and nothing of the kind if the garden row is
  missing. The fields stay as implemented; see the open item below.
- **`judge_e2b` returns `verdicts` = {`fidelity`, `garden_control`, `psnr`} and no single `verdict`**
  (not recorded before): there are two criteria, neither a gate, and a control that Amendment 10 f needs
  for any claim; one top-level field would suggest a single verdict.
- **Differences rounded to 9 decimals, "below" strict, "at most" inclusive; at `rho_cv = 0` a cell is not
  below its own rho = 0.** Recorded: Amendment 10 d; "Amendment 10 session".
- **The PSNR criteria carry the cross-term caveat** in their output (`CROSS_TERM_CAVEAT`). Recorded:
  Amendment 10 a and e.

**The build**

- **`M_even` is its own GN pass, with its own probe draws** (not recorded before): `compute_gn` over the
  even-indexed views starts the seed-0 probe generator afresh, so the even views are probed differently
  than in E2's full pass and `M_even` is not a sub-sum of E2's `M`. That is Amendment 9 b.b's definition
  ("E0's GN pass (probe seed 0) over the even-indexed train views only"), it costs a few seconds, and it
  needs nothing from how E2's pass was ordered.
- **`M_even`'s cache key has a suffix and its own directory** (`|train_views=even_indexed`,
  `gn_cache_even/`; not recorded before): `gm.cache_key` records only the number of views, not which, so
  without them an even-view `M` could be taken for a full `M` of the same view count.
- **CV rows skip the full-pipeline evaluation** (no test PSNR / SSIM / LPIPS, no train PSNR; not recorded
  before): the selection uses the odd-view dMSE only and both criteria use full-`M` rows, and LPIPS over
  the test views plus a pass over the train views is the costliest part of E2's per-row measurement. They
  keep the test dMSE (the second Spearman), `P` on `M_even` and the bytes.
- **Eight lifted checks per scene**, one per metric used: `M_even` at the four rho, the full `M` at the four
  rho. Recorded: Amendment 9 b.g, Amendment 10 c, "Amendment 10 session".
- **Warm-start order: E2's own caches, then the run-5 cache, then this job's cache, then a reclustering,**
  each key-checked. Recorded: `gn_e2b_scene.py`'s docstring and `get_warm_start`. Reason: the first is the
  exact codebook E2 warm-started from, so the rho = 0 row can reproduce E2's.
- **The restore cell copies only the four scenes' `M`** (about 4 x 480 MB rather than E2's 11 files).
  Recorded: the builder's restore cell comment.
- **Both inputs required, each behind an explicit flag; E2's comparator rows from the cloned repo; the rho
  grid fixed in the job; E2b's own configs and CSV; `report_metrics` in `gn_vq`; the CV codebooks also
  measured on the test views.** Recorded with reasons: "E2 results and E2b build".
- **Queue order treehill, garden, flowers, stump.** Recorded: "Amendment 10 session".
- **The dry run first runs E2's own job on the toy** (not recorded before in prose; the dry run's
  docstring lists the stages), so the restore cell and the E2b job read a real E2 output layout
  (`gn_cache/`, `gn2_work/`, `tilequant/e2_kmeans/`) instead of hand-made files, and the reproduction check
  can compare E2b's rho = 0 rows with real E2 rows (on the CPU they come out `identical`). Stump's toy E2
  run reclusters its K = 64 codebook into `gn2_work/`, so the restore path for a K = 65,536 file there
  (which E2's train had) stays covered after train left; flowers deliberately loses E2's warm starts and
  `M`, to exercise the fallbacks and the `inputs_differ` status.
- **Bit-identity of rho = 0 is shown on the CPU only** (the dry run's stage 5 and a test). On a GPU the
  reproduction check decides, with the tolerances above.

### E2b results (2026-09-26)

E2b's bundle was committed and FINDINGS section 11 written. No rule, code or notebook changed.

- **The bundle is committed exactly as downloaded** (80 files, `a3c0099a`, data only). It was unpacked
  straight from `~/Downloads/gn2b_bundle.zip` into `kaggle/gn_e2b/`, and each file was checked byte for byte
  against its zip entry on disk and again as a staged blob (the PNG exactly, the text files
  CR-insensitively). The 4 CSVs carry the writer's CRLF; the JSON files are LF. Nothing needed ignoring.
- **The run used `0d180360`** (`gsplat_commit` in every meta), the Amendment 10 f commit. The branch tip
  then was `25664131`, which changed only this file, so the code that ran is the code on the branch.
- **Section 11 follows Amendment 10 f.** It says the floor works only "in fidelity terms", and only
  because `verdicts.fidelity` is `works` and `verdicts.garden_control` is true in `gn2b_e2b.json`. The PSNR
  criterion is reported beside it with the cross-term caveat, and its failure is stated as plainly as the
  fidelity result.
- **Dace's reading of the bundle was checked against the files before anything was written.** All points
  held except one, which section 11 states in its measured form. "The floor changes bytes by at most about
  0.15%" is true at `rho_cv`: -0.148% to +0.013% on the six cells of treehill, flowers and stump. Over all
  24 rows with `rho` > 0 the change runs from -0.437% (garden, K = 65,536, `rho = 1e-1`) to +0.435%
  (treehill, K = 65,536, `rho = 1e-3`). Section 11 gives both.
- **The limit note is labelled post hoc and carries its caveats.** As `rho` grows, the floored distance
  divided by `rho` tends to `lloyd_trace`'s `tr(M_i)`-weighted Euclidean objective, so the floor interpolates
  between the full matrix and the scalar weighting. The limit is not E2's `lloyd_trace` row, though. E2's
  ridge stays in the update (it scales the weighted mean by 1 / (1 + eps) = 1 / 1.01), as do GN-VQ's warm
  start, clip, iteration cap and quantized final assignment.
- **Other post-hoc observations in section 11, each read from the rows:** at `rho_cv` the train-view
  dMSE rises while the test-view dMSE falls, in all six cells; LPIPS rises slightly in all six; on
  treehill no `rho` in the grid beats `lloyd_trace` in test PSNR; and at `rho = 1e-1` garden's test dMSE
  would have been 1.0704 times its `rho = 0` value at K = 65,536, outside the control's 5%.
- **Section 11 was checked mechanically,** as sections 8-10 were. A script recomputed every number from
  `kaggle/gn_e2b/gn2b/` and E2's committed rows and asserted each appears in the text. Every numeric token
  in the section was matched against its output: 0 failures, and the only unmatched tokens were
  punctuation and commit-hash fragments. **The script is now `bench/gn/check_s11.py`**
  (`python bench/gn/check_s11.py` from any directory): it reads only the committed bundles and FINDINGS,
  counts a quoted number missing from the text, a claim the files contradict, or a numeric token it did
  not recompute as a failure (commit hashes are listed; all-digit hashes such as `25664131` are
  tokenized too), and exits 1 on any failure. It reports 0 failures; changing one quoted number makes it
  fail.

### Amendment 11 and the E2c build (2026-09-26, before any E2c data)

Dace specified E2c; the amendment was committed (`14145093`) before any E2c code. These are the choices
this session made that the request left open, and why. Each is either in Amendment 11 or in code with a
comment.

- **Three rules the request did not state, written into Amendment 11 before any code** (Dace can amend
  them before any E2c data):
  - condition 3 with an **undefined BD-PSNR** (no shared byte range with E2's `gn_vq`) **fails**, because
    "no harm" cannot be shown without an overlap. G2a's rule has a fallback for the win count; a
    no-harm bound has none;
  - the **final codebook runs at every `rho_cv`, 0 included**, and at 0 its reproduction of E2's row is
    reported with Amendment 10 c's statuses. G2c always uses E2c's own row, so condition 3 compares two
    runs even when the floor is not selected;
  - **reported flags**: per curve, whether PSNR rises with K, and per comparison, whether BD-rate and
    BD-PSNR name different curves. E2's treehill H2b win was a cubic artifact of exactly this kind, and
    the flags change no verdict.
- **Amendment 11 f states that E2's `gn_vq` already meets conditions 1 and 2 on these scenes** (domain-scaled
  fit: -3.10% to -6.61% against `lloyd_trace`, mean -5.93% against `lloyd_wopa_area`, computed from
  `kaggle/gn_e2/gn2/` before the amendment). A pass is therefore mostly about "no harm", and
  `n_cells_rho_cv_above_0` says how much of a pass is the floor. This fact comes from E2's committed
  results, not from E2c.
- **No fallback, no override** (Amendment 11 c): the job refuses a missing E2 `M` or warm start before any
  download and a key-mismatched `M` before any GN-VQ run; the notebook has no `ALLOW_WITHOUT_*` flags. E2b
  had fallbacks because it was exploratory. For a gate, a recomputed `M` or reclustered warm start would
  make condition 3 compare different inputs.
- **What "the run-5 output is attached" means to the restore cell:** its `tilequant/run4/<scene>/kmeans`
  caches for the 4 MipNeRF360 scenes, which E2's output does not hold under that path (E2 copied them to
  `tilequant/e2_kmeans/`). They are also used: each job records whether E2's copy equals them
  (`warm_start_equals_run5_cache`). Truck has no run-4 cache; its K = 65,536 warm start is E2's own
  clustering in `gn2_work/`.
- **The final row reads the CV scores back from the CSV** instead of keeping them in memory, so a resumed
  job selects `rho_cv` from exactly the rows G2c will re-judge, and a final row whose `rho` is not its CV
  rows' argmin is treated as missing by `e2c.judge_cell`. The dry run's stage (4) deletes one final row and
  checks it is rerun alone with the same result.
- **E2c's CSV header is E2b's plus `rho_cv` and `cv_odd_scores`,** so neither job can read the other's file.
  E2c's resume directory for `M_even` is `gn2c_cache_even/`, not E2b's `gn_cache_even/`.
- **Rounding:** every comparison in G2c is on values rounded to 9 decimals (win as BD-rate < 0 too), as
  Amendment 11 d says; G2a's own code compared the raw BD-rate. With the domain-scaled fit this matters
  only at exact boundaries.
- **Queue order room, bonsai, kitchen, counter, truck:** longest first by E2's per-row evaluation cost
  (94.1-114.3 s on the MipNeRF360 scenes, 34.2 s on truck). All five are gate scenes, so no scene has to
  come first for a partial session to be useful.
- **`bench/gn/check_s11.py` is in the repo** (`fa768af5`), and pytest runs it, so an edit to FINDINGS
  section 11 that breaks a number fails the suite.
- **The E2c dry run takes 549 s on this machine.** Its first version took 87 minutes because three reruns
  (stages 4-6) found their fake data marker deleted by stage 3 and downloaded the real `360_v2.zip`
  (1.31 GB) into the temp directory. They now recreate the marker, and the dry run replaces
  `tilequant_run4.download_scene` and `tilequant_run5.download_tandt_scene` with a function that raises, so
  a missing marker fails loudly. E2b's dry run has the same structure, but its reruns all stop before any
  download (every row exists, or a refusal), and it took 245 s.

### E2c results (2026-09-27)

E2c's bundle was committed and FINDINGS section 12 written. No rule, code or notebook changed, except the
section-11 checker (below).

- **The bundle is committed exactly as downloaded** (179 files, `b1163f24`, data only). Dace left it in the
  repo root as `gn2c_bundle.zip`, byte-identical to `~/Downloads/gn2c_bundle.zip`. Each file was checked
  byte for byte against its zip entry on disk and again as a staged blob. The 5 CSVs carry the writer's
  CRLF, and the PNG contains `\r\n` only in its signature, so it is compared exactly. The zip stays in the
  repo root, not deleted; since the scouting session (2026-09-27) the root `.gitignore` ignores that one
  name (`/gn2c_bundle.zip`, anchored), as for E0's and E1's bundles.
- **Dace's reading was checked against the files before anything was written.** Two points differed, and
  section 12 states both in their measured form:
  - the test-dMSE ratio against E2's `gn_vq` on bonsai, counter, kitchen and truck is 0.9754-1.0058, not
    0.96-1.006. The 0.96 is room's (0.9636 and 0.9682 at K = 4,096 and 1,024);
  - the BD-rate and BD-PSNR against `lloyd_trace` (-3.21% to -6.95%, +0.0783 to +0.3762 dB) are in
    `gn2c_g2c.json`: they are condition 1's own quantities, so section 12 quotes them from the verdict file,
    not as post hoc.
  Everything else held.
- **What section 12 adds beyond the reading**, each from the files:
  - truck against `gn_vq` is the one comparison whose BD-rate and BD-PSNR disagree in sign (-0.142%,
    -0.0013 dB);
  - four cells have test dMSE above `gn_vq`'s (counter K = 65,536; kitchen K = 1,024, 4,096, 65,536);
  - the MipNeRF360 downloads took 5.1-13.5 times as long as in E2, per scene.
  - Post hoc, labelled: the change in the `lloyd_trace` BD-rate from E2's `gn_vq`, and E1's seed spread
    (0.0075 dB, Amendment 7 e) as the scale for the BD-PSNRs against `gn_vq`.
- **Section 12 was checked mechanically:** 83 recomputed numbers and 220 numeric tokens, 0 failures. The
  check caught one wrong figure before commit (bonsai's BD-rate shift is -0.30 percentage points, not
  -0.31). The script is now `bench/gn/check_s12.py` (`python bench/gn/check_s12.py`, 0 failures; pytest runs
  it), built like `check_s11.py`, with its summary and Sources slices bounded from the start.
- **`bench/gn/check_s11.py` had to be fixed** (`4ec8ab50`): its summary and Sources slices ran to the next
  heading, so they took in section 12's paragraph and entry. They are now bounded to section 11's own. The
  script gives the same 175 tokens and 0 failures as before, and pytest passes.

### Amendment 12 and the E3p build (2026-09-27/28, before any E3p data)

Dace specified E3p. Amendment 12 was committed first, docs only (`aa67e21e`), before any E3p code. These are the
choices this session made that the request left open, and why. No E3p row, bundle or log exists.

- **Amendment 12's one factual claim was checked from the files before it was written:** in all six E2b gap
  cells (treehill, flowers, stump at K = 4,096 and 65,536) both the odd-view score and the full-`M` codebook's
  test dMSE are lowest at `rho` = 1e-1 (`kaggle/gn_e2b/gn2b/gn2b_results_<scene>.csv`). Garden's are not, which
  is why the amendment names the gap cells.
- **The pins were read from the archive, twice.** The zip directory (zip64: 117 entries, directory at
  14,660,617,193, 13,708 bytes) was read by range requests on 2026-09-27; then `e3p_inria.py`'s own reader
  checked it against the pins (no mismatch), fetched both scenes' `cfg_args` and `cameras.json` through
  `fetch_member` (sizes and CRC32 matched), and inflated the first 64 KB of both `.ply` members: 62 INRIA
  properties, 1,532-byte headers, 6,131,954 and 1,026,508 vertices, each file exactly header + 248 bytes per
  splat. Nothing larger than those 64 KB was downloaded here, so the `.ply` SHA-1s will first exist in E3p's
  meta.
- **E3p's own zip reader, not `remotezip`,** for the INRIA members: the directory has to be compared with the
  pins before anything is fetched, the local header's name checked at the pinned offset, and a 1.35 GB member
  streamed with resume; the same code reads a `file://` archive, so pytest and the dry run exercise every
  check. The datasets still use runs 4-5's `remotezip` downloaders.
- **`normalize_world_space` off, and the camera-frame check a stop condition.** INRIA's splats are in COLMAP's
  world frame; gsplat's parser normalizes by default, which would put every camera elsewhere. Checked on real
  data: bicycle's `sparse/0/images.bin` (range-read from `360_v2.zip`) against the fetched `cameras.json`, 194 of
  194 matched, largest differences 8.9e-16 (position) and 3.3e-16 (rotation): `cameras.json`'s `position` and
  `rotation` are the camera-to-world translation and rotation, which is what the check compares.
- **The test-split check is reported, not a stop:** it relies on `cameras.json` listing the test cameras first,
  which the release codebase does and bicycle's file confirms (its first 25 names are every 8th image), but a
  report should not end a session. gsplat's parser and INRIA's reader both take every 8th image by sorted name.
- **Protocol ii, as implemented** (the request fixed the images, the resolution and the 8-bit quantization):
  INRIA's camera, built from `cameras.json` (focal `fx * w / W`, principal point at the image centre, which in
  gsplat's pixel convention is INRIA's `ndc2Pix` centre); `loadCam`'s size rule; PIL, with a resize only when
  `loadCam` changes the size (not for these two scenes); `save_image`'s rounding (`mul(255).add_(0.5)`, clamp,
  uint8). **The metric modules are the runner's** (torchmetrics PSNR, SSIM, LPIPS-VGG as gsplat configures it),
  not INRIA's `ssim` and `lpipsPyTorch`; PSNR is the same formula, SSIM's border handling differs. E3's own
  amendment should say which it uses.
- **The crop.** `PngCompression` keeps the largest square number of splats, dropping the lowest-opacity ones
  (bicycle 1,378, train 339). In the shN-only renders (dMSE) and in `P` a dropped splat keeps its own shN, so
  those measure the codebook alone; the full-pipeline evaluations use the decoded, cropped model, as the codec
  delivers it. With E0-E2c's 1,000,000-splat checkpoints nothing is dropped and this is E2c's arithmetic.
- **The one-copy layout (Amendment 12 a) keeps GN-VQ's code unchanged.** `gn_vq` takes the floored buffer as
  its metric; the unfloored metric it reports on (`report_metrics`) and `predicted_dmse` read through
  `HostMetric`, which answers `quad_form`'s row slices from host memory. The floor is filled slice by slice with
  `e2b.floored_metric` itself. Bit-identity is shown on the CPU (pytest, all 7 `rho`, both metrics, slices
  that do not divide the problem). On the GPU the per-row arithmetic and kernels are the same; that the GPU's
  per-row reductions do not depend on the number of rows in a slice is assumed, not shown.
- **Also not a copy, beyond the request:** the job's `splats_raw` are the runner's own tensors (E2c cloned
  them); the runner is pointed back at them after every swap. Values are unchanged; it saves one model copy
  (1.45 GB at bicycle).
- ~~**`diagnostics.direct_distance` is left as it is**~~ **Superseded on 2026-09-28** (Dace): it is now chunked
  over splats, before any E3p run; see the next subsection.
- **Only out-of-memory errors are caught** (CUDA's OOM, cuBLAS / cuSOLVER / cuDNN allocation failures, host
  `MemoryError`); a job with an OOM exits 0 with its rows missing, because an OOM is a result of the pilot. Any
  other error is recorded and stops the job, and the bundle cell raises at the end.
- **If the PLAS sort runs out of memory,** the job continues with the crop's opacity order (`order_source`
  `crop_only_unsorted`), so the later steps are still measured; their sizes would then not be comparable.
- **The `M` caches go to `/tmp`,** not the output: at bicycle each is about 3 GB, and the output keeps the
  INRIA members (1.77 GB) for a resume instead.
- **The C3DGS check uses pip in a venv over the session's torch,** not the README's conda environment
  (python 3.8, pytorch-cuda 12.1): that is how E3 would run C3DGS next to gsplat on Kaggle, it installs no
  second torch, and it leaves the jobs' environment untouched. The deviations are listed in the JSON, as is
  whether `conda` was on the path. It runs **before the jobs**, so it cannot disturb their timings and runs even
  if a job overruns; it is capped at 3,600 s and never raises. It imports C3DGS's `compression.vq` and
  `gaussian_renderer` as well as the packages, because those are the imports C3DGS itself makes
  (`from weighted_distance._C import weightedDistance`); nothing is called.
- **The published PSNRs** (bicycle 25.246, train 21.097) were read from arXiv 2308.04079v1's HTML, Tables 5 and 8
  (`A4.T5`, `A4.T8`), row Ours-30k. A first table parser attached each caption to the wrong table (the SSIM table
  came out labelled PSNR); the captions were then matched by figure id. Only PSNR is carried, as the request
  asked.
- **"R5 tilequant" is attached for the wheel only.** E3p uses no run-5 checkpoint, sort cache or clustering;
  without it the notebook builds the wheel (`ALLOW_WHEEL_BUILD` is True).
- **Queue:** bicycle on the first GPU, train on the second, as E0 ran two scenes; one job each, so the order only
  matters for the start cutoff.
- **Commit split:** Amendment 12 alone (docs), then the code and tests, then the notebook and the docs.

### `direct_distance` chunked over splats (2026-09-28, before any E3p run)

Dace asked for it before E3p runs. `bench/gn/diagnostics.py` changed; no rule, method or result did.

- **What changed.** `diagnostics.direct_distance` builds the float64 difference `c_i - q_label(i)` one chunk of
  splats at a time and hands each chunk to `quad_form`, instead of building it for all splats first. At
  bicycle's 6.13M splats the old form held three `[N, 15, 3]` float64 tensors at once (2.2 GB each: the
  splats in float64, the gathered centroids, their difference); now each chunk of 262,144 splats needs about
  1 GB (those three at 94 MB each, the chunk's metric in float64, 252 MB, and unpacked, 472 MB). It is called by
  `assign_exact`, `accept_by_cluster` (`cluster_objectives`), `gn_objective` and `lifted_check`.
- **It is method-neutral:** it computes the same quantity, the direct-formula distance of every splat to its
  centroid, from the same inputs; only the order in which memory is used changes.
- **Why it is bit-identical.** A splat's distance depends only on its own row. The difference is element-wise, so
  each chunk holds exactly the values the same rows of the full difference held. And the chunks are
  `quad_form`'s own (the same `chunk` argument, default 262,144, whose boundaries `quad_form` already used), so
  every einsum receives the same values in the same shapes and positions as before. That last part is needed:
  on this machine `quad_form`'s einsum can move a distance by 1 ulp with the splat's position inside its chunk
  (chunks of 1, 3, 7 and 333 splats differ from one chunk of 2,003 by up to 2.05e-16 relative, in the old code
  exactly as in the new), so bit-identity holds for equal chunks and is not claimed for different ones.
- **Checked on the CPU:** `test_direct_distance_chunked_over_splats_is_bit_identical` (2,003 splats, M_i of every
  rank 0-15 with zero ones, chunks from 1 to above N, none but 1 and 2,003 dividing N, both input layouts,
  through `HostMetric`); `test_direct_distance_chunking_changes_nothing_downstream` (GN-VQ's codebook, labels and
  report, the lifted check, `gn_objective`, the cluster objectives and the exact assignment, at the default
  chunk and at 333); and `bench/gn/dryrun/check_chunked_distance.py`, E2c's dry-run path: E2's and E2c's real
  jobs on the toy with the old and the new function give identical rows (25), GN-VQ reports (18) and lifted
  checks, at the default chunk (one chunk, as E2c's 1M-splat scenes) and at 1,000 splats per chunk.
- **Confirmed on the GPU by the E3p run, not before:** CUDA kernels are not the CPU's, so the CPU tests do not
  settle it. E3p's job records `direct_distance_check` once, on the first metric it fills: both forms on the GPU
  over the first 787,432 sorted splats (three full chunks and a partial one), with `identical` and the largest
  difference. It is report only. If it is not identical, the pilot's GN-VQ numbers differ from what the old
  form would have given by that much, and that is the thing to read before quoting them.
- **The rest of the GN-VQ step at 6.13M, checked for the same problem** (per-splat float64 intermediates,
  full-N copies):
  - changed: E3p's `P` (`predicted_dmse`) took a full `[N, 15, 3]` float32 difference of the model's and the
    decoded shN (1.1 GB). It now reads it through `metric_store.ChunkedDifference`, one `quad_form` chunk at a
    time: element-wise, the same values, `P` bit-identical (`test_chunked_difference_keeps_p_bit_identical`).
    E2c's job is not changed;
  - left, under about 1 GB: `quad_form` and `update_centroids` (already chunked at 262,144: about 0.9 GB per
    chunk), the lifted assignment (2,048 splats per chunk, 0.5 GB of scores), `share_in_l2_topk` and
    `shortlist_l2` (4,096 per chunk, 1.07 GB of float32 scores at K = 65,536), `trace_packed` of the metric
    (a `[N, 15]` float32 gather, 368 MB), full-N distances and labels (49 MB each), and `assign_exact`'s
    `x[zero]` (the splats no train view sees; at most 61,407 of 1M in E2's metas, so about 0.07 GB at 6.13M);
  - outside the GN-VQ step and not changed: the writer's input copy of the sorted splats (1.45 GB, float32:
    `PngCompression.compress` edits its input dict in place) and the decoded model's full shN (1.1 GB), which
    the renders need.

### E3p results, Amendment 13 and the E3q build (2026-09-28)

E3p's bundle was committed, FINDINGS section 13 written, E3q pre-registered (Amendment 13) and built, in that
order. No E3q result exists.

- **The bundle is committed exactly as downloaded** (27 files, `25f2133a`, data only). Dace left it in the repo
  root as `gn3p_bundle.zip`, byte-identical to `~/Downloads/gn3p_bundle.zip`; each file was checked byte for byte
  against its zip entry on disk and again as a staged blob (CR-insensitively: the 2 CSVs carry the writer's
  CRLF). The zip stays in the repo root, ignored by an anchored `/gn3p_bundle.zip` line in the root `.gitignore`.
- **Dace's reading was checked against the files before anything was written.** Everything held, with one
  correction and some precision, both carried into section 13 in their measured form:
  - the 11,858 s "CV phase" is the whole `gn_vq_cvfloor` phase (11,857.7 s): the 7 CV rows' steps took
    10,370.7 s and the final row's 1,446.7 s;
  - "11.5 GB" is 10^9-byte GB (11,505,812,480 bytes); bytes +0.12% is +0.117% (bicycle), and train's +0.43% is
    +0.433%.
  - Added from the files: the dMSE steps peaked at 9.00 GB, above GN-VQ; the allocator reserved 15.18 GB of the
    T4's 15.64 GB; the PLAS sort grew 10.4 times from train to bicycle for 6.0 times the splats; GN-VQ ran
    14-17 iterations per run on bicycle against 8-9 on train, so the per-splat GN-VQ rate differs between them.
- **Section 13 was checked mechanically** (`bench/gn/check_s13.py`, run by pytest: 113 recomputed numbers, 276
  numeric tokens, 0 failures). It caught two errors before commit: the text quoted `M_even`'s cache-file size as
  the full `M`'s, and "480 bytes per splat" was not recomputed. The quoted pre-run estimates are checked against
  this file's text.
- **The post-hoc 4-K estimate** is code (`bench/gn/e3p_estimate.py`), so its numbers are recomputed like the rest.
  It takes each part per million splats with the range over bicycle and train, and E2c's and E2's measured
  ratios for the three smaller K; the 13 scenes' splat counts are E3_SCOUTING.md a's. The lower end applies
  train's GN-VQ rate to every scene, the upper bicycle's.
- **Amendment 13's choices the request left open:**
  - **Two fallbacks fixed in advance** (a current `plyfile` if 0.8.1 cannot write and read a `.ply` under the
    session's numpy; one rebuild with `<cstdint>` force-included if an extension's log names a missing
    fixed-width integer type), because both are known ways an old INRIA-era build breaks on a new toolchain, and
    a smoke test that stops on either tells nothing about the rest. Each is recorded as a deviation when used; the
    source is never edited.
  - **The published numbers are Table 9's train row** (per scene), not Table 1's dataset means; their "MB" is MiB,
    shown by the 3DGS column's 242.782 = the pinned `.ply`'s bytes / 2^20.
  - **Protocol ii through `npz2ply.py`**, C3DGS's own decoder, so the harness reads what C3DGS writes rather than a
    reimplementation of its `.npz` format; the amendment states what that loses (the render-time fake quantization).
- **The build lives in the job, not the notebook,** so that every command, deviation and failure lands in the
  meta and the dry run exercises it with a stubbed command runner; E3p's check was a notebook cell.
- **Peak memory through a wrapper** (`e3q_c3dgs_run.py`) that runs `compress.py` unchanged with `runpy`, because
  C3DGS runs in its own process and its code must not change; the allocator's peak is the same measure E3p's steps
  used.
- **Any error is caught in E3q** (a `Steps` subclass), where E3p caught only running out of memory: Amendment 13 f
  asks for every failure to be recorded and the rest to run. Rows that could not be produced are written with
  status `failed` and a `reason`, and a resumed job reruns only those.
- **No smoke tests in E3q:** nothing in it uses the GN code they check.

**Further decisions from the E3p close-out and the E3q build** (added 2026-09-28, before this session was
cleared). Each gives its reason, or where the decision is recorded.

*The C3DGS comparison*

- **C3DGS's published train numbers are Table 9 of arXiv 2401.02436v2** ("Tanks&Temples results", row train),
  not Table 1: Table 1 averages whole datasets, and E3q runs one scene. Recorded: Amendment 13 e,
  `e3q_c3dgs.PUBLISHED_TRAIN`, and a test that compares both with each other.
- **C3DGS's "MB" is MiB.** Shown on its own table: the "3D Gaussian Splatting" column's 242.782 equals the pinned
  train `.ply`'s 254,575,516 bytes / 2^20 (a test asserts it to three decimals). E3_SCOUTING.md b had already found
  the same for Table 1. So every E3q size is reported in bytes, MiB and MB, and C3DGS's own `size` field (MiB of
  the `.npz` alone) is kept beside ours under its own column (`c3dgs_size_MiB_reported`).
- **Their 3DGS train PSNR differs from INRIA's and from ours, and nothing here explains it.** Table 9's "3D
  Gaussian Splatting" train PSNR is 21.770 dB; INRIA's own paper gives 21.097 dB (arXiv 2308.04079v1, Table 8); E3p's
  protocol ii reads the released model at 21.293 dB (`kaggle/gn_e3p/gn3p/`). C3DGS's evaluation
  (`compress.py`'s `render_and_eval`: float renders, its own `ssim` and `lpipsPyTorch`, INRIA's reader) is not
  E3p's protocol ii (8-bit renders, the harness's metric modules), and whether C3DGS evaluated the same released
  checkpoints with the same images is not verified. **Consequence:** E3q's C3DGS-reported PSNR and its protocol-ii
  PSNR are two protocols; only protocol ii compares with the harness and with E3p. Recorded as context in
  Amendment 13 e; the reason for keeping both columns is this one.
- **The published comparison is a sanity check with no criterion** (Amendment 13 e), because C3DGS's numbers come
  from its own evaluation and a different session, and E3q has no pre-stated tolerance to hold them to.

*The build*

- **Into the session's Python environment, not a venv.** E3p's check showed `python -m venv` failing on
  `ensurepip` (Python 3.12.13) and `conda` absent (`conda_on_path` false in `kaggle/gn_e3p/gn3p/gn3p_c3dgs_build.json`),
  so neither the README's conda route nor a venv is available. Recorded: Amendment 13 b. **Consequences** (not
  recorded before): C3DGS's packages stay installed in the session for everything after the build, the harness's
  runner included; the job therefore runs C3DGS before it builds the runner, and installs nothing that gsplat or
  the harness imports (`diff_gaussian_rasterization`, `weighted_distance`, `torch_scatter`, `plyfile`, and `tqdm`
  only if missing). The job takes the interpreter as `--python`, and the notebook passes its own, so pip, C3DGS and
  the job share one environment.
- **`--no-deps` on every pip install** so that pip cannot replace the session's torch (a `torch-scatter` or
  `plyfile` resolution could pull another torch or numpy). Recorded: Amendment 13 b; `e3q_c3dgs.build`.
- **The two fallbacks** (a current `plyfile`; one `<cstdint>` rebuild): see above. Their triggers are checked, not
  guessed: `plyfile` must write and read back a small `.ply` in the session (`PLYFILE_CHECK`), and the rebuild
  happens only if the failed log matches `INT_TYPE_ERROR`. Recorded: Amendment 13 b; `e3q_c3dgs.py`.
- **`torch-scatter` built from source is detected, not assumed** (pip's log names the wheel build), and recorded as
  a deviation when it happens. Recorded: `e3q_c3dgs.build`.
- **The build and each run have timeouts** (5,400 s for the build, 7,200 s per `compress.py` run): nothing about
  either had been measured, and a hung compile must not take the session. Recorded: the job's arguments; the
  reason is this entry.

*The runs and the harness*

- **Protocol ii through C3DGS's own `npz2ply.py`**, so the harness reads what C3DGS's decoder writes rather than a
  reimplementation of its `.npz` format. **What the decoded `.ply` lacks:** C3DGS renders through fake-quantization
  modules applied at render time, and `save_ply` skips two of them: `get_opacity` applies `opacity_qa` after the
  sigmoid and `get_xyz` applies `xyz_qa`, while `save_ply` writes the stored `_opacity` and `_xyz` without them
  (features, scales and rotations are written after their quantizers). So the `.ply` renders without those two. Protocol ii therefore evaluates the decoded model as `npz2ply.py` writes it, which
  is close to, and not identical with, what C3DGS itself renders. Recorded: Amendment 13 d; `e3q_c3dgs.py`.
  **Corrected 2026-09-29 (read from C3DGS's code, not run; FINDINGS section 14):** this holds for a model saved
  straight from training, not for `npz2ply.py`'s output. After the `.npz` round trip, `load_npz` sets `_xyz` to the
  stored float16 values and `_opacity` to the logit of the int8-dequantized opacity, so `save_ply` writes the
  quantized values. What the `.ply` can lack is at most a second application of the quantizers at render time.
- **The peak-memory wrapper runs `compress.py` unchanged** (`e3q_c3dgs_run.py`: `runpy` as `__main__` in C3DGS's
  directory, the arguments after `--`), because C3DGS runs in its own process and its source must not change; it
  resets and reads torch's allocator peak, the same measure as E3p's `Steps`, and records a raised error and its
  traceback instead of losing them. Recorded: Amendment 13 c; the wrapper's docstring.
- **The model directory is hard links to E3p's members** where the filesystem allows (copies otherwise), laid out
  as INRIA's (`point_cloud/iteration_30000/`), because C3DGS's `Scene` searches for the highest iteration and reads
  `cfg_args` from the model path. `--source_path` is given because that `cfg_args` names its authors' paths.
  Recorded: `e3q_c3dgs.layout_model`; Amendment 13 b.
- **A `cfg_args` mismatch is recorded, not a stop, in E3q** (`cfg_args_mismatch` in the meta), where E3p stopped:
  Amendment 13 f asks to record and continue, and C3DGS reads `cfg_args` itself.
- **The uncompressed model's protocol ii is measured again in E3q's session** (7.9 s in E3p), so the decoded models
  have a same-session, same-code reference rather than a number from another run.
- **An attached E3p output's `e3p_inria/` is reused** only after `fetch_member` re-checks each file's size and
  CRC32 (`e3p_inria.py`); nothing else in E3p's output is read.
- **Rows are append-only on resume.** A row with status `failed` is not rewritten: a resumed job appends a new row
  for that config, and the latest row per config is the one to read (`read_rows` returns them all; the summary
  lists them all). Not recorded elsewhere.

*Section 13's checking and estimate*

- **`bench/gn/check_s13.py`** recomputes every number of FINDINGS section 13, its summary paragraph and its
  Sources entry from `kaggle/gn_e3p/gn3p/`, E2b's and E2c's verdict files, E2c's rows and `e3p_estimate.py`, and
  fails on any numeric token it did not recompute; pytest runs it (`test_findings_section_13_numbers_recheck_from_the_repo`).
  **It also reads this file:** section 13 quotes the pre-run estimates of "E3p notebook" (13,105-17,028,
  6,817-7,618, 423-509), and the checker asserts they are still here, so editing that passage breaks pytest.
- **`bench/gn/e3p_estimate.py`** is the post-hoc 4-K cost estimate as code: its construction (per-million-splat
  rates over bicycle and train, E2c's and E2's per-K ratios, the 13 splat counts from E3_SCOUTING.md a), its
  exclusions (downloads, comparators, session steps) and its caveats (linear scaling, which PLAS and the iteration
  counts contradict) are in its docstring. Not recorded before: the two-GPU figure is a lower bound,
  `max(total / 2, longest scene)`, and "sessions" is that bound divided by the 9.5 h start cutoff, rounded up, so
  both are floors, not schedules.
- **Commit order:** the FINDINGS commit (`08902af2`) was docs only, as asked; the checker and the estimate script
  it names came in the code commit (`4ea99729`), so between those two commits section 13's Sources entry names
  files that do not yet exist.
- **Section 13's reading of the protocol gap is labelled post hoc and hedged** ("suggests"): the files show the
  0.598 dB gap on bicycle and 0.001 dB on train, not how the gap splits between gsplat's resize and the extra pixel
  column.

### E3q attempt 1 and the attempt-2 fix (2026-09-28/29, Amendment 13 g)

Order, as asked:
1. the bundle, data only (`59303486`);
2. the diagnosis, read-only, from a scratch clone of C3DGS at `2a234af5` (not in the repo);
3. Amendment 13 g, docs only (`0b11f7ab`);
4. the code and tests;
5. the notebook and this file.

Nothing below was changed after an attempt-2 result: none exists.

- **The bundle is committed exactly as downloaded.** Dace's copy is `~/Downloads/gn3q_bundle.zip`, 13,836 bytes,
  SHA-1 `b93f2975e60153708df0de693ed38b2ce946092c`, and it is the only `gn3q_bundle*.zip` there. It was unpacked
  into `kaggle/gn_e3q/attempt1/` with its arcname `gn3q/` kept, not into `kaggle/gn_e3q/` as the old "After the
  run" said: the attempts are kept apart.
  - Each of the 6 files was checked byte for byte against its zip entry, on disk and as a staged blob. The blob
    check is CR-insensitive for the CSV, whose writer's CRLF `autocrlf` normalizes, as with E3p's bundle.
  - The zip stayed in `~/Downloads`, so nothing needed ignoring.
- **Dace's report was checked against the files first, and it held.** Both runs failed with the reported error;
  the traceback names the call. The bundle adds the location (`utils/splats.py:29`, `extract_rot_scale`), the build's
  and runs' times, and the peak memory.
- **The diagnosis:**
  - **The batch.** `compress_covariance` passes `to_full_cov(compressed_cov)`, where `compressed_cov` is the
    4,096-entry Gaussian codebook joined with every splat kept above `gaussian_importance_include` (3e-6, after the
    `prune_threshold` = 0 prune). So the batch grows with the splat count, up to 1,030,604 on train.
  - **The dtype** is float32: `to_full_cov` allocates with torch's default dtype, and the error names `CUDA_R_32F`.
  - **The kept count is not in attempt 1's files.** The "gaussians keep" line fell outside the 60-line tails, which
    tqdm filled. That is why the wrapper now records every call's `n`.
  - **What else runs batched linear algebra:** `R.det()` on line 34, on the same batch; for batched 3x3 CUDA input,
    torch's LU goes to cuBLAS `getrfBatched`. Nothing else on the compress, fine-tuning (`finetune.py`), save
    (`save_npz`), evaluation or `npz2ply.py` paths does: only single 4x4 `torch.inverse` calls per camera
    (`scene/cameras.py`), which ran fine at scene load, and NumPy on the host.
- **`det` is covered because Dace chose it.** Asked after the diagnosis whether the note and patch should cover
  `eigh` only (as the request wrote) or also `det`, Dace chose both. The reason: `det` never ran in attempt 1, and
  a refusal there would cost another attempt.
- **The patch lives in the wrapper, and `batched.py` is unchanged.** `batched.py` is E0-E3p's code, and its
  behaviour there must not move.
  - The wrapper imports `batched.py` by path and then restores `sys.path`, because C3DGS imports by bare module
    names.
  - It passes `max_batch` = the smaller of 8,192 and the batch that already worked for that op, so `batched.py`'s
    remembered batch is reused. `det`'s name in `batched.py` is `det`, which has no `OP_MAX_BATCH` entry; the
    explicit 8,192 covers it.
- **Why `_like_layout`:** chunked output must keep the op's own memory layout. The first local probe (C3DGS's
  `extract_rot_scale` pattern on CPU) showed:
  - the chunked `eigh`'s values were bitwise equal, but its eigenvectors came back row-major from `torch.cat`,
    where the op returns them column-major;
  - `R.det()` on them then factored the transpose, and the rotations differed by about 1 ulp.

  So the wrapper restores the first chunk's strides (`.mT.contiguous().mT`), and the tests assert bitwise equality
  of the values, the strides and `det` on the result. The claim "chunking does not change what is computed" holds
  bitwise on the CPU. On the GPU it rests on each matrix being independent; the float64 check reports, and a chunk
  of one matrix may take torch's unbatched path (noted in Amendment 13 g).
- **`torch.Tensor.det` is replaced by a plain function**, not a bound method, so that `R.det()` binds `R` as the
  method does.
- **Only a single batch `[N, n, n]` is chunked.** A matrix, two batch dimensions, or `out=` go to the original
  op, and those calls are counted as pass-through. C3DGS's calls are all single batches.
- **The float64 check never touches C3DGS's random stream.** It draws from its own `torch.Generator` (seed 0),
  and a test asserts the global RNG state is unchanged: C3DGS's VQ uses `torch.randint`, and the fine-tuning
  draws cameras afterwards.
  - It samples up to 4,096 matrices per call and runs the original op in float64 on the CPU.
  - For `eigh` it reports the largest eigenvalue difference with the sample's largest eigenvalue for scale, plus
    the reconstruction residual (sign-invariant, unlike comparing eigenvectors). For `det` it reports the largest
    determinant difference.
  - No threshold applies; a failing check is recorded, never raised.
- **The CSV's columns did not change.** The patch record lives in the meta, and the summary adds `c3dgs_runs` and
  `run_deviations`. A new column would have made `assert_e3q_csv` refuse an attempt-1-headed CSV. Attempt 2 should
  not resume from attempt 1 anyway: the uncompressed row is re-measured in the same session by design (decision
  above).
- **The bundle is renamed to `E3q_bundle_2.zip`** in the notebook (`BUNDLE`), as asked, so attempt 2's download
  cannot be confused with attempt 1's `gn3q_bundle.zip` or a browser's `(1)` copy. The arcname stays `gn3q/`.
- **The dry run now runs the real wrapper**, a subprocess around a stand-in `compress.py`, where it used to stub
  it. Only the wrapper's GPU-memory fields are stood in for on the CPU. So the patch, its record and its failure
  path are exercised as they will be on Kaggle.

### E3q attempt 2, section 14 and the C3DGS design (2026-09-29)

The work went in this order:
1. the bundle, data only (`a6c6f725`);
2. FINDINGS section 14, docs only (`6568cf25`);
3. `kaggle/E3_C3DGS_DESIGN.md` and this file.

- **The bundle is committed exactly as downloaded.** It is `~/Downloads/E3q_bundle_2.zip` (15,866 bytes, SHA-1
  `fffbdfbb9963fc3346a4ea57b9b00244239921a4`), the only `E3q_bundle_2*.zip` there.
  - Each of the 6 files was checked byte for byte against its zip entry, on disk and as a staged blob; the CSV
    check was CR-insensitive, as before.
  - The arcname `gn3q/` is kept, under `kaggle/gn_e3q/attempt2/`.
- **Dace's reading was checked against the files before anything was written. Every number held.** The text
  qualifies three of Dace's readings:
  - **"A protocol offset, present with and without compression":** the files do not show it. The uncompressed
    side is C3DGS's *published* 21.770 dB from another session: E3q never ran C3DGS's evaluation on the
    uncompressed model. And the two compressed offsets differ by 0.113 dB (0.523 and 0.410). Section 14 says the
    numbers are compatible with such an offset, lists what the files cannot separate (rasterizer, metric code, the
    8-bit step, the `.npz` round trip), and notes that E3p measured this project's two protocols 0.001 dB apart on
    this checkpoint.
  - **"Ending 0.141 dB above the uncompressed model":** correct. Section 14 adds that fine-tuning is 5,000 more
    training iterations, so this is not like-for-like.
  - **E3p's GN-VQ row:** it is context only, and section 14 says why the codecs are not comparable.
- **A number check caught one error.** A scratch script (not committed; this session's scratchpad,
  `check_s14.py`) recomputes every numeric token in section 14, its summary paragraph and its Sources entry from
  the bundles, E3p's rows and the published row, and fails on any token it did not recompute. It caught one
  error before commit: the `torch-scatter` step was written as 2.0 s, but it took 1.945 s, so 1.9. The checks of
  sections 11-13 still pass (pytest).
  - A committed `bench/gn/check_s14.py` would need a code commit, which this request's commit order did not
    have: an open item.
- **The `.ply` caveat is corrected, not deleted.** Amendment 13 d and the E3q-build decision said the decoded
  `.ply` lacks the `xyz` / opacity quantization. C3DGS's `load_npz` shows it carries the stored quantized values.
  Section 14 and the decision above carry the correction; Amendment 13 d, a pre-registration, is not edited.
- **The design doc is design input, not a pre-registration.** It recommends, with reasons in the doc:
  - the wrapper-patch injection that runs C3DGS's own `vq_features` first, which keeps the random streams paired
    and gives the warm start;
  - `M_i` extended to DC (16x16), because the default quantizes DC with the AC coefficients and
    `color_compress_non_dir = False` looks broken at `2a234af5`;
  - SH-only cross-validation, as frozen;
  - K in {1,024, 4,096, 16,384, 65,536}, after a pilot of the baseline's rate range;
  - no-fine-tuning primary, as Amendment 12 c has it.

  Its compute table was computed by a scratch script (`c3dgs_estimate.py`, not committed) from
  `bench/gn/e3p_estimate.py`'s rates and E3q's train times. It carries the doc's own [estimate] label and is
  not in FINDINGS.

### check_s14, Amendment 14 and the E3r build (2026-09-29)

The work went in this order:
1. `check_s14.py` and its test (`9de1c993`);
2. Amendment 14, docs only (`693a6a4b`);
3. its note f, docs only (`a850c4ea`);
4. the code and tests;
5. the notebook and this file.

No E3r data exists.

- **`check_s14.py`** is the scratch check, rebuilt in `check_s13.py`'s form: recomputed strings must appear in the
  text, claims are re-asserted, and every numeric token is accounted for. It gives 50 recomputed numbers, 132
  tokens, 0 failures. Three planted errors were each caught before it was committed.
- **The threshold grid was Dace's decision.** The request said "values as the design doc recommends", but the
  design doc named the knob without values. Asked, Dace chose "0.2-1.8e-6, x3 steps". That option's label and its
  description disagreed (the description listed {0.2/3, 0.2, 0.6, 1.8, 5.4} x 1e-6); **the amendment follows the
  description, 0.6e-6 x 3^j, j = -2..2**. If the three-point label was meant, a note before any E3r run can
  narrow it.
- **Rule (e) was applied from the estimate.** `bench/gn/e3r_estimate.py` was written before the amendment and is
  quoted by it; the amendment commit precedes the script's commit, like `check_s13` before. With (ii) on both
  scenes, bicycle's upper end is 13.0 h, above Kaggle's 12 h, so (ii) runs on train only. A test asserts the
  amendment's numbers against the script.
- **The GN code was generalized, not copied.**
  - `gn_metric`'s packed helpers read the dimension from the width.
  - `diagnostics._x3` reads the coefficient count, and `update_centroids` its `d`.
  - `e2b.floored_metric` divides by `d`.
  - `gn_vq` takes an optional `quantizer`; None is gsplat's, and every earlier caller passes None.

  At 15 the arithmetic is identical. Every earlier test, the E0-E3q dry runs, `check_chunked_distance` (E2c's path)
  and writer parity pass unchanged, and the notebooks rebuild byte-identical.
- **Why run C3DGS's own `vq_features` first and inject after it:** the random streams stay as in its own run, and
  its codebook is the warm start (Amendment 14 b). **Building E3r showed this pairs nothing across runs:**
  `compress.py` seeds nothing, and torch's default seed differs per process. Hence note f: every run is seeded
  with 0 as `safe_state` seeds. The CPU stand-in then gives identical output across processes, and identical
  output with and without the hooks.
- **The hooks never call `get_features`.** Its `FakeQuantize` observers update on each call, so an extra call
  would change C3DGS's quantizer. The hooks read masks, parameters and quantizer buffers only; a test counts the
  calls.
- **An injection that fails is an error, never C3DGS's own codebook passed off as an injected one.** If the hooks
  cannot install, or the codebook is never injected, the wrapper marks the run an error.
- **The final GN-VQ runs inside the injected C3DGS process** (the metric rows loaded from the harness's cache, by
  the run's own quantized set), not in the harness with the result replayed. The keep threshold reads a
  sensitivity computed with GPU atomics, so the injected run's set may differ from the probe's; running inside
  makes it always right, and the difference is recorded.
- **The two trace shares (16 x 16 and 15 x 15) are equal by construction.** Per band, the sum of squared real SH
  basis values is `(2l + 1) / 4 pi` for any direction (the addition theorem). So `tr(M16_i) = (16 / 15)
  tr(M15_i)` for every splat, and the shares, a ratio, agree. This is known before any data; reporting both is
  then a check. It also means the floor's per-coefficient size, `tr / d`, is the same under both metrics. The dry
  run shows both at 0.256.
- **`gn_vq`'s top-64 shortlist diagnostic is capped at K** in `e3r.run_gn_vq`. It is a diagnostic only; at the
  real K (4,096) it is unchanged.
- **The C3DGS quantizer's `step`** is the AC scale, for `gn_vq`'s log line. Both scales are recorded.
- **The build is a notebook step before the jobs,** because the two jobs share one Python environment and two
  concurrent pip builds could collide. The jobs read its record; a missing record is recorded, not fatal.
- **The GN caches live in `/tmp`** (up to 3.3 GB each on bicycle, recomputable in about 44 s); the probe records,
  which cannot be recomputed without rerunning the probe, live in `gn3r_work/`. The decoded `.ply` files are
  written and deleted one at a time.
- **The dry run shrinks K to the stand-in** (8-64, default 16); every other constant is the job's.

**The memory check and Amendment 14 g** (asked for "before you push"; the request arrived after the push of
`290e62f5`, so g follows the code it constrains; nothing had run):
- **Computed, not estimated.** C3DGS's per-splat allocations are listed from its source with line references. The
  binning buffers depend on the scene, so the tile instances were counted by re-implementing the rasterizer's own
  formulas on the pinned checkpoints, fetched locally (386.6 s for bicycle, 68.5 s for train). A unit test pins
  the formula on a splat whose footprint is known by hand.
- **The accounting is tied to E3q's measured train peak, and the gap is reported, not hidden.** It explains 89.4%.
  The rest, 0.482 GB, is carried to bicycle per splat as the upper end.
- **Bicycle is restricted, by Dace's rule.** Its C3DGS sensitivity pass needs 11.80-14.68 GB even with images on
  the CPU, and 14.16-17.05 GB with them on the GPU, before the CUDA contexts. No method-neutral mitigation touches
  C3DGS's own allocations while its source stays unedited, so the rule ("if none is enough, restrict that step to
  train") applied. Bicycle keeps what fits and measures something the arm needs: the 16 x 16 GN pass at 6.13M.
- **The one-device-copy layout was a real fix, not only a margin.** As coded, the floor's float64 trace alone was
  a 6.67 GB transient at bicycle's size. `metric_store` (E3p's, tested bit-identical) floors slice by slice, and a
  new CPU test pins E3r's path at 16.
- **The splats needing the metric are bounded only by the model.** Every train splat touches a tile, yet C3DGS
  pruned 115,907 of them (E3q), so pruning comes from zero blending weight, not from geometry. The checkpoint
  alone cannot bound the colour-quantized set any tighter, and the check uses the whole model.

**Amendment 14 h, the pruned-splat trace count (2026-09-30, before any E3r run).** Commits: the note `e1e0309e`,
then the code `8d3390c9`. No E3r data existed: no `kaggle/gn_e3r/`, no `gn3r` file in the repo, no E3r bundle in
`~/Downloads`.
- **Why.** C3DGS prunes on zero colour sensitivity. With `clamp_color=False`, DC's gradient is the basis constant
  times the sum of the blending weights, so C3DGS prunes exactly the splats that INRIA's rasterizer gives zero
  weight in every train view. gsplat's GN pass gives the same splats `tr(M) = 0`. The count therefore measures
  whether the two rasterizers agree on which splats are seen at all: the first measurement touching the design
  doc's item 9.2, and only visibility, not render parity.
- **The order check, before any code:**
  - **The prune mask** is C3DGS's `color_importance_n > prune_threshold` over the model as `load_ply` built it
    (`kaggle/e3r_hooks.py:108-110`). `Scene` loads the `.ply` through `GaussianModel.load` ->
    `load_ply` (C3DGS `scene/gaussian_model.py:346-353`, `:355-440`), which builds every tensor from the vertex
    arrays in file order, and `calc_importance` does not reorder.
  - **The `.ply` C3DGS reads is the pinned member,** hard-linked into its model directory (`kaggle/e3q_c3dgs.py:197-205`).
  - **The harness reads the same file in vertex order** (`kaggle/e3p_inria.py:348`, `np.fromfile` then slices).
    `build_runner` assigns those tensors unchanged (`kaggle/gn_e3p_scene.py:286`), and `GaussianScene.put` keeps
    the first component's dict as it is (`gsplat/scene/components/gaussian_scene.py:47-48`).
  - **`compute_gn` runs on those runner tensors** (`kaggle/gn_e3r_scene.py:486, 504`).

  So the mask's index and M's row are the same vertex: no crop, no reorder, no remap.
- **Exact zero on the float32 trace.** Every diagonal entry of `M16_i` is `sum_v s_iv y_k^2 >= 0`, so the trace
  is zero exactly when every entry is. A tolerance would mix in splats that are merely faint.
- **The two limits the note adds**, both beyond the request's text:
  - C3DGS's sensitivity pass renders its **int8-fake-quantized opacity and scales** (`compress.py`
    `calc_importance`), not the stored checkpoint that gsplat's GN pass renders. The two rasterizers therefore see
    slightly different geometry.
  - gsplat's per-view weight `s_iv` is a **16-probe Hutchinson estimate**. It is zero whenever every weight is
    zero, but a zero trace can in principle also come from a cancellation across all 16 probes, or from float32
    underflow of tiny weights.
- **The probe's mask, not the injected run's** (Amendment 14 f: atomics may separate them). It is computed from
  the host copy of the metric in the harness phase, after the full-view GN pass, as its own `Steps` record, with
  errors caught. It is never a CSV column, so `assert_e3r_csv` and the header are unchanged, and a test asserts
  that. `pruned_trace_record` is a pure helper, so pytest covers `ok`, `not_applicable` and `not_computed`.

### E3r results, section 15 and check_s15 (2026-09-30)

The work went in this order:
1. the bundle, data only (`9c28cc89`);
2. Dace's readings checked against the files, and the geometry diagnosis, both read-only and reported in chat;
3. FINDINGS section 15, docs only (`2eb091f9`);
4. `bench/gn/check_s15.py` and its test (`5fda2ea7`);
5. this file.

- **The bundle is committed exactly as downloaded.** `~/Downloads/E3r_bundle.zip` was the only E3r bundle there (no
  `(1)` copy): 61,230 bytes, SHA-1 `e108184f480479a475e31feca10220cf5a2e8c57`. It was unpacked into `kaggle/gn_e3r/`
  with its arcname `gn3r/` kept (17 files).
  - Each file was checked byte for byte against its zip entry, on disk and as a staged blob.
  - The 15 JSONs match exactly. The 2 CSVs carry the writer's CRLF, which `autocrlf` normalizes, so they were compared
    CR-insensitively, as with E3p's and E3q's bundles.
  - The zip stayed in `~/Downloads`, so nothing needed ignoring.
- **Dace's readings (a-g) all held against the files.** Precision the files added, carried into section 15:
  - the 16/15 trace ratio holds to about 8e-9 (float32 sums), not exactly; the two shares differ by 4.4e-11;
  - at save, the fine-tuned run's zero points moved too (DC -81 to -60, AC -4 to -25), and its kept rows moved by up
    to 1.1345;
  - every GN-VQ run was still falling when it stopped at 20 iterations: last relative drops 1.50e-3 to 2.93e-3;
  - bicycle's 594.7 s: the estimate does include the download (E3p's 256.8 s). The excess is a slower download and
    runner, plus two steps it did not count (the cache writes, the member re-check) and 15.4 s outside any step.
- **The geometry SHA-1 diagnosis: (ii), nondeterminism on the GPU, not the injection.** Read from the code, not run.
  C3DGS's source was read at `2a234af5` from a scratch clone outside the repo (session `d4632a74`'s scratchpad).
  - **What the SHA-1 hashes** (`kaggle/e3r_hooks.py:150-160`): the raw bytes of `_gaussian_indices`, `_rotation` and
    `_scaling`, concatenated, as C3DGS's `compress_covariance` returns. That is after `set_gaussian_indexed`
    (C3DGS `compression/vq.py:233-237`), before fine-tuning (`compress.py:178-188` against `:207-221`) and before
    `save_npz`'s Morton sort.
  - **C3DGS's VQ draws from global streams** at `vq.py:27` (`kaiming_uniform_`, CPU), `:35` (`rand_like`, CUDA) and
    `:98` (`randint`, CPU).
  - **Between the colour VQ's return (`vq.py:116`) and the geometry VQ (`:210`), nothing draws:**
    - C3DGS itself: `:108-116`, `:187-192` (`set_color_indexed` only wraps parameters), `:200-206`;
    - the hooks: `e3r_hooks.py:137-148` and `:130-134` only read state;
    - the injection (`:195-239`): GN-VQ, the table quantizer, `torch.load`. The lifted check uses a private generator
      (`bench/gn/diagnostics.py:442-443`), and a grep of every module on the path finds no call without `generator=`;
    - the wrapper seeds after installing the hooks (`kaggle/e3q_c3dgs_run.py:227-243`); its float64 check is private
      (`:165-166`) and runs after the geometry VQ.
  - **The data agrees.** The three K = 4,096 runs have equal draw counts (803,076 colour splats; a geometry batch of
    281,237 in each). The two injected runs run identical code up to the hash, and their SHA-1s still differ. And
    C3DGS's own colour codebook already differs from a bit-identical input (`features_equal`), before any hook computes
    anything.
  - **Not separable from the files:** `torch_scatter`'s atomic sums in the codebook update (`vq.py:40-52`) against the
    sensitivity backward's atomics. The importance values are not recorded.
- **What went into section 15, beyond the request's outline:**
  - train's GN pass against the memory check (1.65 GB measured, 1.41 GB predicted), post hoc; the check scaled
    bicycle's peak by the splat count;
  - C3DGS's peak (4.71-4.74 GB) against E3q's 4.55 GB, post hoc;
  - the injected runs' warm-start objectives, which differ under the same metric, as further evidence that the warm
    starts differ;
  - the statement that only the three K = 4,096 default-threshold runs are shape-paired, so "all 10 SHA-1s differ" is
    not read as evidence for the other seven.
- **Dace's four edits to the draft, before commit:**
  1. The summary's "That is not GN-VQ's effect" became "It cannot be read as GN-VQ's effect". The files show the
     confound, not the absence of an effect.
  2. The zero-trace bullet became post hoc and gained train's figure (102,318 of 1,026,508, 9.97%). A sentence was
     added on why it does not contradict Amendment 14 g's tile count (6,131,774 of 6,131,954): touching a tile is not a
     non-zero blending weight. Without that sentence a reader could take the two counts as inconsistent.
  3. The C3DGS-peak comparison says E3q's run was also unseeded and in another session, and that the files do not
     attribute the difference. The hooks are not the only difference between the runs.
  4. "The two injected runs execute identical code" became "identical code up to the point the SHA-1 is taken
     (fine-tuning comes after it)". The fine-tuned run's code does differ, but only after the hash.
- **`check_s15.py`** is built in `check_s14.py`'s form: 142 recomputed numbers, 406 numeric tokens, 0 failures.
  - It also re-reads this repository's cited lines (the hooks, the wrapper, `diagnostics.py`), so a moved line breaks
    it. It greps the GN modules for a global draw.
  - C3DGS's line numbers are listed constants, because its source is not in the repo.
  - Three planted errors were each caught before commit: a baseline byte count, the bicycle zero-trace percentage
    (post hoc), and `gnvq_k4096`'s geometry SHA-1.
  - pytest: 118 passed.

### Related work, paper direction and E4 decisions (2026-10-01)

Docs only: no code, rule or result. Commits: `04577841` (the related-work notes), `7a954b0d` (the E4 design), then this
flush.

1. **Overlap found.** Two preprints overlap GN-VQ:
   - arXiv **2609.28997** ("OGC", K. Pietroszek, 24 Sep 2026; code `github.com/moholo-founder/ogc-3dgs` at
     `49ccae72`, PolyForm Noncommercial 1.0.0);
   - arXiv **2609.15735** (T. T. Do, P. A. Chou, G. Cheung, 14 Sep 2026).

   Details, with page and file:line for every item, are in `kaggle/RELATED_WORK_OGC.md` (`04577841`). The equivalence
   check found that our `M_i` equals their S2 observation Gram in expectation: the 16-probe Rademacher estimate is
   unbiased for sum_p w_ip^2. The PDFs and the clone stay outside the repository.
2. **Paper direction (Dace's decision).**
   - A pre-registered replication and extension, journal first.
   - Headline: the observation metric's failure on test views, with the cross-validated floor as the fix.
   - The venue is decided after reproducing OGC.
   - **E3a** (gsplat PNG on the 13 INRIA scenes) is **deferred**.
3. **E4: the C3DGS arm, as designed in `kaggle/E4_DESIGN.md` (`7a954b0d`).** The decisions below are final and
   **supersede the design doc's recommendations where they differ.**
   - **a. Co-primary; both must pass:**
     - D1 = `rho_cv` minus ρ = 0;
     - D2 = `rho_cv` minus OGC.

     Both on protocol-ii test-view PSNR, at K = 4,096, the default threshold and no fine-tuning; paired within a
     forked C3DGS process; per-scene mean of 3 processes.
   - **b. The bar, for each difference:**
     - the mean over scenes > 0;
     - positive on at least ceil(0.7 n) scenes (5 of 7);
     - the mean > 2 x SE_noise, where SE_noise = (the pooled within-scene SD of the difference across processes) /
       sqrt(3 n).

     If `rho_cv` = 0 on a scene, D1 there is 0, which does not count as positive.
   - **c. Rows** (one forked process each):
     1. C3DGS's own VQ;
     2. OGC's `vq.gram_kmeans` on our 16 x 16 metric with its defaults (`lam` 1e-3, 15 iterations, its own init);
     3. GN-VQ at ρ = 0;
     4. the scalar tr(M) weighting;
     5. GN-VQ at `rho_cv`, chosen once per scene from a probe run.
   - **d. Scenes and failures:**
     - the INRIA checkpoints of bonsai, counter, kitchen, room, truck, drjohnson and playroom. Disclose that G2c used
       the first five's test views, with this project's own checkpoints;
     - a run out of memory is retried with the images on the CPU; if it still runs out, the scene is dropped before
       any result and n shrinks;
     - a header read (camera count, image sizes, splat count only) is **not touching**; the rule is written before
       the read.
   - **e. Secondaries:**
     - fine-tuning on all 3 processes for C3DGS's own row, OGC and `rho_cv` (OGC minus C3DGS, against the noise
       SE);
     - Gram against scalar;
     - per-array bytes and the index entropy;
     - the threshold-sweep BD in a later session.
   - **f. E4p, a pilot on train first:**
     - the fork end to end, 3 processes, with fine-tuning;
     - OGC's full pipeline against their published train row;
     - the exact S2 Gram against our 16-probe `M` (relative error).

     E4p may only raise the number of repeats or fix bugs.
   - **g. Order:**
     1. Amendment 15 alone (docs);
     2. Deep Blending's header read and a dated note;
     3. E4p code, tests and dry run;
     4. the E4p run and its findings;
     5. E4.

     Kaggle titles: **"E4p C3DGS fork pilot"** and **"E4 C3DGS gate"**.
- **Where the doc's recommendations changed:**
  - its single primary (R5 − R3) became the co-primary D1 and D2;
  - its "pooled mean above C3DGS's own spread" became the 2 x SE_noise rule;
  - fine-tuning moved from one process per scene to all 3 processes for three rows;
  - its scenes-that-do-not-fit options became "retry on the CPU, else drop before results".

  Not stated here, so the doc's recommendations stand as inputs to Amendment 15, not as decisions:
  - the OGC context rows (512 entries; `lam` 1e-6);
  - the arc protocol (left out);
  - R2's statistic (our M16);
  - the ridge-to-mean variant (left out);
  - the licence handling (fetch and import at run time, never vendor).

  **Settled since, in Amendment 15** (below): the arc protocol and the ridge-to-mean variant are out, the licence
  handling and R2's statistic are decided, and of the context rows only `lam` 1e-6 runs.

**Amendment 15 (`573142ac`, 2026-10-01): decisions.** Docs only. Amendment 15 pre-registers E4p and E4 from the
decisions above; nothing of E4 exists beyond it. Dace checked a first draft, asked for edits, then approved it with one
more addition.

- **Dace's decisions (2026-10-01), recorded as decisions, not inputs:**
  - no 90-degree arc protocol in E4;
  - no ridge-toward-the-mean variant of GN-VQ;
  - OGC's code is fetched at run time from a clone pinned at `49ccae72`, imported from there, and never copied into
    the repository or a bundle (PolyForm Noncommercial; noncommercial academic research);
  - OGC's row runs on our 16 x 16 metric, not on their `A_i`;
  - `rho_cv` is chosen once per scene from a separate probe run.
- **The calls made on the draft:**
  - **OGC's settings are their C3DGS host's call** (`hosts/c3dgs_run.py:61`): `iters=15` (`--lloyd_iters`),
    `device="cuda"`, `chunk=100000`. `lam` 1e-3 and `seed` 0 are the function's own defaults. The function's other
    defaults differ (`iters=20`, `device="mps"`, `chunk=150000`, `vq.py:17`), so "its defaults, 15 iterations" was
    read as the host's call. Checked in the scratch clone at `49ccae72`.
  - **Table 19 is E4p's check of OGC's code.** OGC publishes no per-scene train row for its full pipeline: only
    dataset means (Table 4 for Mip-NeRF 360; Table 6 for Tanks & Temples, the mean of train and truck, and truck is a
    gate scene). Its C3DGS host's per-scene Table 12 is Mip-NeRF 360 only. Table 19 (p. 22, uniform degree reduction,
    `run_exps.py` stage `core`) is the only per-scene train row. The replication of their VQ claim is E4's
    pre-fine-tuning `ogc` minus `c3dgs`, not E4p.
  - **Row 2b `ogc_lam1e6`**, report only: row 2 with `lam` 1e-6, the paper's stated value, because which ridge
    produced their tables is unknown (`kaggle/RELATED_WORK_OGC.md`, open question 1). No 512-entry row.
  - **Two pre-fine-tuning secondaries**, each with its `SE_noise`: `ogc` minus `c3dgs` (the replication of OGC's
    +0.49 dB host gain) and `gnvq_cv` minus `c3dgs`; also `ogc_lam1e6` minus `ogc`. **A correction to the request's
    "on new scenes":** bonsai, counter, kitchen and room are among the 9 scenes of OGC's Table 12, on the same INRIA
    checkpoints; only truck, drjohnson and playroom are new to that claim. The amendment says so.
  - **At least 5 scenes:** with n < 5, E4 is `incomplete`, never `pass`.
  - **Failure scope:** only a failure that loses a primary row (rows 2, 3 and 5, measured with protocol ii) reruns
    its process once, whole, with the same seed, and then makes E4 `incomplete`. A secondary failure (fine-tuning,
    row 1, row 4, row 2b, bytes, C3DGS's own evaluation) is recorded, not rerun, and changes no verdict. Row 1 is on
    the secondary list because it enters no primary difference.
  - **The probe run's evaluations are deferred** until the first process's colour steps and saves are done, so a
    memory drop precedes every result. The probe is then evaluated from its decoded `.npz` (C3DGS's evaluation of the
    loaded `.npz`, and protocol ii), unlike E3r's in-memory evaluation of its probe; context only. **A consequence for
    the code:** `compress.py` evaluates in memory straight after its save, so the wrapper has to stop the probe at
    its save and evaluate later.
  - **Every row of a process is saved before any is evaluated** (b.4). The first draft evaluated row 1 inside C3DGS's
    own flow first, which contradicted the deferral.
  - **E4 may span several Kaggle sessions.** Scenes are assigned to sessions in a dated note before any E4 run;
    a scene's probe run, harness phase and 3 processes run in one session.
- **Choices in the draft that Dace let stand:**
  - processes seeded 0, 1 and 2, and the probe seeded 0;
  - if `rho_cv` = 0, row 5 is row 3 and is not run again;
  - `SD_pool` is the root mean of the per-scene variances, and a scene where D1 is 0 by rule enters with 0 (the
    correct SE of the mean when some scene means are fixed zeros);
  - 9-decimal rounding and the `incomplete` verdict, as in G2c;
  - the threshold sweep's details (one process per point seeded 0, rows 1-5, `rho_cv` reused, the domain-scaled
    fit);
  - the seven scenes' archive pins go in note i with the header read;
  - the number of distinct colour indices is recorded with the index entropy;
  - the Gram relative error's definition in f.

### Notes i and ii, and the E4p build (2026-10-01)

Order, each pushed fast-forward after the ancestor check: `573142ac` (Amendment 15), `8b9221d3` (HANDOFF decisions),
`85b43cff` (note i), `15f263a7` (note i's scripts), `fcd1914f` (note ii), then E4p's code, tests, notebook and this
file. No E4p or E4 data exists.

- **Note i's header read stayed inside Amendment 15 d's rule:** the archive's zip directory (matching E3p's pins),
  then for drjohnson and playroom only their `cameras.json` (count and sizes kept, the file deleted) and their `.ply`
  inflated one byte at a time up to `end_header`; 16 HTTP requests. Every pin's size equals E3_SCOUTING's.
  `cfg_args` was not read, so Deep Blending's loaded size is taken as the `cameras.json` size (both under 1,600 wide).
- **Note i's feasibility reproduces the design's table** except bonsai's and counter's colour-step columns, 0.01 GB lower:
  the design gives OGC's chunk as 1.76 GB without its unrounded value. drjohnson is expected to need the CPU retry.
- **Note i's scripts were committed afterwards** at Dace's request (`15f263a7`), unchanged but for a two-line header;
  re-run, the two offline ones regenerate their JSONs byte-identical and the network one re-reads equal fields.
- **Note ii (Dace's design, with two additions of Dace's):** the centre's conditioning, and item e covering note ii a's
  render cost. The draft's references to the amendment's sections were disambiguated from the note's own items.
- **Dace's two conditions for the build, both implemented and tested:**
  1. rows are installed only into a restore of the copy taken when `compress_gaussians` returns. The fork hashes each
     row's installed state just before its save (`e4p.state_sha1`), and
     `test_e4p_fork_installs_rows_into_the_pre_save_copy` shows each hash equals the same row installed into a deep
     copy of the untouched model, in pre-sort order (and that row 1's saved state differs);
  2. OGC's dependencies only into an isolated `--target` (`e4p_ogc.plan_deps`): pip resolves the missing ones against
     the session with `--dry-run --report` first; any replacement of a session package installs nothing and skips
     Table 19 with the reason; otherwise `--no-deps --target` of exactly the report's distributions, and a target that
     would shadow a session module is not used. Never the session's site-packages (a test and dry-run stage 5f).
- **How the fork reaches C3DGS without editing it:** `compress.py` evaluates through its module-global
  `render_and_eval`, looked up at call time (`compress.py:242`), after its save (`:231`), and `scene` and the parameters
  are `run_vq`'s locals. The save hook reads them from `run_vq`'s frame and replaces that global in `run_vq`'s
  globals; `join_features`, called through `compression.vq`'s globals (`vq.py:188`), is wrapped to capture its exact
  inputs. The probe's evaluation from its `.npz` replaces `scene.Scene` with a subclass before `compress.py` imports it
  (C3DGS's own `Scene` and `load_npz`, then `render_and_eval`, then a `Stop` the wrapper records as a success).
- **Choices the request left open:**
  - job ogc **recomputes our `M`** rather than reading job fork's cache: independent of the other GPU's timing and of
    its failures, about 65 s; whether job fork's cache is bit-identical is then reported (a free check of the GN pass's
    reproducibility on the GPU);
  - E3r's harness phase is **reused unchanged** (its CV rows in their own CSV with E3r's columns), with the colour
    share, calibration and pruned-trace count switched off;
  - per-step host RSS through `Steps` subclassing E3q's with a sampler (`psutil`, process and children, 0.25 s), and
    per-row costs inside the fork; the wrapper folds the hooks' per-row GPU-peak resets into the process's peak;
  - the probe's own measurements are a `probe` row (context, `reason` says so);
  - a primary row's failed protocol ii (job side) counts as a primary loss; `results_exist` for the drop rule is any
    row evaluated in the scene.
- **The CPU stand-in now has C3DGS's shape** (`run_vq` with its locals, the global `render_and_eval`, `finetune.py`
  drawing cameras with Python's `random`, `Scene`, `load_npz`, `training_setup`). E3r's tests and dry run pass
  unchanged; its saved arrays are byte-identical (the hooks test compares them).
- **Two checks needed fixing, neither touching what they check:** `test_e3p_pins_and_amendment_12` read every pin row in
  `PREREG_GN.md`, so note i's 21 pins (since `85b43cff`) broke it; it now reads Amendment 12's section only.
  `check_s15.py` re-read the wrapper's cited lines from the working tree; E4p's flags moved them, so it now reads the
  cited lines at `2eb091f9` (section 15's commit; the cited files are identical at E3r's `b27e1421`).
- **Checked:** pytest 130 passed; the E4p dry run (every stage, OGC's real clone, `gram_kmeans` and
  `plugin.observation_gram` on the CPU); E3q's and E3r's dry runs; E0-E3r notebooks rebuild byte-identical.

### E4p results check and the E4q pivot (2026-10-02)

Docs only (this flush). No FINDINGS section, amendment or code changed in this session after the bundle commit.

1. **The E4p bundle is committed data only** (`60c4839b`): `~/Downloads/E4p_bundle.zip`, the only copy (no "(1)"), 298,965
   bytes, SHA-1 `7641a3cdc980e08a2b80cfa726bad68805a9af5a`; 11 files under `kaggle/gn_e4p/gn4p/`, each byte-checked against
   its zip entry on disk and as a staged blob (CSVs CR-insensitively); pytest 130 passed before the commit. **Dace's
   readings were checked against the files in chat** (Claude Code's confirmation report, with three read-only
   diagnoses: the fine-tuned OGC row's smaller `features_rest`, OGC's GPU peak and the gate scenes' feasibility, and
   every difference between the GN-VQ rows and OGC's row). None of it is written up yet; FINDINGS section 16 is next.
2. **Decisions (Dace, via chat):**
   - **a. E4 will be withdrawn before any E4 data**, by Amendment 16 (not yet drafted). The reasons, to be stated there,
     from train (a development scene):
     - D1 is about 0 on the standard test views;
     - D2 compares unequal effective codebooks at equal K: OGC reseeds empty clusters (`vq.py:74-84` at `49ccae72`),
       while GN-VQ keeps the empty ones of C3DGS's warm start. So equal-K PSNR is the wrong primary.
   - **b. Next: E4q**, exploratory, **development scenes only** (train; treehill if it fits after a header read under
     Amendment 15 d's rule). Its contents, to be pre-registered in Amendment 16:
     - a dissection ladder from the frozen GN-VQ toward OGC, one factor at a time: reseed empty clusters; init in
       proportion to tr(G); clip per part (DC, AC) or none; ridge toward the cluster mean in place of the floor;
       iterations 15 and 50; the final int8 assignment off;
     - OGC's `lam` chosen by held-out-train-view cross-validation over {1e-6 .. 1};
     - an equal-rate comparison by the colour-threshold sweep (BD);
     - note ii's measures, plus fidelity as the PSNR of the pooled MSE, the quantizer state at every save, each table's
       range against its int8 grid, and the codewords used per row.

     The new gate hypothesis is written only after E4q's findings. **Train and treehill can never be gate scenes.**
   - **c. The global clip is not patched into the frozen method.** `gn_vq.py:179` clips to one min / max over all 48
     values, DC included, so in C3DGS's layout the AC bound is effectively the DC maximum. A per-part clip is a ladder
     variant in E4q, not a bug fix.
   - **d. Engineering:** fix the reserved-memory recording bug (the wrapper folds the hooks' per-row resets into the
     allocated peak but not the reserved one); OGC's chunk is 25,000 for any gate run, and E4q checks on train that the
     labels equal those at chunk 100,000 (or reports how many differ).
   - **e. Rule (Dace): pytest before every commit, docs-only commits included.**

### Open items (E0-E3r: closed; E4p: ran, not written up; E4: to be withdrawn; E4q: next)

- ~~**Run E0 on Kaggle.**~~ **Done (2026-09-20): G0 passed.** The bundle is committed unchanged in
  `kaggle/gn_e0/gn/` and FINDINGS section 8 quotes it. Nothing in E0 is left to run. The session took
  about 41 minutes of timed steps with the two jobs in parallel.
- ~~**Run E1 on Kaggle.**~~ **Done: G1 failed** on the size rule, all six seeds (FINDINGS section 9,
  `kaggle/gn_e1/gn1/`). Not amended (Amendment 7 a).
- ~~**Run E2 on Kaggle.**~~ **Done: G2a passed** (FINDINGS section 10, `kaggle/gn_e2/gn2/`). The bundle
  is committed unchanged, its BD measures are checked (`kaggle/gn_e2/bd_sensitivity.json`), and the
  measured runtime is in the decision index above. Nothing in E2 is left to run.
- ~~**E2 questions the committed bundle will answer.**~~ **Answered:** all 11 checkpoints and sort caches
  were in the run-5 output, with the pinned sha1s; no job was skipped by the start cutoff, the
  exploratory phase included; the Tanks & Temples jobs at data factor 1 took 2,940.3 s (truck) and
  2,985.3 s (train); eps = 1e-2 took 20 iterations at K = 1,024, 15-20 at 4,096, 12-15 at 16,384 and 9-10
  at 65,536 (FINDINGS section 10); the run-5 wheel's key still matched (install 169.3 s, no build).
- ~~**Run E2b on Kaggle.**~~ **Done (2026-09-25):** fidelity criterion `works`, garden control held,
  PSNR criterion `does not work`; all 8 `rho = 0` cells `identical` to E2's `gn_vq`. The bundle is
  committed unchanged in `kaggle/gn_e2b/gn2b/` (`a3c0099a`), FINDINGS section 11 quotes it, and the
  measured runtime is under "E2b notebook". Nothing in E2b is left to run.
- ~~**FINDINGS section 11 must follow Amendment 10 f.**~~ **Closed:** section 11 says the floor works
  only "in fidelity terms", and only because both `verdicts.fidelity` (`works`) and
  `verdicts.garden_control` (true) hold in `gn2b_e2b.json`. The PSNR criterion is reported alongside
  with the cross-term caveat and supports no such claim. The rule (Amendment 10 f, `0d180360`) still
  governs any later write-up of E2b.
- ~~**Run E2c on Kaggle.**~~ **Done (2026-09-26): G2c passed** (FINDINGS section 12, `kaggle/gn_e2c/gn2c/`,
  `b1163f24`). The measured runtime is under "E2c notebook". Nothing in E2c is left to run.
- ~~**Run E3p on Kaggle.**~~ **Done (2026-09-27/28):** no step out of memory; the C3DGS check failed at the
  venv (FINDINGS section 13, `kaggle/gn_e3p/gn3p/`, `25f2133a`). The measured runtime is under "E3p notebook".
- ~~**Run E3q attempt 2.**~~ **Done (2026-09-28/29):** all three rows `ok`, one chunked `eigh` and one chunked `det`
  per run on 281,237 matrices with no refusal (`kaggle/gn_e3q/attempt2/gn3q/`, `a6c6f725`; FINDINGS section 14,
  `6568cf25`). The measured runtime is under "E3q notebook". Nothing in E3q is left to run.
- ~~**E3q attempt 1 failed; the fix is in, and attempt 2 is ready to run.**~~ Kept for the record: C3DGS **built and imported** and **both
  runs (`c3dgs_ft0`, `c3dgs_ft5000`) failed** in `compress.py`, in `extract_rot_scale` (`utils/splats.py:29`).
  The cause was a cuSOLVER batched-eigen workspace refusal on `torch.linalg.eigh` of one float32 batch of 3x3
  matrices, `CUSOLVER_STATUS_INVALID_VALUE` from `cusolverDnXsyevBatched_bufferSize`, E0's failure class
  (`kaggle/gn_e3q/attempt1/gn3q/`, `59303486`; see "E3q notebook", attempt 1). **Done:**
  1. the bundle, unchanged, in `kaggle/gn_e3q/attempt1/`;
  2. the diagnosis: the batch is the 4,096 codebook entries plus the kept splats, up to 1,030,604 on train, and
     `R.det()` runs on the same batch; see the decisions above;
  3. Amendment 13 g (`0b11f7ab`);
  4. the wrapper's `ChunkedLinalg` for `eigh` and `det`, its tests, the dry run, and the notebook, which now
     writes `E3q_bundle_2.zip`.
- ~~**Run E3r.**~~ **Done (2026-09-30):** every step ran, nothing failed (`kaggle/gn_e3r/gn3r/`, `9c28cc89`; FINDINGS
  section 15, `2eb091f9`; `bench/gn/check_s15.py`, `5fda2ea7`). The measured runtime is under "E3r notebook". Nothing
  in E3r is left to run.
- ~~**E3r's open questions.**~~ **Answered in section 15** (and "E3r results, section 15 and check_s15" above):
  1. **The geometry SHA-1s** differ in all 10 runs, the probe's (`8f72f4ea`) and `gnvq_k4096`'s (`f8641bfb`)
     included. The cause is nondeterminism on the GPU, not the inject path: no global draw lies between the two VQs,
     and C3DGS's own colour codebook already differs from a bit-identical input.
  2. **The stopping rule:** all 9 GN-VQ runs (7 CV, 2 injected) stopped at `max_iters` (20), still falling. E3p's train
     runs took 8-9 iterations, and E2c's at K = 4,096 13-19, all stopping at the rule; not like-for-like.
  3. **`pruned_trace_check`:** 13,627 pruned splats have `tr > 0`, and 38 unpruned ones `tr = 0`. The trace share of
     the 13,627 is not recoverable: the metric cache was not bundled.
- ~~**Next: draft Amendment 15.**~~ **Done (2026-10-01):** `573142ac`, docs only; its decisions are under "Amendment 15
  (`573142ac`, 2026-10-01): decisions" above.
- ~~**Amendment 15 g.2.**~~ **Done (2026-10-01):** note i (`85b43cff`), its scripts (`15f263a7`); note ii (`fcd1914f`).
- ~~**Amendment 15 g.3.**~~ **Done (2026-10-01):** E4p's code, tests and dry run (see "Notes i and ii, and the E4p
  build" above and "E4p notebook").
- ~~**Run E4p on Kaggle.**~~ **Done:** bundle committed data only (`60c4839b`), checked against Dace's readings in chat.
- **Next: draft FINDINGS section 16 (E4p) and `bench/gn/check_s16.py`** (check_s15's form, with planted errors), shown
  to Dace before commit.
- **Then: draft Amendment 16** (dated, before any E4 data): E4 withdrawn, E4q registered (see "E4p results check and the
  E4q pivot (2026-10-02)", item 2), shown to Dace before commit. Then E4q's code, tests, dry run and run.
- ~~**E4's session assignment.**~~ **Moot:** E4 is to be withdrawn. Gate scenes stay untouched.
- **Engineering before E4q runs:** the reserved-memory recording fix; OGC's chunk at 25,000 with E4q's label-equality
  check against 100,000 on train.
- The item below is kept for its checklist.
- ~~**Pre-register the C3DGS arm**~~ **Became E4 (superseded by the item above).** Kept as a checklist:
  (Amendment 12 c) from `kaggle/E3_C3DGS_DESIGN.md` and FINDINGS section 15, in
  a new amendment, before any of its code or data. It must settle every item of the doc's section 9, or state it as a
  limitation. In particular:
  - `M_i` with DC (16x16), an extension of the frozen method;
  - the warm start: C3DGS's own codebook or `lloyd_wopa_area`, and GN-VQ's iteration cap (all of E3r's runs hit 20);
  - the codec's int8 table quantizer as the port's quantizer;
  - SH-only cross-validation;
  - the RD knob and its grid: E3r measured the colour threshold at 2.49x in bytes and K at 1.02x;
  - GPU nondeterminism: seeded runs do not pair (section 15), so repeat runs to measure the baseline's spread, or a
    geometry shared between the baseline and GN-VQ;
  - C3DGS's fine-tuning as a secondary;
  - pins for the seven gate scenes' archive members;
  - Deep Blending's loading;
  - C3DGS's memory above train's size: it has not run on anything larger (Amendment 14 g; the gate scenes reach
    3,405,153 splats).

  The development-scene pilot that this item used to leave open was E3r itself.
- ~~**Optional: commit a section-14 checker.**~~ **Done (2026-09-29):** `bench/gn/check_s14.py`, run by pytest (`9de1c993`).
- **Then: E3's design beyond the C3DGS arm, not written yet; E3a (gsplat PNG on the 13 INRIA scenes) deferred by
  Dace's decision of 2026-10-01** (no amendment, code or notebook; the C3DGS arm's design
  input is `kaggle/E3_C3DGS_DESIGN.md`; `kaggle/E3_SCOUTING.md` has the scouting, which recommends C3DGS as the first host and E3a, gsplat PNG on the INRIA checkpoints, as the
  first step): port the frozen `gn_vq_cvfloor`
  (FINDINGS section 12, "The method, as frozen"; Amendment 11 b) to other codecs, on INRIA checkpoints.
  Pre-register it in a new amendment before any E3 code or data. Inputs for the design, not decisions:
  - **Scenes:** all 11 scenes with pinned checkpoints have now informed a decision or a gate (section 12),
    so any held-out claim in E3 needs new checkpoints, which the INRIA checkpoints would be.
  - **From E2c:** `rho_cv` was 1e-3 to 1e-1, never above. The gain over E2's GN-VQ was clear only on room.
    No seed spread has been measured for differences this small, and no direct reproduction check ran.
  - **From E2b**, from section 11:
    - `rho_cv` was 1e-1, the top of E2b's grid, in all six cells of treehill, flowers and stump. E2c's
      extended grid was never used above 1e-1;
    - on treehill the floor closed most of the test-PSNR gap to `lloyd_trace`, but not all of it;
    - post hoc: large `rho` tends to `lloyd_trace`'s objective (section 11, "The limit of the floor").
- ~~**Commit `check_s12.py`.**~~ **Done (2026-09-27):** `bench/gn/check_s12.py`, run by pytest.
- ~~**Amendment 6's two rows cost little.**~~ **E1 done;** the measured costs are in FINDINGS
  section 9. Kept for the record: They are two more GN-VQ runs per scene at seed 0 and
  K = 65,536, warm-started from a `lloyd_wopa_area` cache that is already on disk, so they add no
  clustering. E0's refines measured the same loop at K = 65,536
  (`gn_e0/gn/gn_refine_<variant>_<scene>.json`): 9.36-9.51 s per assignment and 1.31-1.32 s per
  update, and 72.1-73.6 s for a whole three-iteration variant including the iteration-1 top-64
  diagnostic. GN-VQ runs at most 10 iterations and stops at a relative drop below 1e-3, so each row
  is a few minutes plus one evaluation. The secondary clusterings below still dominate.
- ~~**The secondary clusterings dominate E1's runtime.**~~ **E1 done:** they took 60.9% of garden's
  job and 44.9% of bicycle's (FINDINGS section 9). Kept for the record: `lloyd_trace` and `lloyd_c3dgs` are six fresh
  library-Lloyd runs per scene (two weightings x three seeds) at K = 65,536, and E0 measured that
  clustering at 281-642 s per run, so they are roughly half an hour to an hour per scene against about
  12 s for the GN pass and a few minutes for all the GN-VQ iterations. They are reported, not gating
  (Amendment 5 d), so if a session is short they can be dropped from `CONFIGS` in the notebook's
  config cell and added later by attaching that session's output: every row resumes on its own.
- ~~**The first run crashed in cuSOLVER.**~~ **Fixed (2026-09-20), not re-run.** Both scene jobs died
  right after the GN pass, in `diagnostics.eigen_stats`:
  `cusolverDnXsyevBatched_bufferSize` -> `CUSOLVER_STATUS_INVALID_VALUE`. `eigen_stats` already looped
  in chunks of 65,536, so that was the batch cuSOLVER refused, one more than the CUDA grid limit of
  65,535; the failing call is a workspace-size query, which does not read the matrix values. The
  refines' centroid solve runs at the same size (one system per cluster, K = 65,536), so it would have
  failed next.
  - **Fix:** `bench/gn/batched.py`. `batched_linalg` slices the batch dimension, keeps each call at
    `LINALG_MAX_BATCH` (32,768: half of the batch that failed, below 65,535) or below, and halves the
    batch further on a backend batch or memory error, down to one matrix, recording each reduction in
    `linalg_fallbacks()` (reported by `linalg_scale`). Every batched linalg call in `bench/gn/` and
    `kaggle/gn_e0_scene.py` goes through it.
  - **Not verified on a GPU:** the real cuSOLVER limit is unknown, which is why the helper also
    halves on demand and why `selftest.py`'s `linalg_scale` runs the job's sizes before the jobs start.
    If `linalg_scale` reports reductions, 32,768 was still too large; lower `LINALG_MAX_BATCH`.
  - **Also added:** `finite_report(M)` before any linalg (a non-finite `M` would be a bug in the GN
    pass, and cuSOLVER reports it as an opaque backend error), and the per-job log tails in the bundle.
- ~~**Never executed yet: the CUDA-only paths.**~~ **All ran in E0** (the feature render and its
  backward, the SH / toy / end-to-end checks, TorchPQ at every K, the fp32 lifted assignment with the
  v2 criterion). E1 adds only one new CUDA-only path, GN-VQ's loop, whose pieces (the lifted
  assignment, the ridge update) E0 already exercised at K = 65,536.
- **The batch limit, now measured** (FINDINGS section 8): cuSOLVER refused the eigendecomposition at
  32,768, 16,384 wanted 9.49 GiB, and 8,192 worked. `OP_MAX_BATCH["linalg_eigvalsh"]` is 8,192, and
  `batched_linalg` remembers the batch that worked instead of rediscovering it per chunk. If a future
  run still reports reductions in `linalg_scale`, lower it again.
- ~~**Toy check scene on CUDA.**~~ **Resolved (2026-09-20, PREREG Amendment 4):** `toy_scene` drew
  with the CUDA generator, so the CUDA toy check would have rendered a different scene from the one
  `toy_noise.py` simulated. Dace treated it as a correction. `toy_scene` now draws on the CPU, and
  because a CPU draw is not bitwise reproducible across CPU dispatch and platforms, both checks render
  committed fixtures whose hashes are asserted before rendering (see "Amendment 4 session" above).
- ~~**Assumptions the run will confirm.**~~ **Confirmed by E0 and E1:** the run-5 output held the run-3
  clustering caches (every E1 K = 65,536 `lloyd_wopa_area` row came from them), and the run-5 wheel's
  key matched (E1's install took 170.5 s, with no wheel build). E2 repeats both assumptions and adds the
  run-4 caches; its committed bundle will show.
- ~~**If the 10k lifted check fails in E2.**~~ **It passed on all 11 scenes** (sum excess / sum d_min at
  most 1.32e-08, FINDINGS section 10). In E2b it ran per floored metric, 32 checks over the four scenes,
  and all passed (sum excess / sum d_min at most 2.90e-08, FINDINGS section 11).
- ~~**After the E1 and E2 runs.**~~ **Done:** bundles in `kaggle/gn_e1/gn1/` and `kaggle/gn_e2/gn2/`,
  FINDINGS sections 9 and 10. After the E2b run: see "E2b notebook", "After the run".
- ~~**E1: write down the GN-VQ variant and its size matching before any E1 run.**~~ **Done:**
  PREREG_GN.md Amendment 5, committed before any E1 code.
- ~~**What E1 will settle, and what it will not.**~~ **Settled: G1 failed** (FINDINGS section 9). Kept for
  the record: G1 is only GN-VQ against `lloyd_wopa_area` at
  K = 65,536 over seeds 0-2. The secondary weightings, the dominance flags and the rate-distortion
  curves with BD-rate are reported and cannot pass or fail it. If G1 fails, FINDINGS section 8's
  hypothesis about the quantizer's range is the first thing to check: the clip is what tests it, and
  `fraction_outside_warm_range` with `clusters_rejected_by_clip` say how hard it bit.
- ~~**The dry-run scripts are outside the repo.**~~ **Resolved (2026-09-19):** they moved to
  `bench/gn/dryrun/`, with no temp-folder paths. The toy dry run and the writer-parity check pass from
  there; the temp copies were removed.
- **The 64 flat bundle files in `kaggle/`** are ignored, not deleted. Delete them by hand whenever
  convenient; the committed copy is `kaggle/run5/tilequant/`.
- ~~**`gn_bundle (1).zip` sits untracked in the repo root.**~~ **Ignored (2026-09-21), not
  deleted.** `gn1_bundle.zip` (E1's) is ignored the same way. The root `.gitignore` has
  `/gn_bundle (1).zip` and `/gn1_bundle.zip`, anchored, so `git status` is clean while both files stay in
  the repo root (still there on 2026-09-22). Their contents are committed unpacked in `kaggle/gn_e0/gn/`
  and `kaggle/gn_e1/gn1/`, and no bundle zip is ever committed; delete them or move them to
  `~/Downloads` whenever convenient. E2's and E2b's bundles were unpacked straight from `~/Downloads`
  and never copied into the repo root, so nothing needed ignoring.

## Conventions and gotchas

- **Setup:** MipNeRF360, `examples/benchmarks/compression/mcmc.sh` settings (MCMC, cap 1M, data factor 4
  outdoor / 2 indoor, LPIPS VGG for eval). Data lives under `/tmp`, output in `/kaggle/working`.
  `MAX_JOBS=2` (higher values OOM the build).
- **Size:** `zip_bytes` (`zip -r`, as in `summarize_stats.py`) AND `size_bytes` (raw file bytes).
  Decisions use both.
- **Library shN path:** `_compress_kmeans` runs torchpq `KMeans(65536, distance="manhattan")`, quantizes
  the codebook to 6 bits with one scalar min/max and stores uint16 labels.
  - The codebook is saved **column-major** (via permute), and new encoders must match
    (`np.asfortranarray`), or bytes and sizes differ.
  - The run-2/3 baseline feeds cached float centroids and labels into the unchanged `_compress_kmeans`
    through a stub (`precomputed_kmeans`). Its output is byte-identical to a normal call.
- **torchpq 0.3.0.6 `KMeans`:**
  - Random init (`np.random.choice` of k points, global numpy RNG), `max_iter=100`, `tol=1e-4` on the
    summed squared centroid change (practically unreachable in 100 iterations), `n_redo=1`.
  - manhattan = L1 assignment + MEAN update. Empty clusters become 0. Returned labels come from the last
    assignment.
  - Reproduces to about 0.002 dB across sessions, **not bit-exactly**. The run-3 re-run reproduced
    run-2 seed 0 exactly on garden and bicycle. Run 5's re-run on the 9 run-4 checkpoints was
    identical on 7; bicycle moved +0.0024 dB (`shN.npz` +41 B) and stump +0.00006 dB (+3 B).
- **PLAS sort:** the seed-0 order is cached per checkpoint (`tilequant/sweep/<scene>/cache/seed0/order.pt`,
  keyed by the checkpoint sha1). Run 3 loads it and never rebuilds. Run 4 builds it for new scenes the way
  runs 1–2 did (`torch.manual_seed(0)`, `compute_sort_order`), without the cache's k-means.
- **MipNeRF360 data:**
  - flowers and treehill are in `360_extra_scenes.zip`, the other 7 scenes in `360_v2.zip`.
  - The colmap parser needs `images/` (it resizes full-res JPGs into `images_<f>_png`), `images_<f>/`
    (names), `sparse/` and `poses_bounds.npy`.
- **Run-4 resume rules:**
  - A checkpoint counts only once `torch.load` reads it back with step 29999 (marker
    `ckpts/train_complete.json`).
  - Rows carry the checkpoint sha1. A retrained scene moves its old CSV and gate aside as `.stale-*`.
  - The gate exits with code 3 and stops the queue.
- **Library k-means layout:** the builtin backend returns `[K, D]` centroids; `_compress_kmeans` stores
  the quantized codebook with `np.asfortranarray`, which is a no-op for the TorchPQ path (already
  column-major) and makes both backends write the same bytes for the same values.
- **`zip -r` stores the run directory path**, so zip sizes are only comparable between runs whose run
  directory paths are equally long. The run-5 parity rows are therefore written into the run-3 path
  (`/tmp/tilequant_run3_runs/<scene>/kseed0/lloyd_wopa_area`). Elsewhere each extra path character
  costs 2 x 9 B (9 zip entries: the directory and its 8 files, each named in the local header and the
  central directory). Examples: 126 B for run-5 candidates vs run-4 baselines (0.0008%); 72 B for
  `baseline_lib` vs `baseline`, confirmed in run 5. The decision also compares raw bytes, which are
  path-independent.
- **Peak GPU memory** (`peak_mem_bytes`) is `torch.cuda.max_memory_allocated()` over `compress()` in
  the job process: it includes the loaded scene and whatever else the process holds. The same config
  measured 3.37 GB in the per-scene jobs and 3.61 GB in the parity-gate jobs.
- **Builtin k-means reproducibility:** in run 5, all 18 library candidate rows equal the run-4
  benchmark rows. That is measured, not guaranteed (float64 `index_add_` uses CUDA atomics).
- **Results tables:** each reference row must come from, and be labeled with, that dataset's own
  results CSV (`repo_row_label`). The run-5 notebook once labeled the Tanks & Temples row
  "MipNeRF360.csv" (fixed in `1b4d40f8`).
- **`PngCompression` always asks for 65,536 centroids**, so a CPU dry run has to shrink
  `_compress_kmeans`'s `n_clusters` to make the clustering non-trivial.
- **E0 specifics:**
  - The GN render is a direct `gsplat.rasterization` call with `sh_degree=None` and 17 feature
    channels. 17 is in the default `GSPLAT_NUM_CHANNELS` list, and the notebook builds without
    `NUM_CHANNELS`. The fused path applies `max(SH + 0.5, 0)` to colors, which the GN linearization
    ignores; `clamp_fraction` in `gn_meta_<scene>.json` says how often it bites.
  - E0 writes every codebook through the library's builtin writer with a precomputed codebook
    (`precomputed_codebook` patches `png_compression.weighted_kmeans`). The files are byte-identical
    to run 3's writer (`bench/gn/dryrun/check_writer_parity.py`).
  - Measured errors use `Runner.rasterize_splats(splats=...)` with the eval's keyword arguments;
    `Stage.render` only forwards to it.
  - `gm.gsplat_render` is looked up at call time, so a dry run can swap in `toy_render.render_bruteforce`.
  - The run-3 clustering cache key is `sha1(ckpt_sha1 + seed-0 order bytes)`, the same as E0's.
    The run-3 caches only hold K = 65,536.
  - Exact assignment (`diagnostics.lifted_argmin`) uses `d(i,k) = const_i + u_i . v_k` with 165-dim
    vectors, one fp32 GEMM per chunk. TF32 is off, and coordinates are shifted by the codebook mean.
    `assign_exact` keeps the current centroid unless the new one is strictly closer by the direct
    formula, so assignment never raises the objective. Splats with tr(M) = 0 take their L2-nearest
    centroid.
  - The refines' `mu = 1e-4 * tr(sum M) / 15` is per cluster. Ridge pulls weakly constrained
    directions toward 0; proximal keeps them at the old centroid.
- **Decision rule rounding:** run-4 deltas and means are rounded to 9 decimals before comparing, and
  byte limits are integer compares (`cand * 1000 <= base * 1003`). Float round-off otherwise flips the
  exact-threshold cases: every constructed "exactly -0.02 dB" case had it.
- **Timings** (`timings.json`; job timings are rounded up to the 15 s poll):
  - training: garden 2280 s, bicycle 2145 s (data factor 4, T4); run 4: 1,932-2,243 s at data
    factor 4, 2,983-3,398 s at data factor 2; run 5 Tanks & Temples (data factor 1): train 1,599 s,
    truck 1,736 s
  - gsplat wheel build: 4,404 s (run 5)
  - torchpq manhattan k-means: 398–440 s
  - Lloyd (run 3): up to 7.26 s per iteration, 100 iterations maximum
  - library builtin k-means (run 5): `opacity_area` 297-646 s on MipNeRF360 (mean 510 s), 298 / 357 s
    on truck / train
  - per config (compress + decompress + eval): about 13–15 s
  - per job: about 40 s of process overhead
- **Local environment:**
  - venv `F:\gsplat\.venv` (CPU torch 2.11, Python 3.13).
  - Working copies are CRLF (`autocrlf`): normalize line endings when string-matching.
  - Bash heredocs with nested quotes break: write scripts to files.
  - Use `PYTHONIOENCODING=utf-8` for console output.

## Results so far (details and sources in `FINDINGS.md`)

| Run | Question | Result |
|---|---|---|
| 1 | tile-wise / smooth min/max for PNG params | no win; `pr_worthy` false |
| 2 | shN codebook quantization ranges; loss decomposition | no win on quantization; the decomposition shows clustering is most of the shN loss (0.374 / 0.163 dB garden / bicycle, vs 0.026 / 0.011 dB for the 6-bit codebook) |
| 3 | k-means clustering levers (library format unchanged) | **found `lloyd_wopa_area`**: higher PSNR and lower LPIPS than the baseline on garden and bicycle at all 3 k-means seeds (+0.096 / +0.030 dB mean PSNR). The strict rule (`pr_worthy`) failed only on two garden SSIM cells, both inside the baseline's own SSIM seed spread. |
| 4 | full MipNeRF360 validation of `lloyd_wopa` / `lloyd_wopa_area` (pre-registered rule) | **validated on all 9 scenes.** Both candidates pass; `pr_candidate` = `lloyd_wopa_area`, +0.111 dB mean PSNR at -0.10% mean size, better on every scene. All 7 sanity gates passed. |
| 5 | the same change as library code (`feat/png-weighted-kmeans`): parity gate, MipNeRF360, Tanks & Temples, cost, CPU-only smoke test | **passed.** Parity exact (and all 18 library rows = the run-4 rows); Tanks & Temples +0.052 dB mean PSNR at +0.08% size, both scenes better; k-means 510 s vs 405 s (MipNeRF360), 328 s vs 416 s (T&T); peak GPU memory 3.37-3.62 GB vs 1.03-1.26 GB; CPU smoke test passed; torchpq baseline reproduced to ~0.002 dB. |
| E0 (`bench/gn-vq`) | does a Gauss-Newton metric on shN predict the shN-only render error (G0, `PREREG_GN.md`)? | **G0 passed** (2026-09-20, second attempt; the first crashed in cuSOLVER). 34 non-tied pairs of 36, none misordered, every codebook calibrated within 0.5-2x. The two exploratory refines cut the GN objective 3.7-4.0x and still lost 0.04-0.53 dB after the codec's centroid quantizer. Details in FINDINGS section 8. |
| E1 (`bench/gn-vq`) | does GN-VQ beat `lloyd_wopa_area` at equal size (G1, `PREREG_GN.md` with Amendments 5 and 6)? | **G1 failed** (run 2026-09-20), on the size rule, on all six seeds: GN-VQ was +2.37% (garden) and +0.98% to +1.01% (bicycle) larger, beyond 0.5%, while gaining +0.195 to +0.201 / +0.087 to +0.091 dB. The rate-distortion secondary favours GN-VQ (bicycle BD-rate -9.84%; garden entirely above). Details in FINDINGS section 9. |
| E2 (`bench/gn-vq`) | does GN-VQ (eps 1e-2) beat `lloyd_wopa_area` in rate-distortion on 9 held-out scenes (G2a, `PREREG_GN.md` Amendments 7 and 8)? | **G2a passed** (run 2026-09-21/22): 9 of 9 held-out wins, mean BD-rate -5.37% against -5%. H2b (against `lloyd_trace`) holds with 8 of 9 counted; treehill's computed win is a cubic-fit artifact. Post hoc: PCHIP gives 8 of 9 and -5.66%; the shN stream alone about -31%. Details in FINDINGS section 10. |
| E2b (`bench/gn-vq`) | does an isotropic floor on the metric, selected by train-view cross-validation, improve GN-VQ's fidelity on the test views of treehill, flowers and stump without costing garden (`PREREG_GN.md` Amendments 9 and 10, exploratory)? | **Done (run 2026-09-25), exploratory:** works in fidelity terms, not by the PSNR criterion. R at `rho_cv` is below E2b's own `rho = 0` in 6 of 6 cells (treehill at K = 65,536: 1.1253 to 0.6602), and the garden control holds; treehill stays below `lloyd_trace` in test PSNR at both K (-0.0016 / -0.0120 dB). `rho_cv` = 1e-1, the top of the grid, in all six cells; `rho = 0` reproduces E2 exactly. Details in FINDINGS section 11. |
| E2c (`bench/gn-vq`) | does GN-VQ with the cross-validated floor (`gn_vq_cvfloor`, `PREREG_GN.md` Amendment 11) keep GN-VQ's wins and do no harm on the five scenes no decision has used (gate G2c)? | **G2c passed** (run 2026-09-26): 5 of 5 wins against `lloyd_trace`, mean BD-rate -6.17% against `lloyd_wopa_area`, BD-PSNR against E2's `gn_vq` -0.0013 to +0.0141 dB. `rho_cv` above 0 in all 20 cells (1e-3 to 1e-1). The gain over E2's GN-VQ is clear only on room (test dMSE 0.7603 of `gn_vq`'s at K = 65,536). Method frozen. Details in FINDINGS section 12. |
| E3p (`bench/gn-vq`) | does the frozen pipeline run at INRIA scale on a T4, and what does each step cost (`PREREG_GN.md` Amendment 12, exploratory, no verdict)? | **Ran (2026-09-27/28), no step out of memory** on bicycle (6,131,954 splats) and train: bicycle peaked at 11.51 GB allocated, its GN-VQ at 8.49 GB; its job took 19,699.0 s. Protocol ii reads the uncompressed models at 25.196 / 21.293 dB against INRIA's published 25.246 / 21.097; protocol i reads bicycle 0.598 dB lower. `rho_cv` 3e-1 (bicycle) and 1e-1 (train). The C3DGS build check failed at the venv (no `ensurepip`). Details in FINDINGS section 13. |
| E3q (`bench/gn-vq`) | does the C3DGS host build and run its own compression on INRIA's train model (`PREREG_GN.md` Amendment 13, smoke test, no verdicts)? | **Attempt 1 (2026-09-28): C3DGS built (204.9 s) and imported; both `compress.py` runs failed** after 270.8 / 273.3 s: cuSOLVER refused `torch.linalg.eigh` on one float32 batch of 3x3 matrices in `extract_rot_scale`. Uncompressed protocol ii 21.293 dB, as in E3p. **Attempt 2 (Amendment 13 g: `eigh` and `det` chunked, 281,237 matrices per call, no refusal) ran both runs:** fine-tuned 13.267 MiB, 21.843 dB in C3DGS's own evaluation against its published 21.863 dB; protocol ii 0.410-0.523 dB lower on the compressed models; runs of 351.6 s / 666.4 s. Details in FINDINGS section 14. |
| E3r (`bench/gn-vq`) | does the C3DGS host run with GN-VQ injected, and which knob gives C3DGS rate range (`PREREG_GN.md` Amendment 14 with notes f, g, h; pilot, no verdicts)? | **Ran (2026-09-30), every step, nothing failed.** On train the colour threshold spans 2.49x in `.npz` bytes (protocol ii 20.733-21.205 dB), K 1.02x (20.971-21.024 dB). GN-VQ ran inside C3DGS's own run with and without fine-tuning (`rho_cv` 1e-2; all 9 GN-VQ runs at the 20-iteration cap). The injected row read +0.127 dB at +1.485% bytes against the probe, which cannot be read as GN-VQ's effect: seeded runs are not reproduced on the GPU (geometry and C3DGS's own codebook differ). Bicycle's 16 x 16 GN pass peaked at 8.41 GB, 616,376 bytes above the prediction. Details in FINDINGS section 15. |

## PR plan (`feat/png-weighted-kmeans`)

1. ~~Run 5 on Kaggle.~~ Done (2026-09-18), results in FINDINGS section 6.
2. ~~Fill `PR_DRAFT_weighted_kmeans.md`.~~ Done: both datasets, per-scene deltas, cost, memory,
   reproducibility, extras.
3. ~~Open the PR from `feat/png-weighted-kmeans`.~~ Done (2026-09-18):
   [nerfstudio-project/gsplat#1063](https://github.com/nerfstudio-project/gsplat/pull/1063), open. The
   branch stays library-only; the benchmark lives on `bench/tilequant`.
4. ~~The default flip.~~ Written as the PR's last commit, `61cd1baf` (`kmeans_backend="builtin"`,
   `kmeans_weighting="opacity_area"` as defaults), so maintainers can drop it on its own.
   `benchmarks/compression/results/*.csv` were not regenerated and still hold the TorchPQ numbers.

## Open items

- PR #1063 is open: respond to review only when Dace asks (standing rule: no comments unless asked).
- `benchmarks/compression/results/*.csv` still hold the TorchPQ numbers; regenerating them is a
  follow-up, only if maintainers want the default flip.
- `lint/format-code.sh` and the tests on a CUDA machine (locally only CPU).
- PR #1061 (`fix/png-empty-tensor`): no action unless asked.
- **Related work:** `kaggle/RELATED_WORK_OGC.md` (2026-10-01, literature notes, not findings) on arXiv 2609.28997 (OGC: per-Gaussian S2 observation Gram, matrix-weighted Lloyd, a C3DGS drop-in) and 2609.15735 (global factorised Gram, sqrt + KLT, no VQ), with the metric-equivalence check (our M_i equals their S2 Gram in expectation) and open questions; read it before the C3DGS arm's pre-registration. PDFs and the `ogc-3dgs` clone (`49ccae72`, PolyForm Noncommercial) are not in the repo.
- **E0 / E1 / E2 / E2b / E2c / E3p / E3q / E3r / E3:** see "Open items (E0-E3r: closed; E4p: ran, not written up; E4: to be withdrawn; E4q: next)". E0 is done
  and G0 passed (`kaggle/gn_e0/gn/`, FINDINGS section 8); E1 is done and G1 failed (`kaggle/gn_e1/gn1/`,
  FINDINGS section 9); E2 is done and G2a passed (`kaggle/gn_e2/gn2/`, FINDINGS section 10); E2b
  (Amendments 9 and 10, exploratory) is done, and works in fidelity terms but not by the PSNR criterion
  (`kaggle/gn_e2b/gn2b/`, FINDINGS section 11); E2c (Amendment 11) is done and G2c passed, with the method
  frozen (`kaggle/gn_e2c/gn2c/`, FINDINGS section 12). E3p (Amendment 12, an exploratory engineering pilot on
  INRIA's bicycle and train checkpoints) is done (`kaggle/gn_e3p/gn3p/`, FINDINGS section 13). E3q (Amendment 13,
  a C3DGS smoke test on train) failed in attempt 1 (cuSOLVER batched eigen) and ran in attempt 2 with Amendment 13 g's fix (`kaggle/gn_e3q/`,
  FINDINGS section 14). E3r (Amendment 14, the C3DGS-host pilot) is done (`kaggle/gn_e3r/gn3r/`,
  FINDINGS section 15). The C3DGS arm's integration design is `kaggle/E3_C3DGS_DESIGN.md`; it became E4 (`kaggle/E4_DESIGN.md`,
  decisions of 2026-10-01), pre-registered as E4p and E4 by Amendment 15 (`573142ac`, notes i and ii); E4p ran (`60c4839b`); E4 is to be withdrawn
  for E4q (2026-10-02). **Next: FINDINGS section 16, then Amendment 16.** The rest of E3's design (other codecs, INRIA checkpoints) is not written yet.
