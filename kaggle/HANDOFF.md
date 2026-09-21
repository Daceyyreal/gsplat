# Handoff: gsplat `PngCompression` benchmark (`bench/tilequant`, `bench/gn-vq`)

For a fresh session. Results live in `kaggle/FINDINGS.md`; this file covers how the work is organized.
Runs 1-5 are on `bench/tilequant`. E0 (a Gauss-Newton metric for shN), E1 (GN-VQ, G1 failed) and E2
(GN-VQ on held-out scenes, built, not run) are on `bench/gn-vq`: see the E0, E1 and E2 notebook sections
below and `kaggle/PREREG_GN.md`.

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
| `bench/gn-vq` | off `bench/tilequant` (`12f912eb`): the E0 / E1 / E2 pre-registration (Amendments 1-7), `bench/gn/` (GN metric, diagnostics, G0 / G1 / G2 rules, GN-VQ, smoke tests, scene fixtures), the E0, E1 and E2 jobs and notebooks, and E0's and E1's results; never goes upstream | **E0 done: G0 passed** (2026-09-20, `kaggle/gn_e0/gn/`, FINDINGS section 8). **E1 done: G1 failed** on the size rule (`kaggle/gn_e1/gn1/`, FINDINGS section 9). **E2 is built, not run** (Amendment 7). `gsplat/` and `setup.py` are identical to the run-5 commit, so the run-5 wheel's key matches. Do not modify `feat/png-weighted-kmeans` (PR #1063) from here. |

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
| `HANDOFF.md` | this file |
| `run2/tilequant/` ... `run5/tilequant/` | results bundles (`results_bundle.zip` contents); each session restores the previous one, so files repeat (git stores them once). In `run5/`, `run5_tt_table.csv` and `rd_run5.png` were regenerated locally after the label fix `1b4d40f8` (see FINDINGS sources); the downloaded original is `~/Downloads/results_bundle (3).zip` |
| `PREREG_GN.md` | E0 / E1 pre-registration (`bench/gn-vq`): G0 rule, validity checks, exploratory scope, G1 for E1. Amendment 1: the toy check. Amendment 2: G0 over 9 codebooks per scene with tie-exempt pairs, and exact-assignment refines instead of the shortlist one. Amendment 3: the G0 verdict is the ranking alone (the ratio is reported as calibration), the end-to-end exactness check, the lifted-check criterion v2, and a proximal rise invalidating that variant instead of stopping. Amendment 4: the toy and end-to-end scenes are committed fixtures with pinned hashes (a correction: the CUDA toy check would have drawn a different scene from the simulated one), end-to-end preconditions read from gsplat's render, and a report-only probe-noise diagnostic. Amendment 5 (after G0 passed, before any E1 code): E1's GN-VQ variant, the one-sided size matching that G1's last sentence delegates, the reported secondaries and the exploratory ablations. Amendment 6 (2026-09-21, before any E1 run): two more exploratory rows, GN-VQ at ridge `eps` = 1e-3 and 1e-2, seed 0 at K = 65,536 on both scenes, not judged by anything; and the final codebook's own quantizer range logged beside the warm start's. Amendment 7 (2026-09-21, after E1's results, before any E2 code): G1 failed as pre-registered and is not amended; the project's deviation, stated as one (it continues on Amendment 5 d's rate-distortion evidence; garden and bicycle become development scenes); E2's variant (eps = 1e-2, 20 iterations), 9 held-out scenes, 4 configs x 4 K, G2a (gate) and H2b (reported). Amendment 8 (2026-09-21, before any E2 data): G2a's mean is over all 9 held-out scenes, a scene without a defined BD-rate entering with a substitute (a: `gn_vq` reaches the baseline's best PSNR for fewer bytes; b: the reverse; c: 0%), so it is always defined; exploratory `gn_vq_eps1e4` rows on garden and bicycle, not judged. **Never edit a rule after results exist**; add a dated amendment instead. |
| `gn_e0/gn/` | the E0 results bundle, unpacked as downloaded (22 files); FINDINGS section 8 quotes it |
| `gn_e1/gn1/` | the E1 results bundle, unpacked as downloaded (28 files); FINDINGS section 9 quotes it |
| `gn_e2_scene.py` | E2, one scene per process (Amendment 7): `upstream_l1`, `lloyd_wopa_area`, `lloyd_trace` and `gn_vq` (eps 1e-2, 20 iterations) at K = 1,024-65,536, seed 0, plus `uncompressed`; `--configs gn_vq_eps1e4` adds Amendment 8's exploratory rows (garden and bicycle only). Checks the checkpoint's pinned sha1 first, downloads and later deletes its own scene, reuses E0's / E1's modules unchanged; resumable per (config, K) |
| `build_gn_e2_bench.py` / `gn_e2_bench.ipynb` | E2 notebook (build output; edit the builder, never the JSON). E0's and E1's notebooks are left exactly as they ran |
| `gn_e1_scene.py` | E1, one scene per process: the G1 baseline and GN-VQ at K = 65,536 for seeds 0-2, the two secondary weightings, the seed-0 K grid, the two ablations and an `uncompressed` row. Reuses E0's GN cache; resumable per (config, K, seed) |
| `build_gn_e1_bench.py` / `gn_e1_bench.ipynb` | E1 notebook (build output; edit the builder, never the JSON). E0's notebook is left exactly as it ran |
| `gn_e0_scene.py` | E0, one scene per process: render parity, GN pass (`gn_cache/<scene>.pt`), spectrum, Spearman, the 9 G0 codebooks (predicted vs measured, test and train GT metrics, reproduction fields at K = 65,536), the lifted-assignment check (gates only the refines), the ridge / proximal refines (a proximal rise marks that row invalid). Resumable per (scene, config, K, seed). |
| `build_gn_bench.py` / `gn_bench.ipynb` | E0 notebook (build output; edit the builder, never the JSON) |
| `../bench/gn/` | `sh_basis.py`, `gn_metric.py`, `batched.py` (chunked batched linalg and the finite check; the fix for the first Kaggle crash), `diagnostics.py` (spectrum, Spearman, predicted / measured, lifted exact assignment and its check, refines, end-to-end exactness check), `g0.py` / `g1.py` / `g2.py` (the G0, G1 and G2a / H2b rules as code), `gn_vq.py` (E1's variant and the codec's quantizer), `selftest.py` (the notebook's smoke tests; `--device cpu` is the CPU stand-in), `fixtures/` (the committed toy and end-to-end scenes, `.npz` + `.json`, Amendment 4) and `make_fixtures.py` (wrote them), `toy_render.py` (CPU renderer for tests), `toy_noise.py` / `.json` (Amendment 1), `test_gn.py` (71 CPU tests), `dryrun/` (`fake_env.py` with the shared CPU stand-in, the E0, E1 and E2 dry runs and the writer-parity check; not collected by pytest) |
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

### Open items (E0, E1: closed; E2: open)

- ~~**Run E0 on Kaggle.**~~ **Done (2026-09-20): G0 passed.** The bundle is committed unchanged in
  `kaggle/gn_e0/gn/` and FINDINGS section 8 quotes it. Nothing in E0 is left to run. The session took
  about 41 minutes of timed steps with the two jobs in parallel.
- ~~**Run E1 on Kaggle.**~~ **Done: G1 failed** on the size rule, all six seeds (FINDINGS section 9,
  `kaggle/gn_e1/gn1/`). Not amended (Amendment 7 a).
- **Run E2 on Kaggle** (Dace). `kaggle/gn_e2_bench.ipynb`, attaching the run-5 output (required) and,
  optionally, the E0 or E1 output (for garden's and bicycle's `gn_cache/` only); see "E2 notebook"
  above. The mean question Amendment 7 left open is settled by Amendment 8 (the mean over all 9 with
  substitutes). Nothing about the rules is open before the run.
- **E2 unknowns the run will settle:** whether all 11 checkpoints and sort caches are in the run-5
  output (the restore cell checks before any install); the cost at K = 1,024 and at data factor 1; how
  many iterations eps = 1e-2 takes below K = 65,536.
- **Amendment 6's two rows cost little.** They are two more GN-VQ runs per scene at seed 0 and
  K = 65,536, warm-started from a `lloyd_wopa_area` cache that is already on disk, so they add no
  clustering. E0's refines measured the same loop at K = 65,536
  (`gn_e0/gn/gn_refine_<variant>_<scene>.json`): 9.36-9.51 s per assignment and 1.31-1.32 s per
  update, and 72.1-73.6 s for a whole three-iteration variant including the iteration-1 top-64
  diagnostic. GN-VQ runs at most 10 iterations and stops at a relative drop below 1e-3, so each row
  is a few minutes plus one evaluation. The secondary clusterings below still dominate.
- **The secondary clusterings dominate E1's runtime.** `lloyd_trace` and `lloyd_c3dgs` are six fresh
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
- **Assumptions the run will confirm:**
  - the run-5 output contains the run-3 clustering caches; otherwise the configs are re-clustered, with
    a warning;
  - the run-5 wheel key matches the current Kaggle image; otherwise a build of about 73 min, as in run
    5.
- **If the 10k lifted check fails in E2** on a scene, that scene's GN-VQ rows are skipped and G2a / H2b
  are `incomplete`. In E1 it passed on both scenes (sum excess / sum d_min 4.07e-13 and 2.82e-09).
- ~~**After the E1 run.**~~ **Done:** bundle in `kaggle/gn_e1/gn1/`, FINDINGS section 9. After the E2
  run: see "E2 notebook", "After the run".
- ~~**E1: write down the GN-VQ variant and its size matching before any E1 run.**~~ **Done:**
  PREREG_GN.md Amendment 5, committed before any E1 code.
- **What E1 will settle, and what it will not:** G1 is only GN-VQ against `lloyd_wopa_area` at
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
  deleted.** `gn1_bundle.zip` (E1's) is ignored the same way. The root `.gitignore` now has `/gn_bundle (1).zip`, anchored, matching that one name, so
  `git status` is clean while the file stays where it is. Its contents are committed unpacked in
  `kaggle/gn_e0/gn/`, and no bundle zip is ever committed. Delete it or move it to `~/Downloads`
  whenever convenient. **E1's bundle will arrive the same way:** unpack `gn1_bundle.zip` into
  `kaggle/gn_e1/`, check the files against the zip, and add its download name to `.gitignore` the same
  way (or remove the zip).

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
| E2 (`bench/gn-vq`) | does GN-VQ (eps 1e-2) beat `lloyd_wopa_area` in rate-distortion on 9 held-out scenes (G2a, `PREREG_GN.md` Amendments 7 and 8)? | **built, not run**: `kaggle/gn_e2_bench.ipynb`, 17 rows per scene over 11 scenes plus 8 exploratory rows on garden and bicycle, 71 CPU tests and a dry run pass. |

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
- **E0 / E1 / E2:** see "Open items (E0, E1: closed; E2: open)". E0 is done and G0 passed
  (`kaggle/gn_e0/gn/`, FINDINGS section 8); E1 is done and G1 failed (`kaggle/gn_e1/gn1/`, FINDINGS
  section 9); E2 is built and waiting for a Kaggle run (`kaggle/gn_e2_bench.ipynb`, "E2 notebook"
  above).
