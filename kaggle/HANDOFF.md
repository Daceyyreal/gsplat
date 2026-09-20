# Handoff: gsplat `PngCompression` benchmark (`bench/tilequant`, `bench/gn-vq`)

For a fresh session. Results live in `kaggle/FINDINGS.md`; this file covers how the work is organized.
Runs 1-5 are on `bench/tilequant`. E0 (a Gauss-Newton metric for shN) is on `bench/gn-vq`: see
"E0 notebook" below and `kaggle/PREREG_GN.md`.

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
| `bench/gn-vq` | off `bench/tilequant` (`12f912eb`): the E0 / E1 pre-registration (Amendments 1-5), `bench/gn/` (GN metric, diagnostics, G0 and G1 rules, GN-VQ, smoke tests, scene fixtures), the E0 and E1 jobs and notebooks, and E0's results; never goes upstream | **E0 done: G0 passed** (2026-09-20, `kaggle/gn_e0/gn/`, FINDINGS section 8). **E1 is built, not run.** `gsplat/` and `setup.py` are identical to the run-5 commit, so the run-5 wheel's key matches. Do not modify `feat/png-weighted-kmeans` (PR #1063) from here. |

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
| `PREREG_GN.md` | E0 / E1 pre-registration (`bench/gn-vq`): G0 rule, validity checks, exploratory scope, G1 for E1. Amendment 1: the toy check. Amendment 2: G0 over 9 codebooks per scene with tie-exempt pairs, and exact-assignment refines instead of the shortlist one. Amendment 3: the G0 verdict is the ranking alone (the ratio is reported as calibration), the end-to-end exactness check, the lifted-check criterion v2, and a proximal rise invalidating that variant instead of stopping. Amendment 4: the toy and end-to-end scenes are committed fixtures with pinned hashes (a correction: the CUDA toy check would have drawn a different scene from the simulated one), end-to-end preconditions read from gsplat's render, and a report-only probe-noise diagnostic. Amendment 5 (after G0 passed, before any E1 code): E1's GN-VQ variant, the one-sided size matching that G1's last sentence delegates, the reported secondaries and the exploratory ablations. **Never edit a rule after results exist**; add a dated amendment instead. |
| `gn_e0/gn/` | the E0 results bundle, unpacked as downloaded (22 files); FINDINGS section 8 quotes it |
| `gn_e1_scene.py` | E1, one scene per process: the G1 baseline and GN-VQ at K = 65,536 for seeds 0-2, the two secondary weightings, the seed-0 K grid, the two ablations and an `uncompressed` row. Reuses E0's GN cache; resumable per (config, K, seed) |
| `build_gn_e1_bench.py` / `gn_e1_bench.ipynb` | E1 notebook (build output; edit the builder, never the JSON). E0's notebook is left exactly as it ran |
| `gn_e0_scene.py` | E0, one scene per process: render parity, GN pass (`gn_cache/<scene>.pt`), spectrum, Spearman, the 9 G0 codebooks (predicted vs measured, test and train GT metrics, reproduction fields at K = 65,536), the lifted-assignment check (gates only the refines), the ridge / proximal refines (a proximal rise marks that row invalid). Resumable per (scene, config, K, seed). |
| `build_gn_bench.py` / `gn_bench.ipynb` | E0 notebook (build output; edit the builder, never the JSON) |
| `../bench/gn/` | `sh_basis.py`, `gn_metric.py`, `batched.py` (chunked batched linalg and the finite check; the fix for the first Kaggle crash), `diagnostics.py` (spectrum, Spearman, predicted / measured, lifted exact assignment and its check, refines, end-to-end exactness check), `g0.py` / `g1.py` (the G0 and G1 rules as code), `gn_vq.py` (E1's variant and the codec's quantizer), `selftest.py` (the notebook's smoke tests; `--device cpu` is the CPU stand-in), `fixtures/` (the committed toy and end-to-end scenes, `.npz` + `.json`, Amendment 4) and `make_fixtures.py` (wrote them), `toy_render.py` (CPU renderer for tests), `toy_noise.py` / `.json` (Amendment 1), `test_gn.py` (51 CPU tests), `dryrun/` (`fake_env.py` with the shared CPU stand-in, the E0 and E1 dry runs and the writer-parity check; not collected by pytest) |
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

E1 judges **G1**: does GN-VQ beat `lloyd_wopa_area` at equal size? The variant, the size matching, the
secondaries and the ablations were fixed in `PREREG_GN.md` **Amendment 5**, after G0 passed and before
any E1 code. E0's notebook, builder and job are untouched.

**Kaggle steps:**

1. Import the notebook from
   `https://raw.githubusercontent.com/Daceyyreal/gsplat/bench/gn-vq/kaggle/gn_e1_bench.ipynb`.
2. Attach **both**: the **E0 notebook output** (for `gn_cache/<scene>.pt`, so E1 skips the GN pass -
   the cache version is unchanged) and the **run-5 notebook output** (checkpoints, seed-0 sort caches,
   the run-3 `lloyd_wopa_area` caches for seeds 0-2, the gsplat wheel). To resume, also attach this
   notebook's own earlier output (`gn1/`, `gn1_work/`).
3. GPU T4 x2, Internet on. Save & Run All.
4. Bring back `/kaggle/working/gn1_bundle.zip` (look in `~/Downloads`).

**What it does:** the same smoke tests as E0 (`bench/gn/selftest.py`: fixture hashes, SH, toy,
end-to-end, and `linalg_scale`), then one job per scene in parallel (`kaggle/gn_e1_scene.py`), then the
G1 cell.

**Rows per scene** (`gn1_results_<scene>.csv`, 19):

- `lloyd_wopa_area` at K = 65,536, seeds 0-2: G1's baseline, from the run-3 caches (a missing seed is
  reclustered and the row records `recomputed`);
- `gn_vq` at K = 65,536, seeds 0-2: the variant G1 judges;
- `lloyd_trace`, `lloyd_c3dgs` at K = 65,536, seeds 0-2: the secondary weightings, reported only;
- `lloyd_wopa_area` and `gn_vq` at K = 4,096 and 16,384, seed 0: the rate-distortion grid;
- `gn_vq_noclip`, `gn_vq_noqassign`: the exploratory ablations, seed 0;
- `uncompressed`.

Every row logs predicted vs measured error on train and test, the quantizer's range and step, the
fraction of centroid coordinates outside the warm-start range, the objective before and after
quantization, and `writer_codes_equal` (the job raises if the writer's codes differ from the ones
GN-VQ quantized). Train-view SSIM and LPIPS are **not** computed; train PSNR is.

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

**Local checks:** `pytest bench/gn/test_gn.py` (51 tests) and
`python bench/gn/dryrun/dryrun_gn_e1.py` (the whole job on the toy: all 19 rows, a reclustered seed,
the writer codes, the GN-VQ reports, resume, and the notebook's G1 and bundle cells). Rebuild the
notebook with `python kaggle/build_gn_e1_bench.py` before the dry run.

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

### Open items (E0: closed; E1: open)

- ~~**Run E0 on Kaggle.**~~ **Done (2026-09-20): G0 passed.** The bundle is committed unchanged in
  `kaggle/gn_e0/gn/` and FINDINGS section 8 quotes it. Nothing in E0 is left to run. The session took
  about 41 minutes of timed steps with the two jobs in parallel.
- **Run E1 on Kaggle** (Dace). `kaggle/gn_e1_bench.ipynb`, attaching **both** the E0 output (for
  `gn_cache/`) and the run-5 output; see "E1 notebook" above. Everything resumes per row, so a short
  session can be continued by attaching its own output. The two secondary clusterings per seed are the
  bulk of the cost.
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
- **If the 10k lifted check fails in E1,** the GN-VQ rows are skipped and the Lloyd rows (including
  G1's baseline) still run; G1 is then `incomplete`. In E0 the check passed with 5 to 8 orders of
  margin on both scenes, so this is unlikely.
- **After the E1 run:** commit the bundle under `kaggle/gn_e1/`, read `gn1_g1.json` first (the verdict
  is `pass` / `fail` / `incomplete`), then write FINDINGS section 9 from the files. Check
  `writer_codes_equal` and the `valid` column before quoting any row.
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
| E1 (`bench/gn-vq`) | does GN-VQ beat `lloyd_wopa_area` at equal size (G1, `PREREG_GN.md` with Amendment 5)? | **built, not run**: `kaggle/gn_e1_bench.ipynb`, 51 CPU tests and a dry run pass. |

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
- **E0 / E1:** see "Open items (E0: closed; E1: open)". E0 is done and G0 passed
  (`kaggle/gn_e0/gn/`, FINDINGS section 8); E1 is built and waiting for a Kaggle run
  (`kaggle/gn_e1_bench.ipynb`, "E1 notebook" above).
