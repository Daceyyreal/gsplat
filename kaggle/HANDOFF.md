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
| `bench/gn-vq` | off `bench/tilequant` (`12f912eb`): E0 pre-registration, `bench/gn/` (GN metric, diagnostics, G0 rule), E0 job and notebook; never goes upstream | **E0 ready, not run.** `gsplat/` and `setup.py` are identical to the run-5 commit, so the run-5 wheel's key matches. Do not modify `feat/png-weighted-kmeans` (PR #1063) from here. |

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
| `PREREG_GN.md` | E0 pre-registration (`bench/gn-vq`): G0 rule, validity checks, exploratory scope, G1 for E1. Amendment 1 (toy check) is the only change. **Never edit a rule after results exist**; add a dated amendment instead. |
| `gn_e0_scene.py` | E0, one scene per process: render parity, GN pass (`gn_cache/<scene>.pt`), spectrum, Spearman, the three run-3 configs (predicted vs measured, GT metrics, reproduction fields), GN refine. Resumable per (scene, config, seed). |
| `build_gn_bench.py` / `gn_bench.ipynb` | E0 notebook (build output; edit the builder, never the JSON) |
| `../bench/gn/` | `sh_basis.py`, `gn_metric.py`, `diagnostics.py`, `g0.py` (the G0 rule as code), `toy_render.py` (CPU renderer for tests), `toy_noise.py` / `.json` (Amendment 1), `test_gn.py` (16 CPU tests) |

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
- **CUDA smoke tests** (validity checks) go to `gn/gn_selftest.json`: `sh_basis.cuda_check` (max abs
  error < 1e-5 against gsplat's `spherical_harmonics`) and `gn_metric.toy_exactness(device="cuda")`
  (Amendment 1). The run stops if either fails.
- **Jobs:** garden on GPU 0 and bicycle on GPU 1 in parallel (`run_on_gpus`), each running
  `kaggle/gn_e0_scene.py`. Each job:
  - checks **render parity** first: the direct `rasterization` call against `Runner.rasterize_splats`
    with the eval's keyword arguments; it raises if the max abs difference is above 1e-6;
  - runs the GN pass and caches it;
  - writes the spectrum and Spearman files;
  - runs `upstream_l1`, `plain_l2`, `lloyd_wopa_area`, then `gn_refine`, then an `uncompressed`
    reference row.
- **G0 cell:** `bench/gn/g0.py` on the rows, with the validity dict (selftest, parity per scene,
  reproduction of the run-3 `lloyd_wopa_area` seed-0 row when that row exists), written to
  `gn/gn_g0.json` and plotted in `gn/gn_g0.png`.

**Outputs** (`gn_bundle.zip`, arcname `gn/`):

- `gn_results_<scene>.csv`: one row per (config, seed), with P, D train / test (clamped and raw),
  the ratios, full-pipeline and shN-only GT metrics, bytes per file (`file_bytes` JSON), the
  `shN.npz` members (`shN_centroids_bytes`, `shN_labels_bytes`), the clustering source, timings and the
  run-3 reproduction fields;
- `gn_meta_<scene>.json`: settings, render parity, GN pass (views, pixels, clamp fraction, visible
  splats, zero-trace splats), timings;
- `gn_spectrum_<scene>.csv`, `gn_spectrum_hist_<scene>.csv`, `gn_spectrum_<scene>.png`;
- `gn_spearman_<scene>.csv`;
- `gn_refine_<scene>.json`: per iteration, the objective, recall, changed labels and times;
- `gn_selftest.json`, `gn_g0.json`, `gn_g0.png`, `timings.json`.

Not bundled but kept in the output for resuming: `gn_cache/<scene>.pt` (M is 1,000,000 x 120 fp32 =
480 MB per scene), `gn_work/` (runner stats, E0 clustering cache, job logs).

**Local checks** (no GPU):

- `.venv\Scripts\python.exe -m pytest bench/gn/test_gn.py`: 16 CPU tests.
- In `C:\Users\Dace\AppData\Local\Temp\claude\F--\51b5c32d-122a-4650-8d8a-533b96ba5785\scratchpad\gn\`:
  - `dryrun_gn_e0.py`: the whole job on a 4,096-splat toy with a fake runner and the CPU renderer,
    checking resume, a parity failure, and the notebook's G0 and bundle cells;
  - `check_writer_parity.py`: E0's writer produces byte-identical files to the run-3 writer for the
    same codebook.

  Logs are next to them; both pass.

**After the run:**

1. Unpack `gn_bundle.zip` into `kaggle/gn_e0/` and check the files against the zip, CR-insensitively.
2. Read `gn_g0.json` first. Its verdict is `pass` / `fail` / `invalid` / `incomplete`.
3. Then fill FINDINGS section 8 from the files.

G1 is not judged in E0. Before E1, write down the exact GN-VQ variant and its size matching, as
PREREG_GN.md requires.

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
    to run 3's writer (`check_writer_parity.py`).
  - Measured errors use `Runner.rasterize_splats(splats=...)` with the eval's keyword arguments;
    `Stage.render` only forwards to it.
  - `gm.gsplat_render` is looked up at call time, so a dry run can swap in `toy_render.render_bruteforce`.
  - The run-3 clustering cache key is `sha1(ckpt_sha1 + seed-0 order bytes)`, the same as E0's.
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
| E0 (`bench/gn-vq`) | does a Gauss-Newton metric on shN predict the shN-only render error (G0, `PREREG_GN.md`)? | **pending**: code, tests and dry runs done; not run on Kaggle. |

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
- **E0:** Dace runs `kaggle/gn_bench.ipynb` on Kaggle (see "E0 notebook"). When the bundle is back:
  commit it under `kaggle/gn_e0/`, read `gn_g0.json`, and fill FINDINGS section 8.
- **E1** (GN-VQ, G1): write down the GN-VQ variant and its size matching before any E1 run.
