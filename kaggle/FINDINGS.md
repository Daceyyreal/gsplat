# Findings: tile-wise PNG quantization and the shN codebook in gsplat `PngCompression`

**Result:** no tile-wise or smooth-range config beats the current global per-channel min/max
scheme, and no shN codebook quantization change beats the library default. Most of the remaining
compression loss is in the shN k-means clustering, not in quantization.

**Run 3 (section 4):** no clustering config passes the pre-set rule either (`pr_worthy` = false).
Weighted Lloyd k-means with opacity x footprint-area weights (`lloyd_wopa_area`) raises PSNR on both
scenes and all 3 k-means seeds and fails the rule only on two garden SSIM cells, by less than the
baseline's own SSIM spread over k-means seeds. Run 4 tests it on all 9 MipNeRF360 scenes.

## Sources

- Sections 1 and 2: every number comes from `kaggle/run2/tilequant/` (one Kaggle session on 2x T4, gsplat
  commit `b0998765`, `bench/tilequant`). That session restored nothing from earlier runs: it rebuilt
  the gsplat wheel, trained both scenes from scratch ("training #2") and ran all run-1 and run-2
  configs on those checkpoints, so the numbers are self-consistent.
- Section 3: training #1 (earlier Kaggle session); numbers from its `decision.json` and CSV,
  as reported by Dace. Its files are not in this repository.
- Section 4: every number comes from `kaggle/run3/tilequant/` (one Kaggle session, gsplat commit
  `f9b61526`). That session restored the run-2 output (training #2 checkpoints, seed-0 PLAS sort,
  run-1 / run-2 rows, gsplat wheel) and ran only the run-3 configs, so run-3 rows pair with the
  run-2 rows of the same checkpoints. 14 of its 22 files are byte-identical to `run2/tilequant/`;
  `timings.json` keeps the run-2 entries, with install / smoke / data overwritten by this session and
  the run-3 jobs added.

Setup: MipNeRF360 garden and bicycle, `examples/benchmarks/compression/mcmc.sh` settings (MCMC, 1M
Gaussians, data factor 4, LPIPS VGG). Size = `zip -r` of the compression directory, as in
`summarize_stats.py` (`zip_bytes`); raw file bytes (`size_bytes`) move the same way here. Unless noted:
PLAS sort seed 0, k-means seed 0. U = uncompressed PSNR of the same checkpoint.

### Sanity gate (`run2/tilequant/sanity_gate.json`)

Current-main CLI run (`mcmc.sh` eval command), no failures, no warnings:

| Scene | U (dB) | Compressed (dB) | Drop (dB) | zip bytes | zip vs repo 1M row | #Gaussians |
|---|---|---|---|---|---|---|
| garden | 27.315 | 26.862 | 0.453 | 16,449,239 | +2.6% | 1,000,000 |
| bicycle | 25.568 | 25.317 | 0.251 | 16,233,444 | +1.2% | 1,000,000 |

The repo row (`results/MipNeRF360.csv`, 1M) is a mean over 9 scenes, so only the drop and #Gaussians
are hard checks.

## 1. Tile-wise and smooth ranges (training #2)

Decision (`run2/tilequant/decision.json`): **`pr_worthy` = false, `tile_effect` = false, `global_only` = false.**

### At baseline bits (means 16, others 8)

| Config | garden dPSNR | garden zip | bicycle dPSNR | bicycle zip |
|---|---|---|---|---|
| `baseline` (global min/max) | 26.871 dB | 16,450,113 B | 25.322 dB | 16,235,216 B |
| `t8_m16_o8` | +0.027 | +27.1% | +0.054 | +29.5% |
| `t16_m16_o8` | +0.023 | +20.8% | +0.052 | +22.7% |
| `t32_m16_o8` | +0.025 | +16.7% | +0.050 | +18.1% |
| `t64_m16_o8` | +0.021 | +13.4% | +0.046 | +14.2% |
| `t128_m16_o8` | +0.016 | +9.9% | +0.045 | +10.2% |
| `smooth_t16_m16_o8` | +0.019 | +15.5% | +0.047 | +16.6% |
| `smooth_t32_m16_o8` | +0.020 | +11.6% | +0.044 | +12.0% |

Tiles reduce quantization error a little, and cost far more bytes than they save in PSNR.

### Boundary cost: plain tiles vs smooth ranges

PNG bytes of the plain tile config minus the smooth-range config at the same tile size and bits
(means 16 bits):

| Tile | Other bits | garden | bicycle |
|---|---|---|---|
| 16 | 8 | 0.87 MB | 0.99 MB |
| 16 | 7 | 0.93 MB | 1.05 MB |
| 16 | 6 | 0.92 MB | 1.04 MB |
| 32 | 8 | 0.85 MB | 0.98 MB |
| 32 | 7 | 0.87 MB | 1.00 MB |
| 32 | 6 | 0.85 MB | 0.97 MB |

Removing the range jumps at tile boundaries saves 0.85-1.05 MB of PNG bytes, but smooth ranges are
still +11.6% / +12.0% zip at tile 32 and baseline bits.

### Against the global reduced-bit front

The global front (baseline plus global configs with fewer bits) is `global_m8_o6`, `global_m10_o6`,
`global_m12_o6`, `global_m12_o7`, `global_m16_o6`, `global_m16_o7`, `baseline` on both scenes. All
22 tile and smooth configs whose zip size falls inside the front's size range sit below it at matched
size (linear interpolation):

| Scene | Best tile/smooth config | Margin to the global front |
|---|---|---|
| garden | `smooth_t16_m16_o6` | -0.068 dB |
| bicycle | `t64_m16_o6` | -0.087 dB |

For reference, `global_m16_o7` (other params at 7 bits): -0.054 / -0.048 dB at -7.8% / -7.8% zip.

### The collapse regime

With 8-bit means the global scheme collapses and tiles recover most of it:

| Other bits | garden global | garden tile 8 | bicycle global | bicycle tile 8 |
|---|---|---|---|---|
| 8 | 14.20 dB | 21.70 dB | 14.53 dB | 20.50 dB |
| 7 | 14.20 dB | 21.70 dB | 14.54 dB | 20.50 dB |
| 6 | 14.26 dB | 21.70 dB | 14.57 dB | 20.49 dB |

gsplat stores the log-transformed means with 16 bits, so its default never enters this regime.

### What is left for the PNG params

| Scene | U - baseline | U - best PNG-param config | Best config |
|---|---|---|---|
| garden | 0.444 dB | 0.417 dB | `t8_m16_o8` (+27.1% zip) |
| bicycle | 0.246 dB | 0.192 dB | `t8_m16_o8` (+29.5% zip) |

Even the best PNG-param config leaves most of the loss in place, which points at shN (section 2).

### PLAS sort-seed robustness

Paired deltas against the baseline of the same sort seed, for the 3 configs picked from seed 0:

| Config | Scene | dPSNR (seeds 0 / 1 / 2) | zip (seeds 0 / 1 / 2) |
|---|---|---|---|
| `global_m16_o7` | garden | -0.054 / -0.054 / -0.054 | -7.84% / -7.84% / -7.84% |
| `global_m16_o7` | bicycle | -0.048 / -0.048 / -0.048 | -7.81% / -7.81% / -7.81% |
| `smooth_t16_m16_o6` | garden | -0.075 / -0.086 / -0.081 | -1.06% / -1.02% / -1.03% |
| `smooth_t16_m16_o6` | bicycle | -0.093 / -0.090 / -0.096 | -0.26% / -0.26% / -0.26% |
| `t64_m16_o6` | garden | -0.091 / -0.093 / -0.090 | -3.21% / -3.24% / -3.20% |
| `t64_m16_o6` | bicycle | -0.103 / -0.100 / -0.095 | -2.61% / -2.66% / -2.63% |

Global configs have the same PSNR for every sort order: one min/max per channel, and the extra sort
seeds reuse the seed-0 k-means centroids (baseline PSNR 26.871 / 25.322 dB on all three seeds). None of
the three configs beats the baseline on any seed.

![Size vs PSNR, run 1 configs](run2/tilequant/rd_size_vs_psnr.png)

### Why

At 16-bit means the PNG params are not where the loss is: U - baseline is 0.444 / 0.246 dB and the
best PNG-param config closes only 0.027 / 0.054 dB of it. Tile ranges make the PNG files larger
(tile 8 at baseline bits: +3.73 / +4.05 MB of PNG bytes) and add per-tile bounds on top
(0.73 / 0.74 MB). Smooth ranges remove 0.85-1.05 MB of that PNG growth, not enough to get near the
baseline size. At matched size the global scheme with fewer bits gives more PSNR (front comparison).

## 2. shN codebook (training #2)

`_compress_kmeans` clusters shN into 65,536 centroids (torchpq, manhattan) and quantizes the
65,536 x 45 codebook to 6 bits with one scalar min/max. Decision (`run2/tilequant/shn_decision.json`):
**`pr_worthy` = false.**

### Decomposition (approximate; PSNR losses are not additive)

| Term | garden | bicycle |
|---|---|---|
| U - baseline (total) | 0.434 dB | 0.237 dB |
| U - P (shN loss; PNG params raw float32) | 0.403 dB | 0.176 dB |
| U - S (PNG-param loss; shN raw float32) | 0.034 dB | 0.063 dB |
| S - F (clustering; float centroids vs raw shN) | 0.374 dB | 0.163 dB |
| F - baseline (6-bit centroid quantization) | 0.026 dB | 0.011 dB |

This run-2 baseline re-runs k-means with a fixed seed and caches the float centroids, so its PSNR
(26.881 / 25.331 dB) differs slightly from the section-1 baseline, which used the k-means result of
the sort cache (26.871 / 25.322 dB).

### Configs at k-means seed 0 (deltas vs the run-2 baseline)

| Config | garden dPSNR | garden zip | bicycle dPSNR | bicycle zip |
|---|---|---|---|---|
| `baseline` | 26.881 dB | 16,449,880 B | 25.331 dB | 16,242,031 B |
| `F_float_centroids` | +0.026 | +57.8% | +0.011 | +57.6% |
| `dim_b5` (45 min/max, 5 bits) | -0.010 | -0.6% | -0.022 | -1.7% |
| `dim_b6` | +0.018 | +1.7% | +0.005 | +0.7% |
| `dim_b7` | +0.024 | +3.4% | +0.010 | +2.5% |
| `dim_b8` | +0.026 | +5.2% | +0.011 | +4.2% |
| `band_b5` (9 min/max, 5 bits) | -0.033 | -1.4% | -0.022 | -2.0% |
| `band_b6` | +0.013 | +0.9% | +0.003 | +0.3% |
| `k32768_dim_b6` | -0.054 | -3.7% | -0.041 | -4.8% |
| `drop3_dim_b6` (SH band 3 dropped) | -1.057 | -3.3% | -0.873 | -4.3% |

- `dim_b8` recovers the centroid-quantization share (+0.026 / +0.011 dB vs F - baseline
  0.026 / 0.011 dB) at +5.2% / +4.2% zip. This matches the decomposition, which supports the
  implementation.
- `k32768_dim_b6` sits -0.019 / -0.003 dB from the section-1 global front at its zip size, within the
  baseline's k-means seed spread below.
- `drop3_dim_b6` loses 1.06 / 0.87 dB. Section 4 (`sh2_render`) shows that SH band 3 itself carries
  1.37 / 0.99 dB of the uncompressed checkpoint.

### k-means seed robustness

Baseline over k-means seeds 0 / 1 / 2:

| Scene | PSNR (dB) | zip bytes | PSNR spread | zip spread (max - min) / mean |
|---|---|---|---|---|
| garden | 26.881 / 26.862 / 26.864 | 16,449,880 / 16,412,541 / 16,440,902 | 0.019 dB | 0.23% |
| bicycle | 25.331 / 25.331 / 25.325 | 16,242,031 / 16,217,116 / 16,268,202 | 0.007 dB | 0.31% |

The two configs picked from seed 0 (`dim_b5`, `band_b5`; smaller than the baseline but lower PSNR)
stay below the baseline on seeds 1 and 2:

| Config | Scene | dPSNR (seeds 1 / 2) | zip (seeds 1 / 2) |
|---|---|---|---|
| `dim_b5` | garden | -0.009 / -0.010 | -0.50% / -0.54% |
| `dim_b5` | bicycle | -0.016 / -0.012 | -1.40% / -1.72% |
| `band_b5` | garden | -0.046 / -0.033 | -1.30% / -1.31% |
| `band_b5` | bicycle | -0.030 / -0.016 | -1.87% / -2.02% |

![Size vs PSNR, shN codebook configs](run2/tilequant/rd_shn.png)

### Why

The codebook quantization costs about 0.026 / 0.011 dB, so better ranges (per dimension or per band)
can gain at most about that; `dim_b8` reaches it at +5.2% / +4.2% zip. The large term is the clustering itself
(0.374 / 0.163 dB): 1M Gaussians share 65,536 centroids.

## 3. Replication: training #1

**Training #1 (earlier Kaggle session; numbers from its `decision.json` and CSV, reported by Dace).**
Its files are not in this repository and were not re-checked here. Training #2 values are from
`kaggle/run2/tilequant/`.

| Metric | Training #1 (garden / bicycle) | Training #2 (garden / bicycle) |
|---|---|---|
| `pr_worthy`, `tile_effect`, `global_only` | false, false, false | false, false, false |
| Best margin to the global front | -0.063 / -0.072 dB | -0.068 / -0.087 dB |
| U - baseline | 0.447 / 0.249 dB | 0.444 / 0.246 dB |
| `t8_m16_o8` | +0.024 / +0.055 dB at +25.9% / +30.3% zip | +0.027 / +0.054 dB at +27.1% / +29.5% zip |
| Global 8-bit means -> tile 8 | 14.62 / 14.53 -> 21.69 / 20.52 dB | 14.20 / 14.53 -> 21.70 / 20.50 dB (other params 8 bits) |
| `global_m16_o7` | -0.062 / -0.053 dB at -7.8% zip | -0.054 / -0.048 dB at -7.8% / -7.8% zip |

The conclusions of section 1 hold for both trainings.

## 4. Run 3: shN k-means clustering levers (training #2)

Every config keeps the library shN format (65,536 centroids, 6-bit scalar codebook quantization in the
unchanged `_compress_kmeans`, uint16 labels, unchanged `_decompress_kmeans`, seed-0 PLAS sort); only
the clustering differs. Reference for every delta: the run-2 `baseline` row (`shn_results.csv`) of the
same scene and k-means seed.

| Config | Clustering |
|---|---|
| `manhattan_log` | torchpq `KMeans(distance="manhattan")` as in the library, with its per-iteration log (check, not a candidate) |
| `euclid` | torchpq `distance="euclidean"` |
| `lloyd_w1` | benchmark Lloyd (euclidean assignment, weighted mean update), weights 1 (control for `euclid`) |
| `lloyd_wopa` | same, weights `sigmoid(opacity)` |
| `lloyd_wopa_area` | same, weights `sigmoid(opacity) * exp(sum of the two largest log-scales)` |
| `iters_x` | torchpq manhattan, `max_iter` 300 (ran because `manhattan_log` stopped at `max_iter` 100) |
| `sh2_render` | no compression: uncompressed checkpoint with SH band 3 set to 0 |

The Lloyd runs use torchpq's init (k data points from `np.random.choice` with the same seed), iteration
budget (100), stopping rule (summed squared centroid change <= 1e-4) and empty-cluster rule.

### Decision (`run3/tilequant/run3_decision.json`), as recorded

**`pr_worthy` = false, `pr_worthy_configs` = [].** Rule: PSNR >=, SSIM >=, LPIPS <=, zip bytes and
raw bytes <= 1.003 x the baseline, on both scenes and k-means seeds 0, 1 and 2. Seeds 1 and 2 ran for
the two configs ranked best at seed 0 (`run3_seed_selection.json`: `lloyd_wopa`, `lloyd_wopa_area`),
so the other candidates are incomplete by design.

| Config | complete | beats_all | Cells that fail the rule |
|---|---|---|---|
| `lloyd_wopa` | true | false | garden seed 2 (LPIPS +0.00005) |
| `lloyd_wopa_area` | true | false | garden seed 0 (SSIM -0.00016), garden seed 2 (SSIM -0.00009) |
| `iters_x` | false | false | garden seed 0 (PSNR -0.0003, SSIM -0.000004) |
| `euclid` | false | false | garden and bicycle seed 0 (PSNR, SSIM, LPIPS) |
| `lloyd_w1` | false | false | garden and bicycle seed 0 (PSNR, SSIM, LPIPS) |

### Check: the torchpq manhattan re-run reproduces run 2

`manhattan_log` (k-means seed 0) against the run-2 seed-0 baseline: PSNR, SSIM and LPIPS are identical,
and so are raw bytes, PNG bytes and `shN.npz` bytes, on both scenes. Only the zip size differs, by
+109 / +108 B; `zip -r` also stores the run directory path, which differs between the two scripts
(`/tmp/tilequant_shn_runs/.../baseline` vs `/tmp/tilequant_run3_runs/.../manhattan_log`). torchpq
k-means is deterministic per seed across sessions, so pairing run-3 rows with run-2 baselines is valid.
This was checked at seed 0; seeds 1 and 2 rely on the same determinism.

torchpq manhattan did not meet its own stopping rule: after 100 iterations the centroid change was
0.0107 / 0.0238 (tol 1e-4) on garden / bicycle.

### k-means seed 0: all configs (garden / bicycle)

| Config | dPSNR (dB) | dSSIM | dLPIPS | zip | Iterations | k-means time |
|---|---|---|---|---|---|---|
| `manhattan_log` | 0 / 0 | 0 / 0 | 0 / 0 | +109 / +108 B | 100 / 100 | 419 / 432 s |
| `euclid` | -0.038 / -0.022 | -0.00045 / -0.00027 | +0.00025 / +0.00023 | +0.18% / +0.05% | 100 / 42 | 422 / 233 s |
| `lloyd_w1` | -0.036 / -0.023 | -0.00043 / -0.00028 | +0.00024 / +0.00021 | +0.18% / +0.05% | 100 / 39 | 628 / 281 s |
| `lloyd_wopa` | +0.032 / +0.016 | +0.00025 / +0.00054 | -0.00033 / -0.00035 | -0.19% / -0.08% | 100 / 44 | 633 / 317 s |
| `lloyd_wopa_area` | +0.082 / +0.027 | -0.00016 / +0.00005 | -0.00083 / -0.00094 | -0.25% / -0.14% | 100 / 56 | 642 / 405 s |
| `iters_x` | -0.0003 / +0.0022 | -0.000004 / +0.00002 | -0.000007 / -0.000009 | -0.0004% / +0.006% | 203 / 269 | 868 / 1155 s |

- `lloyd_w1` matches `euclid` (PSNR within 0.0017 / 0.0010 dB, zip within 163 / 4 B), which supports
  the Lloyd implementation.
- Unweighted euclidean clustering (torchpq or Lloyd) is worse than manhattan: -0.036 to -0.038 dB on
  garden, -0.022 to -0.023 dB on bicycle (seed 0 only).
- `iters_x` converged (203 / 269 iterations) and changed PSNR by -0.0003 / +0.0022 dB at 2.0x / 2.8x
  the baseline compress time, so convergence is not the lever.

### k-means seeds 0 / 1 / 2: `lloyd_wopa` and `lloyd_wopa_area`

| Config | Scene | dPSNR (dB) | dSSIM | dLPIPS | zip | raw bytes | Passes |
|---|---|---|---|---|---|---|---|
| `lloyd_wopa` | garden | +0.032 / +0.051 / +0.039 | +0.00025 / +0.00030 / +0.00027 | -0.00033 / -0.00036 / +0.00005 | -0.19% / -0.08% / -0.25% | -0.20% / -0.08% / -0.25% | yes / yes / no |
| `lloyd_wopa` | bicycle | +0.016 / +0.017 / +0.021 | +0.00054 / +0.00054 / +0.00057 | -0.00035 / -0.00007 / -0.00062 | -0.08% / +0.22% / -0.23% | -0.08% / +0.22% / -0.23% | yes / yes / yes |
| `lloyd_wopa_area` | garden | +0.082 / +0.104 / +0.103 | -0.00016 / +0.00002 / -0.00009 | -0.00083 / -0.00072 / -0.00045 | -0.25% / -0.02% / -0.19% | -0.25% / -0.02% / -0.19% | no / yes / no |
| `lloyd_wopa_area` | bicycle | +0.027 / +0.031 / +0.033 | +0.00005 / +0.00011 / +0.00015 | -0.00094 / -0.00079 / -0.00100 | -0.14% / +0.25% / -0.14% | -0.14% / +0.25% / -0.14% | yes / yes / yes |

Mean dPSNR over the 3 seeds: `lloyd_wopa` +0.041 / +0.018 dB, `lloyd_wopa_area` +0.096 / +0.030 dB.

k-means time over the 3 seeds: `lloyd_wopa` 585-637 s / 251-318 s, `lloyd_wopa_area` 635-642 s /
405-460 s. For comparison, torchpq manhattan took 419 / 432 s in this session and 433-440 s /
398-404 s for the run-2 baselines. On garden the Lloyd runs used all 100 iterations; on bicycle they
converged in 39-64. The Lloyd code is a benchmark implementation (float64 weighted update, chunked
`addmm` assignment), not an optimized one.

![Run 3: size vs PSNR and centroid change per iteration](run3/tilequant/rd_run3.png)

### Interpretation: seed noise (not part of the pre-registered rule)

Noise reference: the run-2 baseline over k-means seeds 0 / 1 / 2 (max - min). PSNR 0.019 / 0.0066 dB,
SSIM 0.000158 / 0.000051, LPIPS 0.00021 / 0.00035, zip 0.23% / 0.31% of the mean.

- **`lloyd_wopa_area`:**
  - PSNR is higher in all 6 cells. The mean gain (+0.096 / +0.030 dB) is 5.1x / 4.6x the baseline
    PSNR seed range.
  - LPIPS is lower in all 6 cells (mean -0.00067 / -0.00091, 3.1x / 2.6x the LPIPS seed range).
  - The two failing cells are garden SSIM -0.000157 (seed 0) and -0.000086 (seed 2). Both lie within
    the garden baseline SSIM range of 0.000158, seed 0 by 0.000001. Bicycle SSIM is higher on all
    three seeds.
  - Zip ranges from -0.246% to +0.251% of the paired baseline, inside the 0.3% tolerance (the baseline
    zip spread over seeds is 0.23% / 0.31%).
- **`lloyd_wopa`:** smaller gains (+0.041 / +0.018 dB mean), SSIM higher in all 6 cells. It fails only
  garden seed 2, on LPIPS by +0.00005, a quarter of the garden LPIPS seed range.
- **The weights carry the gain, not the distance:** unweighted euclidean loses 0.036-0.038 /
  0.022-0.023 dB against manhattan. Opacity weights turn that into a gain, and adding the footprint
  area gains more.
- **Share of the clustering loss:** the mean `lloyd_wopa_area` gain is 26% / 18% of the run-2
  clustering loss S - F (0.374 / 0.163 dB). This mixes a 3-seed mean with a seed-0 decomposition; with
  the seed-0 gains alone it is 22% / 16%.
- **Scope:** 2 scenes, 1 training. Run 4 applies a rule fixed in advance to all 9 MipNeRF360 scenes
  of `mcmc.sh`.

Corrections to the preliminary run-3 summary, checked against the CSVs:
- "zip within +-0.25%": bicycle seed 1 is +0.251%.
- "k-means 635-641 s on garden": the maximum is 641.5 s, so the range rounds to 635-642 s.
- The manhattan re-run zip difference is 109 / 108 B.

### `sh2_render`: SH band 3 matters

| Scene | U (PSNR / SSIM / LPIPS) | `sh2_render` (band 3 = 0, uncompressed) | U - `sh2_render` | run-2 baseline - `drop3_dim_b6` |
|---|---|---|---|---|
| garden | 27.315 / 0.8547 / 0.1338 | 25.947 / 0.8383 / 0.1478 | 1.368 dB | 1.057 dB |
| bicycle | 25.568 / 0.7730 / 0.2167 | 24.581 / 0.7582 / 0.2266 | 0.988 dB | 0.873 dB |

Zeroing band 3 of the uncompressed checkpoint costs 1.37 / 0.99 dB, more than `drop3_dim_b6` lost
against the compressed baseline (1.06 / 0.87 dB). The run-2 drop3 loss was therefore not a bug: band 3
carries that much in these MCMC 1M models (trained with the default `sh_degree` 3).

## 5. Open items

- **Run 4, MipNeRF360 validation:** baseline, `lloyd_wopa` and `lloyd_wopa_area` on all 9 scenes of
  `mcmc.sh`, with the decision rule fixed before the run (`kaggle/tilequant_run4_analysis.py`).
