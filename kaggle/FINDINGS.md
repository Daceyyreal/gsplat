# Findings: tile-wise PNG quantization and the shN codebook in gsplat `PngCompression`

**Result:** no tile-wise or smooth-range config beats the current global per-channel min/max
scheme, and no shN codebook quantization change beats the library default. Most of the remaining
compression loss is in the shN k-means clustering, not in quantization.

## Sources

- Sections 1 and 2: every number comes from `kaggle/run2/tilequant/` (one Kaggle session on 2x T4, gsplat
  commit `b0998765`, `bench/tilequant`). That session restored nothing from earlier runs: it rebuilt
  the gsplat wheel, trained both scenes from scratch ("training #2") and ran all run-1 and run-2
  configs on those checkpoints, so the numbers are self-consistent.
- Section 3: training #1 (earlier Kaggle session); numbers from its `decision.json` and CSV,
  as reported by Dace. Its files are not in this repository.

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
- `drop3_dim_b6` loses 1.06 / 0.87 dB. How much of that is SH band 3 itself is pending the
  `sh2_render` check (render the uncompressed checkpoint with band 3 zeroed).

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

## 4. Open items

- **Run 3, clustering levers:** clustering is the largest term (S - F). Test euclidean k-means (PSNR is
  L2; torchpq's manhattan assignment is paired with a mean update), weighted Lloyd with opacity and
  footprint weights, and convergence, all in the unchanged library shN format.
- **`sh2_render`:** PSNR / SSIM / LPIPS of the uncompressed checkpoint with SH band 3 zeroed, next to U
  and `drop3_dim_b6`.
