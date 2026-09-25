# Findings: tile-wise PNG quantization and the shN codebook in gsplat `PngCompression`

**Runs 1-2 (sections 1-2):** no tile-wise or smooth-range config beats the current global per-channel
min/max scheme, and no shN codebook quantization change beats the library default. Most of the
remaining compression loss is in the shN k-means clustering, not in quantization.

**Run 3 (section 4) found the lever:** weighted Lloyd k-means with opacity x footprint-area weights
(`lloyd_wopa_area`) raises PSNR and lowers LPIPS on garden and bicycle at all 3 k-means seeds
(+0.096 / +0.030 dB mean PSNR). The strict pre-set rule (`pr_worthy`) failed only on two garden SSIM
cells, both inside the baseline's own SSIM spread over k-means seeds.

**Run 4 (section 5) validated it** on all 9 MipNeRF360 scenes of `mcmc.sh`, under a rule fixed before
the run: `lloyd_wopa_area` gains **+0.111 dB** mean PSNR at **-0.10%** mean size and is better on
every scene.

**Run 5 (section 6) measured the library code** (branch `feat/png-weighted-kmeans`). It reproduces
the benchmark rows exactly (parity gate passed, and identical rows on all 9 MipNeRF360 scenes) and
passes the same rule on Tanks & Temples: **+0.052 dB** mean PSNR on train / truck at +0.08% mean size.
Cost: k-means 510 s vs 405 s mean on MipNeRF360 and 328 s vs 416 s on Tanks & Temples; peak GPU
memory 3.37-3.62 GB vs 1.03-1.26 GB.

**E0 (section 8, branch `bench/gn-vq`): G0 passed.** A per-splat Gauss-Newton metric on shN ranks
codebooks by their rendering error: 34 non-tied pairs of 36, none misordered, on both view sets and at
all three codebook sizes, and calibrated within 0.5-2x everywhere. Its two exploratory refines cut the
GN objective by 3.7-4.0x and still lose 0.04-0.53 dB after the codec's centroid quantizer, which E1
(pre-registered in Amendment 5) tests with a range clip.

**E1 (section 9, branch `bench/gn-vq`): G1 failed, on the size rule.** GN-VQ beat `lloyd_wopa_area` at
K = 65,536 on every seed in test PSNR, by +0.195 to +0.201 dB on garden and +0.087 to +0.091 dB on
bicycle, but it was +2.37% and +0.98% to +1.01% larger, beyond the pre-registered 0.5%, so all six
seeds count as negative. The extra bytes are the codebook codes, which compress worse at the same
quantizer step. The pre-registered rate-distortion secondary favours GN-VQ on both scenes (bicycle
BD-rate -9.84%; on garden every GN-VQ point is above every `lloyd_wopa_area` point in PSNR), at seed 0
on the two scenes GN-VQ was designed on.

**E2 (section 10, branch `bench/gn-vq`): G2a passed.** On 9 held-out scenes, GN-VQ with ridge
eps = 1e-2 beat `lloyd_wopa_area` in rate-distortion on all 9, with a mean BD-rate of -5.37% against the
pre-registered -5%. H2b (against the scalar `tr(M)` weighting, reported) holds with 8 of 9 scenes
counted: treehill's computed win is an artifact of the cubic fit. Post hoc, PCHIP interpolation gives
8 of 9 and -5.66% for G2a, and the codebook stream alone (`shN.npz`) shows about -31%.

## Sources

- Sections 1 and 2: every number comes from `kaggle/run2/tilequant/` (one Kaggle session on 2x T4, gsplat
  commit `b0998765`, `bench/tilequant`). That session restored nothing from earlier runs: it rebuilt
  the gsplat wheel, trained both scenes from scratch ("training #2") and ran all run-1 and run-2
  configs on those checkpoints, so the numbers are self-consistent.
- Section 3: training #1 (earlier Kaggle session); numbers from its `decision.json` and CSV,
  as reported by Dace. Its files are not in this repository.
- Section 5: every number comes from `kaggle/run4/tilequant/` (one Kaggle session, gsplat commit
  `f7ce5262`). It restored the run-3 output, reused the garden / bicycle checkpoints and their rows
  (training #2, commits `b0998765` and `f9b61526`) and trained the other 7 scenes itself.
- Section 6: every number comes from `kaggle/run5/tilequant/` (one Kaggle session, gsplat commit
  `3bc8eb6c`). It restored the run-4 output, rebuilt the gsplat wheel, reused the 9 run-4 MipNeRF360
  checkpoints and trained the 2 Tanks & Temples scenes itself. Two bundle files were regenerated
  locally from the bundle's own result files after a label fix (commit `1b4d40f8`):
  `run5_tt_table.csv` (only its reference-row label changes) and `rd_run5.png` (memory in decimal GB,
  titles no longer overlap). Every other file is as downloaded.
- Section 8: every number comes from `kaggle/gn_e0/gn/` (one Kaggle session on 2x T4, 2026-09-20,
  gsplat commit `cd3139c2` on `bench/gn-vq`, the run-5 wheel reused). It restored the run-5 output:
  the garden / bicycle checkpoints, the seed-0 PLAS order, the run-3 clustering caches (used for
  K = 65,536) and `run3_results.csv`. The bundle is unpacked unchanged in `kaggle/gn_e0/gn/`.
- Section 9: every number comes from `kaggle/gn_e1/gn1/` (one Kaggle session on 2x T4, rows timestamped
  2026-09-20, gsplat commit `9b6c5bab` on `bench/gn-vq`, the run-5 wheel reused). It restored the run-5
  output (the garden / bicycle checkpoints, the seed-0 PLAS order, the run-3 `lloyd_wopa_area` caches
  for seeds 0-2) and E0's GN cache. The bundle is unpacked unchanged in `kaggle/gn_e1/gn1/`. A script
  recomputed every number in the section from those files and checked each against the text.
- Section 10: every number comes from `kaggle/gn_e2/gn2/` (one Kaggle session on 2x T4, rows
  timestamped 2026-09-21T23:03 to 2026-09-22T03:49, gsplat commit `515ca4a1` on `bench/gn-vq`, the run-5
  wheel reused) and, for the post-hoc items, from `kaggle/gn_e2/bd_sensitivity.json`, which
  `bench/gn/bd_sensitivity.py` computes from that bundle alone. The session restored the run-5 output
  (the 11 checkpoints, their seed-0 PLAS orders, the run-3 / run-4 `lloyd_wopa_area` K = 65,536 caches)
  and a GN cache for garden and bicycle from an earlier output. The bundle is unpacked unchanged in
  `kaggle/gn_e2/gn2/`. A script recomputed every number in the section from those two sources and
  checked each against the text.
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
This was checked at seed 0; seeds 1 and 2 rely on the same determinism. (Correction from run 5,
section 6: across 9 scenes, torchpq reproduces to about 0.002 dB, not always bit-exactly.)

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

## 5. Run 4: MipNeRF360 validation (pre-registered)

All 9 MipNeRF360 scenes of `examples/benchmarks/compression/mcmc.sh` (scene list, data factors and the
train command parsed from the script), k-means seed 0 and PLAS sort seed 0, library shN format
unchanged. garden and bicycle reuse the training-#2 checkpoints and their run-1 / run-2 / run-3 rows;
the other 7 scenes were trained in this session with the `mcmc.sh` command.

### The rule, fixed before the run

From `run4/tilequant/run4_decision.json`, as recorded: a candidate passes when, over all scenes,
mean dPSNR > 0 AND dPSNR > 0 on all scenes but at most one AND no scene dPSNR < -0.02 dB AND
mean dLPIPS <= 0 AND mean dSSIM >= -0.0002 AND on every scene zip AND raw bytes <= baseline x 1.003;
with every scene measured, every new scene's sanity gate passed and one checkpoint per scene.
`pr_candidate` = the passing candidate with the higher mean dPSNR. Deltas and means are rounded to
9 decimals before comparing, and the byte limits are integer comparisons. The reference is always the
library baseline on the same checkpoint (`_compress_kmeans`, torchpq manhattan, 65,536 clusters).

### Verdict: both candidates pass, `pr_candidate` = `lloyd_wopa_area`

| Candidate | mean dPSNR | mean dSSIM | mean dLPIPS | Scenes with dPSNR <= 0 | Worst scene dPSNR | Worst zip | Worst raw |
|---|---|---|---|---|---|---|---|
| `lloyd_wopa` | +0.0494 dB | +0.00091 | -0.00048 | 1 (treehill) | -0.0099 dB | +0.009% | +0.008% |
| `lloyd_wopa_area` | +0.1109 dB | +0.00072 | -0.00093 | 0 | +0.0049 dB | +0.267% | +0.266% |

All 7 new scenes passed the sanity gate with no warnings: 1,000,000 Gaussians each, U - baseline
between 0.101 dB (treehill) and 0.859 dB (bonsai), baseline zip 0.98x to 1.02x the repo 1M row.

### 9-scene means, upstream format (`run4/tilequant/run4_table.csv`)

| Submethod | PSNR | SSIM | LPIPS | Size [Bytes] | #Gaussians |
|---|---|---|---|---|---|
| `baseline` (library) | 27.4929 | 0.8192 | 0.2147 | 16,005,532 | 1,000,000 |
| `lloyd_wopa` | 27.5423 | 0.8201 | 0.2143 | 15,983,770 | 1,000,000 |
| `lloyd_wopa_area` | 27.6038 | 0.8199 | 0.2138 | 15,988,981 | 1,000,000 |
| repo `results/MipNeRF360.csv`, 1M row | 27.29 | 0.811 | 0.229 | 16,038,022 | 1,000,000 |
| uncompressed checkpoints (U) | 27.9258 | 0.8289 | 0.2033 | - | 1,000,000 |

**The repo row is context, not a control.** Our baseline is 0.20 dB above it at 0.2% smaller size,
because it comes from a different environment: different training run, torch and gsplat build, and
evaluation setup. Nothing here is compared against it. Every delta in this section is paired: candidate
minus the baseline measured in the same session, on the same checkpoint, with the same sort.

### Per-scene (deltas vs the paired baseline)

| Scene | Data factor | U - baseline | `lloyd_wopa` dPSNR | zip | `lloyd_wopa_area` dPSNR | zip |
|---|---|---|---|---|---|---|
| garden | 4 | 0.434 dB | +0.0324 | -0.19% | +0.0819 | -0.25% |
| bicycle | 4 | 0.237 dB | +0.0158 | -0.08% | +0.0266 | -0.14% |
| stump | 4 | 0.290 dB | +0.0252 | -0.13% | +0.0481 | +0.06% |
| bonsai | 2 | 0.859 dB | +0.1326 | -0.33% | +0.2721 | -0.30% |
| counter | 2 | 0.546 dB | +0.0838 | +0.01% | +0.1885 | +0.25% |
| kitchen | 2 | 0.824 dB | +0.1123 | -0.21% | +0.2474 | -0.47% |
| room | 2 | 0.386 dB | +0.0263 | -0.17% | +0.0846 | -0.21% |
| treehill | 4 | 0.101 dB | -0.0099 | -0.09% | +0.0049 | -0.15% |
| flowers | 4 | 0.219 dB | +0.0262 | -0.04% | +0.0442 | +0.27% |

SSIM and LPIPS per scene are in `run4/tilequant/run4_scene_deltas.csv`. `lloyd_wopa_area` has lower
(better) LPIPS on all 9 scenes and higher SSIM on 8 of 9 (garden -0.00016).

`lloyd_wopa_area` recovers 4.9% (treehill) to 34.6% (counter) of that scene's compression loss
(U - baseline), 21.1% on average. The four indoor scenes at data factor 2, where the loss is largest,
gain the most.

### k-means time (seconds, one T4)

| Scene | `baseline` (torchpq manhattan) | `lloyd_wopa` | `lloyd_wopa_area` | Iterations (wopa / wopa_area) |
|---|---|---|---|---|
| garden | 433 | 633 | 642 | 100 / 100 |
| bicycle | 404 | 317 | 405 | 44 / 56 |
| stump | 408 | 290 | 461 | 51 / 81 |
| bonsai | 435 | 281 | 374 | 41 / 55 |
| counter | 409 | 535 | 490 | 94 / 86 |
| kitchen | 434 | 622 | 587 | 92 / 87 |
| room | 408 | 234 | 263 | 41 / 46 |
| treehill | 437 | 326 | 594 | 48 / 87 |
| flowers | 409 | 355 | 567 | 63 / 100 |
| **mean** | **420** | **399** | **487** | |

torchpq manhattan always runs its full 100 iterations (it never meets `tol`); the Lloyd runs stop at
`tol` on 7 of 9 scenes. `lloyd_wopa_area` costs 0.64x to 1.48x the baseline clustering time, 1.16x on
average, and this is the benchmark implementation (float64 weighted update, chunked `addmm`), not a
tuned one.

![Run 4: per-scene PSNR change, means and k-means time](run4/tilequant/rd_run4.png)

### Session

One Kaggle session on 2x T4, 7 new scenes queued over 2 GPUs. Training took 1,932-3,398 s per scene
(data factor 4: 1,932-2,243 s; data factor 2: 2,983-3,398 s), the PLAS sort 69-83 s, and the whole
per-scene job 3,302-5,357 s. The pre-run estimate was 5.70 h for the session
(`run4/tilequant/run4_plan.json`); the jobs summed to 8.3 GPU-hours over 2 GPUs.

## 6. Run 5: the library code on MipNeRF360 and Tanks & Temples

Run 5 measures gsplat's own `PngCompression(kmeans_backend="builtin", kmeans_weighting=...)` from
`feat/png-weighted-kmeans`, with `"opacity"` (row `lloyd_wopa`) and `"opacity_area"`
(`lloyd_wopa_area`), against the library default (TorchPQ manhattan). The rule is the run-4 rule,
unchanged, applied to each dataset separately.

- **MipNeRF360:** the 9 run-4 checkpoints. Candidates are paired against the run-4 baseline rows (same
  checkpoint, same sort). The default path was also re-run (`baseline_lib`) to check it and to measure
  its cost.
- **Tanks & Temples:** `mcmc_tt.sh` (train, truck; data factor 1, 1M Gaussians), trained in this
  session. Candidates are paired against the baseline measured in the same session.

### Parity gate (`run5_parity.json`): passed

The library must reproduce the run-3 benchmark rows (`lloyd_wopa_area`, k-means seed 0). The run-5
rows were written into the run-3 run-directory path, so zip sizes compare directly.

| Scene | PSNR (run 3 = run 5) | zip bytes (both) | raw bytes (both) | k-means time run 3 / run 5 |
|---|---|---|---|---|
| garden | 26.963133 dB | 16,409,335 | 16,405,132 | 641.5 / 645.6 s |
| bicycle | 25.357714 dB | 16,219,981 | 16,216,069 | 404.8 / 339.0 s |

dPSNR, dSSIM and dLPIPS are 0 on both scenes. `iters_equal` is false only because the library does not
report an iteration count: `n_iters` is empty in every run-5 row.

The match goes beyond the gate. On all 9 MipNeRF360 scenes, both library candidates give exactly the
run-4 benchmark rows: PSNR, SSIM, LPIPS, zip bytes and raw bytes. `run5_table.csv` is byte-identical
to `run4_table.csv`.

### Verdict (`run5_decision.json`): both candidates pass on both datasets; `pr_candidate` = `lloyd_wopa_area`

| Dataset | Candidate | mean dPSNR | mean dSSIM | mean dLPIPS | Scenes with dPSNR <= 0 | Worst scene dPSNR | Worst zip | Worst raw |
|---|---|---|---|---|---|---|---|---|
| MipNeRF360 | `lloyd_wopa` | +0.0494 dB | +0.00091 | -0.00048 | 1 (treehill) | -0.0099 dB | +0.009% | +0.008% |
| MipNeRF360 | `lloyd_wopa_area` | +0.1109 dB | +0.00072 | -0.00093 | 0 | +0.0049 dB | +0.267% | +0.266% |
| Tanks & Temples | `lloyd_wopa` | +0.0047 dB | +0.00034 | -0.00013 | 1 (train) | -0.0007 dB | +0.076% | +0.076% |
| Tanks & Temples | `lloyd_wopa_area` | +0.0525 dB | +0.00026 | -0.00028 | 0 | +0.0441 dB | +0.153% | +0.153% |

Both Tanks & Temples sanity gates passed with no warnings: 1,000,000 Gaussians, U - baseline
0.171 / 0.188 dB (train / truck), baseline zip 0.989x / 1.012x the repo `TanksAndTemples.csv` 1M row.

### Means, upstream format, each next to its own repo row

MipNeRF360, 9 scenes (`run5_table.csv`, identical to section 5):

| Submethod | PSNR | SSIM | LPIPS | Size [Bytes] | #Gaussians |
|---|---|---|---|---|---|
| `baseline` (library, run-4 rows) | 27.4929 | 0.8192 | 0.2147 | 16,005,532 | 1,000,000 |
| `lloyd_wopa` | 27.5423 | 0.8201 | 0.2143 | 15,983,770 | 1,000,000 |
| `lloyd_wopa_area` | 27.6038 | 0.8199 | 0.2138 | 15,988,981 | 1,000,000 |
| repo `MipNeRF360.csv` 1M row | 27.29 | 0.811 | 0.229 | 16,038,022 | 1,000,000 |
| uncompressed checkpoints (U) | 27.9258 | 0.8289 | 0.2033 | - | 1,000,000 |

Tanks & Temples, 2 scenes (`run5_tt_table.csv`):

| Submethod | PSNR | SSIM | LPIPS | Size [Bytes] | #Gaussians |
|---|---|---|---|---|---|
| `baseline` (library, this session) | 24.0759 | 0.8549 | 0.1633 | 16,105,621 | 1,000,000 |
| `lloyd_wopa` | 24.0806 | 0.8552 | 0.1632 | 16,107,724 | 1,000,000 |
| `lloyd_wopa_area` | 24.1284 | 0.8551 | 0.1630 | 16,118,242 | 1,000,000 |
| repo `TanksAndTemples.csv` 1M row | 24.03 | 0.857 | 0.163 | 16,100,628 | 1,000,000 |
| uncompressed checkpoints (U) | 24.2557 | 0.8612 | 0.1553 | - | 1,000,000 |

**The repo rows are context, not controls.** They come from a different environment (training run,
torch / gsplat build, evaluation setup). Our baseline is +0.20 dB and -0.20% size against the
MipNeRF360 row, and +0.046 dB, -0.0021 SSIM and +0.03% size against the Tanks & Temples row. Every
delta in this section is paired against our own baseline on the same checkpoints.

### Per-scene deltas (candidate minus the paired baseline)

| Dataset | Scene | U - baseline | `lloyd_wopa` dPSNR | zip | `lloyd_wopa_area` dPSNR | dSSIM | dLPIPS | zip | raw |
|---|---|---|---|---|---|---|---|---|---|
| MipNeRF360 | garden | 0.434 dB | +0.0324 | -0.19% | +0.0819 | -0.00016 | -0.00083 | -0.25% | -0.25% |
| MipNeRF360 | bicycle | 0.237 dB | +0.0158 | -0.08% | +0.0266 | +0.00005 | -0.00094 | -0.14% | -0.14% |
| MipNeRF360 | stump | 0.290 dB | +0.0252 | -0.13% | +0.0481 | +0.00052 | -0.00105 | +0.06% | +0.06% |
| MipNeRF360 | bonsai | 0.859 dB | +0.1326 | -0.33% | +0.2721 | +0.00136 | -0.00020 | -0.30% | -0.30% |
| MipNeRF360 | counter | 0.546 dB | +0.0838 | +0.01% | +0.1885 | +0.00166 | -0.00136 | +0.25% | +0.25% |
| MipNeRF360 | kitchen | 0.824 dB | +0.1123 | -0.21% | +0.2474 | +0.00109 | -0.00127 | -0.47% | -0.47% |
| MipNeRF360 | room | 0.386 dB | +0.0263 | -0.17% | +0.0846 | +0.00066 | -0.00024 | -0.21% | -0.21% |
| MipNeRF360 | treehill | 0.101 dB | -0.0099 | -0.09% | +0.0049 | +0.00040 | -0.00102 | -0.15% | -0.15% |
| MipNeRF360 | flowers | 0.219 dB | +0.0262 | -0.04% | +0.0442 | +0.00090 | -0.00147 | +0.27% | +0.27% |
| T&T | train | 0.171 dB | -0.0007 | +0.076% | +0.0609 | +0.00019 | -0.00038 | +0.001% | +0.001% |
| T&T | truck | 0.188 dB | +0.0101 | -0.048% | +0.0441 | +0.00033 | -0.00017 | +0.153% | +0.153% |

- `lloyd_wopa_area` has lower LPIPS on all 11 scenes and higher SSIM on 10 of 11 (garden -0.00016).
- On Tanks & Temples it recovers 35.6% (train) and 23.4% (truck) of the compression loss
  (U - baseline). The loss there is small (0.17-0.19 dB), so the absolute gain is smaller than on the
  indoor MipNeRF360 scenes.
- `lloyd_wopa` passes the rule on Tanks & Temples too, but only just: +0.0047 dB mean, train
  -0.0007 dB, and truck LPIPS +0.00010.

### Cost (`run5_costs.csv`, one T4 per job)

| Dataset | Config | k-means mean | k-means range | Peak GPU memory |
|---|---|---|---|---|
| MipNeRF360 | `baseline_lib` (TorchPQ) | 405 s | 398-427 s | 1.26 GB |
| MipNeRF360 | `lloyd_wopa` | 411 s | 261-673 s | 3.37-3.38 GB |
| MipNeRF360 | `lloyd_wopa_area` | 510 s | 297-646 s | 3.37-3.62 GB |
| Tanks & Temples | `baseline` (TorchPQ) | 416 s | 415-418 s | 1.03 GB |
| Tanks & Temples | `lloyd_wopa` | 247 s | 241-252 s | 3.37 GB |
| Tanks & Temples | `lloyd_wopa_area` | 328 s | 298-357 s | 3.37 GB |

k-means time per scene (TorchPQ / `lloyd_wopa` / `lloyd_wopa_area`, seconds):

| garden | bicycle | stump | bonsai | counter | kitchen | room | treehill | flowers | train | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| 427 / 673 / 646 | 403 / 283 / 339 | 405 / 326 / 522 | 398 / 261 / 388 | 405 / 608 / 556 | 398 / 583 / 599 | 404 / 264 / 297 | 401 / 303 / 597 | 405 / 402 / 644 | 415 / 241 / 357 | 418 / 252 / 298 |

- `lloyd_wopa_area` takes 0.74x-1.59x the TorchPQ time per MipNeRF360 scene (1.26x on average) and
  0.86x / 0.71x on train / truck. TorchPQ always runs its full 100 iterations; the builtin backend
  stops at `tol` or at 100 iterations.
- **Peak GPU memory** is `torch.cuda.max_memory_allocated()` over `compress()` in the benchmark
  process. It includes everything that process holds on the GPU (the loaded scene, among others), not
  only the clustering. `lloyd_wopa_area` shows 3.61-3.62 GB only on garden and bicycle, whose rows come
  from the parity-gate jobs (that config was the only one in those processes). It shows 3.37-3.38 GB
  everywhere else, with the same inputs and identical output. So that extra 0.24 GB comes from process
  state, not from the clustering.
- The builtin backend's largest buffer is the per-chunk distance matrix, `chunk_size` x `n_clusters`
  float32: 4096 x 65,536 x 4 B = 1.07 GB at the default. That is arithmetic from the code, not a
  measurement. A smaller `chunk_size` shrinks it; the speed cost of doing so was not measured.
  `chunk_size` is an argument of `weighted_kmeans` only: `PngCompression` uses the default and does
  not expose it.
- **Builtin backend reproducibility:** all 18 library candidate rows (9 scenes x 2 weightings) equal
  the run-4 benchmark rows from another session and another implementation of the same algorithm.
  That is measured, not guaranteed: the float64 `index_add_` in the centroid update uses CUDA atomics,
  whose order is not fixed.
- Decompression is unchanged: 0.82-1.30 s for every config.

### Baseline reproduction (`baseline_reproduced_mipnerf360`): to about 0.002 dB, not bit-exact

`baseline_lib` re-ran the unchanged TorchPQ path on the 9 run-4 checkpoints:

| Scene | dPSNR | dSSIM | dLPIPS | raw bytes | zip bytes |
|---|---|---|---|---|---|
| bonsai, counter, kitchen, room, treehill, flowers | 0 | 0 | 0 | 0 | +72 |
| garden | 0 | 0 | 0 | 0 | +90 |
| bicycle | +0.0024 dB | +0.00002 | -0.000005 | +41 (`shN.npz`) | +145 |
| stump | +0.00006 dB | -0.0000002 | +0.000009 | +3 (`shN.npz`) | +78 |

- 7 of 9 scenes are identical. On bicycle and stump the shN codebook itself differs (PNG files and
  `meta.json` are identical), so the difference is in the clustering, not in rendering or evaluation.
- The zip differences come from run-directory path lengths. `zip -r` stores the path in 9 entries
  (the directory and its 8 files), twice each (local header and central directory). `baseline_lib`
  is 4 characters longer than `baseline`: 4 x 9 x 2 = 72 B, exactly the difference on the 6 identical
  run-4-trained scenes. garden's reference is the run-2 row (`tilequant_shn_runs`, 1 character
  shorter than `tilequant_run5_runs`): 5 x 18 = 90 B. bicycle (145 B, also a run-2 reference) and
  stump (78 B) add the changed `shN.npz` on top of 90 / 72 B.
- **Conclusion:** TorchPQ k-means reproduces to about 0.002 dB across sessions, not bit-exactly. This
  corrects "deterministic per seed across sessions" (section 4), which held for the one seed-0
  re-run of garden and bicycle in run 3.
- Side check, not part of the rule: paired against `baseline_lib` instead of the run-4 rows, both
  candidates still pass and `lloyd_wopa_area` gains +0.1106 dB (bicycle +0.0242, stump +0.0481).

### CPU smoke test (`run5_cpu_smoke.json`): passed

On CPU, with TorchPQ blocked (its import raises `ModuleNotFoundError`), the builtin backend compressed
and decompressed 4,096 Gaussians in 0.21 s / 0.013 s. It wrote the 8 usual files (199,335 B), the
decoded shapes match and every value is finite. The codebook has 4,096 centroids:
`PngCompression` asks for 65,536, and the builtin path clamps that to the number of splats.

![Run 5: per-scene PSNR change and clustering cost, both datasets](run5/tilequant/rd_run5.png)

### Session

One Kaggle session on 2x T4. The gsplat wheel was rebuilt (4,404 s), because run 5 changes
`gsplat/`. Jobs:

- parity gate: 990 / 660 s (garden / bicycle)
- MipNeRF360: 750-2,055 s per scene
- Tanks & Temples: 3,286 / 3,106 s (train / truck), including training (1,599 / 1,736 s)

The jobs summed to 6.05 GPU-hours over the 2 GPUs. The notebook's pre-run estimate was 7.55 h
(`run5_plan.json`).

## 7. Open items

- **PR:** open as [nerfstudio-project/gsplat#1063](https://github.com/nerfstudio-project/gsplat/pull/1063)
  (opened 2026-09-18, branch `feat/png-weighted-kmeans` at `61cd1baf`), with the run-5 numbers.
- **Default flip:** written as the PR's last commit (`61cd1baf`: `kmeans_backend="builtin"`,
  `kmeans_weighting="opacity_area"` as defaults), droppable on its own. `benchmarks/compression/results/*.csv`
  still hold the TorchPQ numbers; regenerating them is left to a follow-up.
- **CUDA checks:** `lint/format-code.sh` and the test suite on a CUDA machine. Locally only CPU tests
  run (`tests/test_compression.py` skips its CUDA test).

## 8. E0: a Gauss-Newton metric for the shN codebook (`bench/gn-vq`) — G0 passed

The questions, definitions and decision rules were fixed before any E0 code or result, in
`kaggle/PREREG_GN.md`:

- the original text (commit `464c46a5`);
- Amendment 1: the toy exactness check (commit `223faf91`);
- Amendment 2: G0 over more codebooks with tie-exempt pairs, and the exact-assignment refines (commit
  `48fa5823`);
- Amendment 3: the G0 verdict is the ranking alone, with the predicted/measured ratio reported as
  calibration; an end-to-end exactness validity check; a new criterion for the lifted-assignment
  check; a proximal-objective rise marks that refine invalid instead of stopping the run (commit
  `86e5f35f`);
- Amendment 4: the toy and end-to-end checks render committed scene fixtures whose hashes are asserted
  before rendering; end-to-end preconditions read from gsplat's own render; a report-only probe-noise
  diagnostic (commit `a8f4d9ae`).

The run before this one crashed in cuSOLVER's batched eigendecomposition and produced no rows; the
batch limit is handled in `bench/gn/batched.py` (see the fallback sequence below).

### Verdict (`gn_g0.json`): `pass`

The rule is Amendment 3: the ranking alone decides. Within each K, the 3 config pairs on train and
test views of both scenes give 36 pair checks.

| | Value |
|---|---|
| Verdict | **pass** |
| Pair checks | 36 |
| Non-tied pairs | 34 (minimum required: 6) |
| Misordered non-tied pairs | 0 |
| Ties (exempt, < 5% relative) | 2 |
| Same on unclamped renders | 34 non-tied, 0 misordered, `pass` |
| Validity checks | all 7 true: `sh_basis`, `toy_exactness`, `e2e_exactness`, render parity and reproduction per scene |

Both ties are on bicycle at K = 4,096, test views: `upstream_l1` vs `plain_l2`
(D 3.02307e-4 vs 3.15594e-4) and `upstream_l1` vs `lloyd_wopa_area` (3.02307e-4 vs 2.93374e-4). So
the metric ordered every pair it was asked about, on both view sets, at all three codebook sizes.

### Predicted vs measured (`gn_results_<scene>.csv`)

`P` is the GN prediction, `D` the measured shN-only dMSE on clamped renders; PSNR and bytes are the
full compressed pipeline. Uncompressed reference: garden 27.3150 dB, bicycle 25.5682 dB.

| Scene | K | Config | P | D train | D test | PSNR (dB) | Raw bytes |
|---|---|---|---|---|---|---|---|
| garden | 4,096 | `upstream_l1` | 2.4782e-4 | 3.1545e-4 | 3.1185e-4 | 26.5670 | 14,783,177 |
| garden | 4,096 | `plain_l2` | 2.6277e-4 | 3.4596e-4 | 3.4524e-4 | 26.5222 | 14,790,495 |
| garden | 4,096 | `lloyd_wopa_area` | 2.3429e-4 | 2.8137e-4 | 2.7905e-4 | 26.6977 | 14,778,950 |
| garden | 16,384 | `upstream_l1` | 1.9968e-4 | 2.3623e-4 | 2.3315e-4 | 26.7419 | 15,251,769 |
| garden | 16,384 | `plain_l2` | 2.1723e-4 | 2.6340e-4 | 2.6263e-4 | 26.6851 | 15,243,082 |
| garden | 16,384 | `lloyd_wopa_area` | 1.7895e-4 | 2.0010e-4 | 1.9953e-4 | 26.8461 | 15,244,312 |
| garden | 65,536 | `upstream_l1` | 1.5600e-4 | 1.7412e-4 | 1.7293e-4 | 26.8813 | 16,445,885 |
| garden | 65,536 | `plain_l2` | 1.6645e-4 | 1.8738e-4 | 1.8650e-4 | 26.8450 | 16,475,746 |
| garden | 65,536 | `lloyd_wopa_area` | 1.3508e-4 | 1.4224e-4 | 1.4174e-4 | 26.9631 | 16,405,132 |
| bicycle | 4,096 | `upstream_l1` | 2.2973e-4 | 2.8155e-4 | 3.0231e-4 | 25.1183 | 14,385,340 |
| bicycle | 4,096 | `plain_l2` | 2.4533e-4 | 3.0694e-4 | 3.1559e-4 | 25.1276 | 14,396,759 |
| bicycle | 4,096 | `lloyd_wopa_area` | 2.1216e-4 | 2.5289e-4 | 2.9337e-4 | 25.1487 | 14,396,171 |
| bicycle | 16,384 | `upstream_l1` | 1.6521e-4 | 1.8341e-4 | 2.0492e-4 | 25.2528 | 14,891,479 |
| bicycle | 16,384 | `plain_l2` | 1.7960e-4 | 2.0264e-4 | 2.1649e-4 | 25.2374 | 14,893,593 |
| bicycle | 16,384 | `lloyd_wopa_area` | 1.4590e-4 | 1.6126e-4 | 1.8788e-4 | 25.2782 | 14,909,318 |
| bicycle | 65,536 | `upstream_l1` | 1.1884e-4 | 1.2432e-4 | 1.4107e-4 | 25.3311 | 16,238,295 |
| bicycle | 65,536 | `plain_l2` | 1.2825e-4 | 1.3701e-4 | 1.5018e-4 | 25.3086 | 16,246,985 |
| bicycle | 65,536 | `lloyd_wopa_area` | 9.8475e-5 | 1.0186e-4 | 1.2195e-4 | 25.3577 | 16,216,069 |

`lloyd_wopa_area` has both the lowest `P` and the lowest measured error at every K on both scenes, and
the highest PSNR, which is run 3's and run 4's result seen through the metric.

### Calibration (reported, not judged)

| Quantity | Range over the 18 codebooks |
|---|---|
| `P / D` train, clamped | 0.7595 - 0.9668 |
| `P / D` train, unclamped | 0.7586 - 0.9491 |
| `P / D` test, clamped | 0.7232 - 0.9530 |
| Calibrated (within 0.5-2x) | 18 of 18, clamped and unclamped |
| Cross/diagonal term ratio `D_train_unclamped / P - 1` | 0.0536 - 0.3182 |

The prediction is below the measurement everywhere, and the gap shrinks as K grows:

| K | Cross/diagonal ratio |
|---|---|
| 4,096 | 0.2017 - 0.3182 |
| 16,384 | 0.1189 - 0.2142 |
| 65,536 | 0.0536 - 0.1271 |

That is the direction and the ordering Amendment 3 gave as its reason for dropping the ratio from the
verdict: the GN model is block-diagonal, so the cross terms of neighbouring splats that share a
centroid are missing from `P`, and they matter most where the most neighbours share one.

### Validity checks

| Check | Result |
|---|---|
| `sh_basis` (vs gsplat's CUDA `spherical_harmonics`) | max abs error 6.50e-6 (rule: < 1e-5) |
| `toy_exactness`, 64 probes | summed estimate off by 2.87% (rule: < 5%); per-splat median 12.4%, p90 27.7%, max 58.6% |
| `toy_exactness` plumbing | gradient vs Hutchinson formula 4.84e-7, all-ones channel 2.01e-7 (rule: < 1e-4 each) |
| `e2e_exactness`, non-overlapping | relative error 6.85e-8 (rule: <= 1e-4); preconditions all held |
| Render parity, both scenes | max abs difference 0.0 |
| Reproduction, both scenes | all three K = 65,536 rows equal to run 3: PSNR within 3.6e-15 dB, identical raw bytes |

The reproduction check is exact because the K = 65,536 codebooks came from the restored run-3 caches
(`source` = `run3_cache`); K = 4,096 and 16,384 were clustered in this run (`recomputed`).

Report-only, outside every verdict: the toy check's exact probe-noise `sigma_rel` is 0.014806, whose
implied false-fail probability for the 5% rule under a normal approximation is 7.33e-4. The
overlapping end-to-end variants, also report-only: `P / D` 0.9614 with cross/diagonal +0.0402 for
independent perturbations, and 0.5855 with +0.7079 when all splats share one perturbation - the
correlated case Amendment 3 describes.

### The GN pass (`gn_meta_<scene>.json`)

| | garden | bicycle |
|---|---|---|
| Train / test views | 161 / 24 | 169 / 25 |
| Train pixels | 175,406,280 | 171,702,648 |
| GN pass | 11.75 s | 11.51 s |
| Splats visible in any train view | 998,603 | 986,233 |
| Splats with `tr(M) = 0` | 1,397 | 13,776 |
| Clamp fraction (`SH + 0.5 < 0`, logged only) | 2.894% | 10.328% |
| max abs entry of M, all finite | 18,842.88 | 36,533.97 |
| Original render above 1 before clamping, train | 0.0217-0.0235% per channel | 2.177-3.233% |
| Original render above 1 before clamping, test | 0.0394-0.0445% | 0.136-0.187% |
| Train-metric code vs `Runner.eval` on test views | 0.0 on PSNR, SSIM, LPIPS | 0.0 |

No pixel of either original render fell below 0.

The lifted fp32 assignment against the exhaustive float64 minimum, on 10,000 sampled splats:

| | garden | bicycle |
|---|---|---|
| Evaluated (`tr(M) > 0`) / sampled | 9,989 / 10,000 | 9,866 / 10,000 |
| `sum excess / sum d_min` (rule: <= 1e-4) | 4.07e-13 | 3.43e-10 |
| Worst `excess / scale` (rule: <= 1e-4) | 1.99e-12 | 9.89e-10 |
| Splats over the tolerance | 0 | 0 |
| Same index as brute force | 99.95% | 98.90% |
| Splats with `d_min = 0` | 61 | 75 |
| Splats with `d_min < 1e-3 * scale` | 8,197 | 9,489 |
| Time | 29.05 s | 28.77 s |

Amendment 3 replaced the Amendment-2 criterion (excess relative to `d_min`) for exactly this reason,
and the run shows why: that measure reaches 0.197 on garden and 90,600.9 on bicycle, because for
thousands of splats `d_min` is at or near 0, while the criterion actually used is met with 5 to 8
orders of magnitude of margin.

### Spectrum of `M_i` (`gn_spectrum_<scene>.csv`), over splats with `tr(M) > 0`

| Metric | garden unweighted | garden trace-weighted | bicycle unweighted | bicycle trace-weighted |
|---|---|---|---|---|
| Participation ratio, mean | 2.221 | 2.403 | 2.846 | 2.222 |
| Participation ratio, median | 1.946 | 2.014 | 2.307 | 1.280 |
| Participation ratio, p95 | 4.505 | 5.460 | 5.721 | 5.573 |
| Top-1 eigenvalue fraction, mean | 0.673 | 0.660 | 0.623 | 0.741 |
| Top-1 fraction, median | 0.665 | 0.647 | 0.573 | 0.877 |
| Top-3 fraction, mean | 0.932 | 0.916 | 0.875 | 0.919 |
| Top-3 fraction, median | 0.973 | 0.970 | 0.955 | 0.998 |

Out of 15 possible directions, a splat's metric occupies 2-3, and its top 3 eigenvalues carry about
90% of the energy. The metric is strongly anisotropic, which is what a Mahalanobis codebook can use
and plain L2 cannot.

### Rank correlations (`gn_spearman_<scene>.csv`)

Spearman over all 1,000,000 splats; the `tr(M) > 0` subsets differ by at most 0.011.

| Pair | garden | bicycle |
|---|---|---|
| `tr(M)` vs `opacity_area` (run 3's weight) | 0.5678 | 0.5373 |
| `tr(M)` vs `F` (footprint) | 0.9330 | 0.9479 |
| `tr(M)` vs C3DGS-style weight | 0.9362 | 0.9434 |
| `opacity_area` vs `F` | 0.5988 | 0.5799 |
| `opacity_area` vs C3DGS-style | 0.6383 | 0.6513 |
| `F` vs C3DGS-style | 0.9944 | 0.9923 |

`tr(M)` is close to the footprint and to the C3DGS-style weight, and only moderately related to the
weight that won run 3. So the scalar part of the metric is not what `opacity_area` measures.

### The two exploratory refines (`gn_refine_<variant>_<scene>.json`, excluded from G0)

Both warm-started from the `lloyd_wopa_area` K = 65,536 seed-0 codebook, 3 iterations, both **valid**
(the proximal objective never rose).

| | garden ridge | garden prox | bicycle ridge | bicycle prox |
|---|---|---|---|---|
| Objective at the warm start | 1.22224e-4 | 1.22224e-4 | 9.07304e-5 | 9.07304e-5 |
| After iteration 1 (assign / update) | 5.60388e-5 / 3.68247e-5 | 5.60388e-5 / 3.68244e-5 | 4.72653e-5 / 3.01872e-5 | 4.72653e-5 / 3.01867e-5 |
| After iteration 2 | 3.48557e-5 / 3.19806e-5 | 3.48570e-5 / 3.19574e-5 | 2.84626e-5 / 2.60212e-5 | 2.84630e-5 / 2.60165e-5 |
| After iteration 3 | 3.15780e-5 / 3.05895e-5 | 3.15614e-5 / 3.05785e-5 | 2.56508e-5 / 2.47791e-5 | 2.56475e-5 / 2.47794e-5 |
| **Unquantized objective, final** | 3.05895e-5 | 3.05785e-5 | 2.47791e-5 | 2.47794e-5 |
| **After the codec's centroid quantization (= `P`)** | 1.38351e-4 | 3.58568e-4 | 2.21117e-4 | 2.98634e-4 |
| Warm start's own quantized `P` | 1.35080e-4 | 1.35080e-4 | 9.84749e-5 | 9.84749e-5 |
| Measured D train (clamped) | 1.4502e-4 | 3.6931e-4 | 2.1080e-4 | 2.6484e-4 |
| PSNR (dB) | 26.9261 | 26.4300 | 25.1850 | 25.0972 |
| PSNR minus `lloyd_wopa_area` (dB) | -0.0371 | -0.5331 | -0.1727 | -0.2605 |
| Raw bytes | 16,211,546 | 16,041,615 | 15,496,773 | 15,537,571 |
| vs `lloyd_wopa_area` bytes | -1.18% | -2.22% | -4.43% | -4.18% |
| `centroids.npy` in `shN.npz` | 1,235,196 | 1,065,266 | 874,221 | 914,982 |
| vs `lloyd_wopa_area` centroid bytes | -13.5% | -25.4% | -45.2% | -42.6% |
| Refine time | 73.6 s | 73.4 s | 72.4 s | 72.1 s |

Per iteration, for the ridge variant: labels changed 84.9% / 37.0% / 18.0% on garden and 78.0% /
43.7% / 24.4% on bicycle; the assignment guard kept the current centroid for 168 / 241 / 448 splats on
garden and 9,240 / 9,239 / 9,534 on bicycle; clusters with `tr(sum M) = 0` that kept `q_old`:
65 / 27 / 18 and 420 / 278 / 257.

The share of exact argmins inside the plain-L2 top-64 shortlist falls from 0.431 to 0.075 to 0.060 on
garden, and from 0.510 to 0.303 to 0.269 on bicycle. A shortlist of 64 would have missed most of the
exact assignments after the first update, which is why Amendment 2 replaced it with the exact lifted
assignment. The shortlist diagnostic cost 12.4-13.0 s per iteration against 9.4-9.5 s for the
assignment it checks, and the centroid update 1.3 s.

**So the exact-Mahalanobis refine works on the objective it optimizes and loses on the objective that
matters:** the unquantized GN objective falls by 4.0x (garden) and 3.7x (bicycle), while after the
codec's centroid quantization every refined row is worse than its own warm start, by 0.04 to 0.53 dB.
The refines are also smaller, by 1.2-4.4% of total bytes, with the `centroids.npy` member 13-45%
smaller.

**Hypothesis (not tested by this run):** the refines' loss comes from the range of the shN centroid
quantizer. gsplat's `_compress_kmeans` quantizes the whole codebook with **one global scalar min/max
and 6 bits** (`mins = min(centroids) + 1e-6`, `maxs = max(centroids)`, step `(maxs - mins) / 63`, 64
levels for all K x 45 coordinates). The GN update is only constrained where `M` has curvature, so it
is free to move centroid coordinates far in weakly constrained directions; a few such coordinates
widen `maxs - mins`, and the step coarsens for every coordinate at once. Two observations from this
run are consistent with it, and neither proves it: the quantized objective gets worse while the
unquantized one improves 3.7-4.0x, and the `centroids.npy` member shrinks by 13-45%, which is what a
coarser step does when many centroids collapse onto the same codes. The proximal variant, which pulls
weakly constrained directions to the old centroid rather than to 0, ends up worse than the ridge
variant on both scenes, which also fits: ridge at least shrinks those coordinates toward 0. E1 tests
this directly by clipping every centroid coordinate to the warm-start codebook's range, per
`PREREG_GN.md` Amendment 5.

### The eigvalsh batch fallback

`bench/gn/batched.py` starts each batched linalg call at `LINALG_MAX_BATCH` = 32,768 and halves on a
backend error. What the run recorded:

| Where | Sequence | Result |
|---|---|---|
| Smoke cell, `linalg_scale`, 1,000,000 float64 matrices | 32,768 -> `CUSOLVER_STATUS_INVALID_VALUE` -> 16,384 -> CUDA out of memory (9.49 GiB requested; 9.82 GiB for the last, odd chunk) -> 8,192 | 8,192 worked; 30 reductions logged (15 cuSOLVER, 15 out-of-memory); eigvalsh over 1M matrices 6.545 s, the 65,536-system solve 0.222 s |
| Both scene jobs, `eigen_stats` over 1,000,000 splats | 32,768 -> `CUSOLVER_STATUS_INVALID_VALUE` -> 16,384 | 16,384 worked (the job process held less memory); 15 reductions in each job's last 200 log lines, one per chunk; spectrum step 8.21 s garden, 8.39 s bicycle |

So 32,768 is refused outright by cuSOLVER's batched eigendecomposition on a T4, and 16,384 only fits
when little else is allocated. The helper recovered every time, but it re-discovered the same
reduction on every chunk, which is why E1 starts at 8,192 and remembers the batch that worked.

### Timings (`timings.json`, `gn_meta_<scene>.json`, `gn_results_<scene>.csv`)

| Step | Seconds |
|---|---|
| Restore inputs | 15.6 |
| Install (restored run-5 wheel, no build) | 151.2 |
| Smoke tests (including the 1M-matrix scale check) | 14.5 |
| MipNeRF360 data for both scenes | 130.1 |
| `gn_e0_garden` job | 2,145.1 |
| `gn_e0_bicycle` job | 2,085.1 |

The two jobs ran in parallel, one T4 each; the timed steps sum to about 41 minutes. Inside a job: GN pass
11.7 / 11.5 s, spectrum 8.2 / 8.4 s, Spearman 4.8 / 4.8 s, lifted check 29.0 / 28.8 s, the two refines
73.6 + 73.4 s (garden) and 72.4 + 72.1 s (bicycle). The rest is clustering and the 12 evaluated rows
per scene.

Clustering time per codebook (garden / bicycle):

| Config | K = 4,096 | K = 16,384 | K = 65,536 |
|---|---|---|---|
| `upstream_l1` (TorchPQ manhattan, 100 iterations) | 102.5 / 99.0 | 104.7 / 105.4 | 419.1 / 432.1 |
| `plain_l2` (library Lloyd) | 37.0 / 34.1 | 149.6 / 141.5 | 628.1 / 281.3 |
| `lloyd_wopa_area` (library Lloyd, weighted) | 36.2 / 34.1 | 150.1 / 141.9 | 641.5 / 404.8 |

The K = 65,536 columns are the recomputation cost only; those rows used the restored run-3 codebooks.
Lloyd stopped early on bicycle at K = 65,536 (39 iterations for `plain_l2`, 56 for `lloyd_wopa_area`),
which is why its times there are below garden's 100-iteration runs.

### What E0 settles, and what it does not

- **G0 passed:** the GN metric ranks these codebooks by rendering error, on both view sets, at all
  three sizes, with no misordered pair.
- It is also calibrated within 0.5-2x everywhere, though that was not part of the verdict, and the
  missing cross terms account for the gap in the direction Amendment 3 predicted.
- **G1 is untouched.** The refines show that lowering the GN objective does not by itself beat
  `lloyd_wopa_area` once the codec's quantizer has its say; that is what E1 is for, with the range
  clip and the size rule pre-registered in Amendment 5 before any E1 code.

## 9. E1: GN-VQ against `lloyd_wopa_area` at equal size (`bench/gn-vq`) — G1 failed

**G1 failed.** Every GN-VQ seed was more than 0.5% larger than `lloyd_wopa_area` at the same seed:
+2.37% on garden and +0.98% to +1.01% on bicycle. Under Amendment 5 b a seed that breaks the size rule
counts as negative whatever its PSNR did, so all six seeds are negative and G1 fails on both scenes.
The PSNR half of the rule held on its own: GN-VQ gained +0.195 to +0.201 dB on garden and +0.087 to
+0.091 dB on bicycle, no seed below zero, means +0.1972 and +0.0893 dB against the +0.05 dB threshold.
No seed dominates (smaller and better). The extra bytes are all in the shN codebook file, at a
quantizer range and step identical to `lloyd_wopa_area`'s.

The rules were fixed before any E1 code or result: G1 in the original `PREREG_GN.md` (commit
`464c46a5`); the GN-VQ variant, the one-sided size rule, the secondaries and the ablations in
Amendment 5 (`48b6c0fe`); the two ridge rows and the extra quantizer logging in Amendment 6
(`d35c4492`). The verdict below is `bench/gn/g1.py`'s, written by the notebook into `gn1_g1.json`;
nothing here re-judges it.

### Verdict (`gn1_g1.json`): `fail`

K = 65,536, raw bytes of the compressed directory, test PSNR of the full compressed pipeline.

| Scene | Seed | GN-VQ bytes | `lloyd_wopa_area` bytes | Size | GN-VQ PSNR | `lloyd_wopa_area` PSNR | dPSNR (dB) | Negative | Dominates |
|---|---|---|---|---|---|---|---|---|---|
| garden | 0 | 16,793,118 | 16,405,132 | +2.37% | 27.1642 | 26.9631 | +0.2011 | True | False |
| garden | 1 | 16,794,029 | 16,405,784 | +2.37% | 27.1618 | 26.9665 | +0.1953 | True | False |
| garden | 2 | 16,794,112 | 16,405,503 | +2.37% | 27.1620 | 26.9667 | +0.1953 | True | False |
| bicycle | 0 | 16,380,637 | 16,216,069 | +1.01% | 25.4445 | 25.3577 | +0.0868 | True | False |
| bicycle | 1 | 16,412,866 | 16,253,928 | +0.98% | 25.4520 | 25.3618 | +0.0902 | True | False |
| bicycle | 2 | 16,402,556 | 16,242,429 | +0.99% | 25.4488 | 25.3580 | +0.0908 | True | False |

| Scene | Mean dPSNR (dB) | Negative seeds | Size violations | Dominating seeds |
|---|---|---|---|---|
| garden | +0.1972 | 3 | 3 | 0 |
| bicycle | +0.0893 | 3 | 3 | 0 |

The paired difference barely moves with the k-means seed: its spread over seeds 0-2 is 0.0058 dB on
garden and 0.0040 dB on bicycle. Every seed is negative for the same reason, the size.

### Where the extra bytes are

Between a GN-VQ row and `lloyd_wopa_area` at the same seed, only `shN.npz` differs (`file_bytes`), and
inside it almost all of the difference is `centroids.npy`, the 6-bit codebook codes stored with
`np.savez_compressed` (the member's compressed size):

| Scene | Seed | Size difference (B) | `centroids.npy` (B) | Change | `labels.npy` change (B) |
|---|---|---|---|---|---|
| garden | 0 | +387,986 | 1,428,796 -> 1,816,769 | +27.2% | +13 |
| garden | 1 | +388,245 | 1,429,456 -> 1,817,674 | +27.2% | +27 |
| garden | 2 | +388,609 | 1,429,172 -> 1,817,760 | +27.2% | +21 |
| bicycle | 0 | +164,568 | 1,594,007 -> 1,759,141 | +10.4% | -566 |
| bicycle | 1 | +158,938 | 1,631,935 -> 1,791,627 | +9.8% | -754 |
| bicycle | 2 | +160,127 | 1,620,433 -> 1,781,013 | +9.9% | -453 |

The clip held exactly: each GN-VQ row wrote the same quantizer min and max as `lloyd_wopa_area` at the
same seed (garden seed 0: [-1.076333, 1.078585], step 0.034205; bicycle seed 0: [-0.802158, 0.883134],
step 0.026751), and so the same step. GN-VQ did not lose bytes to a wider or coarser quantizer. At the
same step its codes compress worse.

### Rate-distortion (Amendment 5 d; seed 0, reported, not gating)

| Scene | Config | K | Raw bytes | PSNR (dB) |
|---|---|---|---|---|
| garden | `lloyd_wopa_area` | 4,096 | 14,778,950 | 26.6977 |
| garden | `lloyd_wopa_area` | 16,384 | 15,244,312 | 26.8461 |
| garden | `lloyd_wopa_area` | 65,536 | 16,405,132 | 26.9631 |
| garden | `gn_vq` | 4,096 | 14,803,113 | 27.0065 |
| garden | `gn_vq` | 16,384 | 15,318,406 | 27.0959 |
| garden | `gn_vq` | 65,536 | 16,793,118 | 27.1642 |
| bicycle | `lloyd_wopa_area` | 4,096 | 14,396,171 | 25.1487 |
| bicycle | `lloyd_wopa_area` | 16,384 | 14,909,318 | 25.2782 |
| bicycle | `lloyd_wopa_area` | 65,536 | 16,216,069 | 25.3577 |
| bicycle | `gn_vq` | 4,096 | 14,429,045 | 25.3335 |
| bicycle | `gn_vq` | 16,384 | 14,959,165 | 25.4016 |
| bicycle | `gn_vq` | 65,536 | 16,380,637 | 25.4445 |

| Scene | BD-rate of `gn_vq` vs `lloyd_wopa_area` | PSNR overlap of the two curves |
|---|---|---|
| garden | undefined (NaN) | none |
| bicycle | **-9.84%** | 25.3335-25.3577 dB (0.0242 dB) |

**Why garden's BD-rate is undefined.** BD-rate averages the log-byte difference between the two
curves over the PSNR range both of them cover, and `g1.bd_rate` returns NaN when there is no such range
rather than extrapolate (a choice recorded before the run). On garden there is none: GN-VQ's lowest
point (K = 4,096, 27.0065 dB) is above `lloyd_wopa_area`'s highest (K = 65,536, 26.9631 dB), so the
whole GN-VQ curve lies above the whole baseline curve. It is undefined in GN-VQ's favour: GN-VQ at
K = 4,096 (14,803,113 B) is 9.77% smaller than `lloyd_wopa_area` at K = 65,536 (16,405,132 B) and
+0.0434 dB better.

**Bicycle's -9.84%** comes from a narrow overlap, 0.0242 dB wide, which lies inside both measured
curves (the bottom of GN-VQ's, the top of `lloyd_wopa_area`'s), so both are interpolated there, not
extrapolated. Recomputing it from the bundled points with the same function gives the bundled value
(-9.8445%). Across K on bicycle, GN-VQ at K = 16,384 is 7.75% smaller than `lloyd_wopa_area` at
K = 65,536 and +0.0438 dB better, and GN-VQ at K = 4,096 is 3.22% smaller than `lloyd_wopa_area` at
K = 16,384 and +0.0554 dB better.

**At equal K, the size premium grows with K and the PSNR gain shrinks:**

| Scene | K | GN-VQ bytes vs `lloyd_wopa_area` | dPSNR (dB) |
|---|---|---|---|
| garden | 4,096 | +0.16% | +0.3089 |
| garden | 16,384 | +0.49% | +0.2499 |
| garden | 65,536 | +2.37% | +0.2011 |
| bicycle | 4,096 | +0.23% | +0.1848 |
| bicycle | 16,384 | +0.33% | +0.1234 |
| bicycle | 65,536 | +1.01% | +0.0868 |

At K = 4,096 and 16,384 GN-VQ was within 0.5% of `lloyd_wopa_area`'s bytes on both scenes; G1 was
judged at K = 65,536 only, as pre-registered.

### The scalar weightings (Amendment 5 d; reported, not gating)

GN-VQ against the `tr(M)`-weighted and the C3DGS-weighted Lloyd, same K, same seeds, same size rule.
Both comparisons `fail` for the same reason G1 does:

| Comparison | Scene | Mean dPSNR (dB) | GN-VQ size vs the weighting | Negative | Size violations | Dominating |
|---|---|---|---|---|---|---|
| GN-VQ vs `lloyd_trace` | garden | +0.1519 | +2.28% to +2.46% | 3 | 3 | 0 |
| GN-VQ vs `lloyd_trace` | bicycle | +0.0520 | +0.94% to +1.06% | 3 | 3 | 0 |
| GN-VQ vs `lloyd_c3dgs` | garden | +0.1621 | +2.12% to +2.47% | 3 | 3 | 0 |
| GN-VQ vs `lloyd_c3dgs` | bicycle | +0.0613 | +0.98% to +1.05% | 3 | 3 | 0 |

**Post hoc, not pre-registered:** the scalar weightings themselves against `lloyd_wopa_area`, with the
same `judge_pair` code. Neither pays in bytes:

| Weighting | Scene | dPSNR vs `lloyd_wopa_area` (dB), seeds 0-2 | Mean (dB) | Size vs `lloyd_wopa_area` | Dominating seeds |
|---|---|---|---|---|---|
| `lloyd_trace` | garden | +0.0398 to +0.0521 | +0.0454 | -0.089% to +0.081% | 2 |
| `lloyd_trace` | bicycle | +0.0364 to +0.0382 | +0.0373 | -0.073% to +0.077% | 2 |
| `lloyd_c3dgs` | garden | +0.0312 to +0.0420 | +0.0351 | -0.102% to +0.242% | 2 |
| `lloyd_c3dgs` | bicycle | +0.0258 to +0.0302 | +0.0280 | -0.067% to +0.033% | 2 |

So a scalar per-splat weight taken from `M` (its trace) already beats `lloyd_wopa_area` on all six
seeds at about equal bytes, by less than GN-VQ does.

### Ablations and ridge rows (Amendments 5 e and 6; seed 0, K = 65,536, reported only)

| Scene | Config | Raw bytes | vs `lloyd_wopa_area` | `centroids.npy` | PSNR | dPSNR (dB) | Quantizer range | Step | Iterations | Clusters rejected by the clip |
|---|---|---|---|---|---|---|---|---|---|---|
| garden | `lloyd_wopa_area` | 16,405,132 | | 1,428,796 | 26.9631 | | [-1.0763, 1.0786] | 0.03421 | | |
| garden | `gn_vq` | 16,793,118 | +2.37% | 1,816,769 | 27.1642 | +0.2011 | [-1.0763, 1.0786] | 0.03421 | 10 (rel_tol) | 262,736 |
| garden | `gn_vq_noclip` | 16,224,242 | -1.10% | 1,247,887 | 27.0655 | +0.1023 | [-3.5480, 3.1656] | 0.10656 | 10 (rel_tol) | 0 |
| garden | `gn_vq_noqassign` | 16,793,099 | +2.36% | 1,816,769 | 27.1614 | +0.1982 | [-1.0763, 1.0786] | 0.03421 | 10 (rel_tol) | 262,789 |
| garden | `gn_vq_eps1e3` | 16,619,127 | +1.30% | 1,642,786 | 27.1620 | +0.1989 | [-1.0763, 1.0786] | 0.03421 | 10 (rel_tol) | 265,205 |
| garden | `gn_vq_eps1e2` | 16,495,694 | +0.55% | 1,519,339 | 27.1581 | +0.1950 | [-0.9133, 1.0215] | 0.03071 | 10 (rel_tol) | 290,509 |
| bicycle | `lloyd_wopa_area` | 16,216,069 | | 1,594,007 | 25.3577 | | [-0.8022, 0.8831] | 0.02675 | | |
| bicycle | `gn_vq` | 16,380,637 | +1.01% | 1,759,141 | 25.4445 | +0.0868 | [-0.8022, 0.8831] | 0.02675 | 10 (rel_tol) | 220,728 |
| bicycle | `gn_vq_noclip` | 15,816,038 | -2.47% | 1,195,222 | 25.4104 | +0.0527 | [-2.4160, 2.5484] | 0.07880 | 10 (rel_tol) | 0 |
| bicycle | `gn_vq_noqassign` | 16,381,718 | +1.02% | 1,759,141 | 25.4430 | +0.0853 | [-0.8022, 0.8831] | 0.02675 | 10 (rel_tol) | 220,902 |
| bicycle | `gn_vq_eps1e3` | 16,301,438 | +0.53% | 1,679,910 | 25.4502 | +0.0925 | [-0.8022, 0.8831] | 0.02675 | 10 (rel_tol) | 251,419 |
| bicycle | `gn_vq_eps1e2` | 16,213,719 | -0.01% | 1,592,107 | 25.4478 | +0.0901 | [-0.8022, 0.8831] | 0.02675 | 9 (rel_tol) | 266,304 |

The warm start for every GN-VQ row here is the `lloyd_wopa_area` codebook in the first row of its
scene, so its range and step are the warm-start range and step (`warm_quant_*`).

- **No clip** widened the range about threefold: to [-3.5480, 3.1656] on garden (step 0.10656,
  3.1155 times the warm step) and [-2.4160, 2.5484] on bicycle (0.07880, 2.9458 times). Against the
  clipped GN-VQ it lost 0.0987 dB on garden and 0.0341 dB on bicycle, and it was 3.39% and 3.45%
  smaller. That is section 8's quantizer hypothesis seen from both sides: the widened range costs
  PSNR, and the coarser step makes the codes compress better. It still beat `lloyd_wopa_area` by
  +0.1023 / +0.0527 dB while being 1.10% / 2.47% smaller, so it **dominates `lloyd_wopa_area` on both
  scenes**, at this single seed.
- **No final quantized assignment:** within 0.01% of GN-VQ's bytes, -0.0029 / -0.0015 dB. Its objective
  after quantization was higher (garden 4.0827e-05 against GN-VQ's 3.8165e-05; bicycle 3.1020e-05
  against 2.8409e-05), so the final assignment did what it was for, by a small margin in PSNR.
- **eps = 1e-3:** 1.04% / 0.48% smaller than GN-VQ, -0.0022 / +0.0057 dB, range unchanged. Against
  `lloyd_wopa_area`: +1.30% and +0.53%, both outside the 0.5% tolerance.
- **eps = 1e-2:** 1.77% / 1.02% smaller than GN-VQ, -0.0061 / +0.0033 dB. On garden the range
  **shrank** to [-0.9133, 1.0215], step 0.03071 (0.8979 of the warm step); on bicycle it kept the warm
  range. Against `lloyd_wopa_area`: +0.55% on garden (outside the 0.5% tolerance) and -0.01% on bicycle
  at +0.0901 dB, so on bicycle it **dominates `lloyd_wopa_area`**.

None of these rows enters a verdict. Each is one seed.

One reading note on the logged steps. `quant_*` is computed from the written codebook on the CPU and
`warm_quant_*` from the warm-start codebook on the GPU. Where the two ranges are identical, the two
steps still differ by up to 3.73e-09 (float32 rounding on the two devices), which is not a range
change; compare the min and max, or compare `quant_step` with the `lloyd_wopa_area` row's, which is
computed the same way (they are equal).

### Measured error, train vs test views (against `lloyd_wopa_area` at the same K and seed)

`D` is the measured shN-only dMSE on clamped renders, as in E0. The ratio is the row's `D` over
`lloyd_wopa_area`'s; the PSNR columns are the row's gain over `lloyd_wopa_area`. Rows marked (s0) are
compared with `lloyd_wopa_area` K = 65,536 seed 0.

| Scene | Config | K | Seed | `D` ratio, train | `D` ratio, test | Train PSNR gain (dB) | Test PSNR gain (dB) |
|---|---|---|---|---|---|---|---|
| garden | `gn_vq` | 65,536 | 0 | 0.314 | 0.337 | +0.2729 | +0.2011 |
| garden | `gn_vq` | 65,536 | 1 | 0.316 | 0.338 | +0.2761 | +0.1953 |
| garden | `gn_vq` | 65,536 | 2 | 0.316 | 0.338 | +0.2724 | +0.1953 |
| garden | `gn_vq` | 4,096 | 0 | 0.391 | 0.413 | +0.4356 | +0.3089 |
| garden | `gn_vq` | 16,384 | 0 | 0.360 | 0.381 | +0.3440 | +0.2499 |
| garden | `gn_vq_noclip` (s0) | 65,536 | 0 | 0.581 | 0.613 | +0.1444 | +0.1023 |
| garden | `gn_vq_noqassign` (s0) | 65,536 | 0 | 0.328 | 0.348 | +0.2686 | +0.1982 |
| garden | `gn_vq_eps1e3` (s0) | 65,536 | 0 | 0.317 | 0.339 | +0.2698 | +0.1989 |
| garden | `gn_vq_eps1e2` (s0) | 65,536 | 0 | 0.329 | 0.353 | +0.2579 | +0.1950 |
| garden | `lloyd_trace` | 65,536 | 0 | 0.858 | 0.863 | +0.0604 | +0.0521 |
| garden | `lloyd_c3dgs` | 65,536 | 0 | 0.881 | 0.886 | +0.0501 | +0.0420 |
| bicycle | `gn_vq` | 65,536 | 0 | 0.308 | 0.463 | +0.1092 | +0.0868 |
| bicycle | `gn_vq` | 65,536 | 1 | 0.306 | 0.468 | +0.1078 | +0.0902 |
| bicycle | `gn_vq` | 65,536 | 2 | 0.309 | 0.465 | +0.1069 | +0.0908 |
| bicycle | `gn_vq` | 4,096 | 0 | 0.380 | 0.531 | +0.2082 | +0.1848 |
| bicycle | `gn_vq` | 16,384 | 0 | 0.347 | 0.517 | +0.1524 | +0.1234 |
| bicycle | `gn_vq_noclip` (s0) | 65,536 | 0 | 0.484 | 0.674 | +0.0709 | +0.0527 |
| bicycle | `gn_vq_noqassign` (s0) | 65,536 | 0 | 0.326 | 0.469 | +0.1091 | +0.0853 |
| bicycle | `gn_vq_eps1e3` (s0) | 65,536 | 0 | 0.313 | 0.472 | +0.1068 | +0.0925 |
| bicycle | `gn_vq_eps1e2` (s0) | 65,536 | 0 | 0.339 | 0.491 | +0.0956 | +0.0901 |
| bicycle | `lloyd_trace` | 65,536 | 0 | 0.782 | 0.777 | +0.0382 | +0.0382 |
| bicycle | `lloyd_c3dgs` | 65,536 | 0 | 0.825 | 0.813 | +0.0344 | +0.0279 |

- GN-VQ cuts the shN-only error to about a third of `lloyd_wopa_area`'s on train views on both scenes
  (0.306-0.316 at K = 65,536). **Out of sample the cut is smaller, much more so on bicycle:** 0.337-0.338
  on garden's test views, 0.463-0.468 on bicycle's. That is the pattern Amendment 6 recorded from E0,
  where bicycle's weighting advantage also shrank from train to test views.
- The train PSNR gain exceeds the test PSNR gain for every GN-VQ row on both scenes.
- The ridge rows did not narrow bicycle's train/test gap: the `D` ratios are 0.308 / 0.463 at
  eps = 1e-4, 0.313 / 0.472 at 1e-3 and 0.339 / 0.491 at 1e-2. What the larger ridge changed was the
  bytes (previous section).
- The scalar weightings show no such gap on bicycle (`lloyd_trace` 0.782 train, 0.777 test).

Uncompressed reference (`uncompressed` rows): garden 27.3150 dB test, 28.5197 dB train; bicycle
25.5682 dB test, 24.2350 dB train.

### GN-VQ iterations (`gn1_<config>_k<K>_s<seed>_<scene>.json`)

- At K = 65,536 every row stopped on the 1e-3 relative-drop rule, at iteration 10 (iteration 9 for
  bicycle at eps = 1e-2). At K = 4,096 and 16,384 GN-VQ ran into the 10-iteration cap on both scenes.
- The GN objective, warm start -> before quantization -> after quantization, for GN-VQ at K = 65,536
  seed 0: garden 1.2222e-04 -> 2.9381e-05 -> 3.8165e-05; bicycle 9.0730e-05 -> 2.3698e-05 ->
  2.8409e-05.
- The clip's per-cluster acceptance rejected more clusters every iteration: on garden seed 0, from 785
  at iteration 1 to 54,463 at iteration 10 (262,736 in total); on bicycle from 665 to 49,571 (220,728).
- The top-64 L2 share at iteration 1, K = 65,536: 0.431-0.433 on garden, 0.510-0.511 on bicycle.

### Validity and engineering checks

| Check | Result |
|---|---|
| CUDA smoke tests (`gn1_selftest.json`) | pass: `sh_basis` 6.50e-06, toy check 2.87%, end-to-end `pass`, `linalg_scale` pass |
| Render parity, both scenes | max abs difference 0.0 |
| GN metric | E0's cache reused on both scenes (neither job ran a GN pass), `M` finite |
| Lifted check (10,000 splats) | pass; sum excess / sum d_min 4.07e-13 garden, 2.82e-09 bicycle; worst excess / scale 1.99e-12, 9.89e-10 |
| Writer codes | `writer_codes_equal` True on all 40 compressed rows; `valid` True on all 42 rows |
| Batched linalg | no fallbacks in either job |
| Warm starts | `lloyd_wopa_area` K = 65,536 seeds 0-2 from the run-3 caches on both scenes (none reclustered); K = 4,096 and 16,384 clustered in the job |
| Both jobs | exit code 0, `gsplat_commit` `9b6c5babdf24` |

### Timings (`timings.json`, `gn1_meta_<scene>.json`, `gn1_results_<scene>.csv`)

| Step | Seconds |
|---|---|
| Restore inputs | 23.2 |
| Install (restored run-5 wheel, no build) | 170.5 |
| Smoke tests | 15.8 |
| MipNeRF360 data for both scenes | 149.0 |
| `gn_e1_garden` job | 6,450.4 |
| `gn_e1_bicycle` job | 4,455.3 |

The two jobs ran in parallel, one T4 each. Inside them:

- `lloyd_trace` / `lloyd_c3dgs` clustering at K = 65,536: 651.7-655.7 s per run on garden, 266.6-451.8 s
  on bicycle, six runs per scene, 3,921.6 s and 1,997.3 s in total: 60.9% of garden's job and 44.9% of
  bicycle's.
- `lloyd_wopa_area` clustering: 39.8 / 37.7 s at K = 4,096 and 165.2 / 158.4 s at K = 16,384 (garden /
  bicycle).
- GN-VQ per row: 143.5-156.7 s on garden and 138.9-153.7 s on bicycle at K = 65,536; 62.7 / 61.8 s at
  K = 4,096; 80.8 / 78.8 s at K = 16,384.

### What E1 settles, and what it does not

- **G1 failed, as pre-registered, on the size rule, on all six seeds.** The PSNR condition was met with
  room to spare; the bytes were not.
- **Equal K is not equal bytes.** GN-VQ's codes compress worse than `lloyd_wopa_area`'s at the same
  quantizer step, and the premium grows with K. The rate-distortion secondary of Amendment 5 d was
  pre-registered for exactly this case, and it favours GN-VQ on both scenes: bicycle's BD-rate is
  -9.84%, and on garden every GN-VQ point is above every `lloyd_wopa_area` point in PSNR.
- **Limits of that evidence:** it is seed 0, two scenes, and three points per curve, and those two
  scenes are the ones GN-VQ was designed and tuned on. On bicycle the gain also shrinks markedly from
  train to test views. E1 cannot say whether the rate-distortion advantage holds on scenes GN-VQ has
  not seen.
- **Exploratory, one seed:** without the clip GN-VQ dominated `lloyd_wopa_area` on both scenes, and at
  eps = 1e-2 on bicycle; a scalar `tr(M)` weight beat `lloyd_wopa_area` at about equal bytes on all six
  seeds (post hoc).

## 10. E2: GN-VQ against `lloyd_wopa_area` on 9 held-out scenes (`bench/gn-vq`) — G2a passed

**G2a passed.** E2's GN-VQ (ridge eps = 1e-2, at most 20 iterations) won on all 9 held-out scenes, and
its mean BD-rate against `lloyd_wopa_area` over the 9 is **-5.37%**, inside the pre-registered -5%.
Every scene had a defined BD-rate, so no Amendment 8 substitute entered the mean. The margins are
thin: the mean clears the threshold by 0.37 percentage points, and treehill wins at -1.20%.

**H2b (reported, its own verdict) passed as computed, 9 of 9 against `lloyd_trace`, but treehill's
win is an artifact of the cubic fit, so any claim uses 8 of 9**, which still meets H2b's 7.

The rules were fixed before any E2 code or data: the variant, the scenes, the configs, G2a and H2b in
Amendment 7 (`3e074edd`); G2a's mean over all 9 scenes with substitutes, and the exploratory
eps = 1e-4 rows, in Amendment 8 (`3061cd6f`). The verdicts below are `bench/gn/g2.py`'s, written by
the notebook into `gn2_g2.json`; nothing here re-judges them. Items marked **post hoc** were computed
after the results, by `bench/gn/bd_sensitivity.py` (`kaggle/gn_e2/bd_sensitivity.json`) or from the
result CSVs.

### Verdicts (`gn2_g2.json`)

Degree-3 Bjontegaard over the four points per curve (K = 1,024, 4,096, 16,384 and 65,536, seed 0):
raw bytes of the compressed directory against test PSNR of the full compressed pipeline. A negative
BD-rate and a positive BD-PSNR favour GN-VQ.

| | G2a (gate): GN-VQ vs `lloyd_wopa_area` | H2b (reported): GN-VQ vs `lloyd_trace` |
|---|---|---|
| Verdict | **pass** | pass as computed; 8 of 9 counted (below) |
| Wins | 9 of 9 (needs 8) | 9 of 9 (needs 7) |
| Mean BD-rate over the 9 held-out scenes | **-5.37%** (needs at most -5%) | -6.64% (reported; no condition) |
| Scenes with a defined BD-rate / substituted | 9 / 0 | 9 / 0 |

| Scene | G2a BD-rate | G2a BD-PSNR (dB) | G2a | H2b BD-rate | H2b BD-PSNR (dB) | H2b |
|---|---|---|---|---|---|---|
| stump | -7.32% | +0.2014 | win | -5.61% | +0.1272 | win |
| bonsai | -5.79% | +0.4485 | win | -5.53% | +0.3752 | win |
| counter | -4.06% | +0.2660 | win | -3.10% | +0.1921 | win |
| kitchen | -5.60% | +0.3837 | win | -4.88% | +0.2971 | win |
| room | -7.76% | +0.3038 | win | -6.61% | +0.2348 | win |
| treehill | -1.20% | +0.0320 | win | -24.04% | -0.0303 | win |
| flowers | -5.26% | +0.0767 | win | -3.46% | +0.0393 | win |
| train | -4.92% | +0.1130 | win | -2.35% | +0.0446 | win |
| truck | -6.45% | +0.1479 | win | -4.19% | +0.0797 | win |

Every scene was decided by its BD-rate; the BD-PSNR is reported alongside, as Amendment 7 h asks.

### Treehill against `lloyd_trace`: a fit artifact, so H2b counts 8 of 9

- **GN-VQ's treehill curve is not monotone:** its test PSNR is 23.2453 dB at K = 16,384
  (14,701,250 B) and 23.2302 dB at K = 65,536 (15,843,536 B). It is the only one of E2's 46 curves
  whose PSNR does not rise with K (`bd_sensitivity.json`, `monotonicity`).
- **The cubic's two measures disagree:** BD-rate -24.04% says GN-VQ needs fewer bytes, BD-PSNR
  -0.0303 dB says it has the lower PSNR. This is the only one of the 39 comparisons in `gn2_g2.json`
  where BD-rate and BD-PSNR name different curves as the better one (`sign_disagreement`).
- **At equal K, `lloyd_trace` has the higher PSNR at all four K** (by 0.0489, 0.0477, 0.0279 and
  0.0444 dB), **but fewer bytes at only three:** at K = 65,536 it has 0.42% more bytes than GN-VQ, so
  it does not dominate there.
- With PCHIP instead of the cubic (post hoc, below) treehill is a loss on both measures: +9.34% and
  -0.0371 dB.

So treehill is not counted as a win: **H2b holds with 8 of 9 scenes**, and the mean BD-rate over the
other 8 is -4.47% (post hoc, the bundle's values; `held_out_without_flagged_scenes_post_hoc`). G2a has
no such case: none of its 9 comparisons is flagged under the cubic.

### The numbers of record, and how exactly they reproduce

`bd_sensitivity.py` recomputes every value in `gn2_g2.json` from the committed CSVs with `g2`'s own
functions. All 39 comparisons reproduce every outcome, deciding measure, win, mean-term source, sign,
count and verdict; the values agree within 1.77e-3 percentage points (BD-rate, at H2b treehill) and
1.60e-7 dB (BD-PSNR), inside the tolerances of 2e-3 pp and 1e-6 dB. They do not agree bit for bit,
because `g2`'s cubic is `np.polyfit` on uncentred PSNR (and on uncentred log10 bytes), whose last digits
depend on the machine's LAPACK. Against the exact interpolating cubic in 50-digit arithmetic, the
bundle's values are within 6.09e-4 pp and 1.46e-7 dB, and this machine's within 1.16e-3 pp and
1.13e-7 dB. **The bundle's values are the numbers of record**, and they are the ones quoted here; at the
precision quoted (0.01 pp, 0.0001 dB) the exact cubic gives the same figures.

### PCHIP instead of the cubic (post hoc)

The same BD-rate and BD-PSNR with a monotone piecewise-cubic interpolant (`scipy`'s
`PchipInterpolator` on the same axes, integrated exactly), and the same per-scene rule and Amendment 8
substitute:

| Scene | G2a cubic | G2a PCHIP | G2a PCHIP BD-PSNR (dB) | H2b cubic | H2b PCHIP | H2b PCHIP BD-PSNR (dB) |
|---|---|---|---|---|---|---|
| stump | -7.32% | -7.61% | +0.1632 | -5.61% | -5.90% | +0.1199 |
| bonsai | -5.79% | -6.39% | +0.4031 | -5.53% | -5.92% | +0.3440 |
| counter | -4.06% | -4.35% | +0.2606 | -3.10% | -3.39% | +0.1865 |
| kitchen | -5.60% | -6.00% | +0.3547 | -4.88% | -5.26% | +0.2942 |
| room | -7.76% | -9.00% | +0.2757 | -6.61% | -7.54% | +0.2207 |
| treehill | -1.20% | +0.51% | +0.0156 | -24.04% | +9.34% | -0.0371 |
| flowers | -5.26% | -5.46% | +0.0710 | -3.46% | -3.64% | +0.0427 |
| train | -4.92% | -5.89% | +0.0996 | -2.35% | -3.57% | +0.0488 |
| truck | -6.45% | -6.74% | +0.1357 | -4.19% | -4.49% | +0.0757 |
| **Wins / mean** | 9 / -5.37% | **8 / -5.66%** | | 9 / -6.64% | 8 / -3.38% | |

- **G2a with PCHIP: 8 of 9 wins, mean -5.66%**, which would still meet both of G2a's thresholds.
  Treehill's BD-rate becomes +0.51% while its BD-PSNR stays positive (+0.0156 dB); that is the only
  comparison PCHIP's two measures disagree on.
- H2b with PCHIP: 8 of 9 (treehill is the loss), mean -3.38%.
- On the other 8 held-out scenes PCHIP's BD-rate is more negative than the cubic's, in both
  comparisons.

### The shN stream alone (post hoc)

At equal K, a GN-VQ row and the `lloyd_wopa_area` row differ only in `shN.npz` (the codebook codes and
the labels); the PNG files have identical sizes, and `meta.json` differs by at most 2 bytes
(`file_bytes`). The `shN_bytes` column is **10.0-22.9% of `size_bytes`** across the 184 compressed rows
(mean 15.0%). BD-rate of GN-VQ against `lloyd_wopa_area` with `shN_bytes` as the rate:

| Scene | Cubic (`g2.bd_rate`) | PCHIP |
|---|---|---|
| stump | -39.23% | -40.02% |
| bonsai | -33.56% | -35.43% |
| counter | -24.55% | -25.56% |
| kitchen | -32.02% | -33.19% |
| room | -42.30% | -46.22% |
| treehill | -8.35% | +0.70% |
| flowers | -30.88% | -31.39% |
| train | -30.60% | -33.90% |
| truck | -36.19% | -37.26% |
| **Mean of the 9** | -30.85% | -31.37% |

- **Cubic:** -24.5% to -42.3% on the 8 scenes other than treehill; treehill -8.35%.
- **PCHIP:** -25.6% to -46.2% on those 8; treehill +0.70%.
- The mean is about -31% either way. The whole-file BD-rates above are smaller in magnitude because the
  other 77-90% of each file's bytes (the PNG files and `meta.json`) have the same size for both configs.

### Equal K (Amendment 7 h, reported)

PSNR difference of GN-VQ against the baseline at the same K, then GN-VQ's bytes against the
baseline's; "dominates" means bytes `<=` and PSNR `>=` (`gn2_g2.json`, `reported.equal_k`).

Against `lloyd_wopa_area`:

| Scene | K = 1,024 | K = 4,096 | K = 16,384 | K = 65,536 |
|---|---|---|---|---|
| stump | +0.4459 / +0.52% | +0.2120 / +0.14% | +0.1590 / +0.16% | +0.1221 / +0.07% |
| bonsai | +1.0013 / +0.55% | +0.5797 / +0.26% | +0.4062 / +0.12% | +0.2696 / +0.17% |
| counter | +0.6757 / +0.32% | +0.4305 / +0.12% | +0.2572 / +0.15% | +0.1439 / +0.29% |
| kitchen | +0.8855 / +0.51% | +0.5112 / +0.16% | +0.3466 / +0.07% | +0.2408 / +0.23% |
| room | +0.6782 / +0.59% | +0.4078 / +0.29% | +0.2819 / +0.03% | +0.1526 / -0.09% (dominates) |
| treehill | +0.0617 / +0.44% | +0.0192 / +0.20% | +0.0284 / +0.17% | -0.0083 / -0.04% |
| flowers | +0.1372 / +0.31% | +0.0900 / +0.13% | +0.0701 / +0.14% | +0.0600 / +0.18% |
| train | +0.4414 / +0.85% | +0.1859 / +0.42% | +0.0798 / +0.14% | +0.0513 / +0.10% |
| truck | +0.4314 / +0.55% | +0.2355 / +0.21% | +0.1270 / +0.08% | +0.0632 / -0.15% (dominates) |
| garden (dev) | +0.4074 / +0.33% | +0.2982 / +0.10% | +0.2363 / +0.16% | +0.1949 / +0.55% |
| bicycle (dev) | +0.3642 / +0.56% | +0.1835 / +0.20% | +0.1235 / +0.15% | +0.0897 / -0.01% (dominates) |

Against `lloyd_trace`:

| Scene | K = 1,024 | K = 4,096 | K = 16,384 | K = 65,536 |
|---|---|---|---|---|
| stump | +0.2005 / +0.25% | +0.1522 / +0.26% | +0.1266 / +0.41% | +0.1013 / +0.56% |
| bonsai | +0.7387 / +0.40% | +0.5130 / +0.23% | +0.3817 / +0.12% | +0.1788 / +0.24% |
| counter | +0.4395 / +0.27% | +0.2950 / +0.14% | +0.1953 / +0.20% | +0.1124 / +0.38% |
| kitchen | +0.5534 / +0.17% | +0.4117 / +0.08% | +0.2983 / +0.13% | +0.2045 / +0.15% |
| room | +0.5106 / +0.45% | +0.3379 / +0.26% | +0.2253 / +0.02% | +0.1156 / -0.08% (dominates) |
| treehill | -0.0489 / +0.33% | -0.0477 / +0.16% | -0.0279 / +0.08% | -0.0444 / -0.42% |
| flowers | +0.0723 / +0.27% | +0.0613 / +0.17% | +0.0444 / +0.18% | +0.0334 / +0.21% |
| train | +0.1322 / +0.49% | +0.0876 / +0.33% | +0.0424 / +0.18% | +0.0312 / +0.11% |
| truck | +0.2009 / +0.47% | +0.1250 / +0.24% | +0.0765 / +0.15% | +0.0412 / +0.14% |
| garden (dev) | +0.3092 / +0.23% | +0.2445 / +0.12% | +0.1936 / +0.23% | +0.1428 / +0.47% |
| bicycle (dev) | +0.1640 / +0.45% | +0.1096 / +0.27% | +0.0730 / +0.21% | +0.0516 / -0.09% (dominates) |

- Against `lloyd_wopa_area`, GN-VQ has the higher PSNR in every cell except treehill at K = 65,536
  (-0.0083 dB), and the gain falls as K grows on every scene except treehill. Its byte premium is
  largest at K = 1,024 (+0.31% to +0.85% on the held-out scenes) and between -0.15% and +0.29% at
  K = 65,536; it dominates in two held-out cells (room and truck at K = 65,536).
- Against `lloyd_trace`, GN-VQ has the higher PSNR in every cell except the four of treehill.

### Train and test views against `lloyd_trace` (post hoc)

At K = 65,536, GN-VQ's PSNR gain over `lloyd_trace` on the test views and on the train views (`PSNR`
and `train_PSNR` columns), and their ratio:

| Scene | Train views | Data factor | Test PSNR gain (dB) | Train PSNR gain (dB) | Test / train |
|---|---|---|---|---|---|
| stump | 109 | 4 | +0.1013 | +0.1933 | 0.52 |
| bonsai | 255 | 2 | +0.1788 | +0.2300 | 0.78 |
| counter | 210 | 2 | +0.1124 | +0.1555 | 0.72 |
| kitchen | 244 | 2 | +0.2045 | +0.2396 | 0.85 |
| room | 272 | 2 | +0.1156 | +0.1894 | 0.61 |
| treehill | 123 | 4 | -0.0444 | +0.0420 | -1.06 |
| flowers | 151 | 4 | +0.0334 | +0.1049 | 0.32 |
| train | 263 | 1 | +0.0312 | +0.0669 | 0.47 |
| truck | 219 | 1 | +0.0412 | +0.0739 | 0.56 |
| garden (dev) | 161 | 4 | +0.1428 | +0.1975 | 0.72 |
| bicycle (dev) | 169 | 4 | +0.0516 | +0.0574 | 0.90 |

The test gain is below the train gain on all 11 scenes. **The three lowest ratios are treehill
(-1.06), flowers (0.32) and train (0.47)**; the other eight are 0.52-0.90. On treehill GN-VQ gains on
the train views (+0.0420 dB) and loses on the test views (-0.0444 dB). `M` is accumulated over the train
views, so this is the out-of-sample gap E1 saw on bicycle, here measured against the scalar weighting
built from the same `M`.

### Reported extras (Amendments 7 h and 8 b)

- **Against `upstream_l1`:** BD-rate -3.93% (treehill) to -10.48% (kitchen) on 8 held-out scenes.
  Room's curves share no PSNR range, so its BD-rate is undefined; its BD-PSNR is +0.3691 dB and its
  substitute (a, GN-VQ reaching `upstream_l1`'s best PSNR for fewer bytes) is -11.40%. Garden -11.74%,
  bicycle -6.54%.
- **Development scenes:** garden -7.95% against `lloyd_wopa_area` and -6.44% against `lloyd_trace`;
  bicycle -5.55% and -3.37%; all wins, never read by a verdict.
- **Exploratory eps = 1e-4 (Amendment 8 b):** against `lloyd_wopa_area`, -8.17% on garden and -5.53% on
  bicycle. E2's variant (eps = 1e-2) against it: +0.22% on garden (BD-PSNR -0.0042 dB) and -0.18% on
  bicycle (+0.0075 dB). On the development curves the ridge moved the BD-rate by less than a quarter of
  a percentage point.

### GN-VQ iterations (`gn2_gn_vq_k<K>_s0_<scene>.json`, results CSVs)

- K = 65,536: 9-10 iterations, every row stopped on the 1e-3 relative-drop rule. K = 16,384: 12-15,
  all on the relative drop.
- K = 4,096: 15-20; train, garden and bicycle stopped on the 20-iteration cap. K = 1,024: 20 on every
  scene, on the cap everywhere except room, whose 20th iteration met the relative drop.
- Time per GN-VQ row: 109.6-112.5 s at K = 1,024, 92.9-120.7 s at 4,096, 94.3-115.4 s at 16,384 and
  134.6-161.6 s at 65,536.
- The clip held: every GN-VQ row's quantizer range (`quant_mins`, `quant_maxs`) lies inside its warm
  start's (`warm_quant_*`).

### Validity and engineering checks

| Check | Result |
|---|---|
| CUDA smoke tests (`gn2_selftest.json`) | pass: `sh_basis` 6.50e-06, toy check 2.87%, end-to-end `pass`, `linalg_scale` pass |
| Checkpoints | all 11 sha1s equal Amendment 7's pins, checked before any install (21.4 s) and in every job |
| Render parity, 11 scenes | max abs difference 0.0 |
| GN metric | computed on the 9 held-out scenes (8.4-32.8 s per pass); garden and bicycle reused a restored cache; `M` finite |
| Lifted check (10,000 splats), 11 scenes | pass; sum excess / sum d_min at most 1.32e-08, worst excess / scale at most 9.56e-09, no splat over the tolerance |
| Rows | 195 (17 per scene, plus 8 exploratory); `writer_codes_equal` True on all 184 compressed rows, `valid` True on all 195 |
| Completeness | every meta's `missing_rows` empty; `missing_exploratory` empty on garden and bicycle |
| Jobs | all 13 (11 scene jobs, 2 exploratory) exit code 0; none skipped by the start cutoff |
| Warm starts | `lloyd_wopa_area` K = 65,536 from the run-3 / run-4 caches on the 9 MipNeRF360 scenes, reclustered on train and truck; K = 1,024-16,384 clustered in the job |
| Batched linalg | no fallbacks in any job |
| Install | the run-5 wheel reused (169.3 s, no wheel build); `gsplat_commit` `515ca4a1ed01` |

### Timings (`timings.json`, `gn2_meta_<scene>.json`)

| Scene | Data factor | Download | GN pass | Job (`timings.json`, queue) | Job (meta) |
|---|---|---|---|---|---|
| stump | 4 | 38.4 | 8.4 | 2,475.3 | 2,458.4 |
| bonsai | 2 | 67.6 | 30.5 | 3,465.4 | 3,456.8 |
| counter | 2 | 53.5 | 31.6 | 3,525.3 | 3,506.7 |
| kitchen | 2 | 78.1 | 32.8 | 3,600.3 | 3,586.9 |
| room | 2 | 89.3 | 29.7 | 3,780.3 | 3,766.3 |
| treehill | 4 | 62.1 | 9.5 | 2,325.2 | 2,318.4 |
| flowers | 4 | 72.7 | 12.8 | 2,580.3 | 2,573.1 |
| train | 1 | 500.0 | 16.2 | 2,985.3 | 2,975.0 |
| truck | 1 | 233.2 | 15.8 | 2,940.3 | 2,923.0 |
| garden | 4 | 73.5 | reused | 2,640.3 | 3,315.9 |
| bicycle | 4 | 63.0 | reused | 2,190.1 | 2,877.8 |

Seconds. The queue timing is wall time at the queue's 15 s poll, so up to 15 s late. The meta's job time
adds up over invocations, so for garden and bicycle it includes the exploratory job too (690.1 s and
720.1 s in `timings.json`). Session steps: checkpoint sha1s 21.4 s, restore 24.5 s, install 169.3 s,
smoke tests 15.7 s. The longest job was room.

### What E2 settles, and what it does not

- **G2a passed, as pre-registered, on 9 scenes GN-VQ was not designed or tuned on.** This is the gate
  Amendment 7 moved to held-out scenes after G1 failed.
- **The margins are thin.** The mean clears -5% by 0.37 pp, and treehill's -1.20% turns into +0.51%
  under PCHIP (post hoc), which would still leave 8 of 9 wins and a mean of -5.66%.
- **H2b: the per-splat matrix beats the scalar `tr(M)` weighting on 8 of 9 held-out scenes.** Treehill is
  not one of them, whatever the cubic's BD-rate says.
- **The gain is in the codebook stream:** about -31% BD-rate on `shN.npz` alone (post hoc), diluted to
  the whole-file figures by the 77-90% of each file whose size does not change.
- **Limits:** k-means seed 0 only, four points per curve, and the train-to-test transfer against
  `lloyd_trace` is weakest on treehill, flowers and train (post hoc).

Next: `PREREG_GN.md` Amendment 9 (`1ae0c05b`, written after these results) pre-registers E2b, an
exploratory run on treehill, flowers and train with garden as the control, which adds an isotropic
floor to the metric in the assignment as well as the update and selects its strength by train-view
cross-validation. From that amendment on, treehill, flowers and train are development scenes.
