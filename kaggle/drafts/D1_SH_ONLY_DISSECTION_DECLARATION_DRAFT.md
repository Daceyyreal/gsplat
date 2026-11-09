# DRAFT D1 — declaration: the lead before fine-tuning, split into DC observation-weighting and AC Gram structure (exploratory)

**Status: ready to commit on Dace's go; then committed and pushed alone, before any of its code. Not in PREREG; nothing
run.** Supersedes `kaggle/drafts/SH_ONLY_CHECK_DECLARATION_DRAFT.md` (first session's per-splat-DC design), on Dace's
framing. Numbers quoted are FINDINGS' (sections 19, 21, 22, 25, 26). Revised 2026-11-09 (R18): the main arm is stated as a
metric dissection, an optional AC-only pair is added, the second-host thresholds move onto that pair, and both versions
are costed.

## What it is

**Exploratory, no verdict, on the seven gate scenes (not held out), before fine-tuning.** Two arms:

- **The main arm, a metric dissection inside the joint container.** A split of P1 (E5: 0.3558 dB) into the part that
  comes from weighting the **DC** coefficient by what the training views observed and the part that comes from the
  **AC** (SH degree >= 1) Gram structure. All its rows quantize the same joint 48-dimensional colour vector (16
  coefficients x 3 channels) into C3DGS's default container, so their bytes are the same kind of bytes; only the
  metric's DC block differs. **It is not a MesonGS proxy:** a host that clusters only the AC (MesonGS, MesonGS++) stores
  the DC per splat outside the codebook, and the main arm does not.
- **The optional pair, a true AC-only container** (`ogc_ac_only`, `c3dgs_ac_only`): the DC stored per splat outside the
  codebook, as MesonGS stores it before its RAHT, and the 15 AC coefficients clustered. **The second-host decision rests
  on this pair only.**

## Rows per process (E5's fork, protocol and processes; no fine-tuning)

| Row | Arm | Colour VQ | Container | Metric per colour-quantized splat (coefficient 0 = DC) |
|---|---|---|---|---|
| `c3dgs` (exists) | main | C3DGS's own `vq_features` on 48 dims | joint (C3DGS's default) | (C3DGS's importance-weighted squared error) |
| `ogc_gram` (exists) | main | OGC's `gram_kmeans`, `metric="gram"` | joint | the 16-probe Gram estimate G_i (16 x 16), as E5-E7 |
| **`ogc_gram_ac`** (new) | main | OGC's `gram_kmeans` on the same inputs | joint | **G_i with its DC row and column replaced:** DC weight = the scene-wide mean of G_i[0, 0] (the same for every splat), DC-AC cross terms 0, AC block G_i[1:, 1:] unchanged |
| **`c3dgs_ac_only`** (optional) | pair | C3DGS's own `vq_features` on the 45 AC dims (its `color_compress_non_dir=False` clustering) | AC-only (below) | (C3DGS's importance-weighted squared error on the AC) |
| **`ogc_ac_only`** (optional) | pair | OGC's `gram_kmeans` on the 45 AC dims | AC-only (below) | the AC block G_i[1:, 1:] (15 x 15) |

- **Main arm, "DC as C3DGS" means:** the DC is still part of every codeword (`color_compress_non_dir` on), but it is
  weighted the same for every splat. The three main rows use the same container, so **their bytes are directly
  comparable**.
- **Confound, stated now:** dropping the DC-AC cross terms changes more than the DC weighting; their share of G_i's
  Frobenius norm is reported per scene (descriptive).
- **The pair's AC-only container, exactly:** in the same `.npz` written by C3DGS's `save_npz` with quantization on:
  - **DC:** one row per splat, unindexed: `features_dc` of shape (n, 1, 3), int8 under C3DGS's per-tensor DC grid
    (`features_dc_qa`'s scale and zero point, saved as `features_dc_scale` and `features_dc_zero_point`, as C3DGS saves
    them). No codebook, no transform and no entropy coding beyond the `.npz`'s own: MesonGS's RAHT is **not** applied,
    so the pair's DC bytes are not MesonGS's.
  - **AC:** the codebook plus the kept splats' own rows (C3DGS's `join_features` layout), `features_rest` of shape
    (rows, 15, 3), int8 under C3DGS's AC grid, with `feature_indices` (int32, one per splat) as C3DGS stores them.
  - **Bytes** are the `.npz` file's size, as in E5, so the DC's 3 bytes per splat (before the `.npz`'s compression) are
    counted. Both pair rows write a byte-identical DC array (same values, same grid), so the pair compares the two AC
    quantizers at equal DC; the pair's bytes are not comparable with the main arm's.
  - **Decode:** the fork's AC-only colour mode, which returns the per-splat DC concatenated with the indexed AC rows
    (the reference decode of `kaggle/drafts/d1_regression_tests_draft.py`). C3DGS's own `False` branch is never used
    unfixed: it is unreachable from C3DGS's command line (a `store_true` flag defaulting to True,
    `arguments/__init__.py:30-31`, `:81`) and, called directly, decodes the first AC coefficient as the DC
    (`bench/gn/test_gn.py::test_c3dgs_color_compress_non_dir_false_loses_the_dc`).

## The R9 regression contract (R13, 2026-11-08; part of the plan from 2026-11-09)

Two contracts, drafted as tests in `kaggle/drafts/d1_regression_tests_draft.py` (run by hand on 2026-11-08 against
`F:\c3dgs_src_2a234af5` at `2a234af5`: 3 passed):

1. **The main arm** (`test_d1_rows_use_the_default_container`): D1's hook wraps `compress_color` and **refuses
   `color_compress_non_dir=False`** for the main rows; the call decodes 16 coefficients and kept splats exactly.
2. **The pair's AC-only container** (`assert_ac_only_contract`): 16 decoded coefficients, **the decoded DC is the true
   per-splat DC**, the 15 AC coefficients come from the AC codebook, kept splats decode exactly.
   `test_unfixed_c3dgs_branch_fails_the_ac_only_contract` shows C3DGS's branch at `2a234af5` failing it;
   `test_reference_ac_only_decode_passes_the_contract` shows the reference decode passing it.

**In the plan:** the three tests move into `bench/gn/test_gn.py` in D1's code commit (after the declaration is committed
alone), with `test_reference_ac_only_decode_passes_the_contract` pointed at the fork's own AC-only decode instead of the
sketch. From then on they run in every gate (`kaggle/gate.sh` sets `GN_C3DGS_SRC`); they skip without it, so the gate
log's skip list must show them not skipped before any D1 commit.

## The readout (each a BD-PSNR over the seed-0 curve, E5's registered fit, with section 28's PCHIP beside it)

Main arm (descriptive dissection):

1. **Full vs C3DGS:** `ogc_gram` against `c3dgs`, which should reproduce E5's P1 (new runs; seeded C3DGS runs are not
   reproduced on the GPU, CLAIMS E2, so agreement is within cross-run noise, not exact).
2. **DC flat vs C3DGS:** `ogc_gram_ac` against `c3dgs` (**P_ACmetric**): the AC Gram structure with DC weighted flat,
   inside the joint container.
3. **Full vs DC flat:** `ogc_gram` against `ogc_gram_ac`: what weighting DC by observation adds.

Pair (only if run):

4. **AC-only OGC vs AC-only C3DGS:** `ogc_ac_only` against `c3dgs_ac_only` (**P_ACpair**).

Per scene and mean, the j = 0 seed pair for noise (`SD_pool`, `SE` = `SD_pool` / sqrt(n)); 1 + 3 is not exactly 2
(BD values do not add across different curves); the difference is reported.

## What would make a second host not worth building (fixed now; on the pair only)

**If the pair is not run, D1 makes no second-host decision**: P_ACmetric is a metric dissection in the joint container and
is not used for it.

- **Training-free rows:** a second host is worth building only if **mean P_ACpair >= 0.10 dB and P_ACpair > 0 on at
  least 5 of 7 scenes**.
- **Fine-tuned rows:** informative only if **mean P_ACpair >= 0.25 dB**, because C3DGS's fine-tuning kept 0.1195 of the
  lead (E6's P_FT against P1 replicated, section 22 b), and 0.1195 x 0.25 = 0.030 dB is about the smallest mean effect
  E7's within-process noise detects (superiority threshold 0.016-0.032 dB at `SE` 0.0056-0.0111, section 26).
  **This is a planning assumption, not a measurement (R14):** it assumes the AC part of the lead shrinks under
  fine-tuning by the same fraction as the full lead. It might not: 0.1195 is a ratio of means whose per-scene values
  run from -0.017 to +0.331 (section 22 b), and E7 found neither the colour grids' re-fit nor codebook training
  explains the shrink (CLAIMS H1, H2), so it comes from parameters that train alongside (opacity, geometry, the DC
  itself), which need not erode the DC and AC parts in proportion.
- Above both: MesonGS for both regimes; between: training-free rows only; below 0.10 dB: no second host.

## Cost (`bench/gn/e5_estimate.py`, four processes per scene as E5's, none fine-tuned; seven gate scenes)

| Version | Rows per process | Jobs, low-high | Largest scene (room) | Sessions |
|---|---|---|---|---|
| The first D1 draft's figure | priced as `c3dgs` + three OGC-sized rows | 9.1-12.9 h | 1.52-2.23 h | two |
| **Main only** | `c3dgs`, `ogc_gram`, `ogc_gram_ac` (two OGC-sized rows) | **7.4-10.9 h** | 1.22-1.89 h | two |
| **Main + pair** | the main three + `ogc_ac_only` (an OGC-sized row) + `c3dgs_ac_only` (C3DGS's own VQ time) | **11.5-15.5 h** | 1.93-2.67 h | two |

- The first draft's 9.1-12.9 h priced one OGC row more than D1's main arm has; corrected above.
- `c3dgs_ac_only` is priced at C3DGS's measured colour-VQ time per process (128-139 s in the estimate's units, against
  29-98 s for one OGC row on these scenes), with a row's save, evaluation and decode as every other row; the AC-only
  clustering has 45 dimensions instead of 48, so this is slightly high.
- With the cached cu130 wheel; each session's setup as E5-E7's restored case.

## Fork code it needs (none written yet)

- **C3DGS's own files:** none changed for the main arm (the hook refuses the `False` branch). **Only with the pair:**
  `scene/gaussian_model.py` (an AC-only colour mode: a per-splat DC beside the indexed AC table in `set_color_indexed`
  and `get_features`, the `.npz` save and load of the unindexed DC, fine-tuning's parameter groups left as C3DGS's since
  D1 does not fine-tune) and `compression/vq.py` (`compress_color`'s `False` branch handing the true DC over).
- `kaggle/d1_hooks.py`: E5's colour-call injection (`kaggle/e5_hooks.py`) with the new rows, and the guard on
  `color_compress_non_dir` (contract 1); `bench/gn/ogc_derived.py` (or OGC's `gram_kmeans` from the private dataset or
  URL, Amendment 18 note 3's source chain) receives the modified matrices; the matrix edit itself in a new function in
  `bench/gn/metric_store.py` (`ac_only_metric(G, dc_weight)`), and the AC block for the pair (`ac_block(G)`).
- `kaggle/d1_scenes.py` (rows and processes, the pair behind one switch) and `kaggle/build_gn_d1_bench.py` (the
  notebook), from E5's.
- **Licence headers:** any new file that adapts OGC's code carries `ogc_derived.py`'s PolyForm Noncommercial header and
  both "Required Notice" lines verbatim; files that only call it carry none. No OGC file is committed, pushed or bundled.
- **Tests (before any session):** the edited matrix is PSD, its AC block equals G_i's, its DC weight is constant across
  splats, its cross terms are 0; `ogc_gram_ac` with the edit disabled equals `ogc_gram` bit for bit; the three main rows
  write the same container layout; with the pair, the two pair rows write byte-identical DC arrays and pass contract 2;
  R13's three contract tests in the gate (above); a CPU dry run of one stand-in scene through every stage (as E5's).
- **Code identity:** every session runs one pinned commit; no run-path file changes between its sessions.
