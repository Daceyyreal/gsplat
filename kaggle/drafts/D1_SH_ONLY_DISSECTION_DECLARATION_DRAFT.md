# DRAFT D1 — declaration: the lead before fine-tuning, split into DC observation-weighting and AC Gram structure (exploratory)

**Status: ready to commit on Dace's go; then committed and pushed alone, before any of its code. Not in PREREG; nothing
run.** Supersedes `kaggle/drafts/SH_ONLY_CHECK_DECLARATION_DRAFT.md` (first session's per-splat-DC design), on Dace's
framing. Numbers quoted are FINDINGS' (sections 19, 21, 22, 25, 26).

## What it is

**Exploratory, no verdict, on the seven gate scenes (not held out), before fine-tuning.** A dissection of P1 (E5:
0.3558 dB) into the part that comes from weighting the **DC** coefficient by what the training views observed and the
part that comes from the **AC** (SH degree >= 1) Gram structure. Its second purpose: the AC-only lead is what a host
that clusters only the AC (MesonGS, MesonGS++) could show at most, so it decides whether a second host is worth building.

## Rows per process (E5's fork, protocol and processes; no fine-tuning)

| Row | Colour VQ | Metric per colour-quantized splat (16 x 16, coefficient 0 = DC) |
|---|---|---|
| `c3dgs` (exists) | C3DGS's own `vq_features` | (C3DGS's importance-weighted squared error) |
| `ogc_gram` (exists) | OGC's `gram_kmeans`, `metric="gram"` | the 16-probe Gram estimate G_i, as E5-E7 |
| **`ogc_gram_ac`** (new) | OGC's `gram_kmeans` on the same inputs | **G_i with its DC row and column replaced:** DC weight = the scene-wide mean of G_i[0, 0] (the same for every splat, so no per-splat observation weighting on DC), DC-AC cross terms 0, AC block G_i[1:, 1:] unchanged |

- **"DC as C3DGS" means:** the DC is still part of every codeword (C3DGS's default container, `color_compress_non_dir`
  on), but it is weighted the same for every splat, as C3DGS's squared error weights it within a splat. All three rows
  use the same container, so **their bytes are directly comparable** (unlike an AC-only codebook with a per-splat DC).
- **Confound, stated now:** dropping the DC-AC cross terms changes more than the DC weighting; their share of G_i's
  Frobenius norm is reported per scene (descriptive).
- C3DGS's own `color_compress_non_dir=False` switch is not used: it is unreachable from C3DGS's command line (a
  `store_true` flag defaulting to True, `arguments/__init__.py:30-31`, `:81`) and, called directly, decodes the first AC
  coefficient as the DC (`bench/gn/test_gn.py::test_c3dgs_color_compress_non_dir_false_loses_the_dc`, which reproduces it).

## The R9 regression contract (R13, added 2026-11-08, fourth session)

**D1 has no AC-only container.** All three rows keep C3DGS's default container (the DC is a coefficient of every
codeword); `ogc_gram_ac` changes the *metric* only. So "the decoded DC is the true DC" cannot hold for D1's rows: their
DC is quantized with the codeword, as in E5. What D1 must guarantee, and what any later AC-only path must guarantee, are
two separate contracts, drafted as tests in `kaggle/drafts/d1_regression_tests_draft.py` (run by hand on 2026-11-08
against `F:\c3dgs_src_2a234af5` at `2a234af5`: 3 passed; not collected by the gate):

1. **D1's rows** (`test_d1_rows_use_the_default_container`): D1's hook wraps `compress_color` and **refuses
   `color_compress_non_dir=False`**; the call it makes decodes 16 coefficients and kept splats exactly.
2. **Any AC-only container** (an AC codebook with a per-splat DC: a MesonGS-like second host, or a later D1 variant):
   the decoded model has 16 coefficients, **the decoded DC is the true per-splat DC**, the 15 AC coefficients come from
   the AC codebook, kept splats decode exactly (`assert_ac_only_contract`).
   `test_unfixed_c3dgs_branch_fails_the_ac_only_contract` shows C3DGS's branch at `2a234af5` failing it (R9 as a failing
   contract); `test_reference_ac_only_decode_passes_the_contract` shows a sketch of the fork change passing it.

The three tests move into `bench/gn/test_gn.py` with D1's code, after D1's declaration is committed alone.

## The readout (each a BD-PSNR over the seed-0 curve, E5's registered fit, with section 28's PCHIP beside it)

1. **Full vs C3DGS:** `ogc_gram` against `c3dgs`, which should reproduce E5's P1 (new runs; seeded C3DGS runs are not
   reproduced on the GPU, CLAIMS E2, so agreement is within cross-run noise, not exact).
2. **AC-only vs C3DGS:** `ogc_gram_ac` against `c3dgs` (**P_AC**): the AC Gram structure with DC weighted flat.
3. **Full vs AC-only:** `ogc_gram` against `ogc_gram_ac`: what weighting DC by observation adds.

Per scene and mean, the j = 0 seed pair for noise (`SD_pool`, `SE` = `SD_pool` / sqrt(n)); 1 + 3 is not exactly 2
(BD values do not add across different curves); the difference is reported.

## What would make a second host not worth building (fixed now)

- **Training-free rows:** worth building only if **mean P_AC >= 0.10 dB and P_AC > 0 on at least 5 of 7 scenes**.
- **Fine-tuned rows:** informative only if **mean P_AC >= 0.25 dB**, because C3DGS's fine-tuning kept 0.1195 of the lead
  (E6's P_FT against P1 replicated, section 22 b), and 0.1195 x 0.25 = 0.030 dB is about the smallest mean effect E7's
  within-process noise detects (superiority threshold 0.016-0.032 dB at `SE` 0.0056-0.0111, section 26).
  **This is a planning assumption, not a measurement (R14):** it assumes the AC part of the lead shrinks under
  fine-tuning by the same fraction as the full lead. It might not: 0.1195 is a ratio of means whose per-scene values
  run from -0.017 to +0.331 (section 22 b), and E7 found neither the colour grids' re-fit nor codebook training
  explains the shrink (CLAIMS H1, H2), so it comes from parameters that train alongside (opacity, geometry, the DC
  itself), which need not erode the DC and AC parts in proportion.
- Above both: MesonGS for both regimes; between: training-free rows only; below 0.10 dB: no second host.

## Cost

Each process like E5's j = +-1 processes (four unfine-tuned rows, no fine-tuning; `bench/gn/e5_estimate.py`): the seven
gate scenes **9.1-12.9 h of jobs, two sessions** with the cached cu130 wheel.

## Fork code it needs (none written yet)

**Files and tests, as listed for R13 (2026-11-08):** C3DGS's own files: **none changed** for D1's three rows (the
default container; the hook refuses the `False` branch). Only an AC-only container would need fork changes to
`scene/gaussian_model.py` (a colour mode with a per-splat DC beside the indexed AC table: `set_color_indexed`,
`get_features`, the npz save and load, fine-tuning's parameter groups) and `compression/vq.py` (`compress_color`'s
`False` branch handing the DC over); D1 does not add one. Ours: the files below. Tests: the three of
`kaggle/drafts/d1_regression_tests_draft.py`, plus the list under "Tests" below.

- `kaggle/d1_hooks.py`: E5's colour-call injection (`kaggle/e5_hooks.py`) with the third row, and the guard on
  `color_compress_non_dir` (contract 1); `bench/gn/ogc_derived.py` (or
  OGC's `gram_kmeans` from the private dataset or URL, Amendment 18 note 3's source chain) receives the modified
  matrices; the matrix edit itself in a new function in `bench/gn/metric_store.py` (`ac_only_metric(G, dc_weight)`).
- `kaggle/d1_scenes.py` (rows and processes) and `kaggle/build_gn_d1_bench.py` (the notebook), from E5's.
- **Licence headers:** any new file that adapts OGC's code carries `ogc_derived.py`'s PolyForm Noncommercial header and
  both "Required Notice" lines verbatim; files that only call it carry none. No OGC file is committed, pushed or bundled.
- **Tests (before any session):** the edited matrix is PSD, its AC block equals G_i's, its DC weight is constant across
  splats, its cross terms are 0; `ogc_gram_ac` with the edit disabled equals `ogc_gram` bit for bit; the three rows write
  the same container layout; R13's contract tests (above; skipped without `GN_C3DGS_SRC`, so the gate log's skip list
  must show them not skipped); a CPU dry run of one stand-in scene through every stage (as E5's).
- **Code identity:** every session runs one pinned commit; no run-path file changes between its sessions.
