# DRAFT: FINDINGS section 26 (E7), not part of FINDINGS until filled

**Status: DRAFT skeleton, written 2026-10-29 before E7 S7.** It holds no E7 value: every number is a placeholder,
a name in angle brackets, filled after S11 by `bench/gn/check_s26.py --fill` from all eleven committed E7 bundles. It is not part of
`kaggle/FINDINGS.md` until it is filled, checked by `bench/gn/check_s26.py` and moved there (procedure: `kaggle/HANDOFF.md`,
"E7 write-up skeleton and check_s26"). The three parts below go to three places in FINDINGS: the summary paragraph to the
summaries at the top (after section 25's), the Sources entry to "## Sources" (after section 25's), and the section to the
end. Prose outside the placeholders may be reworded by hand after filling; every number must stay a filled string, and
`check_s26.py` must pass on the result. Each block quotes the PREREG lines it reports as `` `kaggle/PREREG_GN.md:N`: «...» ``;
the checker verifies each quote against that line.

## Summary paragraph (goes after section 25's summary)

**E7 (section 26, branch `bench/gn-vq`): the within-process dissection of why OGC's lead shrinks with fine-tuning
(Amendment 20 f-h, with notes 1-5), on E5's seven scenes.**
- **Δ_refit (both colour grids frozen minus moving): ⟨outcome_refit⟩,** ⟨fragile_refit⟩. **Δ_cb (the codewords frozen
  minus training, grids fixed): ⟨outcome_cb⟩,** ⟨fragile_cb⟩.
- ⟨summary_bdbar⟩; superiority and a TOST at δ = 0.07 dB, each Holm-corrected over the two primaries.
- **Disclosure:** the hypotheses were formed after E5's and E6's results on these scenes; E7 makes no held-out claim.

## Sources entry (goes after section 25's)

- Section 26: every number comes from the E7 session bundles, ⟨sources_bundles⟩, each unpacked unchanged, through
  `bench/gn/e7.py`; the exceptions are E5's and E6's values quoted beside E7's (secondary 4 and section 25's standard
  error) from their own bundles, and the quoted lines of `kaggle/PREREG_GN.md`. `bench/gn/check_s26.py` recomputes every
  number in the section from those files, with its own BD fit and tests beside the tool's and the sessions' summaries,
  and checks each against the text.

## 26. E7: a within-process dissection of why OGC's lead shrinks with fine-tuning (pre-registered) — ⟨headline⟩

**E7 is the test Amendment 20 pre-registered** (`53dde708`, with notes 1-5, all written before the data they govern).
- **The scenes and checkpoints:** E5's seven, INRIA's 30k checkpoints. `kaggle/PREREG_GN.md:3954`: «**Scenes:** E5's seven (bonsai, counter, kitchen, room, truck, drjohnson, playroom)»
- **The processes:** six per scene; the four full ones (j = -1, 0, +1 with seed 0, j = 0 with seed 1) compute every row,
  `j-1_p1` and `j+1_p1` arm R and the unfine-tuned pair only.
- **The rows:** `c3dgs` and `ogc_gram` with the shared pre-fine-tuning grid; `c3dgs_own` and `ogc_gram_own`, the same
  tables with each method's own grid; and per method five fine-tuned arms (5,000 iterations each, from the same forked
  copy): R (C3DGS's own fine-tuning), G (both colour grids frozen), CG (grids frozen and the K codewords frozen), K (only
  the codewords train) and O (each method's own grid, frozen), and in `p0` a replicate of R.
  `kaggle/PREREG_GN.md:3979`: «`c3dgs_own` and `ogc_gram_own`: the same tables, each saved with **its own grid**»
  `kaggle/PREREG_GN.md:3994`: «| **R, replicate** (`p0` only) | `c3dgs_ft_r2`, `ogc_gram_ft_r2` | as R | as R |»
- **Disclosure:** `kaggle/PREREG_GN.md:3936`: «E7's primaries test interventions within each scene; **E7 makes no»
  held-out claim.
- **The sources:** ⟨sources_bundles⟩. A split scene's seed-0 half and seed-1 half come from two sessions; later bundles
  carry each scene unchanged (notes 3-5).

### The verdict (Amendment 20 f, h)

`kaggle/PREREG_GN.md:4040`: «**Δ_refit,s = BD_G,s - BD_R,s:**»
`kaggle/PREREG_GN.md:4042`: «**Δ_cb,s = BD_CG,s - BD_G,s:**»
`kaggle/PREREG_GN.md:4109`: «The verdict is computed locally over every E7 session's committed bundle by E7's verdict module (k)»

- **Δ_refit: ⟨outcome_refit⟩;** ⟨fragile_refit⟩. Reading (20 h): ⟨reading_refit⟩.
- **Δ_cb: ⟨outcome_cb⟩;** ⟨fragile_cb⟩. Reading (20 h): ⟨reading_cb⟩.
- ⟨h_bullets⟩.
- **Scenes:** ⟨n⟩; dropped: ⟨dropped⟩. Completeness: Δ_refit ⟨incomplete_refit⟩; Δ_cb ⟨incomplete_cb⟩.
  `kaggle/PREREG_GN.md:4103`: «**Completeness:** a primary is `incomplete` if n < 5 after drops»
  `kaggle/PREREG_GN.md:4156`: «A **fragile** outcome (f) is reported as such beside its reading.»

The tool (`bench/gn/e7.py`, over every E7 bundle, `--out` outside the repository) printed:

⟨block:tool⟩

### The primaries per scene (20 f)

`kaggle/PREREG_GN.md:4033`: «For an arm X, **BD_X** is the BD-PSNR of `ogc_gram_ft_X` against `c3dgs_ft_X`»
`kaggle/PREREG_GN.md:4034`: «computed exactly as E6's P_FT (Amendment 19 c)»

Per scene (dB; 9 decimals as Note 2 C7 compares them; a missing BD enters as 0 and is not positive, Note 2 C2; the last
column says whether arm G's seed-0 curve / its j = 0 seed pair spans Kaggle images, Note 2 C6):

| Scene | BD_R | BD_G | BD_CG | Δ_refit | Δ_cb | BD missing | Spans images (curve / pair) |
|---|---|---|---|---|---|---|---|
⟨block:primary_rows⟩

### The noise and the standard error (20 f, b)

`kaggle/PREREG_GN.md:4048`: «`d_s` the mean of the two, `v_s` the sum of squared deviations»
`kaggle/PREREG_GN.md:4051`: «**`SE` = `SD_pool` / sqrt(n).**»
`kaggle/PREREG_GN.md:3945`: «**E7's standard error is `SD_pool` / sqrt(n), not E5's and E6's `SD_pool` / sqrt(2 n)**»

Per scene (dB): `d_sp` is the primary's paired difference in protocol ii's PSNR at j = 0 with seed p (for Δ_refit
(`ogc_gram_ft_g` - `c3dgs_ft_g`) - (`ogc_gram_ft` - `c3dgs_ft`), for Δ_cb the same with CG and G), `d_s` their mean and
`v_s` their sum of squared deviations:

| Scene | Δ_refit `d_s0` | `d_s1` | `d_s` | `v_s` | Δ_cb `d_s0` | `d_s1` | `d_s` | `v_s` |
|---|---|---|---|---|---|---|---|---|
⟨block:noise_rows⟩

| | Δ_refit | Δ_cb |
|---|---|---|
⟨block:primary_stats⟩

- **Beside E5's and E6's convention** (20 b; section 25): with `SD_pool` / sqrt(n), E5's P1 and P2 and E6's P_FT still
  meet their criterion 3, E6's P_FT at ⟨s25_pft⟩ ⟨s25_reg⟩. `kaggle/PREREG_GN.md:3950`: «E7's write-up reports this beside E7's own numbers, and it changes no earlier verdict.»

### Superiority, Holm-corrected (20 f)

`kaggle/PREREG_GN.md:4056`: «Each test type is one family of two hypotheses (one per»
`kaggle/PREREG_GN.md:4060`: «**Order:** the primary with the larger |`Δ_bar`| / `SE` first.»
`kaggle/PREREG_GN.md:4063`: «#{s : sign(r(Δ_s)) = σ} >= (7 n + 9) // 10 (5 of 7);»
`kaggle/PREREG_GN.md:4064`: «c_sup,1 = t(1 - 0.0125, n) (two-sided 0.025; 2.841 at n = 7).»
`kaggle/PREREG_GN.md:4066`: «c_sup,2 = t(1 - 0.025, n) (two-sided 0.05; 2.365 at n = 7). If the first is not superior, neither is the second.»
`kaggle/PREREG_GN.md:4362`: «alone, at the first Holm position's critical values»

Critical values: ⟨crit⟩. Holm's order: ⟨sup_order⟩. The p-values are descriptive: the two-sided p of r(`Δ_bar`) /
r(`SE`) with n degrees of freedom, and its Holm adjustment in this order (the first doubled, the second no smaller than
the first's); an adjusted p below 0.05 is criterion 3 at the position tested, and superiority also needs criteria 1-2
and the first's pass.

| Primary | Holm position | Tested | abs(r(`Δ_bar`)) / r(`SE`) | Critical t | Sign | Scenes of that sign | Criteria 1 / 2 / 3 | Superior | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|---|---|
⟨block:sup_rows⟩

### Equivalence (TOST at δ = 0.07 dB), Holm-corrected (20 f)

`kaggle/PREREG_GN.md:4067`: «**Equivalence (two one-sided tests)** at margin **δ = 0.07 dB**»
`kaggle/PREREG_GN.md:4070`: «**Order:** the primary with the larger m first.»
`kaggle/PREREG_GN.md:4071`: «**The first** is equivalent if m > c_eq,1 = t(1 - 0.025, n) (each one-sided test at 0.025; 2.365 at n = 7).»
`kaggle/PREREG_GN.md:4072`: «**The second** is tested only if the first is equivalent, with c_eq,2 = t(1 - 0.05, n)»
`kaggle/PREREG_GN.md:4075`: «**Why δ = 0.07 dB** (22% of the 0.3131 dB gap).»

Holm's order: ⟨eq_order⟩. The interval is r(`Δ_bar`) ± (critical t) x r(`SE`); equivalent when it lies inside ±δ.
p is the larger one-sided p of the TOST (that of m), Holm-adjusted as above.

| Primary | Holm position | Tested | m | Critical t | Interval (dB) | Equivalent | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|
⟨block:eq_rows⟩

### The robustness reading and the fragile flag (20 f)

`kaggle/PREREG_GN.md:4091`: «**A pre-registered robustness reading for BD-level noise:**»
`kaggle/PREREG_GN.md:4093`: «`SD_BD` = sqrt(mean over the n scenes of (BD_R,s from the seed-0 curve - BD_R,s from the seed-1 curve)^2 / 2) (g.5):»
`kaggle/PREREG_GN.md:4100`: «A primary whose outcome (h) differs between `SE` and `SE_BD` is reported **fragile**; the outcome with `SE` is the one»
`kaggle/PREREG_GN.md:4396`: «a scene without a seed-1 BD_R is left out of the mean of squared differences»

BD_R per scene from the two curves (dB):

| Scene | BD_R, seed-0 curve | BD_R, seed-1 curve | Difference |
|---|---|---|---|
⟨block:sdbd_rows⟩

⟨sd_bd⟩. Under `SE_BD`, superiority's order: ⟨bd_sup_order⟩; equivalence's: ⟨bd_eq_order⟩.

| Primary | Holm position | Tested | abs(r(`Δ_bar`)) / r(`SE_BD`) | Critical t | Sign | Scenes of that sign | Criteria 1 / 2 / 3 | Superior | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|---|---|
⟨block:bd_sup_rows⟩

| Primary | Holm position | Tested | m | Critical t | Interval (dB) | Equivalent | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|
⟨block:bd_eq_rows⟩

| Primary | Outcome with `SE` (of record) | Outcome with `SE_BD` | Fragile |
|---|---|---|---|
⟨block:fragile_rows⟩

### How the numbers are computed, and three sources that agree

- **The tool** (`bench/gn/e7.py`): each BD through `bench/gn/e5.py`'s `bd` (Amendment 9 a's fit, degree 2 through the
  three seed-0 points, averaged over the shared byte range), the noise through `e5.noise` with `SE` = `SD_pool` /
  sqrt(n), the tests with scipy's t quantiles.
- **A separate recompute** (`bench/gn/check_s26.py`), importing none of `e7.py`, `e5.py` or `g2.py`: every BD with
  `numpy.polyfit` over log10 bytes, the primaries, the noise, both Holm families, the outcomes, `SD_BD` and the fragile
  flag; and **each session's own `per_scene_parts`** (`gn5_summary_<session>.json`). Every BD agrees below 1e-9 among
  the three, and the outcomes with `SE` and with `SE_BD` are equal.

### Secondaries (20 g; report only, no bars)

**1. The codebook-alone share.** `kaggle/PREREG_GN.md:4114`: «**The codebook-alone share:** BD_K - BD_CG and BD_K - BD_R per scene»

| Scene | BD_K - BD_CG | BD_K - BD_R |
|---|---|---|
⟨block:g1_rows⟩

**2. The own grid in fine-tuning (arm O).** `kaggle/PREREG_GN.md:4117`: «**The own grid:** BD_O - BD_G per scene»
O against G measures the removed clamping and the changed step together (20 g.2, m).

| Scene | BD_O - BD_G |
|---|---|
⟨block:g2_rows⟩

**3. P1 with each method's own grid.** `kaggle/PREREG_GN.md:4122`: «**P1 with each method's own grid:** P1_own = BD-PSNR(`ogc_gram_own` against `c3dgs_own`)»
P1_own against P1 measures OGC's removed clamping and its coarser step together; it does not by itself say whether P1
understates OGC (20 g.3).

| Scene | P1 | P1_own | P1_own - P1 | `D_s`, P1 | `D_s`, P1_own |
|---|---|---|---|---|---|
⟨block:g3_rows⟩

**4. P1 and P_FT replicated, with E5's three criteria as a description.** `kaggle/PREREG_GN.md:4126`: «**P1 and P_FT replicated** (BD_R), with E5's three criteria evaluated as a description»
They re-judge nothing.

| | `P_bar` | Positive | `SE_noise` (E5's, `SD_pool` / sqrt(2 n)) | Criteria | `SD_pool` / sqrt(n) | Criteria | Beside |
|---|---|---|---|---|---|---|---|
⟨block:g4_rows⟩

**5. BD noise (CLAIMS gap 8).** `kaggle/PREREG_GN.md:4128`: «**BD noise (gap 8):** P1 and P_FT (BD_R) from the seed-1 curve»
This bears on E5's and E6's P1 and P_FT as an estimate of BD-level noise, labelled so.

| Scene | P1, seed 0 | P1, seed 1 | Difference | P_FT, seed 0 | P_FT, seed 1 | Difference |
|---|---|---|---|---|---|---|
⟨block:g5_rows⟩

⟨g5_sd⟩.

**6. Within-process fine-tuning noise (the replicate).** `kaggle/PREREG_GN.md:4131`: «**Within-process fine-tuning noise:** per scene, (`ogc_gram_ft_r2` - `c3dgs_ft_r2`) - (`ogc_gram_ft` - `c3dgs_ft`)»

| Scene | Pair: replicate minus R | `c3dgs`: replicate minus R | `ogc_gram`: replicate minus R |
|---|---|---|---|
⟨block:g6_rows⟩

Against ⟨g6_sd_pool⟩.

**7. The logs of 20 e, per arm.** `kaggle/PREREG_GN.md:4133`: «**The logs of e,** per arm: values outside the pre-fine-tuning grid and outside the saved grid, for AC and DC,»
`kaggle/PREREG_GN.md:4019`: «**The inverted no-movement guard,** at every fine-tuned row's save»

Rows, rows with a failed guard (`frozen_changed`), and the colour grids' scale after over before (min to max; `c3dgs`
DC / `c3dgs` AC / `ogc_gram` DC / `ogc_gram` AC):

| Arm | Rows | `frozen_changed` | Scale after / before |
|---|---|---|---|
⟨block:g7_scale_rows⟩

Values outside a grid, summed over rows, with the largest distance beyond it in grid steps (pre-fine-tuning grid:
codewords DC, AC, kept DC, AC; then the saved grid in the same order):

| Arm | Pre: cw DC | Pre: cw AC | Pre: kept DC | Pre: kept AC | Saved: cw DC | Saved: cw AC | Saved: kept DC | Saved: kept AC |
|---|---|---|---|---|---|---|---|---|
⟨block:g7_outside_rows⟩

The own grid's step over the shared grid's step (`*_own` rows and arm O; min to max); ⟨g7_own_beyond⟩ (note 2 b
reads "no value outside it" as at most 0.5):

| Method | DC | AC |
|---|---|---|
⟨block:g7_own_rows⟩

The geometry quantizers' scale at save (min to max; 20 e, m: they re-fit in every arm):

| Arm | `opacity_qa` | `scaling_qa` | `scaling_factor_qa` | `rotation_qa` |
|---|---|---|---|---|
⟨block:g7_geo_rows⟩

**8. Bytes, BD-rate, C3DGS's evaluation, SSIM, LPIPS, time and memory.** `kaggle/PREREG_GN.md:4137`: «**Bytes per arm and point;** BD-rate per arm pair»
`ogc_gram`'s `.npz` bytes over `c3dgs`'s at `j-1_p0` / `j0_p0` / `j0_p1` / `j+1_p0`; BD-rate with log10 bytes as a
parabola in PSNR, over the shared PSNR range (not interpretable below 0.1 dB, section 21's convention):

| Pair | Scene | Bytes ratio | BD-rate | Shared PSNR range (dB) |
|---|---|---|---|---|
⟨block:g8_rows⟩

At `j0_p0`, OGC's fine-tuned row minus C3DGS's (mean over scenes; scenes where OGC's row is better):

| Arm | Measure | Mean difference | OGC better |
|---|---|---|---|
⟨block:g8_eval_rows⟩

Per fine-tuned row, `row_time_s` (s) and `row_cuda_peak_allocated` (GiB), min to max:

| Arm | Time | GPU peak |
|---|---|---|
⟨block:g8_time_rows⟩

### Drops, missing rows and reruns (17 d, 17 i)

`kaggle/PREREG_GN.md:4164`: «E5's harness phase per scene; the data, start devices, drop rules,»

| Scene | Dropped | Rows `ok` | Missing or failed | Processes rerun | OOM | `stopped_17i` | Deviations | Deadline cuts | Failed / skipped steps | Rows with `frozen_changed` |
|---|---|---|---|---|---|---|---|---|---|---|
⟨block:drop_rows⟩

Primary rows missing after the rerun: ⟨missing_primary⟩.

### How the sessions ran (note 1; notes 3-5)

`kaggle/PREREG_GN.md:4336`: «**Eleven sessions.**»
`kaggle/PREREG_GN.md:4447`: «**Verification at ingest.**»
`kaggle/PREREG_GN.md:4502`: «**a. Scope.** This note applies to a scene extended in the ingested session»

Per session: gsplat commit, the halves it ran, Kaggle images, setup (s; preflight, restore, install, C3DGS build) and
setup plus jobs (s):

| Session | gsplat commit | Halves | Images | Setup | Setup and jobs |
|---|---|---|---|---|---|
⟨block:session_rows⟩

Each half's job (s) against note 1's estimate (low - high), and the ratio to each end:

| Scene | Half | Session | Job | Note 1 | / low | / high |
|---|---|---|---|---|---|---|
⟨block:half_time_rows⟩

Per process, the wall time (s) and its ratio to Note 4 (a)'s guard value for E7 (`j-1_p0` / `j0_p0` / `j+1_p0` / `j0_p1` /
`j-1_p1` / `j+1_p1`):

| Scene | `j-1_p0` | `j0_p0` | `j+1_p0` | `j0_p1` | `j-1_p1` | `j+1_p1` |
|---|---|---|---|---|---|---|
⟨block:process_rows⟩

Images per scene (Note 2 C6): the sessions, the number of Kaggle images, and whether arm G's seed-0 curve / its j = 0
seed pair spans images. Across E7: ⟨images⟩.
`kaggle/PREREG_GN.md:4411`: «**Per-scene image spans** (Note 2 C6)»

| Scene | Sessions | Images | Spans (curve / pair) |
|---|---|---|---|
⟨block:image_rows⟩

### What section 26 settles, and what it does not

- **Settled, as pre-registered:** Δ_refit ⟨outcome_refit⟩ (⟨fragile_refit⟩): ⟨reading_refit⟩. Δ_cb ⟨outcome_cb⟩
  (⟨fragile_cb⟩): ⟨reading_cb⟩.
- **Not settled, or limits** (20 m, stated before any data):
  - E7 is not a held-out test: its hypotheses came from E5's, E6's and section 24's results on these scenes;
  - in CG, opacity, the geometry tables and the kept colours still train and can make up for frozen codewords;
  - under a fixed grid OGC's codewords outside it stay clamped; O and the `*_own` rows remove the clamping but change
    the step, so they cannot separate the two;
  - `SE` takes the j = 0 point's noise for the BD's and leaves out scene-to-scene spread; the robustness reading checks
    the first;
  - only arm R and the unfine-tuned pair have a seed-1 curve; the geometry quantizers re-fit in every arm;
  - seven scenes, INRIA's checkpoints, K = 4,096, one GPU type (T4).

## Placeholder index (not part of FINDINGS; each placeholder with the PREREG lines it reports)

| Placeholder (in angle brackets above, without them here) | What | PREREG |
|---|---|---|
| `outcome_refit`, `outcome_cb` | 20 h's outcome per primary | 4140-4151 |
| `reading_refit`, `reading_cb` | 20 h's reading for that outcome, read from the table's cell | 4146-4151 |
| `fragile_refit`, `fragile_cb`, `block:fragile_rows` | the fragile reading (outcome under `SE_BD` against `SE`) | 4100-4101, 4156 |
| `h_bullets` | 20 h's conditional readings (both equivalent; Δ_cb superior and positive) | 4153-4155 |
| `headline`, `summary_bdbar` | the outcomes and `Δ_bar`, `SE` for the summary | 4040-4043, 4051 |
| `n`, `dropped`, `incomplete_refit`, `incomplete_cb`, `missing_primary` | n after drops, drops, completeness | 4103-4107; 17 d |
| `block:tool` | `bench/gn/e7.py`'s printed lines | 4109-4110 |
| `block:primary_rows` | BD_R, BD_G, BD_CG, Δ_refit, Δ_cb per scene, missing BDs, image spans | 4033-4043; note 2 c.11 |
| `block:noise_rows`, `block:primary_stats` | `d_sp`, `d_s`, `v_s`; `Δ_bar`, `SD_pool`, `SE` = `SD_pool` / sqrt(n), the statistics | 4045-4055; 3945-3950 |
| `s25_pft`, `s25_reg` | section 25's corrected standard error beside E7's | 3949-3950 |
| `crit` | c_sup,1, c_sup,2, c_eq,1, c_eq,2 at n | 4064-4073 |
| `sup_order`, `block:sup_rows` | superiority: Holm's order, criteria 1-3, superior, p and Holm-adjusted p | 4056-4066; note 2 a, c.2-3 |
| `eq_order`, `block:eq_rows` | TOST at δ = 0.07 dB: order, m, interval, equivalent, p and Holm-adjusted p | 4067-4074 |
| `block:sdbd_rows`, `sd_bd`, `bd_sup_order`, `bd_eq_order`, `block:bd_sup_rows`, `block:bd_eq_rows` | the robustness reading with `SE_BD` | 4091-4101; note 2 c.5 |
| `block:g1_rows` | g.1 BD_K - BD_CG, BD_K - BD_R | 4114-4116 |
| `block:g2_rows` | g.2 BD_O - BD_G | 4117-4121 |
| `block:g3_rows` | g.3 P1_own against P1 | 4122-4125 |
| `block:g4_rows` | g.4 P1 and P_FT replicated with E5's criteria, beside E5's P1 and E6's P_FT | 4126-4127 |
| `block:g5_rows`, `g5_sd` | g.5 the seed-1 curve and `SD_BD` for P1 and P_FT | 4128-4130 |
| `block:g6_rows`, `g6_sd_pool` | g.6 the replicate pair against `SD_pool` | 4131-4132 |
| `block:g7_scale_rows`, `block:g7_outside_rows`, `block:g7_own_rows`, `g7_own_beyond`, `block:g7_geo_rows` | g.7 the logs of e, own-grid step ratios | 4133-4136; 4017-4030; note 2 b |
| `block:g8_rows`, `block:g8_eval_rows`, `block:g8_time_rows` | g.8 bytes, BD-rate, C3DGS's evaluation, SSIM, LPIPS, time and memory | 4137-4139 |
| `block:drop_rows` | drops, 17 i reruns and stops, deviations, cuts, failed guards | 4103-4107, 4164 |
| `block:session_rows`, `block:half_time_rows`, `block:process_rows` | sessions, setup, jobs and process walls against note 1 and Note 4 (a) | 4262-4352 (note 1) |
| `block:image_rows`, `images` | images per scene and spans (Note 2 C6) | 4411-4412 |
| `sources_bundles` | the bundles and their commits | 4226-4231 (20 l); notes 3-5 |
