# DRAFT: FINDINGS section 26 (E7), filled, not part of FINDINGS

**Status: filled by `bench/gn/check_s26.py --fill` from `kaggle/drafts/FINDINGS_s26_E7_DRAFT.md` and all eleven committed E7 bundles, and checked by `bench/gn/check_s26.py --findings`.** FINDINGS section 26, CLAIMS and the paper wait for Dace's reading.

## Summary paragraph (goes after section 25's summary)

**E7 (section 26, branch `bench/gn-vq`): the within-process dissection of why OGC's lead shrinks with fine-tuning
(Amendment 20 f-h, with notes 1-5), on E5's seven scenes.**
- **Δ_refit (both colour grids frozen minus moving): equivalent; not superior,** not fragile (the same outcome under `SE_BD`). **Δ_cb (the codewords frozen
  minus training, grids fixed): equivalent; not superior,** not fragile (the same outcome under `SE_BD`).
- Δ_refit `Δ_bar` -0.020086807 dB, Δ_cb `Δ_bar` 0.008438079 dB (n = 7; `SE` 0.005561656 and 0.011134979); superiority and a TOST at δ = 0.07 dB, each Holm-corrected over the two primaries.
- **Disclosure:** the hypotheses were formed after E5's and E6's results on these scenes; E7 makes no held-out claim.

## Sources entry (goes after section 25's)

- Section 26: every number comes from the E7 session bundles, `kaggle/gn_e7/S1/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S2/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S3/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S4/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S5/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S6/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S7/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S8/gn7/` (`64fd4494`), `kaggle/gn_e7/S9/gn7/` (`505c2305`), `kaggle/gn_e7/S10/gn7/` (`ccb7a4f9`), `kaggle/gn_e7/S11/gn7/` (`32758a0e`), each unpacked unchanged, through
  `bench/gn/e7.py`; the exceptions are E5's and E6's values quoted beside E7's (secondary 4 and section 25's standard
  error) from their own bundles, and the quoted lines of `kaggle/PREREG_GN.md`. `bench/gn/check_s26.py` recomputes every
  number in the section from those files, with its own BD fit and tests beside the tool's and the sessions' summaries,
  and checks each against the text.

## 26. E7: a within-process dissection of why OGC's lead shrinks with fine-tuning (pre-registered) — Δ_refit: equivalent; not superior; Δ_cb: equivalent; not superior

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
- **The sources:** `kaggle/gn_e7/S1/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S2/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S3/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S4/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S5/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S6/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S7/gn7/` (`c8ba01f3`), `kaggle/gn_e7/S8/gn7/` (`64fd4494`), `kaggle/gn_e7/S9/gn7/` (`505c2305`), `kaggle/gn_e7/S10/gn7/` (`ccb7a4f9`), `kaggle/gn_e7/S11/gn7/` (`32758a0e`). A split scene's seed-0 half and seed-1 half come from two sessions; later bundles
  carry each scene unchanged (notes 3-5).

### The verdict (Amendment 20 f, h)

`kaggle/PREREG_GN.md:4040`: «**Δ_refit,s = BD_G,s - BD_R,s:**»
`kaggle/PREREG_GN.md:4042`: «**Δ_cb,s = BD_CG,s - BD_G,s:**»
`kaggle/PREREG_GN.md:4109`: «The verdict is computed locally over every E7 session's committed bundle by E7's verdict module (k)»

- **Δ_refit: equivalent; not superior;** not fragile (the same outcome under `SE_BD`). Reading (20 h): the re-fit does not explain the shrink beyond ±δ.
- **Δ_cb: equivalent; not superior;** not fragile (the same outcome under `SE_BD`). Reading (20 h): codebook training does not explain the shrink beyond ±δ.
- 20 h's case: both equivalent to 0, so the shrink is carried mostly by what neither primary holds fixed, which secondary 1 reads directly.
- **Scenes:** n = 7; dropped: none. Completeness: Δ_refit complete; Δ_cb complete.
  `kaggle/PREREG_GN.md:4103`: «**Completeness:** a primary is `incomplete` if n < 5 after drops»
  `kaggle/PREREG_GN.md:4156`: «A **fragile** outcome (f) is reported as such beside its reading.»

The tool (`bench/gn/e7.py`, over every E7 bundle, `--out` outside the repository) printed:

```
E7 (n = 7; dropped -)
  Delta_refit: equivalent; not superior; Delta_bar -0.020086807, SE 0.005561656; complete
  Delta_cb: equivalent; not superior; Delta_bar 0.008438079, SE 0.011134979; complete
```

### The primaries per scene (20 f)

`kaggle/PREREG_GN.md:4033`: «For an arm X, **BD_X** is the BD-PSNR of `ogc_gram_ft_X` against `c3dgs_ft_X`»
`kaggle/PREREG_GN.md:4034`: «computed exactly as E6's P_FT (Amendment 19 c)»

Per scene (dB; 9 decimals as Note 2 C7 compares them; a missing BD enters as 0 and is not positive, Note 2 C2; the last
column says whether arm G's seed-0 curve / its j = 0 seed pair spans Kaggle images, Note 2 C6):

| Scene | BD_R | BD_G | BD_CG | Δ_refit | Δ_cb | BD missing | Spans images (curve / pair) |
|---|---|---|---|---|---|---|---|
| bonsai | 0.073024691 | 0.081184560 | 0.127371491 | 0.008159869 | 0.046186931 | - | no / no |
| counter | 0.081245494 | 0.050225540 | 0.076107457 | -0.031019954 | 0.025881917 | - | no / no |
| drjohnson | -0.009509652 | 0.016629770 | -0.018899789 | 0.026139422 | -0.035529559 | - | no / yes |
| kitchen | 0.041540814 | -0.108538329 | -0.065458844 | -0.150079143 | 0.043079485 | - | no / no |
| playroom | 0.066939629 | 0.058965785 | 0.056049488 | -0.007973844 | -0.002916298 | - | no / no |
| room | 0.042698598 | 0.052853950 | 0.064883552 | 0.010155352 | 0.012029602 | - | no / no |
| truck | 0.031003175 | 0.035013826 | 0.005348303 | 0.004010650 | -0.029665522 | - | no / no |

### The noise and the standard error (20 f, b)

`kaggle/PREREG_GN.md:4048`: «`d_s` the mean of the two, `v_s` the sum of squared deviations»
`kaggle/PREREG_GN.md:4051`: «**`SE` = `SD_pool` / sqrt(n).**»
`kaggle/PREREG_GN.md:3945`: «**E7's standard error is `SD_pool` / sqrt(n), not E5's and E6's `SD_pool` / sqrt(2 n)**»

Per scene (dB): `d_sp` is the primary's paired difference in protocol ii's PSNR at j = 0 with seed p (for Δ_refit
(`ogc_gram_ft_g` - `c3dgs_ft_g`) - (`ogc_gram_ft` - `c3dgs_ft`), for Δ_cb the same with CG and G), `d_s` their mean and
`v_s` their sum of squared deviations:

| Scene | Δ_refit `d_s0` | `d_s1` | `d_s` | `v_s` | Δ_cb `d_s0` | `d_s1` | `d_s` | `v_s` |
|---|---|---|---|---|---|---|---|---|
| bonsai | +0.047967 | +0.044378 | +0.046172 | 6.44e-06 | +0.078513 | +0.036523 | +0.057518 | 8.82e-04 |
| counter | +0.012487 | -0.016571 | -0.002042 | 4.22e-04 | +0.037485 | +0.084192 | +0.060839 | 1.09e-03 |
| drjohnson | +0.020344 | +0.025228 | +0.022786 | 1.19e-05 | -0.043430 | -0.021234 | -0.032332 | 2.46e-04 |
| kitchen | -0.121190 | -0.135211 | -0.128200 | 9.83e-05 | +0.080558 | +0.148505 | +0.114531 | 2.31e-03 |
| playroom | -0.034575 | -0.000479 | -0.017527 | 5.81e-04 | +0.033241 | +0.070939 | +0.052090 | 7.11e-04 |
| room | +0.021672 | +0.046189 | +0.033930 | 3.01e-04 | +0.020372 | +0.000026 | +0.010199 | 2.07e-04 |
| truck | +0.016554 | +0.030340 | +0.023447 | 9.50e-05 | -0.023152 | +0.012366 | -0.005393 | 6.31e-04 |

| | Δ_refit | Δ_cb |
|---|---|---|
| `Δ_bar` (n = 7) | -0.020086807 | 0.008438079 |
| `SD_pool` | 0.014714759 | 0.029460386 |
| scenes with both j = 0 values | 7 | 7 |
| `SE` = `SD_pool` / sqrt(n) | 0.005561656 | 0.011134979 |
| r(`Δ_bar`) / r(`SE`) | 3.612 | 0.758 |
| m = (δ - abs(r(`Δ_bar`))) / r(`SE`) | 8.975 | 5.529 |

- **Beside E5's and E6's convention** (20 b; section 25): with `SD_pool` / sqrt(n), E5's P1 and P2 and E6's P_FT still
  meet their criterion 3, E6's P_FT at 3.45 x the corrected standard error against the registered 4.9. `kaggle/PREREG_GN.md:3950`: «E7's write-up reports this beside E7's own numbers, and it changes no earlier verdict.»

### Superiority, Holm-corrected (20 f)

`kaggle/PREREG_GN.md:4056`: «Each test type is one family of two hypotheses (one per»
`kaggle/PREREG_GN.md:4060`: «**Order:** the primary with the larger |`Δ_bar`| / `SE` first.»
`kaggle/PREREG_GN.md:4063`: «#{s : sign(r(Δ_s)) = σ} >= (7 n + 9) // 10 (5 of 7);»
`kaggle/PREREG_GN.md:4064`: «c_sup,1 = t(1 - 0.0125, n) (two-sided 0.025; 2.841 at n = 7).»
`kaggle/PREREG_GN.md:4066`: «c_sup,2 = t(1 - 0.025, n) (two-sided 0.05; 2.365 at n = 7). If the first is not superior, neither is the second.»
`kaggle/PREREG_GN.md:4362`: «alone, at the first Holm position's critical values»

Critical values: c_sup,1 = 2.841, c_sup,2 = 2.365, c_eq,1 = 2.365, c_eq,2 = 1.895 at n = 7. Holm's order: Δ_refit, then Δ_cb. The p-values are descriptive: the two-sided p of r(`Δ_bar`) /
r(`SE`) with n degrees of freedom, and its Holm adjustment in this order (the first doubled, the second no smaller than
the first's); an adjusted p below 0.05 is criterion 3 at the position tested, and superiority also needs criteria 1-2
and the first's pass.

| Primary | Holm position | Tested | abs(r(`Δ_bar`)) / r(`SE`) | Critical t | Sign | Scenes of that sign | Criteria 1 / 2 / 3 | Superior | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|---|---|
| Δ_refit | 1 | yes | 3.612 | 2.841 | - | 3 of 7 (5 needed) | holds / fails / holds | no | 0.00861 | 0.0172 |
| Δ_cb | 2 | no | 0.758 | 2.365 | + | 4 of 7 (5 needed) | holds / fails / fails | no | 0.473 | 0.473 |

### Equivalence (TOST at δ = 0.07 dB), Holm-corrected (20 f)

`kaggle/PREREG_GN.md:4067`: «**Equivalence (two one-sided tests)** at margin **δ = 0.07 dB**»
`kaggle/PREREG_GN.md:4070`: «**Order:** the primary with the larger m first.»
`kaggle/PREREG_GN.md:4071`: «**The first** is equivalent if m > c_eq,1 = t(1 - 0.025, n) (each one-sided test at 0.025; 2.365 at n = 7).»
`kaggle/PREREG_GN.md:4072`: «**The second** is tested only if the first is equivalent, with c_eq,2 = t(1 - 0.05, n)»
`kaggle/PREREG_GN.md:4075`: «**Why δ = 0.07 dB** (22% of the 0.3131 dB gap).»

Holm's order: Δ_refit, then Δ_cb. The interval is r(`Δ_bar`) ± (critical t) x r(`SE`); equivalent when it lies inside ±δ.
p is the larger one-sided p of the TOST (that of m), Holm-adjusted as above.

| Primary | Holm position | Tested | m | Critical t | Interval (dB) | Equivalent | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|
| Δ_refit | 1 | yes | 8.975 | 2.365 | [-0.033238, -0.006936] | yes | 2.17e-05 | 4.35e-05 |
| Δ_cb | 2 | yes | 5.529 | 1.895 | [-0.012658, +0.029534] | yes | 0.00044 | 0.00044 |

### The robustness reading and the fragile flag (20 f)

`kaggle/PREREG_GN.md:4091`: «**A pre-registered robustness reading for BD-level noise:**»
`kaggle/PREREG_GN.md:4093`: «`SD_BD` = sqrt(mean over the n scenes of (BD_R,s from the seed-0 curve - BD_R,s from the seed-1 curve)^2 / 2) (g.5):»
`kaggle/PREREG_GN.md:4100`: «A primary whose outcome (h) differs between `SE` and `SE_BD` is reported **fragile**; the outcome with `SE` is the one»
`kaggle/PREREG_GN.md:4396`: «a scene without a seed-1 BD_R is left out of the mean of squared differences»

BD_R per scene from the two curves (dB):

| Scene | BD_R, seed-0 curve | BD_R, seed-1 curve | Difference |
|---|---|---|---|
| bonsai | 0.073024691 | 0.021120642 | 0.051904050 |
| counter | 0.081245494 | 0.116094295 | -0.034848801 |
| kitchen | 0.041540814 | 0.083000525 | -0.041459711 |
| room | 0.042698598 | 0.034709769 | 0.007988829 |
| truck | 0.031003175 | 0.012600548 | 0.018402627 |
| drjohnson | -0.009509652 | 0.004275303 | -0.013784955 |
| playroom | 0.066939629 | 0.022288224 | 0.044651405 |

`SD_BD` = 0.024221637 over 7 scenes with both curves; `SE_BD` = sqrt(2) x `SD_BD` / sqrt(n) = 0.012947010. Under `SE_BD`, superiority's order: Δ_refit, then Δ_cb; equivalence's: Δ_cb, then Δ_refit.

| Primary | Holm position | Tested | abs(r(`Δ_bar`)) / r(`SE_BD`) | Critical t | Sign | Scenes of that sign | Criteria 1 / 2 / 3 | Superior | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|---|---|
| Δ_refit | 1 | yes | 1.551 | 2.841 | - | 3 of 7 (5 needed) | holds / fails / fails | no | 0.165 | 0.329 |
| Δ_cb | 2 | no | 0.652 | 2.365 | + | 4 of 7 (5 needed) | holds / fails / fails | no | 0.535 | 0.535 |

| Primary | Holm position | Tested | m | Critical t | Interval (dB) | Equivalent | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|
| Δ_refit | 2 | yes | 3.855 | 1.895 | [-0.044616, +0.004442] | yes | 0.00313 | 0.00313 |
| Δ_cb | 1 | yes | 4.755 | 2.365 | [-0.022177, +0.039053] | yes | 0.00104 | 0.00207 |

| Primary | Outcome with `SE` (of record) | Outcome with `SE_BD` | Fragile |
|---|---|---|---|
| Δ_refit | equivalent; not superior | equivalent; not superior | no |
| Δ_cb | equivalent; not superior | equivalent; not superior | no |

### Exploratory sensitivity, not pre-registered: drjohnson's j = 0 seed pair out of the noise pooling

**exploratory, declared at `c95c369f` before unblinding; the verdict is the pre-registered computation.**
- **Why:** drjohnson's j = 0 seed pair is the one comparison inside `SD_pool` that spans Kaggle's two images (see "How the
  sessions ran").
- **What changes:** `SD_pool` = sqrt(mean `v_s` over the other scenes), with one degree of freedom per pooled scene, and
  each primary's `SE` from it, still `SD_pool` / sqrt(n) with n = 7, the scenes `Δ_bar` averages.
- **What does not:** every per-scene Δ, `Δ_bar`, criterion 2's count and the robustness reading's `SD_BD`.
- **Where:** `bench/gn/check_s26.py` computes it; `bench/gn/e7.py` does not. Holm's order is each test's own under this
  `SE`.

Beside the pre-registered values:

| | Δ_refit, pre-registered | Δ_refit, exploratory | Δ_cb, pre-registered | Δ_cb, exploratory |
|---|---|---|---|---|
| `Δ_bar` (n = 7) | -0.020086807 | -0.020086807 | 0.008438079 | 0.008438079 |
| `SD_pool` | 0.014714759 | 0.015831109 | 0.029460386 | 0.031169125 |
| scenes pooled for the noise, and the t's degrees of freedom | 7 | 6 | 7 | 6 |
| `SE` = `SD_pool` / sqrt(n), n = 7 | 0.005561656 | 0.005983597 | 0.011134979 | 0.011780822 |
| r(`Δ_bar`) / r(`SE`) | 3.612 | 3.357 | 0.758 | 0.716 |
| m = (δ - abs(r(`Δ_bar`))) / r(`SE`) | 8.975 | 8.342 | 5.529 | 5.226 |

Critical values, exploratory: c_sup,1 = 2.969, c_sup,2 = 2.447, c_eq,1 = 2.447, c_eq,2 = 1.943 at 6 degrees of freedom. Superiority's order: Δ_refit, then Δ_cb; equivalence's: Δ_refit, then Δ_cb.

Superiority, exploratory (p and Holm-adjusted p as above, at these degrees of freedom):

| Primary | Holm position | Tested | abs(r(`Δ_bar`)) / r(`SE`) | Critical t | Sign | Scenes of that sign | Criteria 1 / 2 / 3 | Superior | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|---|---|
| Δ_refit | 1 | yes | 3.357 | 2.969 | - | 3 of 7 (5 needed) | holds / fails / holds | no | 0.0153 | 0.0306 |
| Δ_cb | 2 | no | 0.716 | 2.447 | + | 4 of 7 (5 needed) | holds / fails / fails | no | 0.501 | 0.501 |

Equivalence, exploratory:

| Primary | Holm position | Tested | m | Critical t | Interval (dB) | Equivalent | p | Holm-adjusted p |
|---|---|---|---|---|---|---|---|---|
| Δ_refit | 1 | yes | 8.342 | 2.447 | [-0.034728, -0.005445] | yes | 8.06e-05 | 0.000161 |
| Δ_cb | 2 | yes | 5.226 | 1.943 | [-0.014454, +0.031330] | yes | 0.000983 | 0.000983 |

| Primary | Outcome of record (pre-registered) | Under this exploratory reading | Differs |
|---|---|---|---|
| Δ_refit | equivalent; not superior | equivalent; not superior | no |
| Δ_cb | equivalent; not superior | equivalent; not superior | no |

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
| bonsai | 0.278322498 | 0.332669298 |
| counter | 0.216516623 | 0.211378585 |
| drjohnson | 0.238412847 | 0.229022710 |
| kitchen | 0.415772416 | 0.308772758 |
| playroom | 0.290866250 | 0.279976109 |
| room | 0.292453622 | 0.314638576 |
| truck | 0.131137997 | 0.105483125 |
| mean | 0.266211750 | 0.254563023 |
| `SD_pool` | 0.037468532 | 0.026179949 |
| `SE` = `SD_pool` / sqrt(n) | 0.014161774 | 0.009895091 |

**2. The own grid in fine-tuning (arm O).** `kaggle/PREREG_GN.md:4117`: «**The own grid:** BD_O - BD_G per scene»
O against G measures the removed clamping and the changed step together (20 g.2, m).

| Scene | BD_O - BD_G |
|---|---|
| bonsai | 0.006992617 |
| counter | 0.040869978 |
| drjohnson | -0.036128645 |
| kitchen | 0.095868516 |
| playroom | 0.019037905 |
| room | -0.014206680 |
| truck | -0.010314904 |
| mean | 0.014588398 |
| `SD_pool` | 0.026242742 |
| `SE` = `SD_pool` / sqrt(n) | 0.009918824 |

**3. P1 with each method's own grid.** `kaggle/PREREG_GN.md:4122`: «**P1 with each method's own grid:** P1_own = BD-PSNR(`ogc_gram_own` against `c3dgs_own`)»
P1_own against P1 measures OGC's removed clamping and its coarser step together; it does not by itself say whether P1
understates OGC (20 g.3).

| Scene | P1 | P1_own | P1_own - P1 | `D_s`, P1 | `D_s`, P1_own |
|---|---|---|---|---|---|
| bonsai | 0.481061137 | 0.540573161 | 0.059512024 | +0.658978 | +0.639502 |
| counter | 0.319291809 | 0.431714406 | 0.112422597 | +0.449972 | +0.441493 |
| drjohnson | 0.275725931 | 0.295928524 | 0.020202593 | +0.445531 | +0.437793 |
| kitchen | 0.400761720 | 0.473270326 | 0.072508605 | +0.627047 | +0.615933 |
| playroom | 0.462868732 | 0.464454045 | 0.001585314 | +0.692013 | +0.681593 |
| room | 0.389364824 | 0.441560247 | 0.052195423 | +0.542418 | +0.525153 |
| truck | 0.148715982 | 0.177956509 | 0.029240527 | +0.248258 | +0.242147 |
| mean | 0.353970019 | 0.403636745 | - | - | - |

**4. P1 and P_FT replicated, with E5's three criteria as a description.** `kaggle/PREREG_GN.md:4126`: «**P1 and P_FT replicated** (BD_R), with E5's three criteria evaluated as a description»
They re-judge nothing.

| | `P_bar` | Positive | `SE_noise` (E5's, `SD_pool` / sqrt(2 n)) | Criteria | `SD_pool` / sqrt(n) | Criteria | Beside |
|---|---|---|---|---|---|---|---|
| P1 replicated | 0.353970019 | 7 of 7 | 0.003494087 | holds | 0.004941385 | holds | E5's P1 0.3558 |
| P_FT replicated (BD_R) | 0.046706107 | 6 of 7 | 0.007570177 | holds | 0.010705847 | holds | E6's P_FT 0.0425 |

**5. BD noise (CLAIMS gap 8).** `kaggle/PREREG_GN.md:4128`: «**BD noise (gap 8):** P1 and P_FT (BD_R) from the seed-1 curve»
This bears on E5's and E6's P1 and P_FT as an estimate of BD-level noise, labelled so.

| Scene | P1, seed 0 | P1, seed 1 | Difference | P_FT, seed 0 | P_FT, seed 1 | Difference |
|---|---|---|---|---|---|---|
| bonsai | 0.481061137 | 0.486661607 | -0.005600470 | 0.073024691 | 0.021120642 | 0.051904050 |
| counter | 0.319291809 | 0.333855656 | -0.014563848 | 0.081245494 | 0.116094295 | -0.034848801 |
| drjohnson | 0.275725931 | 0.296002618 | -0.020276687 | -0.009509652 | 0.004275303 | -0.013784955 |
| kitchen | 0.400761720 | 0.422425213 | -0.021663492 | 0.041540814 | 0.083000525 | -0.041459711 |
| playroom | 0.462868732 | 0.467764910 | -0.004896179 | 0.066939629 | 0.022288224 | 0.044651405 |
| room | 0.389364824 | 0.395563986 | -0.006199162 | 0.042698598 | 0.034709769 | 0.007988829 |
| truck | 0.148715982 | 0.157310516 | -0.008594534 | 0.031003175 | 0.012600548 | 0.018402627 |

`SD_BD`: P1 0.009487548 (7 scenes), P_FT 0.024221637 (7 scenes).

**6. Within-process fine-tuning noise (the replicate).** `kaggle/PREREG_GN.md:4131`: «**Within-process fine-tuning noise:** per scene, (`ogc_gram_ft_r2` - `c3dgs_ft_r2`) - (`ogc_gram_ft` - `c3dgs_ft`)»

| Scene | Pair: replicate minus R | `c3dgs`: replicate minus R | `ogc_gram`: replicate minus R |
|---|---|---|---|
| bonsai | +0.041913 | -0.016061 | +0.025853 |
| counter | +0.017660 | -0.014329 | +0.003331 |
| kitchen | -0.011623 | +0.030842 | +0.019219 |
| room | -0.010663 | +0.002373 | -0.008289 |
| truck | -0.024807 | +0.013816 | -0.010991 |
| drjohnson | -0.057807 | +0.014673 | -0.043134 |
| playroom | -0.041697 | +0.002979 | -0.038718 |
| across-scene SD | 0.033988 | 0.016669 | 0.026460 |

Against the primaries' `SD_pool`: Δ_refit 0.014715, Δ_cb 0.029460.

**7. The logs of 20 e, per arm.** `kaggle/PREREG_GN.md:4133`: «**The logs of e,** per arm: values outside the pre-fine-tuning grid and outside the saved grid, for AC and DC,»
`kaggle/PREREG_GN.md:4019`: «**The inverted no-movement guard,** at every fine-tuned row's save»

Rows, rows with a failed guard (`frozen_changed`), and the colour grids' scale after over before (min to max; `c3dgs`
DC / `c3dgs` AC / `ogc_gram` DC / `ogc_gram` AC):

| Arm | Rows | `frozen_changed` | Scale after / before |
|---|---|---|---|
| R | 84 | 0 | 0.466 to 1.050 / 0.795 to 1.133 / 0.581 to 1.056 / 1.586 to 3.227 |
| G | 56 | 0 | 1.000 to 1.000 / 1.000 to 1.000 / 1.000 to 1.000 / 1.000 to 1.000 |
| CG | 56 | 0 | 1.000 to 1.000 / 1.000 to 1.000 / 1.000 to 1.000 / 1.000 to 1.000 |
| K | 56 | 0 | 1.000 to 1.000 / 1.000 to 1.000 / 1.000 to 1.000 / 1.000 to 1.000 |
| O | 56 | 0 | 0.425 to 0.984 / 0.710 to 1.000 / 0.552 to 0.984 / 1.590 to 3.177 |
| R2 | 14 | 0 | 0.549 to 0.800 / 0.945 to 1.133 / 0.745 to 1.031 / 1.588 to 2.843 |

Values outside a grid, summed over rows, with the largest distance beyond it in grid steps (pre-fine-tuning grid:
codewords DC, AC, kept DC, AC; then the saved grid in the same order):

| Arm | Pre: cw DC | Pre: cw AC | Pre: kept DC | Pre: kept AC | Saved: cw DC | Saved: cw AC | Saved: kept DC | Saved: kept AC |
|---|---|---|---|---|---|---|---|---|
| R | 16 (10.25) | 3614 (404.21) | 51 (16.29) | 646 (23.98) | 45 (1.00) | 53 (0.54) | 70 (1.32) | 68 (0.84) |
| G | 13 (1.93) | 2391 (398.15) | 30 (1.15) | 430 (0.81) | 13 (1.93) | 2391 (398.15) | 30 (1.15) | 430 (0.81) |
| CG | 8 (1.93) | 2238 (398.15) | 28 (1.44) | 435 (0.78) | 8 (1.93) | 2238 (398.15) | 28 (1.44) | 435 (0.78) |
| K | 11 (1.93) | 2549 (398.15) | 10 (0.00) | 28 (0.00) | 11 (1.93) | 2549 (398.15) | 10 (0.00) | 28 (0.00) |
| O | 8 (2.35) | 2354 (400.25) | 20 (1.29) | 377 (24.49) | 78 (1.36) | 43 (0.70) | 88 (1.62) | 362 (1.03) |
| R2 | 4 (5.29) | 549 (266.39) | 6 (2.01) | 106 (22.92) | 9 (1.06) | 6 (0.43) | 16 (0.61) | 11 (0.59) |

The own grid's step over the shared grid's step (`*_own` rows and arm O; min to max); the largest distance of an own grid's table beyond it is 0.500 steps (note 2 b
reads "no value outside it" as at most 0.5):

| Method | DC | AC |
|---|---|---|
| c3dgs | 0.425 to 0.984 | 0.710 to 1.000 |
| ogc_gram | 0.552 to 0.984 | 1.590 to 3.177 |

The geometry quantizers' scale at save (min to max; 20 e, m: they re-fit in every arm):

| Arm | `opacity_qa` | `scaling_qa` | `scaling_factor_qa` | `rotation_qa` |
|---|---|---|---|---|
| R | 0.003922 to 0.003922 | 0.003922 to 0.003922 | 0.0436 to 0.05775 | 0.00837 to 0.009798 |
| G | 0.003922 to 0.003922 | 0.003922 to 0.003922 | 0.04307 to 0.05732 | 0.008361 to 0.009807 |
| CG | 0.003922 to 0.003922 | 0.003922 to 0.003922 | 0.04477 to 0.05832 | 0.008349 to 0.009741 |
| K | 0.003922 to 0.003922 | 0.003922 to 0.003922 | 0.04026 to 0.05156 | 0.006692 to 0.006695 |
| O | 0.003922 to 0.003922 | 0.003922 to 0.003922 | 0.04318 to 0.05695 | 0.008328 to 0.009753 |
| R2 | 0.003922 to 0.003922 | 0.003922 to 0.003922 | 0.04465 to 0.05689 | 0.008588 to 0.00979 |

**8. Bytes, BD-rate, C3DGS's evaluation, SSIM, LPIPS, time and memory.** `kaggle/PREREG_GN.md:4137`: «**Bytes per arm and point;** BD-rate per arm pair»
`ogc_gram`'s `.npz` bytes over `c3dgs`'s at `j-1_p0` / `j0_p0` / `j0_p1` / `j+1_p0`; BD-rate with log10 bytes as a
parabola in PSNR, over the shared PSNR range (not interpretable below 0.1 dB, section 21's convention):

| Pair | Scene | Bytes ratio | BD-rate | Shared PSNR range (dB) |
|---|---|---|---|---|
| R | bonsai | 0.9514 / 1.0077 / 1.0093 / 1.0363 | -8.64% | 0.3347 |
| R | counter | 0.9242 / 0.9856 / 0.9853 / 1.0366 | -12.17% | 0.2799 |
| R | drjohnson | 0.9761 / 1.0336 / 1.0349 / 1.0632 | +5685.00% (overlap < 0.1 dB, not interpretable) | 0.0216 |
| R | kitchen | 0.9543 / 0.9962 / 0.9995 / 1.0348 | -6.72% | 0.3653 |
| R | playroom | 0.9976 / 1.0673 / 1.0696 / 1.0786 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| R | room | 0.9552 / 1.0010 / 1.0016 / 1.0585 | -12.17% (overlap < 0.1 dB, not interpretable) | 0.0788 |
| R | truck | 0.9561 / 1.0175 / 1.0175 / 1.0443 | -12.24% (overlap < 0.1 dB, not interpretable) | 0.0629 |
| G | bonsai | 1.0246 / 1.0419 / 1.0424 / 1.0556 | -9.76% | 0.2687 |
| G | counter | 1.0285 / 1.0476 / 1.0485 / 1.0635 | -8.83% | 0.2530 |
| G | drjohnson | 1.0430 / 1.0617 / 1.0627 / 1.0705 | +16.25% (overlap < 0.1 dB, not interpretable) | 0.0387 |
| G | kitchen | 1.0190 / 1.0347 / 1.0376 / 1.0455 | +10.08% | 0.4215 |
| G | playroom | 1.0614 / 1.0817 / 1.0840 / 1.0941 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| G | room | 1.0381 / 1.0581 / 1.0589 / 1.0738 | -57.85% (overlap < 0.1 dB, not interpretable) | 0.0276 |
| G | truck | 1.0407 / 1.0554 / 1.0577 / 1.0631 | -10.56% (overlap < 0.1 dB, not interpretable) | 0.0771 |
| CG | bonsai | 1.0251 / 1.0424 / 1.0421 / 1.0557 | -12.49% | 0.3295 |
| CG | counter | 1.0286 / 1.0482 / 1.0488 / 1.0641 | -11.47% | 0.3029 |
| CG | drjohnson | 1.0437 / 1.0610 / 1.0626 / 1.0707 | +17.20% (overlap < 0.1 dB, not interpretable) | 0.0111 |
| CG | kitchen | 1.0187 / 1.0342 / 1.0380 / 1.0461 | +2.37% | 0.4379 |
| CG | playroom | 1.0613 / 1.0818 / 1.0849 / 1.0944 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| CG | room | 1.0383 / 1.0585 / 1.0594 / 1.0744 | -100.00% (overlap < 0.1 dB, not interpretable) | 0.0474 |
| CG | truck | 1.0411 / 1.0557 / 1.0582 / 1.0639 | -9.41% (overlap < 0.1 dB, not interpretable) | 0.0481 |
| K | bonsai | 1.0242 / 1.0409 / 1.0414 / 1.0536 | -29.63% | 0.1282 |
| K | counter | 1.0279 / 1.0456 / 1.0462 / 1.0601 | -25.07% | 0.2198 |
| K | drjohnson | 1.0428 / 1.0591 / 1.0610 / 1.0677 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| K | kitchen | 1.0190 / 1.0336 / 1.0371 / 1.0436 | -24.52% | 0.2471 |
| K | playroom | 1.0595 / 1.0788 / 1.0808 / 1.0895 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| K | room | 1.0379 / 1.0570 / 1.0577 / 1.0719 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| K | truck | 1.0410 / 1.0546 / 1.0573 / 1.0622 | -25.41% (overlap < 0.1 dB, not interpretable) | 0.0030 |
| O | bonsai | 0.9412 / 1.0027 / 1.0039 / 1.0333 | -11.98% | 0.2685 |
| O | counter | 0.9146 / 0.9783 / 0.9782 / 1.0340 | -13.45% | 0.2728 |
| O | drjohnson | 0.9707 / 1.0310 / 1.0328 / 1.0618 | -15.59% (overlap < 0.1 dB, not interpretable) | 0.0255 |
| O | kitchen | 0.9468 / 0.9926 / 0.9962 / 1.0335 | -0.78% | 0.3564 |
| O | playroom | 0.9924 / 1.0643 / 1.0660 / 1.0771 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| O | room | 0.9506 / 0.9957 / 0.9969 / 1.0560 | +8.32% (overlap < 0.1 dB, not interpretable) | 0.0257 |
| O | truck | 0.9492 / 1.0140 / 1.0146 / 1.0424 | -10.44% (overlap < 0.1 dB, not interpretable) | 0.0545 |
| unfine-tuned | bonsai | 1.0247 / 1.0417 / 1.0422 / 1.0547 | -30.78% | 0.1307 |
| unfine-tuned | counter | 1.0284 / 1.0465 / 1.0470 / 1.0613 | -24.52% | 0.2627 |
| unfine-tuned | drjohnson | 1.0433 / 1.0599 / 1.0618 / 1.0686 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| unfine-tuned | kitchen | 1.0193 / 1.0342 / 1.0377 / 1.0443 | -24.83% | 0.2838 |
| unfine-tuned | playroom | 1.0600 / 1.0796 / 1.0815 / 1.0904 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| unfine-tuned | room | 1.0386 / 1.0580 / 1.0586 / 1.0732 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| unfine-tuned | truck | 1.0417 / 1.0557 / 1.0583 / 1.0635 | -24.61% (overlap < 0.1 dB, not interpretable) | 0.0134 |
| own grid | bonsai | 0.9304 / 1.0020 / 1.0025 / 1.0372 | -32.08% | 0.1401 |
| own grid | counter | 0.9098 / 0.9752 / 0.9756 / 1.0329 | -27.95% | 0.2714 |
| own grid | drjohnson | 0.9743 / 1.0315 / 1.0334 / 1.0608 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| own grid | kitchen | 0.9470 / 0.9907 / 0.9933 / 1.0328 | -27.18% | 0.2921 |
| own grid | playroom | 0.9962 / 1.0648 / 1.0667 / 1.0761 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| own grid | room | 0.9476 / 1.0018 / 1.0043 / 1.0580 | n/a (overlap < 0.1 dB, not interpretable) | 0.0000 |
| own grid | truck | 0.9492 / 1.0182 / 1.0187 / 1.0474 | -25.36% (overlap < 0.1 dB, not interpretable) | 0.0229 |

At `j0_p0`, OGC's fine-tuned row minus C3DGS's (mean over scenes; scenes where OGC's row is better):

| Arm | Measure | Mean difference | OGC better |
|---|---|---|---|
| R | c3dgs_PSNR | +0.0691 | 6 of 7 |
| R | SSIM_ii | +0.0012 | 6 of 7 |
| R | LPIPS_ii | -0.0032 | 7 of 7 |
| G | c3dgs_PSNR | +0.0892 | 7 of 7 |
| G | SSIM_ii | +0.0014 | 5 of 7 |
| G | LPIPS_ii | -0.0033 | 7 of 7 |
| CG | c3dgs_PSNR | +0.1065 | 6 of 7 |
| CG | SSIM_ii | +0.0016 | 6 of 7 |
| CG | LPIPS_ii | -0.0040 | 7 of 7 |
| K | c3dgs_PSNR | +0.4382 | 7 of 7 |
| K | SSIM_ii | +0.0085 | 7 of 7 |
| K | LPIPS_ii | -0.0113 | 7 of 7 |
| O | c3dgs_PSNR | +0.0703 | 7 of 7 |
| O | SSIM_ii | +0.0012 | 6 of 7 |
| O | LPIPS_ii | -0.0031 | 7 of 7 |

Per fine-tuned row, `row_time_s` (s) and `row_cuda_peak_allocated` (GiB), min to max:

| Arm | Time | GPU peak |
|---|---|---|
| R | 337.6 to 828.2 | 3.27 to 7.57 |
| G | 333.4 to 818.3 | 3.29 to 7.57 |
| CG | 333.2 to 819.9 | 3.29 to 7.57 |
| K | 314.8 to 815.7 | 3.14 to 7.50 |
| O | 333.8 to 821.6 | 3.29 to 7.57 |
| R2 | 346.7 to 785.2 | 3.33 to 7.48 |

### Drops, missing rows and reruns (17 d, 17 i)

`kaggle/PREREG_GN.md:4164`: «E5's harness phase per scene; the data, start devices, drop rules,»

| Scene | Dropped | Rows `ok` | Missing or failed | Processes rerun | OOM | `stopped_17i` | Deviations | Deadline cuts | Failed / skipped steps | Rows with `frozen_changed` |
|---|---|---|---|---|---|---|---|---|---|---|
| bonsai | - | 66 | 0 | - | - | - | 0 | 0 | 0 / 0 | 0 |
| counter | - | 66 | 0 | - | - | - | 0 | 0 | 0 / 0 | 0 |
| kitchen | - | 66 | 0 | - | - | - | 0 | 0 | 0 / 0 | 0 |
| room | - | 66 | 0 | - | - | - | 0 | 0 | 0 / 0 | 0 |
| truck | - | 66 | 0 | - | - | - | 0 | 0 | 0 / 0 | 0 |
| drjohnson | - | 66 | 0 | - | - | - | 0 | 0 | 0 / 0 | 0 |
| playroom | - | 66 | 0 | - | - | - | 0 | 0 | 0 / 0 | 0 |

Primary rows missing after the rerun: no primary row missing.

### How the sessions ran (note 1; notes 3-5)

`kaggle/PREREG_GN.md:4336`: «**Eleven sessions.**»
`kaggle/PREREG_GN.md:4447`: «**Verification at ingest.**»
`kaggle/PREREG_GN.md:4502`: «**a. Scope.** This note applies to a scene extended in the ingested session»

- **The image, named as 18 b requires:** Kaggle's image changed between S10 and S11: S1-S10 ran on `py3.13.15|torch2.11.0+cu128|cuda12.8|cudnn91900|driver580.178.04`, S11 on `py3.13.16|torch2.11.0+cu130|cuda13.0|cudnn92700|driver580.178.04`. Spanning images (Note 2 C6): the j = 0 seed pair of drjohnson; the seed-0 curve of no scene; the seed-1 curve of no scene;
  nothing else changes. `kaggle/PREREG_GN.md:2785`: «E4p or E4q is reported with both images named.»
  `kaggle/PREREG_GN.md:3184`: «(one process per point) or j = 0 pair (two processes) whose processes ran on different images is reported as»
  `kaggle/PREREG_GN.md:3185`: «spanning images; nothing else changes (18 b).»
- **S10's inputs, a departure from 20 l with no data effect:** S10's restore read only the S2, S3, S5, S7 and S9 bundle
  datasets; S1, S4, S6 and S8 were attached after it had run. S9's copy of every scene was among those it read, and
  every copy it restored is byte-identical to its S9 reference copy. `kaggle/PREREG_GN.md:4230`: «Each session attaches every earlier E7 session's output»

Per session: gsplat commit, the halves it ran, Kaggle images, setup (s; preflight, restore, install, C3DGS build; a wheel
built in the session counts once, inside install) and setup plus jobs (s):

| Session | gsplat commit | Halves | Images | Setup | Setup and jobs |
|---|---|---|---|---|---|
| S1 | `f029da7a` | room seed 0 | 1 | 465.3 | 26,880.9 |
| S2 | `5418fec3` | kitchen seed 0 | 1 | 385.1 | 32,590.8 |
| S3 | `2334728c` | drjohnson seed 0 | 1 | 393.6 | 26,404.1 |
| S4 | `43751390` | bonsai seed 0 | 1 | 502.1 | 23,752.5 |
| S5 | `444c79d3` | counter seed 0 | 1 | 374.0 | 26,549.5 |
| S6 | `33a4bdfd` | playroom seed 0 | 1 | 402.7 | 23,158.2 |
| S7 | `5e189d0d` | truck seed 0 | 1 | 381.6 | 16,686.9 |
| S8 | `8c1d6d16` | kitchen seed 1, playroom seed 1 | 1 | 424.0 | 24,724.5 |
| S9 | `425dad52` | bonsai seed 1, counter seed 1 | 1 | 381.8 | 21,937.2 |
| S10 | `7fb734a7` | room seed 1, truck seed 1 | 1 | 383.9 | 21,834.4 |
| S11 | `82725251` | drjohnson seed 1 | 1 | 4,464.3 | 17,904.6 |

Each half's job (s) against note 1's estimate (low - high), and the ratio to each end:

| Scene | Half | Session | Job | Note 1 | / low | / high |
|---|---|---|---|---|---|---|
| room | 0 | S1 | 26,415.5 | 24,288 - 38,180 | 1.088 | 0.692 |
| kitchen | 0 | S2 | 32,205.6 | 23,457 - 37,666 | 1.373 | 0.855 |
| drjohnson | 0 | S3 | 26,010.5 | 21,228 - 36,988 | 1.225 | 0.703 |
| bonsai | 0 | S4 | 23,250.4 | 23,636 - 36,645 | 0.984 | 0.634 |
| counter | 0 | S5 | 26,175.5 | 21,880 - 34,430 | 1.196 | 0.760 |
| playroom | 0 | S6 | 22,755.5 | 19,148 - 32,789 | 1.188 | 0.694 |
| truck | 0 | S7 | 16,305.3 | 16,483 - 28,759 | 0.989 | 0.567 |
| kitchen | 1 | S8 | 14,205.3 | 12,169 - 19,873 | 1.167 | 0.715 |
| playroom | 1 | S8 | 10,095.2 | 9,700 - 16,952 | 1.041 | 0.596 |
| bonsai | 1 | S9 | 10,380.2 | 12,231 - 18,760 | 0.849 | 0.553 |
| counter | 1 | S9 | 11,175.2 | 11,348 - 17,493 | 0.985 | 0.639 |
| room | 1 | S10 | 13,020.2 | 12,571 - 19,937 | 1.036 | 0.653 |
| truck | 1 | S10 | 8,430.2 | 8,056 - 14,030 | 1.046 | 0.601 |
| drjohnson | 1 | S11 | 13,440.3 | 10,855 - 20,028 | 1.238 | 0.671 |

Per process, the wall time (s) and its ratio to Note 4 (a)'s guard value for E7 (`j-1_p0` / `j0_p0` / `j+1_p0` / `j0_p1` /
`j-1_p1` / `j+1_p1`):

| Scene | `j-1_p0` | `j0_p0` | `j+1_p0` | `j0_p1` | `j-1_p1` | `j+1_p1` |
|---|---|---|---|---|---|---|
| bonsai | 6,768.8 (0.614) | 7,693.1 (0.553) | 6,435.0 (0.584) | 5,949.1 (0.495) | 1,573.9 (0.522) | 1,517.2 (0.503) |
| counter | 7,587.4 (0.735) | 8,757.9 (0.669) | 7,226.5 (0.700) | 6,662.0 (0.590) | 1,695.1 (0.615) | 1,643.8 (0.596) |
| drjohnson | 7,495.4 (0.726) | 8,556.3 (0.552) | 7,128.2 (0.691) | 7,542.2 (0.550) | 2,010.9 (0.734) | 1,934.4 (0.706) |
| kitchen | 9,522.4 (0.869) | 10,858.9 (0.723) | 9,083.9 (0.829) | 8,388.3 (0.638) | 2,147.0 (0.717) | 2,063.0 (0.689) |
| playroom | 6,621.9 (0.696) | 7,554.4 (0.581) | 6,277.9 (0.660) | 5,737.7 (0.507) | 1,562.8 (0.640) | 1,506.8 (0.617) |
| room | 7,619.4 (0.675) | 8,768.4 (0.589) | 7,113.2 (0.630) | 7,684.9 (0.593) | 1,897.3 (0.608) | 1,891.9 (0.606) |
| truck | 4,741.1 (0.543) | 5,276.8 (0.500) | 4,384.1 (0.502) | 4,665.3 (0.521) | 1,348.1 (0.623) | 1,283.2 (0.593) |

Images per scene (Note 2 C6): the sessions, the number of Kaggle images, and whether arm G's seed-0 curve / its j = 0
seed pair spans images. Across E7: 2 Kaggle images.
`kaggle/PREREG_GN.md:4411`: «**Per-scene image spans** (Note 2 C6)»

| Scene | Sessions | Images | Spans (curve / pair) |
|---|---|---|---|
| bonsai | S4, S9 | 1 | no / no |
| counter | S5, S9 | 1 | no / no |
| drjohnson | S11, S3 | 2 | no / yes |
| kitchen | S2, S8 | 1 | no / no |
| playroom | S6, S8 | 1 | no / no |
| room | S1, S10 | 1 | no / no |
| truck | S10, S7 | 1 | no / no |

### What section 26 settles, and what it does not

- **Settled, as pre-registered:** Δ_refit equivalent; not superior (not fragile (the same outcome under `SE_BD`)): the re-fit does not explain the shrink beyond ±δ. Δ_cb equivalent; not superior
  (not fragile (the same outcome under `SE_BD`)): codebook training does not explain the shrink beyond ±δ.
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
| `image_S1_S10`, `image_S11`, `image_change`, `image_span` | both images named, from the rows' strings against each session's env file; the spans from the rows against the tool | 2785; 3183-3185 |
| `exp_label`, `exp_crit`, `exp_sup_order`, `exp_eq_order`, `block:exp_stats`, `block:exp_sup_rows`, `block:exp_eq_rows`, `block:exp_outcome_rows` | the exploratory sensitivity, not pre-registered, beside the tests | none: `bench/gn/check_s26.py`, `kaggle/HANDOFF.md` |
| `sources_bundles` | the bundles and their commits | 4226-4231 (20 l); notes 3-5 |
