# R33 smoke checklist: what to confirm in session 1's counter log before session 2 (Amendment 21 j)

Session 1 runs `kaggle/drafts/gn_r33_s1_counter.ipynb` (Kaggle title "R33 training-view ratio S1"; GPU T4 x2,
Internet on; attach the dataset that holds `wheels/`, see HANDOFF). Everything below is read from the notebook's
output (the scene's log is printed there), from `/kaggle/working/r33/log_counter.txt`, `r33/setup.json` and
`r33/counter/r33_counter.json`, or from the zip `r33_s1_bundle.zip`. **This is a mechanics check.** The per-fold gains,
r_f and D_f print in the log and will be seen; they count under Amendment 21 and are not judged here (no value decides
anything about session 2).

## 1. Setup

- [ ] `setup.json`: `"commit"` is the pin in HANDOFF; `"session": "s1"`; `"gpu"` is a Tesla T4.
- [ ] The line `wheel key <key> restored` (or `built`, which costs about 73 min and is not a fault); record the key.
  Expected with the cached wheel: `33734f894b0a-95f8de51-torch2.11.0_cu130-sm7.5` if Kaggle's image still has torch
  2.11.0+cu130. A different torch version is not a fault in session 1, but session 2 must have the same key (section 7).
- [ ] `c3dgs_build` has `"ok": true`.

## 2. The guard never fired

A guard that fires raises `Forbidden` and stops the job; there is no silent firing. So:
- [ ] The line `counter: guards installed (scene readers, the test-camera accessor, the evaluation routine, psnr x2, lpips x2)` is there.
- [ ] The word `Forbidden` appears nowhere in the log.
- [ ] The notebook prints `counter gpu 0 exit 0`, and the log ends with `counter: finished`.

## 3. No test camera reached the job

- [ ] `counter: n_train 210, test infos dropped 30, fold sizes [53, 53, 52, 52]`. Counter has 240 images: INRIA's
  split puts every 8th (30) in test (E5's `gn5_meta_counter.json`: `n_test` 30, `n_train` 210). Any other count: stop.
- [ ] `r33_counter.json`: `n_test_dropped` 30, `n_train` 210, and `fold_of` names 210 images. (The test images are
  dropped before any camera exists, so they cannot appear there; whether a name in `fold_of` is a test image can be
  checked at ingest against E5's counter split.)

## 4. Memory against the T4 and the host

- [ ] Each fold line ends `peak cuda <allocated>/<reserved> GB, host <rss> GB`. Reserved must be well under the T4's
  15.6 GB.
- [ ] **Projection to the largest scene (treehill, 3.78 M splats, 3.09 x counter's 1.22 M):** the job's host arrays
  scale with splats: `M_total` (float64) 1.24 GB on counter, 3.83 GB on treehill; each fold's `M_f` (float32) 0.62 and
  1.92 GB; the row selections about as much again. So treehill's host peak is about counter's host peak + 7 GB.
  It must stay under the session's RAM (shown on Kaggle, about 29-31 GB) with a few GB to spare. The GPU side holds no
  `[n, 136]` array beyond one GN pass (as E5-E7's), so the T4 projection is the same as E5's for these scenes.
- If the projection does not fit, **do not start session 2**: write it down; a fix is a code change (section 7).

## 5. Timing and the projected time for the nine

- [ ] Each fold line ends `(<s> s)`; `counter: GN total over 4 folds in <s> s`; the JSON's `seconds_total`.
- [ ] **Projection:** the nine scenes hold 17.69 x counter's splats in total (`e6_scenes.N_SPLATS`), so the nine take
  about 17.7 x counter's `seconds_total`, plus 9 x the data and model fetch (5-15 min each, the notebook's time
  between `counter gpu 0 exit 0` and the setup's end gives counter's), plus setup (C3DGS build 10-15 min; the wheel).
  Splats are a rough proxy (views and resolution also vary; drjohnson keeps its images on the CPU).
- [ ] It must fit one Kaggle session (12 h) with an hour to spare. If it does not, **do not start session 2**: a split
  of the nine into two sessions is a deviation from Amendment 21 j and needs a dated note first (same pin, no code
  change, so counter would not re-run).

## 6. Values present and finite (not judged)

For each of the four folds, in the fold lines and in the JSON's `folds`:
- [ ] `gain_in` and `gain_out` finite.
- [ ] `r_f` finite, **or** `None` exactly when `gain_in <= 0`: that is a legitimate outcome under Amendment 21 d (the
  fold is left out of r_s), not a fault. A `None` with `gain_in > 0`, or a `nan`/`inf`, is a fault.
- [ ] `D_f` finite and between 0 and about 2 (it is a ratio of norms; 1 would mean the held-out metric is as far from the
  in-fold one as the in-fold one is from zero).
- [ ] `D_probe` present on fold 0 only, finite (the JSON also has it at top level).
- [ ] `GN checksum rel diff` small (below about 1e-4): fold f's recomputed metric equals the first pass up to the GPU's
  summation order. A large value means the recomputation did not reproduce the fold, which is a fault.
- [ ] `n_cq` (colour-quantized splats) similar across folds, `ogc iters 15` (OGC's `info`).
- [ ] The final JSON block has `ratio`, `ratio_se`, `D_s`, `D_probe`, `seconds_total`.

## 7. The bundle

- [ ] The notebook's output holds `r33_s1_bundle.zip` with `setup.json`, `log_counter.txt` and `counter/r33_counter.json`.
- [ ] Create the Kaggle dataset **"R33 S1 bundle"** from it (as "E7 S1 bundle" was named). It is not attached to
  session 2 (the nine do not read counter's output).
- [ ] Ingest into the repository as data only (`kaggle/gn_r33/S1/`, through `gated_commit.py`, EOL-exact as the E7
  bundles: the zip entry and the committed blob byte-identical).

## 8. What counts as a code change (counter then re-runs in session 2)

Counter re-runs with the nine, the session-2 counter counts and session 1's values are reported beside it, labelled,
if between the sessions **any** of these changes:
- the pin, or any file in Amendment 21 k's list (`kaggle/drafts/r33_cv_ratio.py`, `kaggle/e3r_hooks.py`,
  `kaggle/e3q_c3dgs_run.py`, `kaggle/e3q_c3dgs.py`, `kaggle/e3p_inria.py`, `kaggle/e5_data.py`, `kaggle/e6_scenes.py`,
  `bench/gn/gn_metric.py`, `bench/gn/ogc_derived.py`, `bench/gn/e3r.py`, `gsplat/`);
- the environment, so that the wheel key differs (torch version or GPU) or C3DGS's build differs (`c3dgs_build`).

**Not** a code change (recorded in HANDOFF, counter does not re-run): a notebook-only change that leaves the pin and
the environment as they were, such as attaching a dataset, a fetch retry, or toggling `PARALLEL`. Building the session-2
notebook from the same builder at the same pin is the declared procedure, not a change.

**If session 1 does not finish** (no `counter: finished`): read nothing from it; fix, commit, re-pin, and counter runs
in session 2 under the new pin.
