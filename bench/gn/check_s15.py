"""Re-check every number in kaggle/FINDINGS.md section 15 (E3r), its summary paragraph and its Sources entry
against the committed files.

    python bench/gn/check_s15.py

It reads only committed files: the E3r bundle (``kaggle/gn_e3r/gn3r/``), the memory check
(``kaggle/gn_e3r_memory/e3r_memory.json``), the runtime estimate (``bench/gn/e3r_estimate.py``, ``estimate_g``), E3p's
rows and metas (``kaggle/gn_e3p/gn3p/``), E3q attempt 2's rows (``kaggle/gn_e3q/attempt2/gn3q/``), E2c's rows
(``kaggle/gn_e2c/gn2c/``, iteration counts) and this repository's own hook, wrapper and diagnostics source for the
file:line citations. C3DGS's line numbers were read from its source at ``2a234af5``, which is not in the repository;
they are listed constants. Each quoted number is recomputed and must appear in the text, each claim is re-asserted,
and every numeric token of the text must be a recomputed string or a listed constant. It prints each failure and
exits 1 if there is any, 0 otherwise. Built like ``check_s14.py``.
"""

import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
E3R = os.path.join(REPO, "kaggle", "gn_e3r", "gn3r")
MEM = os.path.join(REPO, "kaggle", "gn_e3r_memory", "e3r_memory.json")
E3P = os.path.join(REPO, "kaggle", "gn_e3p", "gn3p")
E3Q = os.path.join(REPO, "kaggle", "gn_e3q", "attempt2", "gn3q")
E2C = os.path.join(REPO, "kaggle", "gn_e2c", "gn2c")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
# the amendments', protocols' and codecs' own constants, dates, names and the SH identity, not results
CONSTANTS = {"0", "1", "2", "3", "4", "12", "13", "14", "15", "16", "30", "4,096", "8,192", "2^20", "10^6", "0.6e-6",
             "5,000", "2026-09-30", "1e-3", "0.5-2x", "-2", "+2"}
# C3DGS's source at 2a234af5 (compression/vq.py, compress.py), read for the diagnosis; not in this repository
C3DGS_LINES = {"27", "35", "98", "116", "210", "187", "192", "200", "206", "40", "52", "178", "188", "207", "221"}
COMMITS = {"693a6a4b", "a850c4ea", "b06f9293", "e1e0309e", "9c28cc89", "2a234af5", "673a963a", "b27e1421"}
RHO_LABEL = {0.0: "0", 0.001: "1e-3", 0.01: "1e-2", 0.1: "1e-1", 0.3: "3e-1", 1.0: "1", 3.0: "3"}


def section_text(txt: str) -> str:
    start = txt.index("## 15. E3r")
    nxt = txt.find("\n## ", start + 1)
    sec = txt[start:] if nxt < 0 else txt[start:nxt]
    s0 = txt.index("**E3r (section 15")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 15:")
    end = txt.find("\n- Section", s1 + 1)
    src = txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


def fmt_e(v: float) -> str:
    """0.6e-6 x 3^j as the table writes it: three significant digits, no exponent padding (6.67e-8)."""
    m, e = f"{v:.3g}".split("e")
    return f"{m}e{int(e)}"


def e1(v: float) -> str:
    """One decimal, no exponent padding (8.0e-9)."""
    m, e = f"{v:.1e}".split("e")
    return f"{m}e{int(e)}"


def lines_of(path: str, a: int, b: int) -> str:
    return "".join(open(os.path.join(REPO, path), encoding="utf-8").readlines()[a - 1:b])


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    exp, fails = set(), []
    f = float

    def want(s, why=""):
        exp.add(s)
        if s not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    def rows_of(path):
        return {r["config"]: r for r in csv.DictReader(open(path, newline=""))}

    R = rows_of(os.path.join(E3R, "gn3r_results_train.csv"))
    RB = rows_of(os.path.join(E3R, "gn3r_results_bicycle.csv"))
    M = {s: json.load(open(os.path.join(E3R, f"gn3r_meta_{s}.json"))) for s in ("bicycle", "train")}
    S = json.load(open(os.path.join(E3R, "gn3r_summary.json")))
    ST = S["scenes"]["train"]
    env = json.load(open(os.path.join(E3R, "gn3r_env.json")))
    tim = json.load(open(os.path.join(E3R, "timings.json")))
    build = json.load(open(os.path.join(E3R, "gn3r_c3dgs_build.json")))
    tails = {s: json.load(open(os.path.join(E3R, f"gn_e3r_{s}_log_tail.json"))) for s in ("bicycle", "train")}
    cv = {}
    for p in glob.glob(os.path.join(E3R, "gn3r_cv_rho*_train.json")):
        r = json.load(open(p))
        cv[RHO_LABEL[r["rho"]]] = r
    runs = M["train"]["runs"]
    steps = {s: {x["name"]: x for x in M[s]["steps"]} for s in M}
    p, g, ft = R["c3dgs_k4096"], R["gnvq_k4096"], R["gnvq_k4096_ft5000"]
    KS = ["c3dgs_k1024", "c3dgs_k4096", "c3dgs_k16384", "c3dgs_k65536"]
    JS = ["c3dgs_k4096_j-2", "c3dgs_k4096_j-1", "c3dgs_k4096_j+1", "c3dgs_k4096_j+2"]
    C3 = KS + JS + ["gnvq_k4096", "gnvq_k4096_ft5000"]

    # what ran
    check(all(t["exit_code"] == 0 and not t["skipped_by_cutoff"] for t in tails.values()), "both jobs exit 0, not skipped")
    check(len(R) == 18 and all(r["status"] == "ok" for r in R.values()) and list(RB) == ["uncompressed"]
          and RB["uncompressed"]["status"] == "ok", "18 train rows ok, one bicycle row")
    want(f"wrote all {len(R)} rows", "rows")
    for s in M:
        sc = S["scenes"][s]
        check(M[s]["done"] and not M[s]["failed_steps"] and not M[s]["skipped_steps"] and not M[s]["missing_or_failed"]
              and sc["missing_or_failed"] == [] and all(x["status"] == "ok" for x in M[s]["steps"]), f"{s}: nothing failed")
    check(set(runs) == set(C3) and all(R[c]["data_device"] == "cuda" and R[c]["retried_on_cpu"] == "False" for c in C3)
          and all(runs[c]["wrapper"]["seed"] == 0 for c in C3), "10 C3DGS runs on cuda, no retry, seeded 0")
    want(f"all {len(C3)} C3DGS runs ran with `--data_device cuda`", "runs")

    # inputs and checks
    check(env["gpu"]["count"] == 2 and env["gpu"]["name"] == "Tesla T4" and env["wheel_restored"], "2x T4, restored wheel")
    want(f"torch {env['torch']} (CUDA {env['torch_cuda']}), driver {env['nvidia_smi'].split(', ')[1]}, cuDNN {env['cudnn']}, "
         f"Python {env['python']}, {env['gpu']['count']}x Tesla T4; gsplat commit `{env['gsplat_commit'][:8]}`, the restored "
         f"wheel (install {tim['install_s']:.1f} s)", "session")
    e3pm = {s: json.load(open(os.path.join(E3P, f"gn3p_meta_{s}.json"))) for s in ("bicycle", "train")}
    for s in M:
        mem = M[s]["inria"]["members"]
        check(len(mem) == 3 and all(v["source"] == "present" and v["bytes"] == v["bytes_pinned"] and v["crc32"] == v["crc32_pinned"]
                                    for v in mem.values()), f"{s}: members present, pins match")
        check(mem["ply"]["sha1"] == e3pm[s]["inria"]["members"]["ply"]["sha1"], f"{s}: ply SHA-1 = E3p's")
        want(f"SHA-1 `{mem['ply']['sha1'][:8]}`, E3p's", f"{s} sha1")
        check(M[s]["cfg_args_mismatch"] == [], f"{s}: cfg_args")
        cf, sp = M[s]["camera_frame_check"], M[s]["split_check"]
        check(cf["pass"] and cf["n_matched"] == cf["n_runner"] == cf["n_cameras_json"] and sp["equals_cameras_json_head"], f"{s}: frame, split")
        pos, rot = f"{cf['max_position_diff']:.2e}", f"{cf['max_rotation_diff']:.2e}"
        want(f"pass: {cf['n_matched']} of {cf['n_runner']}", f"{s} frame")
        want(pos, f"{s} position")
        want(rot, f"{s} rotation")
        want(f"equal, {sp['n_test']} test views", f"{s} split")
        u = (RB if s == "bicycle" else R)["uncompressed"]
        e3p_u = rows_of(os.path.join(E3P, f"gn3p_results_{s}.csv"))["uncompressed"]
        check(all(u[k] == e3p_u[k] for k in ("PSNR_ii", "SSIM_ii", "LPIPS_ii")), f"{s}: uncompressed = E3p's, every digit")
        w, h = json.loads(u["resolution_ii"])[0]
        want(f"{f(u['PSNR_ii']):.3f} / {f(u['SSIM_ii']):.4f} / {f(u['LPIPS_ii']):.4f} at {w}x{h}", f"{s} uncompressed")
    want("1.33e-15 (position), 3.33e-16 (rotation)", "bicycle frame text")
    bs = {x["name"]: x for x in build["steps"]}
    check(build["ok"] and build["failed_step"] is None and len(build["deviations"]) == 6
          and not any(n.endswith(("_current", "_cstdint")) for n in bs) and bs["torch_scatter"]["built_from_source"] is False
          and all(v["ok"] for v in build["imports"].values()) and build["head"] == "2a234af55fbe8b90c8829c1436ce80088c4b622b"
          and build["submodules"][0].startswith("673a963a"), "build ok, no fallback, six deviations, wheel")
    want(f"ok in {build['total_time_s']:.1f} s, head `2a234af5` with glm `673a963a`, {len(build['imports'])} of "
         f"{len(build['imports'])} imports", "build")
    want(f"took {bs['diff_gaussian_rasterization']['time_s']:.1f} s and {bs['weighted_distance']['time_s']:.1f} s, and "
         f"`torch-scatter` came from a PyG wheel ({bs['torch_scatter']['time_s']:.1f} s)", "build steps")
    eig, dets = [], []
    for c in C3:
        lp = runs[c]["wrapper"]["linalg_patch"]
        e, d = lp["calls"]
        check(lp["n_chunked_calls"] == {"eigh": 1, "det": 1} and lp["fallbacks"] == [] and e["n"] == d["n"] == 281237
              and e["dtype"] == "torch.float32" and e["shape"][1:] == [3, 3] and e["reductions"] == d["reductions"] == 0
              and lp["working_batches"]["linalg_eigh"] == lp["working_batches"]["det"] == 8192
              and e["check"]["n_sampled"] == d["check"]["n_sampled"] == 4096, f"{c}: patch record")
        eig.append(e["check"])
        dets.append(f"{d['check']['max_abs_det_diff']:.2e}")
    want(f"each on {281237:,} float32 3x3 matrices", "batch")
    ed = sorted(x["max_abs_eigenvalue_diff"] for x in eig)
    ev = sorted(x["max_abs_eigenvalue"] for x in eig)
    want(f"from {ed[0]:.2e} to {ed[-1]:.2e}, against largest eigenvalues of {ev[0]:.2f} to {ev[-1]:.2f}", "eigh check")
    check(len(set(dets)) == 1, "det check equal in every run")
    want(f"`det`: {dets[0]} in every run", "det check")

    # the baseline table
    def thr(r):
        j = int(r["threshold_j"])
        return f"{fmt_e(f(r['color_importance_include']))} ({j:+d})" if j else f"{fmt_e(f(r['color_importance_include']))} (0)"

    def trip(r, pre):
        return f"{f(r[pre + 'PSNR']):.3f} / {f(r[pre + 'SSIM']):.4f} / {f(r[pre + 'LPIPS']):.4f}"

    def ii(r):
        return f"{f(r['PSNR_ii']):.3f} / {f(r['SSIM_ii']):.4f} / {f(r['LPIPS_ii']):.4f}"

    for c in KS + JS:
        r = R[c]
        b = int(r["npz_bytes"])
        check(abs(f(r["size_MiB"]) - b / 2 ** 20) < 1e-9 and abs(f(r["size_MB"]) - b / 1e6) < 1e-9
              and f(r["size_MiB"]) == f(r["c3dgs_size_MiB_reported"]) and r["finetune_iterations"] == "0", f"{c}: sizes")
        name = f"`{c}` (the probe run)" if c == "c3dgs_k4096" else f"`{c}`"
        want(f"| {name} | {int(r['color_codebook_size']):,} | {thr(r)} | {int(r['n_pruned']):,} / {int(r['n_kept_colour']):,} / "
             f"{int(r['n_colour_quantized']):,} | {b:,} | {f(r['size_MiB']):.3f} | {f(r['size_MB']):.3f} | {trip(r, 'c3dgs_')} | {ii(r)} |", c)
    kk, kt = ST["knob_K"], ST["knob_threshold"]
    check(kk["configs"] == KS and sorted(kt["configs"]) == sorted(["c3dgs_k4096"] + JS), "knob configs")
    for lab, kn, n in (("K, 1,024-65,536", kk, 4), ("threshold, j = -2 to +2", kt, 5)):
        check(kn["n_points"] == n, f"{lab}: points")
        want(f"| {lab} | {n} | {int(kn['bytes_min']):,}-{int(kn['bytes_max']):,} | {kn['bytes_max_over_min']:.3f} | "
             f"{kn['PSNR_ii_min']:.3f}-{kn['PSNR_ii_max']:.3f} | {kn['c3dgs_PSNR_min']:.3f}-{kn['c3dgs_PSNR_max']:.3f} |", lab)
    for seq in (KS, ["c3dgs_k4096_j+2", "c3dgs_k4096_j+1", "c3dgs_k4096", "c3dgs_k4096_j-1", "c3dgs_k4096_j-2"]):
        for k in ("npz_bytes", "PSNR_ii", "c3dgs_PSNR"):
            v = [f(R[c][k]) for c in seq]
            check(all(a < b for a, b in zip(v, v[1:])), f"monotone {k} over {seq}")
    want(f"from {int(R['c3dgs_k4096_j+2']['n_kept_colour']):,} splats keeping their own colour at j = +2 to "
         f"{int(R['c3dgs_k4096_j-2']['n_kept_colour']):,} at j = -2", "kept range")
    check(len({R[c]["n_pruned"] for c in C3}) == 1, "the same pruning in every run")
    want(f"Pruning is the same {int(p['n_pruned']):,} splats in every run", "pruning")
    cl = [f(R[c]["c3dgs_clustering_s"]) for c in KS]
    want(f"{cl[0]:.1f} s at K = 1,024, {cl[1]:.1f} s at 4,096, {cl[2]:.1f} s at 16,384 and {cl[3]:,.1f} s at 65,536", "clustering")
    cj = [f(R[c]["c3dgs_clustering_s"]) for c in JS]
    want(f"The four other thresholds took {min(cj):.1f}-{max(cj):.1f} s at K = 4,096", "threshold clustering")
    want(f"that is {cl[0] / cl[1]:.2f}, {cl[2] / cl[1]:.2f} and {cl[3] / cl[1]:.2f} times the K = 4,096 time", "clustering ratios")

    # the colour-quantized splats
    cs = ST["colour_share"]
    check(cs["n_ckpt"] == int(p["n_ckpt"]) and cs["n_pruned"] == int(p["n_pruned"]) and cs["n_colour_quantized"] == cs["n_selected"]
          == int(p["n_colour_quantized"]) and cs["n_ckpt"] == cs["n_pruned"] + cs["n_kept_colour"] + cs["n_colour_quantized"], "counts")
    want(f"pruned {cs['n_pruned']:,} of train's {cs['n_ckpt']:,} splats. Of the rest, {cs['n_kept_colour']:,} kept their own colour "
         f"and {cs['n_colour_quantized']:,} ({cs['n_colour_quantized'] / cs['n_ckpt'] * 100:.2f}% of the checkpoint)", "split")
    want(f"a share of {cs['16x16']['share']:.8f}", "share")
    for lab, k in (("16 x 16 (bands 0-3)", "16x16"), ("15 x 15 (bands 1-3, the frozen metric)", "15x15")):
        c = cs[k]
        want(f"| {lab} | {c['total_trace']:,.2f} | {c['selected_trace']:,.2f} | {c['share']:.10f} |", k)
    want(f"differ by {abs(cs['16x16']['share'] - cs['15x15']['share']):.1e}", "share difference")
    rt = cs["16x16"]["total_trace"] / cs["15x15"]["total_trace"] / (16 / 15) - 1
    rs = cs["16x16"]["selected_trace"] / cs["15x15"]["selected_trace"] / (16 / 15) - 1
    want(f"16/15 times (1 + {e1(rt)}), and the quantized splats' 16/15 times (1 + {e1(rs)})", "trace ratios")

    # the pruned-splat trace count
    pc = ST["pruned_trace_check"]
    check(pc["status"] == "ok" and pc["mask_source"] == "probe" and pc["n_splats"] == cs["n_ckpt"] and pc["n_pruned"] == cs["n_pruned"]
          and pc["n_tr0_all"] == M["train"]["gn"]["full"]["splats_zero_trace"], "pruned_trace_check")
    check(S["scenes"]["bicycle"]["pruned_trace_check"]["status"] == "not_applicable", "bicycle not_applicable")
    want(f"| {pc['n_splats']:,} | {pc['n_pruned']:,} | {pc['n_pruned_tr0']:,} | {pc['n_tr0_all']:,} |", "ptc row")
    want(f"{pc['n_pruned_tr0']:,} of the {pc['n_pruned']:,} pruned splats ({pc['n_pruned_tr0'] / pc['n_pruned'] * 100:.2f}%)", "ptc share")
    want(f"nonzero trace:** {pc['n_pruned'] - pc['n_pruned_tr0']:,}", "ptc pruned tr>0")
    want(f"zero trace:** {pc['n_tr0_all'] - pc['n_pruned_tr0']:,} (", "ptc unpruned tr0")
    want(f"The trace share of the {pc['n_pruned'] - pc['n_pruned_tr0']:,} is not recoverable", "ptc not recoverable")
    want(f"how the {pc['n_pruned'] - pc['n_pruned_tr0']:,} pruned splats with a nonzero trace split", "settles ptc")

    # GN-VQ inside C3DGS: the cross-validation
    mt = M["train"]
    want(f"GN-VQ on the {int(p['n_colour_quantized']):,} probe-run splats with the even-view 16 x 16 metric ({mt['n_even_views']} views)",
         "cv set")
    want(f"on the {mt['n_odd_views']} odd-indexed train views", "odd views")
    sc = ST["cv_odd_scores"]
    order = ["0", "1e-3", "1e-2", "1e-1", "3e-1", "1", "3"]
    check(all(abs(sc[k] - cv[k]["measured_odd"]["clamped"]) == 0 for k in order), "scores = CV reports")
    best = min(order, key=lambda k: (sc[k], order.index(k)))
    check(RHO_LABEL[ST["rho_cv"]] == best == "1e-2", "rho_cv is the argmin")
    want("| clamped (the score) | " + " | ".join(f"**{sc[k]:.4e}**" if k == best else f"{sc[k]:.4e}" for k in order) + " |", "scores")
    want("| unclamped | " + " | ".join(f"{cv[k]['measured_odd']['raw']:.4e}" for k in order) + " |", "raw")
    want(f"**`rho_cv` is {best},** ahead of 1e-1 by {(sc['1e-1'] / sc[best] - 1) * 100:.2f}% of its score", "margin")
    want(f"`rho_cv` was {best}", "summary rho")
    e3p_rows = [r for r in csv.DictReader(open(os.path.join(E3P, "gn3p_results_train.csv"), newline=""))]
    check(RHO_LABEL[f(e3pm["train"]["rho_cv"])] == "1e-1" if "rho_cv" in e3pm["train"] else
          any(r["config"] == "gn_vq_cvfloor" and f(r["rho"]) == 0.1 for r in e3p_rows), "E3p train rho_cv 1e-1")

    # iterations
    inj = {c: runs[c]["wrapper"]["e3r"]["inject"] for c in ("gnvq_k4096", "gnvq_k4096_ft5000")}
    allv = [cv[k] for k in order] + [inj[c]["gn_vq"] for c in inj]
    check(len(allv) == 9 and all(v["iterations"] == v["max_iters"] == 20 and v["stopped_because"] == "max_iters"
                                 and v["rel_tol"] == 1e-3 for v in allv), "all 9 at max_iters 20")
    check(all(R[c]["vq_iterations"] == "20" and R[c]["vq_stopped_because"] == "max_iters"
              for c in R if R[c]["vq_iterations"]) and sum(1 for c in R if R[c]["vq_iterations"]) == 9, "csv agrees")
    want(f"All {len(allv)} GN-VQ runs", "nine")
    want(f"all {len(allv)} GN-VQ runs stopped at the {allv[0]['max_iters']}-iteration cap", "summary nine")
    want(f"all {len(allv)} runs stopped at the {allv[0]['max_iters']}-iteration cap", "settles cap")
    last = sorted(v["history"][-1]["relative_drop"] for v in (cv[k] for k in order))
    li = [inj[c]["gn_vq"]["history"][-1]["relative_drop"] for c in inj]
    check(all(x > 1e-3 for x in last + li), "still falling above 1e-3")

    def e3(x):
        m, e = f"{x:.2e}".split("e")
        return f"{m}e{int(e)}"

    want(f"were {e3(last[0])} to {e3(last[-1])} in the CV runs, and {e3(li[0])} and {e3(li[1])} in the two injected", "last drops")
    first = {k: [h for h in cv[k]["history"] if h["step"] == "update"][0] for k in order}
    lo_k = min(order, key=lambda k: first[k]["relative_drop"])
    hi_k = max(order, key=lambda k: first[k]["relative_drop"])
    want(f"cut the floored objective by {first[lo_k]['relative_drop'] * 100:.1f}% (`rho` = {lo_k}) to "
         f"{first[hi_k]['relative_drop'] * 100:.1f}% (`rho` = {hi_k})", "first-iteration range")
    b0 = cv[best]
    want(f"it by {first[best]['relative_drop'] * 100:.1f}% ({b0['warm_start']['objective']:.4e} to {first[best]['objective']:.4e}), "
         f"and 20 iterations reached {b0['objective_before_quantization']:.4e}", "first iteration at rho_cv")
    it3p = sorted({int(r["vq_iterations"]) for r in e3p_rows if r.get("vq_iterations")})
    check(all(r["vq_stopped_because"] == "rel_tol" for r in e3p_rows if r.get("vq_iterations")), "E3p train: rel_tol")
    want(f"took {it3p[0]}-{it3p[-1]} iterations, all stopping at the relative-drop rule", "E3p iterations")
    e2c = []
    for pth in sorted(glob.glob(os.path.join(E2C, "gn2c_results_*.csv"))):
        e2c += [r for r in csv.DictReader(open(pth, newline="")) if r["config"].startswith("gn_vq_cvfloor")]
    c4 = [r for r in e2c if r["n_clusters"] == "4096"]
    cvi = sorted({int(r["vq_iterations"]) for r in c4 if r["config"] == "gn_vq_cvfloor_cv"})
    fni = sorted({int(r["vq_iterations"]) for r in c4 if r["config"] == "gn_vq_cvfloor"})
    check(all(r["vq_stopped_because"] == "rel_tol" for r in c4), "E2c K=4096: all rel_tol")
    want(f"E2c at K = 4,096 took {cvi[0]}-{cvi[-1]} iterations in its CV runs and {fni[0]}-{fni[-1]} in its final runs", "E2c 4096")
    f1 = [r for r in e2c if r["n_clusters"] == "1024" and r["config"] == "gn_vq_cvfloor"]
    check(len(f1) == 5 and all(r["vq_iterations"] == "20" and r["vq_stopped_because"] == "max_iters" for r in f1), "E2c 1024 finals")
    want(f"E2c at K = 1,024: all {len(f1)} final runs stopped at 20", "E2c 1024")

    # the injected runs
    for c in inj:
        i = inj[c]
        sv = i["set_vs_probe"]
        check(sv["same_set"] and sv["only_probe"] == sv["only_injected"] == 0 and sv["features_equal"]
              and R[c]["set_same_as_probe"] == "True", f"{c}: set, features")
        check(i["quantizer_at_injection"] == runs["c3dgs_k4096"]["wrapper"]["e3r"]["qa_at_colour_vq"], f"{c}: qa at injection")
        check(i["lifted_check"]["pass"] and i["lifted_check"]["n"] == 10000 and i["lifted_check"]["criterion_version"] == 2, f"{c}: lifted")
        check(i["metric_device_bytes"] == 544 * int(p["n_colour_quantized"]), f"{c}: metric bytes")
    a, b = inj["gnvq_k4096"], inj["gnvq_k4096_ft5000"]
    want(f"| Injection (GN-VQ itself) | {a['time_s']:.1f} s ({a['gn_vq']['time_s']:.1f} s) | {b['time_s']:.1f} s ({b['gn_vq']['time_s']:.1f} s) |",
         "inject time")
    want("| " + " | ".join(f"{i['gn_vq']['warm_start']['objective']:.4e}, {i['gn_vq']['objective_before_quantization']:.4e}, "
                           f"{i['gn_vq']['objective_after_quantization']:.4e}" for i in (a, b)) + " |", "objectives")
    want("| " + " | ".join(f"{i['gn_vq']['final_assignment_labels_changed_fraction'] * 100:.2f}%" for i in (a, b)) + " |", "labels changed")
    want(f"Lifted check, {a['lifted_check']['n']:,} splats", "lifted n")
    want(f"pass: sum excess / sum d_min {a['lifted_check']['sum_excess_over_sum_dmin']:.2e}, same index "
         f"{a['lifted_check']['same_index_fraction']:.1f} | pass: {b['lifted_check']['sum_excess_over_sum_dmin']:.2e}, "
         f"{b['lifted_check']['same_index_fraction']:.1f} |", "lifted")
    want(f"{a['metric_device_bytes']:,} bytes, one copy for the {int(p['n_colour_quantized']):,} quantized splats "
         f"({a['metric_device_bytes'] // int(p['n_colour_quantized'])} bytes each)", "metric")
    qa0 = runs["c3dgs_k4096"]["wrapper"]["e3r"]["qa_at_colour_vq"]
    check(all(runs[c]["wrapper"]["e3r"]["qa_at_colour_vq"] == qa0 for c in C3), "the same quantizer in all 10 runs")
    check(all(runs[c]["wrapper"]["e3r"]["qa_at_save"] == qa0 for c in C3 if c != "gnvq_k4096_ft5000"), "unchanged without fine-tuning")
    want(f"the same in all {len(C3)} runs", "qa all runs")
    want(f"DC: scale {qa0['dc_scale']:.6f}, zero point {qa0['dc_zero_point']};", "qa dc")
    want(f"AC: scale {qa0['rest_scale']:.7f}, zero point {qa0['rest_zero_point']}.", "qa ac")
    cal = ST["calibration"]
    check(RHO_LABEL[cal["rho"]] == best and cal["n_views"] == mt["n_even_views"] and cal["report_only"]
          and abs(cal["predicted"] / cv[best]["objectives_under"]["M"]["objective_after_quantization"] - 1) < 1e-6,
          "calibration's P is the rho_cv codebook's objective under M_even (to 1e-6: computed separately)")
    want(f"`P` = {cal['predicted']:.4e} against the measured unclamped dMSE of {cal['measured_even_raw']:.4e} (clamped "
         f"{cal['measured_even_clamped']:.4e}), a ratio of {cal['ratio_raw']:.3f}", "calibration")
    check(0.5 <= cal["ratio_raw"] <= 2, "inside 0.5-2x")

    # the injected row against the probe
    d = {k: f(g[k]) - f(p[k]) for k in ("PSNR_ii", "c3dgs_PSNR", "SSIM_ii", "c3dgs_SSIM", "LPIPS_ii", "c3dgs_LPIPS")}
    want(f"| PSNR (dB) | {d['PSNR_ii']:+.4f} | {d['c3dgs_PSNR']:+.4f} |", "dPSNR")
    want(f"| SSIM | {d['SSIM_ii']:+.4f} | {d['c3dgs_SSIM']:+.4f} |", "dSSIM")
    want(f"| LPIPS | {d['LPIPS_ii']:+.4f} | {d['c3dgs_LPIPS']:+.4f} |", "dLPIPS")
    nb, npb = int(g["npz_bytes"]), int(p["npz_bytes"])
    want(f"| `.npz` bytes | {nb - npb:+,} ({(nb / npb - 1) * 100:+.3f}%) |", "dbytes")
    want(f"read {d['PSNR_ii']:+.3f} dB in protocol ii at {(nb / npb - 1) * 100:+.3f}% bytes", "summary delta")
    sha = {c: runs[c]["wrapper"]["e3r"]["geometry_sha1"] for c in C3}
    check(len(set(sha.values())) == len(C3) and all(sha[c] == R[c]["geometry_sha1"] for c in C3)
          and ST["probe_geometry_sha1"] == sha["c3dgs_k4096"], "all 10 SHA-1s differ")
    want(f"`{sha['c3dgs_k4096'][:8]}` (probe) against `{sha['gnvq_k4096'][:8]}` (`gnvq_k4096`); the fine-tuned injected run has "
         f"`{sha['gnvq_k4096_ft5000'][:8]}`. All {len(C3)} runs' SHA-1s differ", "sha1s")
    want(f"(`{sha['gnvq_k4096'][:8]}` against `{sha['gnvq_k4096_ft5000'][:8]}`)", "two injected sha1s")
    rng = {"c3dgs_k4096": cv[best]["warm_start"]["quantizer"]["dc"],
           "gnvq_k4096": a["gn_vq"]["warm_start"]["quantizer"]["dc"],
           "gnvq_k4096_ft5000": b["gn_vq"]["warm_start"]["quantizer"]["dc"]}
    check(all(cv[k]["warm_start"]["quantizer"]["dc"] == rng["c3dgs_k4096"] for k in order), "every CV report holds the probe's codebook")
    check(len({(v["codebook_min"], v["codebook_max"]) for v in rng.values()}) == 3, "three different warm starts")
    check(a["gn_vq"]["warm_start"]["objective"] != b["gn_vq"]["warm_start"]["objective"], "warm objectives differ")
    for c, lab in (("c3dgs_k4096", "`c3dgs_k4096` (probe; its record, via the CV reports)"), ("gnvq_k4096", "`gnvq_k4096`"),
                   ("gnvq_k4096_ft5000", "`gnvq_k4096_ft5000`")):
        want(f"| {lab} | [{rng[c]['codebook_min']:.6f}, {rng[c]['codebook_max']:.6f}] |", f"{c} dc range")
    ng = {c: [x["n"] for x in runs[c]["wrapper"]["linalg_patch"]["calls"]] for c in ("c3dgs_k4096", "gnvq_k4096", "gnvq_k4096_ft5000")}
    check(all(v == [281237, 281237] for v in ng.values()) and all(R[c]["n_colour_quantized"] == p["n_colour_quantized"]
                                                                  for c in ng), "equal draw shapes")
    want(f"the same {int(p['n_colour_quantized']):,} colour splats at K = 4,096, and a geometry batch of {281237:,} in each", "draw shapes")
    # this repository's own citations: the cited lines hold what the text says they hold
    check("self.orig[\"vq_features\"](*args, **kwargs)" in lines_of("kaggle/e3r_hooks.py", 137, 137), "e3r_hooks.py:137")
    h = lines_of("kaggle/e3r_hooks.py", 150, 160)
    check(h.lstrip().startswith("def compress_covariance") and "_gaussian_indices, g._rotation, g._scaling" in h
          and "geometry_sha1" in h, "e3r_hooks.py:150-160")
    check(lines_of("kaggle/e3r_hooks.py", 195, 195).strip().startswith("def _inject")
          and lines_of("kaggle/e3r_hooks.py", 239, 239).strip() == "return C, L", "e3r_hooks.py:195-239")
    inj_src = lines_of("kaggle/e3r_hooks.py", 195, 239)
    check(not re.search(r"\brand|manual_seed|np\.random|random\.", inj_src), "no draw in _inject")
    dg = lines_of("bench/gn/diagnostics.py", 442, 443)
    check("torch.Generator(device=\"cpu\").manual_seed(seed)" in dg and "randperm(" in dg and "generator=g" in dg, "diagnostics.py:442-443")
    wr = lines_of("kaggle/e3q_c3dgs_run.py", 227, 243)
    check(wr.lstrip().startswith("if a.observe or record or inject") and "torch.manual_seed(a.seed)" in wr
          and wr.index("e3r_hooks.install") < wr.index("torch.manual_seed"), "e3q_c3dgs_run.py:227-243")
    ck = lines_of("kaggle/e3q_c3dgs_run.py", 165, 166)
    check("torch.Generator().manual_seed(self.check_seed)" in ck and "generator=g" in ck, "e3q_c3dgs_run.py:165-166")
    for mod in ("e3r", "metric_store", "diagnostics", "gn_vq", "e2b", "g1", "g2", "gn_metric", "sh_basis", "batched"):
        src = open(os.path.join(HERE, f"{mod}.py"), encoding="utf-8").read()
        for mm in re.finditer(r"torch\.(rand|randn|randint|randperm|rand_like|randn_like|multinomial|bernoulli|normal)\(([^)]*)", src):
            check("generator" in mm.group(2), f"{mod}.py: a global draw {mm.group(0)[:60]}")
        check(not re.search(r"np\.random\.|(?<![\w.])random\.(random|randint|shuffle|choice|seed)", src), f"{mod}.py: numpy/python random")
    for cite in ("kaggle/e3r_hooks.py:150-160", "`kaggle/e3r_hooks.py:137`", "(`:195-239`)", "`bench/gn/diagnostics.py:442-443`",
                 "`kaggle/e3q_c3dgs_run.py:227-243`", "(`:165-166`)"):
        want(cite, "citation")
    e3q = rows_of(os.path.join(E3Q, "gn3q_results_train.csv"))
    q0, q5 = e3q["c3dgs_ft0"], e3q["c3dgs_ft5000"]
    want(f"the probe run is {npb - int(q0['npz_bytes']):,} bytes larger ({(npb / int(q0['npz_bytes']) - 1) * 100:+.2f}%) and reads "
         f"{f(p['PSNR_ii']) - f(q0['PSNR_ii']):+.3f} dB in protocol ii and {f(p['c3dgs_PSNR']) - f(q0['c3dgs_PSNR']):+.3f} dB", "E3q context")

    # fine-tuning
    sv = runs["gnvq_k4096_ft5000"]["wrapper"]["e3r"]["save_check"]
    check(sv["labels_survived"] and sv["n_labels_changed"] == 0 and R["gnvq_k4096_ft5000"]["finetune_iterations"] == "5000", "labels survived")
    check(runs["gnvq_k4096"]["wrapper"]["e3r"]["save_check"]["codebook_max_abs_change"] == 0.0, "no-ft injected table unmoved")
    want(f"5,000 fine-tuning iterations ({f(ft['c3dgs_finetune_s']):.1f} s)", "ft time")
    want(f"({sv['n_labels_changed']} changed)", "labels changed")
    want(f"changed by up to {sv['codebook_max_abs_change']:.4f} and the kept rows by up to {sv['kept_rows_max_abs_change']:.4f}", "moved")
    qs = runs["gnvq_k4096_ft5000"]["wrapper"]["e3r"]["qa_at_save"]
    want(f"DC: scale {qa0['dc_scale']:.6f} to {qs['dc_scale']:.6f}, zero point {qa0['dc_zero_point']} to {qs['dc_zero_point']};", "ft dc")
    want(f"AC: scale {qa0['rest_scale']:.7f} to {qs['rest_scale']:.7f}, zero point {qa0['rest_zero_point']} to {qs['rest_zero_point']}.", "ft ac")
    bf = int(ft["npz_bytes"])
    want(f"{bf:,} bytes ({f(ft['size_MiB']):.3f} MiB, {f(ft['size_MB']):.3f} MB); C3DGS's evaluation {trip(ft, 'c3dgs_')}; protocol ii {ii(ft)}", "ft row")
    want(f"{f(ft['PSNR_ii']) - f(g['PSNR_ii']):+.3f} dB in protocol ii at {nb - bf:,} fewer bytes", "ft vs injected")
    want(f"{int(q5['npz_bytes']):,} bytes and {f(q5['PSNR_ii']):.3f} dB in protocol ii", "E3q ft")

    # time and memory
    def gb(x):
        return f"{x / 1e9:.2f}"

    sb, stt = steps["bicycle"], steps["train"]
    runner_t = [x for x in M["train"]["steps"] if x["name"] == "build_runner"]
    check(len(runner_t) == 2, "train builds the runner twice")
    tbl = [
        ("INRIA members (re-checked)", f"{sb['fetch_inria']['time_s']:.1f} | {gb(sb['fetch_inria']['cuda_peak']['allocated'])} | "
                                       f"{stt['fetch_inria']['time_s']:.1f} | {gb(stt['fetch_inria']['cuda_peak']['allocated'])}"),
        ("dataset download", f"{sb['download_dataset']['time_s']:.1f} | 0.00 | {stt['download_dataset']['time_s']:.1f} | 0.00"),
        ("runner", f"{sb['build_runner']['time_s']:.1f} | {gb(sb['build_runner']['cuda_peak']['allocated'])} | "
                   f"{runner_t[0]['time_s']:.1f}, and {runner_t[1]['time_s']:.1f} for protocol ii | "
                   f"{gb(runner_t[0]['cuda_peak']['allocated'])}, {gb(runner_t[1]['cuda_peak']['allocated'])}"),
        ("uncompressed, protocol ii", f"{sb['eval_ii_uncompressed']['time_s']:.1f} | {gb(sb['eval_ii_uncompressed']['cuda_peak']['allocated'])} | "
                                      f"{stt['eval_ii_uncompressed']['time_s']:.1f} | {gb(stt['eval_ii_uncompressed']['cuda_peak']['allocated'])}"),
        ("GN pass 16 x 16, all train views", f"{sb['gn_pass16_full']['time_s']:.1f} | {gb(sb['gn_pass16_full']['cuda_peak']['allocated'])} | "
                                             f"{stt['gn_pass16_full']['time_s']:.1f} | {gb(stt['gn_pass16_full']['cuda_peak']['allocated'])}"),
        ("GN pass 16 x 16, even views", f"{sb['gn_pass16_even']['time_s']:.1f} | {gb(sb['gn_pass16_even']['cuda_peak']['allocated'])} | "
                                        f"{stt['gn_pass16_even']['time_s']:.1f} | {gb(stt['gn_pass16_even']['cuda_peak']['allocated'])}"),
        ("the two GN cache writes", f"{sb['gn_cache_write16_full']['time_s']:.1f} and {sb['gn_cache_write16_even']['time_s']:.1f} | - | "
                                    f"{stt['gn_cache_write16_full']['time_s']:.1f} and {stt['gn_cache_write16_even']['time_s']:.1f} | -"),
    ]
    for lab, body in tbl:
        want(f"| {lab} | {body} |", lab)
    for s in ("bicycle", "train"):
        check(steps[s]["download_dataset"]["cuda_peak"]["allocated"] == 0, f"{s}: download on no GPU")
    cvt = [x for x in M["train"]["steps"] if x["name"].startswith("gn_vq_cv_")]
    dmt = [x for x in M["train"]["steps"] if x["name"].startswith("dmse_cv_")]
    check(len(cvt) == len(dmt) == 7, "7 CV steps")
    ta = [x["time_s"] for x in cvt]
    pa = [x["cuda_peak"]["allocated"] for x in cvt]
    want(f"| CV: GN-VQ, 7 runs | - | - | {min(ta):.1f}-{max(ta):.1f} each, {sum(ta):.1f} in all | {gb(min(pa))}-{gb(max(pa))} |", "cv steps")
    td = {f"{x['time_s']:.1f}" for x in dmt}
    pd = [x["cuda_peak"]["allocated"] for x in dmt]
    check(len(td) == 1, "dMSE steps equal to 0.1 s")
    want(f"| CV: dMSE, 7 rows | - | - | {td.pop()} each | {gb(min(pd))}-{gb(max(pd))} |", "dmse steps")
    want(f"| calibration | - | - | {stt['calibration']['time_s']:.1f} | {gb(stt['calibration']['cuda_peak']['allocated'])} |", "calibration step")
    n2 = [x["time_s"] for x in M["train"]["steps"] if x["name"].startswith("npz2ply_")]
    ev = [x for x in M["train"]["steps"] if x["name"].startswith("eval_ii_c3dgs") or x["name"].startswith("eval_ii_gnvq")]
    check(len(n2) == len(ev) == 10 and len({gb(x["cuda_peak"]["allocated"]) for x in ev}) == 1, "decoded rows")
    want(f"| per decoded row: `npz2ply.py` / protocol ii | - | - | {min(n2):.1f}-{max(n2):.1f} / "
         f"{min(x['time_s'] for x in ev):.1f}-{max(x['time_s'] for x in ev):.1f} | - / {gb(ev[0]['cuda_peak']['allocated'])} |", "decoded")

    def c3row(c, lab=None):
        r = R[c]
        ftv = f"{f(r['c3dgs_finetune_s']):.1f}" if r["c3dgs_finetune_s"] else "-"
        return (f"| {lab or '`' + c + '`'} | {f(r['c3dgs_wall_s']):,.1f} | {f(r['c3dgs_sensitivity_s']):.1f} | "
                f"{f(r['c3dgs_clustering_s']):,.1f} | {ftv} | {f(r['c3dgs_encode_s']):.1f} | {gb(int(r['peak_allocated_bytes']))} |")

    for c in KS + ["gnvq_k4096", "gnvq_k4096_ft5000"]:
        want(c3row(c), c + " time row")
    jr = [R[c] for c in JS]
    pk = {gb(int(r["peak_allocated_bytes"])) for r in jr}
    check(len(pk) == 1, "threshold peaks equal to 0.01 GB")
    want(f"| `c3dgs_k4096_j-2` / `j-1` / `j+1` / `j+2` | {' / '.join(f'{f(r['c3dgs_wall_s']):.1f}' for r in jr)} | "
         f"{min(f(r['c3dgs_sensitivity_s']) for r in jr):.1f}-{max(f(r['c3dgs_sensitivity_s']) for r in jr):.1f} | "
         f"{min(cj):.1f}-{max(cj):.1f} | - | {min(f(r['c3dgs_encode_s']) for r in jr):.1f}-{max(f(r['c3dgs_encode_s']) for r in jr):.1f} | "
         f"{pk.pop()} |", "threshold time row")
    want(f"clustering took {f(g['c3dgs_clustering_s']):.1f} s and {f(ft['c3dgs_clustering_s']):.1f} s, against the probe's "
         f"{f(p['c3dgs_clustering_s']):.1f} s", "injection in clustering")
    pks = [int(R[c]["peak_allocated_bytes"]) for c in C3]
    rs_ = {R[c]["peak_reserved_bytes"] for c in C3}
    check(len(rs_) == 1, "one reserved peak")
    want(f"was {gb(min(pks))}-{gb(max(pks))} GB allocated in every run, and {gb(int(rs_.pop()))} GB reserved", "c3dgs peaks")
    want(f"E3q measured {gb(int(q0['peak_allocated_bytes']))} GB for the unhooked run", "E3q peak")
    mem = json.load(open(MEM))
    pred = mem["report"]["e3r"]["bicycle_mitigated_images_cuda"]["gn_pass"]
    check(all(v["gn_pass"] == pred for k, v in mem["report"]["e3r"].items() if k.startswith("bicycle")), "one bicycle prediction")
    full, even = sb["gn_pass16_full"]["cuda_peak"], sb["gn_pass16_even"]["cuda_peak"]
    want(f"peaked at {full['allocated']:,} bytes against the predicted {int(pred):,} ({full['allocated'] - int(pred):+,} bytes)", "bicycle gn")
    want(f"the even-view pass peaked at {even['allocated']:,}", "bicycle even")
    want(f"the allocator reserved {gb(max(full['reserved'], even['reserved']))} GB", "reserved")
    gnb = M["bicycle"]["gn"]["full"]
    check(gnb["M_bytes"] == mem["report"]["m16"]["bicycle_all_splats"] == 544 * int(RB["uncompressed"]["n_splats"]), "M bytes as predicted")
    want(f"the metric is {gnb['M_bytes']:,} bytes ({mem['report']['m16']['bytes_per_splat']} per splat), as predicted", "M bytes")
    tp = mem["report"]["e3r"]["train_mitigated_images_cuda"]["gn_pass"]
    want(f"1.65 GB against a predicted {gb(tp)} GB" if gb(stt["gn_pass16_full"]["cuda_peak"]["allocated"]) == "1.65"
         else f"{gb(stt['gn_pass16_full']['cuda_peak']['allocated'])} GB against a predicted {gb(tp)} GB", "train gn prediction")
    nbz, nb_all = gnb["splats_zero_trace"], int(RB["uncompressed"]["n_splats"])
    ntz, nt_all = M["train"]["gn"]["full"]["splats_zero_trace"], int(R["uncompressed"]["n_splats"])
    tiles = mem["tiles"]
    want(f"{nbz:,} of bicycle's {nb_all:,} splats ({nbz / nb_all * 100:.2f}%) have zero trace over all "
         f"{M['bicycle']['n_train_views']} train views ({M['bicycle']['gn']['even']['splats_zero_trace']:,} over the "
         f"{M['bicycle']['n_even_views']} even ones), and {ntz:,} of train's {nt_all:,} ({ntz / nt_all * 100:.2f}%) over its "
         f"{M['train']['n_train_views']}", "zero trace")
    check(M["bicycle"]["gn"]["full"]["n_views"] == M["bicycle"]["n_train_views"] and tiles["bicycle"]["n_splats"] == nb_all
          and tiles["train"]["n_splats"] == nt_all, "tile counts are over the same models")
    want(f"{tiles['bicycle']['seen_in_any_train_view']:,} of bicycle's splats (and all {tiles['train']['seen_in_any_train_view']:,} "
         f"of train's) touch a tile", "tiles")
    check(tiles["train"]["seen_in_any_train_view"] == nt_all, "all train splats touch a tile")

    # the jobs against the estimate
    sys.path.insert(0, HERE)
    try:
        import e3r_estimate as est
    finally:
        sys.path.pop(0)
    eg = est.estimate_g()
    mm = est.measured()
    jt, jb = M["train"]["timings_s"]["job"], M["bicycle"]["timings_s"]["job"]
    want(f"{jt:,.1f} s by its own clock ({tim['gn_e3r_train_s']:,.1f} s in the queue), inside the estimated "
         f"{eg['train'][0]:,.0f}-{eg['train'][1]:,.0f} s", "train job")
    check(eg["train"][0] <= jt <= eg["train"][1], "train inside")
    setup = tim["restore_s"] + tim["install_s"] + tim["c3dgs_build_s"]
    sess = setup + tim["gn_e3r_train_s"]
    want(f"took {setup:.1f} s against {eg['setup_s']:.0f} s. With the train job, that is {sess:,.1f} s ({sess / 3600:.1f} h), "
         f"inside the session estimate of {eg['session'][0]:,.0f}-{eg['session'][1]:,.0f} s", "setup, session")
    check(eg["session"][0] <= sess <= eg["session"][1], "session inside")
    bl, bh = eg["bicycle_gn_only"]
    want(f"{jb:.1f} s ({tim['gn_e3r_bicycle_s']:.1f} s in the queue), above the estimated {bl:.0f}-{bh:.0f} s", "bicycle job")
    dl = mm["download_s"]["bicycle"]
    want(f"as E3p's {dl:.1f} s", "estimate download")
    want(f"the download took {sb['download_dataset']['time_s']:.1f} s, {sb['download_dataset']['time_s'] - dl:.1f} s more than E3p's", "download excess")
    want(f"the runner took {sb['build_runner']['time_s']:.1f} s against E3p's {mm['runner_s']['bicycle']:.1f} s", "runner excess")
    cw = sb["gn_cache_write16_full"]["time_s"] + sb["gn_cache_write16_even"]["time_s"]
    want(f"the two cache writes ({cw:.1f} s) and the member re-check ({sb['fetch_inria']['time_s']:.1f} s)", "uncounted")
    sum_steps = sum(x["time_s"] for x in M["bicycle"]["steps"])
    want(f"{jb - sum_steps:.1f} s of the job outside its steps", "outside steps")
    ev3p = next(s["time_s"] for s in e3pm["bicycle"]["steps"] if s["name"] == "eval_ii_uncompressed")
    gnp = sb["gn_pass16_full"]["time_s"] + sb["gn_pass16_even"]["time_s"]
    g0, g1 = mm["gn_passes_s"]["bicycle"], mm["gn_passes_s"]["bicycle"] * 136 / 120
    want(f"Protocol ii ({sb['eval_ii_uncompressed']['time_s']:.1f} s against {ev3p:.1f} s) and the GN passes ({gnp:.1f} s against "
         f"{g0:.1f}-{g1:.1f} s) were as estimated", "as estimated")
    check(g0 <= gnp <= g1, "GN passes inside")
    check(abs(bl + (sb["download_dataset"]["time_s"] - dl) + (sb["build_runner"]["time_s"] - mm["runner_s"]["bicycle"])
              + (sb["eval_ii_uncompressed"]["time_s"] - ev3p) + (gnp - g0) + cw + sb["fetch_inria"]["time_s"]
              + sb["layout_model"]["time_s"] + sb["protocol_ii_views"]["time_s"] + (jb - sum_steps) - jb) < 1e-6, "the excess adds up")
    stamps = sorted(r["timestamp"] for r in R.values())
    want(f"timestamped {RB['uncompressed']['timestamp']} (bicycle) and {stamps[0][11:]} to {stamps[-1][11:]} (train)", "stamps")
    want(f"rows timestamped {min(RB['uncompressed']['timestamp'], stamps[0])[:16]} to {stamps[-1][11:16]}", "sources stamps")
    want(f"one Kaggle session on {env['gpu']['count']}x T4", "sources session")

    # the title, the summary and what E3r settles
    kr, trr = kk["bytes_max_over_min"], kt["bytes_max_over_min"]
    want(f"the colour-threshold knob spans {trr:.2f}x in bytes, K {kr:.2f}x", "title")
    want(f"spans {trr:.2f}x in `.npz` bytes (protocol ii {kt['PSNR_ii_min']:.3f}-{kt['PSNR_ii_max']:.3f} dB), K only {kr:.2f}x "
         f"({kk['PSNR_ii_min']:.3f}-{kk['PSNR_ii_max']:.3f} dB)", "summary knobs")
    want(f"{trr:.2f}x in `.npz` bytes on train, against {kr:.2f}x for K", "settles knobs")
    want(f"at {nb_all / 1e6:.2f}M splats is measured:** {gb(full['allocated'])} GB, within {full['allocated'] - int(pred):,} bytes", "settles gn")

    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | C3DGS_LINES | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 15: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
