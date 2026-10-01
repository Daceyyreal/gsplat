"""Re-check every number in kaggle/FINDINGS.md section 16 (E4p), its summary paragraph and its Sources entry against
the committed files.

    python bench/gn/check_s16.py

It reads only committed files: the E4p bundle (``kaggle/gn_e4p/gn4p/``), E3r's and E3p's rows and summary
(``kaggle/gn_e3r/gn3r/``, ``kaggle/gn_e3p/gn3p/``), the pre-run estimate (``kaggle/HANDOFF.md``, "E4p notebook"), note i's
feasibility and Amendment 15 e (``kaggle/PREREG_GN.md``), ``kaggle/E4_DESIGN.md`` (fact 1) and this repository's own job,
hooks, wrapper and ``bench/gn/`` source for the file:line citations, read at the commit E4p ran at. OGC's line numbers
(``vq.py`` at ``49ccae72``) and C3DGS's (at ``2a234af5``) were read from their sources, which are not in the repository;
they are listed constants. Each quoted number is recomputed and must appear in the text, each claim is re-asserted, and
every numeric token of the text must be a recomputed string or a listed constant. It prints each failure and exits 1 if
there is any, 0 otherwise. Built like ``check_s15.py``.
"""

import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
E4P = os.path.join(REPO, "kaggle", "gn_e4p", "gn4p")
E3R = os.path.join(REPO, "kaggle", "gn_e3r", "gn3r")
E3P = os.path.join(REPO, "kaggle", "gn_e3p", "gn3p")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
HANDOFF = os.path.join(REPO, "kaggle", "HANDOFF.md")
PREREG = os.path.join(REPO, "kaggle", "PREREG_GN.md")
DESIGN = os.path.join(REPO, "kaggle", "E4_DESIGN.md")
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
# the amendments', notes', protocols' and codecs' own constants, the design's seeds, angles, K, chunks and caps, names and
# dates; not results
CONSTANTS = {"0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "12", "13", "14", "15", "16", "20", "21", "48", "4,096",
             "100,000", "25,000", "256", "1e-3", "1e-2", "1e-1", "3e-1", "1e-6", "10^9", "-40", "-20", "-10", "+10", "+20",
             "+40", "40", "2026-10-01", "5,000", "2b", "360"}
# OGC's vq.py at 49ccae72 (its lines and its literal 2.0) and C3DGS's scene/gaussian_model.py at 2a234af5; not in this
# repository
OUTSIDE_LINES = {"74", "-84", "84", "47", "-49", "49", "2.0", "80", "183"}
COMMITS = {"573142ac", "85b43cff", "fcd1914f", "60c4839b", "24950320", "2a234af5", "49ccae72"}
# this repository's citations: (path, line, text the line must contain), read at the commit E4p ran at
CITED_AT = "24950320"
CITED = [("kaggle/e3q_c3dgs.py", 247, 'out["ok"] = res["returncode"] == 0 and os.path.exists(npz) and out.get("results") is not None',
          "(`kaggle/e3q_c3dgs.py:247`)"),
         ("kaggle/gn_e4p_scene.py", 597, '"geometry_sha1": (rep or {}).get("geometry_sha1", "")', "(`kaggle/gn_e4p_scene.py:597`)"),
         ("kaggle/e4p_hooks.py", 308, '"geometry_sha1_equal": _geometry_sha1(g) == self.rep.get("geometry_sha1")',
          "(`kaggle/e4p_hooks.py:308`)"),
         ("kaggle/e4p_hooks.py", 139, "torch.cuda.reset_peak_memory_stats()", "(`kaggle/e4p_hooks.py:139`)"),
         ("kaggle/e4p_hooks.py", 130, "self.cuda_peak = max(self.cuda_peak, torch.cuda.max_memory_allocated())", "(`:130`)"),
         ("kaggle/e3q_c3dgs_run.py", 308,
          'rec["max_memory_allocated"] = max(rec["max_memory_allocated"], rec["e4p"].get("cuda_peak_allocated_process") or 0)',
          "(`kaggle/e3q_c3dgs_run.py:308`)"),
         ("kaggle/e3q_c3dgs_run.py", 306, "max_memory_reserved=torch.cuda.max_memory_reserved()", "(`:306`)"),
         ("bench/gn/diagnostics.py", 547, "Clusters with ``tr(sum M) = 0`` (empty, or all members unseen) keep ``q_old``",
          "(`bench/gn/diagnostics.py:547`)"),
         ("bench/gn/gn_vq.py", 179, "lo, hi = float(C0.detach().float().min()), float(C0.detach().float().max())",
          "`bench/gn/gn_vq.py:179`"),
         ("bench/gn/gn_vq.py", 207, "C_new = C_new.clamp(lo, hi)", "the clip at `:207`"),
         ("bench/gn/e3r.py", 70, "torch.quantize_per_tensor(part.contiguous(), scale, zp, torch.qint8)", "`bench/gn/e3r.py:70`")]
P = (0, 1, 2)
ROWS = ["c3dgs", "ogc", "ogc_lam1e6", "gnvq_rho0", "scalar", "gnvq_cv", "c3dgs_ft", "ogc_ft", "gnvq_cv_ft"]
PRIMARY = ["c3dgs", "ogc", "ogc_lam1e6", "gnvq_rho0", "scalar", "gnvq_cv"]
ANG = ["-40", "-20", "-10", "0", "10", "20", "40"]
RHO_LABEL = {0.0: "0", 0.001: "1e-3", 0.01: "1e-2", 0.1: "1e-1", 0.3: "3e-1", 1.0: "1", 3.0: "3"}


def section_text(txt: str) -> str:
    start = txt.index("## 16. E4p")
    nxt = txt.find("\n## ", start + 1)
    sec = txt[start:] if nxt < 0 else txt[start:nxt]
    s0 = txt.index("**E4p (section 16")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 16:")
    end = txt.find("\n- Section", s1 + 1)
    src = txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


def lines_of(path: str, a: int, b: int) -> str:
    import subprocess

    text = subprocess.run(["git", "-C", REPO, "show", f"{CITED_AT}:{path}"], capture_output=True, check=True,
                          encoding="utf-8").stdout
    return "".join(text.splitlines(keepends=True)[a - 1:b])


def sg(v: float, d: int = 4) -> str:
    """Signed, d decimals, ASCII minus."""
    return f"{v:+.{d}f}"


def e2(v: float) -> str:
    """Two decimals in e-notation with no exponent padding (2.82e-3)."""
    m, e = f"{v:.2e}".split("e")
    return f"{m}e{int(e)}"


def gb(b) -> str:
    return f"{float(b) / 1e9:.2f}"


def comp(v):
    d = sum(v) / len(v)
    var = sum((x - d) ** 2 for x in v) / (len(v) - 1)
    sd = math.sqrt(var)
    return d, var, sd, sd / math.sqrt(len(v))


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

    R = {r["config"]: r for r in csv.DictReader(open(os.path.join(E4P, "gn4p_results_train.csv"), newline=""))}
    CV = {r["config"]: r for r in csv.DictReader(open(os.path.join(E4P, "gn4p_cv_train.csv"), newline=""))}
    M = json.load(open(os.path.join(E4P, "gn4p_meta_fork_train.json")))
    MO = json.load(open(os.path.join(E4P, "gn4p_meta_ogc_train.json")))
    S = json.load(open(os.path.join(E4P, "gn4p_summary.json")))
    ST = S["scenes"]["train"]
    OG = json.load(open(os.path.join(E4P, "gn4p_ogc_train.json")))
    env = json.load(open(os.path.join(E4P, "gn4p_env.json")))
    tim = json.load(open(os.path.join(E4P, "timings.json")))
    build = json.load(open(os.path.join(E4P, "gn4p_c3dgs_build.json")))
    tails = {j: json.load(open(os.path.join(E4P, f"gn_e4p_{j}_train_log_tail.json"))) for j in ("fork", "ogc")}
    runs = M["runs"]
    W = {p: runs[f"p{p}_a0"]["wrapper"] for p in P}
    FK = {p: W[p]["e4p"]["fork"] for p in P}
    meas = {p: M["processes"][str(p)]["measured"]["0"] for p in P}
    names = M["split_check"]["test_names"]
    row = lambda r, p: R[f"p{p}_{r}"]  # noqa: E731
    arr = lambda r, p: json.loads(row(r, p)["arrays"])  # noqa: E731
    col = lambda r, k: [f(row(r, p)[k]) for p in P]  # noqa: E731

    def diff(a, b, k="PSNR_ii"):
        return [f(row(a, p)[k]) - f(row(b, p)[k]) for p in P]

    # what ran
    check(all(t["exit_code"] == 0 and not t["skipped_by_cutoff"] for t in tails.values()), "both jobs exit 0, not skipped")
    fork_rows = [c for c in R if c.startswith("p") and c != "probe"]
    check(len(R) == 29 and all(r["status"] == "ok" for r in R.values()) and len(fork_rows) == 27
          and {"uncompressed", "probe"} <= set(R), "29 rows ok: uncompressed, probe, 27 fork rows")
    want(f"The fork job wrote {len(R)} rows", "rows")
    want(f"the `probe` and {len(fork_rows)} fork rows ({len(fork_rows) // 3} per process)", "fork rows")
    check(M["done"] and not M["failed_steps"] and not M["skipped_steps"] and not M["missing_or_failed"]
          and not ST["deviations"] and not ST["failed_steps"] and ST["dropped"] is None
          and all(s["status"] == "ok" for s in M["steps"]) and all(s["status"] == "ok" for s in MO["steps"])
          and not MO["failed_steps"] and not MO["skipped_steps"], "nothing failed, no deviation")
    for p in P:
        att = M["processes"][str(p)]["attempts"]
        check(len(att) == 1 and att[0]["attempt"] == 0 and att[0]["device"] == "cuda" and not att[0]["oom"]
              and not att[0]["lost_rows"] and not att[0]["checks_failed"] and att[0]["status"] == "ok", f"process {p} one attempt")
        check(all(row(r, p)["data_device"] == "cuda" and row(r, p)["attempt"] == "0" for r in ROWS), f"process {p} rows on cuda")
    pe = runs["probe_eval"]
    check(pe["ok"] is False and pe["returncode"] == 0 and pe["wrapper"]["status"] == "ok"
          and R["probe"]["status"] == "ok", "probe evaluation run: ok false, return code 0, wrapper ok")

    # inputs and checks
    check(env["gpu"]["count"] == 2 and env["gpu"]["name"] == "Tesla T4" and env["wheel_restored"], "2x T4, restored wheel")
    want(f"torch {env['torch']} (CUDA {env['torch_cuda']}), driver {env['nvidia_smi'].split(', ')[1]}, cuDNN {env['cudnn']}, "
         f"Python {env['python']}, {env['gpu']['count']}x Tesla T4; gsplat commit `{env['gsplat_commit'][:8]}`, the restored "
         f"wheel (install {tim['install_s']:.1f} s)", "session")
    mem = M["inria"]["members"]
    e3pm = json.load(open(os.path.join(E3P, "gn3p_meta_train.json")))
    check(len(mem) == 3 and all(v["source"] == "present" and v["bytes"] == v["bytes_pinned"] and v["crc32"] == v["crc32_pinned"]
                                for v in mem.values()), "members present, pins match")
    check(mem["ply"]["sha1"] == e3pm["inria"]["members"]["ply"]["sha1"], "ply SHA-1 = E3p's")
    want(f"`point_cloud.ply` SHA-1 `{mem['ply']['sha1'][:8]}`, E3p's", "ply sha1")
    check(M["cfg_args_mismatch"] == [] and MO["cfg_args_mismatch"] == [], "no cfg_args mismatch")
    cf = M["camera_frame_check"]
    check(cf["pass"] and cf == MO["camera_frame_check"], "camera frame, both jobs")
    want(f"pass: {cf['n_matched']} of {cf['n_runner']}, largest differences {cf['max_position_diff']:.2e} (position), "
         f"{cf['max_rotation_diff']:.2e} (rotation)", "camera frame")
    check(M["split_check"]["equals_cameras_json_head"], "split equal")
    want(f"equal, {M['split_check']['n_test']} test views", "split")
    u = R["uncompressed"]
    want(f"{f(u['PSNR_ii']):.3f} / {f(u['SSIM_ii']):.4f} / {f(u['LPIPS_ii']):.4f} at "
         f"{json.loads(u['resolution_ii'])[0][0]}x{json.loads(u['resolution_ii'])[0][1]}", "uncompressed")
    u3 = {s: next(r for r in csv.DictReader(open(os.path.join(d, n), newline="")) if r["config"] == "uncompressed")
          for s, d, n in (("e3r", E3R, "gn3r_results_train.csv"), ("e3p", E3P, "gn3p_results_train.csv"))}
    check(all(u3["e3r"][k] == u3["e3p"][k] and u[k] != u3["e3r"][k] for k in ("PSNR_ii", "SSIM_ii", "LPIPS_ii")),
          "E3p = E3r, E4p's digits differ")
    dps = abs(f(u["PSNR_ii"]) - f(u3["e3r"]["PSNR_ii"]))
    def e1(v: float) -> str:
        m, e = f"{v:.1e}".split("e")
        return f"{m}e{int(e)}"

    dsl = {e1(abs(f(u[k]) - f(u3['e3r'][k]))) for k in ("SSIM_ii", "LPIPS_ii")}
    check(len(dsl) == 1, "SSIM and LPIPS differ by the same, to two digits")
    want(f"differ from it by {e1(dps)} dB in PSNR and {dsl.pop()} in SSIM and LPIPS", "uncompressed digits")
    check(build["ok"] and build["head"].startswith("2a234af5") and len(build["deviations"]) == 6, "build ok, six deviations")
    want(f"{build['build_time_s']:.1f} s of builds, {build['total_time_s']:.1f} s in all", "build")
    check(ST["ogc_clone"]["head"].startswith("49ccae72") and OG["clone"]["head"].startswith("49ccae72")
          and ST["ogc_clone"]["dest"] == "/tmp/ogc_fork" and OG["clone"]["dest"] == "/tmp/ogc_ogc", "OGC clones")

    # the fork
    for p in P:
        rws = FK[p]["rows"]
        check(all(rws[r]["checks"]["ok"] and all(rws[r]["checks"].values()) for r in PRIMARY) and FK[p]["checks_failed"] == [],
              f"process {p}: every primary check")
    want(f"({3 * len(PRIMARY)} of {3 * len(PRIMARY)}; `checks_failed` empty in every process)", "18 checks")
    cnt = {(row(r, p)["n_pruned"], row(r, p)["n_kept_colour"], row(r, p)["n_colour_quantized"]) for r in ROWS for p in P}
    e3r = {r["config"]: r for r in csv.DictReader(open(os.path.join(E3R, "gn3r_results_train.csv"), newline=""))}
    pr = e3r["c3dgs_k4096"]
    check(len(cnt) == 1 and cnt.pop() == (pr["n_pruned"], pr["n_kept_colour"], pr["n_colour_quantized"]), "counts = E3r probe's")
    nq = int(row("c3dgs", 0)["n_colour_quantized"])
    want(f"{int(pr['n_pruned']):,} pruned, {int(pr['n_kept_colour']):,} keeping their own colour and {nq:,}\ncolour-quantized"
         .replace("\n", " "), "counts")
    sha = {p: row("c3dgs", p)["geometry_sha1"] for p in P}
    check(all(row(r, p)["geometry_sha1"] == sha[p] for r in ROWS for p in P), "one SHA-1 per process")
    check(len(set(sha.values()) | {R["probe"]["geometry_sha1"]}) == 4, "four SHA-1s differ")
    want(f"| probe (seed 0) | `{R['probe']['geometry_sha1'][:8]}` |", "probe sha")
    for p in P:
        want(f"| process {p} | `{sha[p][:8]}` |", f"sha {p}")
    rot = [arr(r + "_ft", p)["rotation"]["compressed_bytes"] - arr(r, p)["rotation"]["compressed_bytes"]
           for r in ("c3dgs", "ogc", "gnvq_cv") for p in P]
    sca = [arr(r, p)["scaling"]["compressed_bytes"] - arr(r + "_ft", p)["scaling"]["compressed_bytes"]
           for r in ("c3dgs", "ogc", "gnvq_cv") for p in P]
    check(min(rot) > 0 and min(sca) > 0, "fine-tuning changes the geometry")
    want(f"the compressed `rotation` array grew by {min(rot):,}-{max(rot):,} bytes and `scaling` shrank by "
         f"{min(sca):,}-{max(sca):,}", "geometry moved")
    for p in P:
        for r in ("c3dgs", "ogc", "gnvq_cv"):
            rec = FK[p]["rows"][r + "_ft"]
            check(rec["pre_finetune_sha1"] == FK[p]["rows"][r]["pre_save_sha1"] and rec["labels_survived"]
                  and row(r + "_ft", p)["labels_survived"] == "True", f"p{p} {r}_ft started from its row, labels survived")
    for r in ("ogc", "ogc_lam1e6"):
        for k in ("features_dc", "features_rest", "feature_indices"):
            check(len({arr(r, p)[k]["compressed_bytes"] for p in P}) == 1, f"{r} {k} same in every process")
        check(len({row(r, p)["index_entropy_bits"] for p in P}) == 1, f"{r} entropy identical")
    oa = arr("ogc", 0)
    want(f"({oa['features_dc']['compressed_bytes']:,}, {oa['features_rest']['compressed_bytes']:,} and "
         f"{oa['feature_indices']['compressed_bytes']:,} bytes)", "ogc arrays")
    qa = [W[p]["e4p"]["qa_at_colour_vq"] for p in P]
    check(all(q == qa[0] for q in qa), "the quantizer at the colour step the same")
    want(f"(DC scale {qa[0]['dc_scale']:.6f},\nzero point {qa[0]['dc_zero_point']}; AC scale {qa[0]['rest_scale']:.5g}, zero point "
         f"{qa[0]['rest_zero_point']})".replace("\n", " "), "quantizer")

    # rho_cv and iterations
    rs = sorted(CV.values(), key=lambda r: f(r["rho"]))
    check([RHO_LABEL[f(r["rho"])] for r in rs] == list(RHO_LABEL.values()), "7 rho")
    sc = {RHO_LABEL[f(r["rho"])]: f(r["measured_odd_clamped"]) for r in rs}
    check(sc == {k: v for k, v in ST["cv_odd_scores"].items()}, "scores = summary's")
    best = min(sc, key=lambda k: (sc[k], list(RHO_LABEL.values()).index(k)))
    check(best == "1e-2" and ST["rho_cv"] == 0.01 and all(f(row(r, p)["rho_cv"]) == 0.01 for r in ROWS for p in P), "rho_cv 1e-2")
    cells = " | ".join(f"**{sc[k]:.4e}**" if k == best else f"{sc[k]:.4e}" for k in RHO_LABEL.values())
    want(f"| clamped (the score) | {cells} |", "cv clamped")
    want("| unclamped | " + " | ".join(f"{f(r['measured_odd_raw']):.4e}" for r in rs) + " |", "cv raw")
    second = sorted(sc.values())[1]
    want(f"ahead of 1e-1 by {(second - sc[best]) / sc[best] * 100:.2f}% of its score", "margin")
    check(second == sc["1e-1"], "1e-1 second")
    s3 = json.load(open(os.path.join(E3R, "gn3r_summary.json")))
    check(s3["scenes"]["train"]["rho_cv"] == 0.01, "E3r's rho_cv 1e-2")
    gv = ST["gn_vq_runs"]
    check(len(gv) == 16 and all(x["iterations"] == "20" and x["stopped_because"] == "max_iters" for x in gv)
          and all(r["vq_iterations"] == "20" for r in CV.values()), "16 runs at the cap")
    want(f"all {len(gv)} ({len(CV)} CV,", "16")

    def drops(r):
        return [FK[p]["rows"][r]["gn_vq"]["history_last3"][-1]["relative_drop"] for p in P]

    want(f"{e2(min(drops('gnvq_rho0')))} to {e2(max(drops('gnvq_rho0')))} (`gnvq_rho0`), {e2(min(drops('gnvq_cv')))} to "
         f"{e2(max(drops('gnvq_cv')))} (`gnvq_cv`)\nand {e2(min(drops('scalar')))} to {e2(max(drops('scalar')))} (`scalar`)"
         .replace("\n", " "), "last drops")
    check(min(min(drops(r)) for r in ("gnvq_rho0", "gnvq_cv", "scalar")) > 1e-3, "all above the rule")
    check(M["n_even_views"] == 132 and M["n_odd_views"] == 131 and M["n_train_views"] == 263, "views")
    want(f"metric ({M['n_even_views']} views)", "even views")
    want(f"the\n{M['n_odd_views']} odd-indexed train views".replace("\n", " "), "odd views")
    want(f"probe run's {nq:,} splats", "cv splats")

    # the rows
    want(f"among the {nq:,} quantized\nsplats' stored indices (K = 4,096); the index entropy is over the whole stored index array "
         f"({FK[0]['n_colour_quantized'] + int(row('c3dgs', 0)['n_kept_colour']):,} indices)".replace("\n", " "), "index counts")
    check(meas[0]["ogc"]["npz"]["n_indices"] == FK[0]["n_colour_quantized"] + int(row("c3dgs", 0)["n_kept_colour"]), "n_indices")
    for r in ROWS:
        ps = col(r, "PSNR_ii")
        cw = col(r, "codebook_distinct")
        ie = col(r, "index_entropy_bits")
        cws = f"{int(min(cw)):,}" if min(cw) == max(cw) else f"{int(min(cw)):,}-{int(max(cw)):,}"
        ies = f"{min(ie):.2f}" if f"{min(ie):.2f}" == f"{max(ie):.2f}" else f"{min(ie):.2f}-{max(ie):.2f}"
        if r in ("c3dgs_ft", "gnvq_cv_ft"):
            src = r[:-3]
            check(col(r, "codebook_distinct") == col(src, "codebook_distinct") and col(r, "index_entropy_bits") == col(src, "index_entropy_bits"),
                  f"{r} labels as {src}")
            cws = ies = f"as `{src}`"
        lab = {"c3dgs": "`c3dgs` (row 1)", "ogc": "`ogc` (2)", "ogc_lam1e6": "`ogc_lam1e6` (2b)", "gnvq_rho0": "`gnvq_rho0` (3)",
               "scalar": "`scalar` (4)", "gnvq_cv": "`gnvq_cv` (5)"}.get(r, f"`{r}`")
        want(f"| {lab} | {sum(ps) / 3:.3f} ({min(ps):.3f}-{max(ps):.3f}) | {sum(col(r, 'SSIM_ii')) / 3:.4f} | "
             f"{sum(col(r, 'LPIPS_ii')) / 3:.4f} | {sum(col(r, 'c3dgs_PSNR')) / 3:.3f} | {sum(col(r, 'npz_bytes')) / 3:,.0f} | "
             f"{cws} | {ies} |", f"row {r}")
    pb = R["probe"]
    want(f"the uncompressed model reads {f(u['PSNR_ii']):.3f} dB; the probe run, evaluated from its decoded `.npz`, "
         f"{f(pb['PSNR_ii']):.3f} / {f(pb['SSIM_ii']):.4f} / {f(pb['LPIPS_ii']):.4f} at {int(pb['npz_bytes']):,} bytes "
         f"(C3DGS's evaluation {f(pb['c3dgs_PSNR']):.3f})", "probe")
    c1 = col("c3dgs", "PSNR_ii")
    want(f"an SD of {comp(c1)[2]:.4f} dB in protocol ii\nand {comp(col('c3dgs', 'npz_bytes'))[2]:,.0f} bytes".replace("\n", " "),
         "row 1 spread")

    # the primary components
    pc = ST["primary_components"]
    D = {}
    for name, (a, b) in (("D1", ("gnvq_cv", "gnvq_rho0")), ("D2", ("gnvq_cv", "ogc"))):
        v = diff(a, b)
        D[name] = comp(v) + (v,)
        sm = pc[name]["PSNR_ii"]
        check(abs(sm["D_bar"] - D[name][0]) < 1e-12 and abs(sm["SE_noise"] - D[name][3]) < 1e-12
              and abs(sm["SD_pool"] - D[name][2]) < 1e-12, f"{name} = summary's")
    d1, d2 = D["D1"], D["D2"]
    want(f"| `D_sp`, processes 0 / 1 / 2 (dB) | {' / '.join(sg(x) for x in d1[4])} | {' / '.join(sg(x) for x in d2[4])} |", "D_sp")
    want(f"| `D_s` | {sg(d1[0])} | {sg(d2[0])} |", "D_s")
    want(f"| `v_s` | {d1[1]:.3e} | {d2[1]:.3e} |", "v_s")
    want(f"| `SD_pool` | {d1[2]:.4f} | {d2[2]:.4f} |", "SD_pool")
    want(f"| `SE_noise` | {d1[3]:.4f} | {d2[3]:.4f} |", "SE")

    def ds(a, b, k):
        c = comp(diff(a, b, k))
        return c[0], c[3]

    s1, s2 = ds("gnvq_cv", "gnvq_rho0", "SSIM_ii"), ds("gnvq_cv", "ogc", "SSIM_ii")
    want(f"| SSIM: `D_s` (`SE_noise`) | {s1[0]:+.6f} ({s1[1]:.6f}) | {s2[0]:+.4f} ({s2[1]:.5f}) |", "ssim")
    l1, l2 = ds("gnvq_cv", "gnvq_rho0", "LPIPS_ii"), ds("gnvq_cv", "ogc", "LPIPS_ii")
    want(f"| LPIPS | {l1[0]:+.5f} ({l1[1]:.5f}) | {l2[0]:+.4f} ({l2[1]:.5f}) |", "lpips")
    b1, b2 = ds("gnvq_cv", "gnvq_rho0", "npz_bytes"), ds("gnvq_cv", "ogc", "npz_bytes")
    want(f"| `.npz` bytes | {b1[0]:,.0f} ({b1[1]:,.0f}) | {b2[0]:,.0f} ({b2[1]:,.0f}) |", "bytes")
    cc1, cc2 = ds("gnvq_cv", "gnvq_rho0", "c3dgs_PSNR"), ds("gnvq_cv", "ogc", "c3dgs_PSNR")
    want(f"| C3DGS's evaluation, PSNR | {sg(cc1[0])} ({cc1[1]:.4f}) | {sg(cc2[0])} ({cc2[1]:.4f}) |", "c3dgs eval")
    check(sum(x < 0 for x in d1[4]) == 1 and all(x < 0 for x in d2[4]), "D1: one negative; D2: all negative")
    check(0.3 < d1[0] / d1[3] < 0.4, "D1 about a third of SE")
    want(f"{abs(d2[0]) / d2[3]:.1f} times its `SE_noise`", "D2 / SE")
    pcx = ST["note_ii"]["power_check"]
    for name, c in (("D1", d1), ("D2", d2)):
        thr = 2 * c[2] / math.sqrt(21)
        check(abs(pcx[name]["threshold"] - thr) < 1e-12 and pcx[name]["n"] == 7 and pcx[name]["processes"] == 3, f"{name} threshold")
    t1, t2 = 2 * d1[2] / math.sqrt(21), 2 * d2[2] / math.sqrt(21)
    want(f"| `s` = sqrt(`v_s`) | {d1[2]:.4f} | {d2[2]:.4f} |", "s")
    want(f"2 `s` / sqrt(21) | {t1:.4f} | {t2:.4f} |", "thresholds")
    want(f"| observed `D_s` | {sg(d1[0])} | {sg(d2[0])} |", "observed")
    pp = next(q for q in range(3, 10_000) if 2 * d1[2] / math.sqrt(7 * q) < abs(d1[0]))
    check(pp == pcx["D1"]["proposal"] and pcx["D2"]["proposal"] is None and abs(d1[0]) < t1 and abs(d2[0]) > t2, "proposals")
    want(f"| proposed process count | {pp} | none |", "proposal")
    want(f"the report proposes P = {pp}", "P")
    want(f"|`D_s`| is {abs(d2[0]) / t2:.1f} times the threshold", "D2 over threshold")

    # the secondaries
    SEC = [("`ogc` - `c3dgs`", "ogc", "c3dgs", "ogc_minus_c3dgs"), ("`gnvq_cv` - `c3dgs`", "gnvq_cv", "c3dgs", "gnvq_cv_minus_c3dgs"),
           ("`ogc_lam1e6` - `ogc`", "ogc_lam1e6", "ogc", "ogc_lam1e6_minus_ogc"),
           ("`gnvq_rho0` - `scalar`", "gnvq_rho0", "scalar", "gnvq_rho0_minus_scalar"),
           ("after fine-tuning: `ogc` - `c3dgs`", "ogc_ft", "c3dgs_ft", "ogc_minus_c3dgs_ft"),
           ("after fine-tuning: `gnvq_cv` - `ogc`", "gnvq_cv_ft", "ogc_ft", "gnvq_cv_minus_ogc_ft")]
    base_bytes = sum(col("c3dgs", "npz_bytes")) / 3
    SD = {}
    for lab, a, b, key in SEC:
        v = diff(a, b)
        d, _, sd, se = comp(v)
        SD[key] = sd
        check(abs(ST["secondaries"][key]["D_bar"] - d) < 1e-12, f"{key} = summary's")
        by = comp(diff(a, b, "npz_bytes"))[0]
        bs = f"{by:+,.0f}"
        if b == "c3dgs":
            bs += f" ({by / base_bytes * 100:+.2f}%)"
        want(f"| {lab} | {' / '.join(sg(x) for x in v)} | {sg(d)} | {se:.4f} | {bs} |", key)
    oc = comp(diff("ogc", "c3dgs"))
    gc = comp(diff("gnvq_cv", "c3dgs"))
    gcb = comp(diff("gnvq_cv", "c3dgs", "npz_bytes"))[0] / base_bytes * 100
    want(f"+{gc[0]:.4f} dB at +{gcb:.2f}% bytes, with the\ngeometry and warm start shared".replace("\n", " "), "paired")
    e3d = f(e3r["gnvq_k4096"]["PSNR_ii"]) - f(pr["PSNR_ii"])
    e3b = (int(e3r["gnvq_k4096"]["npz_bytes"]) - int(pr["npz_bytes"])) / int(pr["npz_bytes"]) * 100
    want(f"E3r's unpaired injected row read {sg(e3d)} dB at {e3b:+.3f}% (section 15)", "E3r context")
    ocb = comp(diff("ogc", "c3dgs", "npz_bytes"))[0] / base_bytes * 100
    want(f"is +{oc[0]:.4f} dB at +{ocb:.2f}% bytes", "ogc - c3dgs")
    pre = open(PREREG, encoding="utf-8").read()
    check("+0.49 dB, the mean of 9 Mip-NeRF 360 scenes" in re.sub(r"\s+", " ", pre), "the published +0.49 dB, 9 scenes")
    lam = comp(diff("ogc_lam1e6", "ogc"))
    check(all(x < 0 for x in diff("ogc_lam1e6", "ogc")), "lam 1e-6 below in every process")
    want(f"reads {abs(lam[0]):.4f} dB below the code's 1e-3", "lam")
    ft1, ft2 = SD["ogc_minus_c3dgs_ft"], SD["gnvq_cv_minus_ogc_ft"]
    want(f"(`SD_pool` {ft1:.4f} and {ft2:.4f} against {oc[2]:.4f} and {d2[2]:.4f} before: {ft1 / oc[2]:.1f} and "
         f"{ft2 / d2[2]:.1f}\ntimes)".replace("\n", " "), "ft spreads")
    for key in ("ogc_minus_c3dgs_ft", "gnvq_cv_minus_ogc_ft"):
        sec = ST["secondaries"][key]
        check(abs(sec["D_bar"]) < 2 * sec["SE_noise"], f"{key} not beyond 2 SE")
    # per-array
    for p in P:
        a, c = arr("ogc", p), arr("c3dgs", p)
        dd = {k: a[k]["compressed_bytes"] - c[k]["compressed_bytes"] for k in a}
        nz = {k for k, v in dd.items() if v}
        check(nz == {"features_dc", "features_rest", "feature_indices"}, f"p{p} only the colour arrays differ")
        tot = int(row("ogc", p)["npz_bytes"]) - int(row("c3dgs", p)["npz_bytes"])
        check(sum(dd.values()) == tot, f"p{p} arrays add up")
        want(f"| {p} | {dd['features_dc']:+,} | {dd['features_rest']:+,} | {dd['feature_indices']:+,} | {tot:+,} | "
             f"{dd['feature_indices'] / tot * 100:.1f}% |", f"split {p}")
    gi = [arr("gnvq_cv", p)["feature_indices"]["compressed_bytes"] - arr("c3dgs", p)["feature_indices"]["compressed_bytes"] for p in P]
    gt = [int(row("gnvq_cv", p)["npz_bytes"]) - int(row("c3dgs", p)["npz_bytes"]) for p in P]
    gr = [arr("gnvq_cv", p)["features_rest"]["compressed_bytes"] - arr("c3dgs", p)["features_rest"]["compressed_bytes"] for p in P]
    gd = [arr("gnvq_cv", p)["features_dc"]["compressed_bytes"] - arr("c3dgs", p)["features_dc"]["compressed_bytes"] for p in P]
    sh = [i / t * 100 for i, t in zip(gi, gt)]
    want(f"`feature_indices` {' / '.join(f'{x:+,}' for x in gi)} of {' / '.join(f'{x:+,}' for x in gt)} in\nall "
         f"({min(sh):.1f}-{max(sh):.1f}%), `features_rest` {' / '.join(f'{x:+,}' for x in gr)}, `features_dc` "
         f"{' / '.join(f'{x:+,}' for x in gd)}".replace("\n", " "), "gnvq split")
    check(all(abs(gi[k] + gr[k] + gd[k] - gt[k]) == 0 for k in range(3)), "gnvq split adds up")
    des = open(DESIGN, encoding="utf-8").read()
    check("only the colour-index stream" in des, "E4_DESIGN fact 1")

    # codewords used
    used = {r: [int(row(r, p)["codebook_distinct"]) for p in P] for r in PRIMARY}
    check(used["ogc"] == used["ogc_lam1e6"] == [4096] * 3, "OGC uses all")
    for p in P:
        want(f"| {p} | {used['c3dgs'][p]:,} | {used['gnvq_rho0'][p]:,} | {used['scalar'][p]:,} | {used['gnvq_cv'][p]:,} | "
             f"4,096 | {4096 / used['gnvq_cv'][p]:.3f} |", f"used {p}")
    empty = [4096 - x for x in used["c3dgs"]]
    want(f"leaves {min(empty):,}-{max(empty):,} of its 4,096 entries without a splat", "empty")
    kz = [FK[p]["rows"]["gnvq_cv"]["gn_vq"]["history_last3"][-1]["clusters_kept_zero_M"] for p in P]
    want(f"{min(kz):,}-{max(kz):,} of them at `gnvq_cv`'s last iteration", "kept zero M")
    inc = {r: [used[r][p] - used["c3dgs"][p] for p in P] for r in ("gnvq_cv", "gnvq_rho0", "scalar")}
    want(f"{min(inc['gnvq_cv'])}-{max(inc['gnvq_cv'])} (`gnvq_cv`), {min(inc['gnvq_rho0'])}-{max(inc['gnvq_rho0'])} (`gnvq_rho0`) "
         f"and {min(inc['scalar'])}-{max(inc['scalar'])} (`scalar`) more entries", "increase")
    rat = [4096 / x for x in used["gnvq_cv"]]
    want(f"OGC's is {min(rat):.2f}-{max(rat):.2f} times `gnvq_cv`'s", "ratio")
    want(f"its `.npz` is {abs(d2[0] if False else comp(diff('gnvq_cv', 'ogc', 'npz_bytes'))[0]):,.0f} bytes larger", "D2 bytes")

    # fidelity per angle
    fpa = ST["note_ii"]["fidelity_per_angle"]
    FD = {}
    for name, (a, b), key in (("D1", ("gnvq_cv", "gnvq_rho0"), "D1"), ("D2", ("gnvq_cv", "ogc"), "D2"),
                              ("lam", ("ogc_lam1e6", "ogc"), "ogc_lam1e6_minus_ogc")):
        FD[name] = {}
        for ang in ANG:
            v = [json.loads(row(a, p)["fidelity_psnr"])[ang] - json.loads(row(b, p)["fidelity_psnr"])[ang] for p in P]
            c = comp(v)
            FD[name][ang] = c
            check(abs(fpa[key][ang]["D_bar"] - c[0]) < 1e-12, f"fidelity {name} {ang} = summary's")
    for ang in ANG:
        al = ang if ang in ("0",) or ang.startswith("-") else "+" + ang
        want(f"| {al} | {sg(FD['D1'][ang][0])} ({FD['D1'][ang][3]:.4f}) | {sg(FD['D2'][ang][0])} ({FD['D2'][ang][3]:.4f}) | "
             f"{sg(FD['lam'][ang][0])} ({FD['lam'][ang][3]:.4f}) |", f"fidelity {ang}")
    check(all(FD["D1"][a][0] > 0 for a in ANG) and all(FD["D2"][a][0] < 0 for a in ANG), "D1 > 0, D2 < 0 at every angle")
    want(f"{sg(FD['D1']['0'][0])} dB to the uncompressed model, {sg(d1[0])} dB to the\nground truth".replace("\n", " "), "angle 0")

    def pview(a, b, ang, i):
        return [meas[p][a]["fidelity"]["per_view"][ang][i] - meas[p][b]["fidelity"]["per_view"][ang][i] for p in P]

    def without(a, b, ang, i):
        return sum(sum(meas[p][a]["fidelity"]["per_view"][ang][j] - meas[p][b]["fidelity"]["per_view"][ang][j]
                       for j in range(38) if j != i) / 37 for p in P) / 3

    big = set()
    for name, (a, b) in (("D1", ("gnvq_cv", "gnvq_rho0")), ("D2", ("gnvq_cv", "ogc")), ("lam", ("ogc_lam1e6", "ogc"))):
        for ang in ANG:
            for i in range(38):
                if max(abs(x) for x in pview(a, b, ang, i)) >= 10:
                    big.add((name, ang, names[i]))
    check(big == {("D1", "10", "00297"), ("D2", "10", "00297"), ("D1", "20", "00201"), ("lam", "40", "00049")},
          "the views reaching 10 dB")
    i297, i201, i049 = names.index("00297"), names.index("00201"), names.index("00049")
    fv = lambda r, i: meas[2][r]["fidelity"]["per_view"]["10"][i]  # noqa: E731
    for nm, ang in (("00297", "+10"), ("00201", "+20"), ("00049", "+40")):
        want(f"`{nm}` at {ang} degrees:", f"outlier {nm}")
    want(f"process 2's `gnvq_cv` render reads {fv('gnvq_cv', i297):.2f} dB against {fv('gnvq_rho0', i297):.2f} dB (`gnvq_rho0`) and "
         f"{fv('ogc', i297):.2f} dB\n(`ogc`)".replace("\n", " "), "00297 renders")
    want(f"D1's per-process differences there are {' / '.join(f'{x:+.2f}' for x in pview('gnvq_cv', 'gnvq_rho0', '10', i297))} dB",
         "00297 D1")
    want(f"Without that view D1 at +10 degrees is\n{sg(without('gnvq_cv', 'gnvq_rho0', '10', i297))} instead of "
         f"{sg(FD['D1']['10'][0])}, and D2 {sg(without('gnvq_cv', 'ogc', '10', i297))} instead of {sg(FD['D2']['10'][0])}"
         .replace("\n", " "), "00297 without")
    want(f"D1 {' / '.join(f'{x:+.2f}' for x in pview('gnvq_cv', 'gnvq_rho0', '20', i201))} dB; without it D1 is "
         f"{sg(without('gnvq_cv', 'gnvq_rho0', '20', i201))} instead of {sg(FD['D1']['20'][0])}", "00201")
    want(f"`ogc_lam1e6` minus `ogc` {' / '.join(f'{x:+.2f}' for x in pview('ogc_lam1e6', 'ogc', '40', i049))} dB, in every "
         f"process; without it that\ndifference is {sg(without('ogc_lam1e6', 'ogc', '40', i049))} instead of "
         f"{sg(FD['lam']['40'][0])}".replace("\n", " "), "00049")
    check(all(x > 0 for x in pview("ogc_lam1e6", "ogc", "40", i049)), "00049 in every process")
    fm = {r: [sum(json.loads(row(r, p)["fidelity_psnr"])[a] for p in P) / 3 for a in ANG] for r in ROWS}
    ftv = [x for r in ("c3dgs_ft", "ogc_ft", "gnvq_cv_ft") for x in fm[r]]
    pre_v = [x for r in PRIMARY for x in fm[r]]
    check(max(ftv) < min(pre_v), "fine-tuned rows farther from the uncompressed model")
    want(f"({min(ftv):.2f}-{max(ftv):.2f} dB across rows and angles, against\n{min(pre_v):.2f}-{max(pre_v):.2f} dB before "
         f"fine-tuning)".replace("\n", " "), "ft fidelity")
    check(all(sum(col(r + "_ft", "PSNR_ii")) > sum(col(r, "PSNR_ii")) for r in ("c3dgs", "ogc", "gnvq_cv")), "higher in protocol ii")
    want(f"Mean over the {len(names)} test cameras", "38")

    # terciles
    G = ST["note_ii"]["geometry"]
    angs = G["test_nearest_train_angle_deg"]
    want(f"runs from {min(angs):.3f} to {max(angs):.3f}\ndegrees".replace("\n", " "), "angle range")
    check(len(angs) == len(names) == 38, "one angle per test camera")
    want("Per test camera, in degrees: " + ", ".join(f"`{n}` {x:.3f}" for n, x in zip(names, angs)) + ".", "every angle")
    for t, (lab, idx, rng) in enumerate(zip(("near", "middle", "far"), G["terciles"], G["tercile_angle_ranges"])):
        check(rng == [min(angs[i] for i in idx), max(angs[i] for i in idx)], f"tercile {t} range")
        upv = json.loads(u["psnr_ii_per_view"])
        cells = []
        for a, b in (("gnvq_cv", "gnvq_rho0"), ("gnvq_cv", "ogc")):
            v = []
            for p in P:
                pa, pb2 = json.loads(row(a, p)["psnr_ii_per_view"]), json.loads(row(b, p)["psnr_ii_per_view"])
                v.append(sum(pa[i] for i in idx) / len(idx) - sum(pb2[i] for i in idx) / len(idx))
            c = comp(v)
            cells.append(f"{sg(c[0])} ({c[3]:.4f})")
            key = "D1" if b == "gnvq_rho0" else "D2"
            check(abs(ST["note_ii"]["terciles"][key][str(t)]["D_bar"] - c[0]) < 1e-12, f"tercile {t} {key} = summary's")
        want(f"| {lab} | {len(idx)} | {rng[0]:.3f}-{rng[1]:.3f} | {sum(upv[i] for i in idx) / len(idx):.3f} | {cells[0]} | {cells[1]} |",
             f"tercile {t}")

    # geometry and coverage
    cond = G["conditioning"]
    ev = G["eigenvalues_over_n"]
    want(f"is\n{ev[0]:.3f} per camera (the others {ev[1]:.3f} and {ev[2]:.3f})".replace("\n", " "), "eigen")
    check(abs(cond["lambda_min_over_n_train"] - ev[0]) < 1e-15 and G["n_train"] == 263, "conditioning")
    want(f"over the {G['n_train']} training cameras", "263")
    want(f"lie {cond['train_distance_min']:.2f} to {cond['train_distance_max']:.2f} world units from the centre (median "
         f"{cond['train_distance_median']:.2f})", "distances")
    nb = round(cond["test_share_beyond_farthest_train"] * G["n_test"])
    check(cond["orbit_share_beyond_farthest_train"] == cond["test_share_beyond_farthest_train"], "orbit share = test share")
    want(f"{nb} of the {G['n_test']} test cameras ({cond['test_share_beyond_farthest_train'] * 100:.2f}%), and so "
         f"{cond['orbit_share_beyond_farthest_train'] * 100:.2f}% of the {cond['n_orbit_cameras']} orbit cameras", "beyond")
    cov = ST["note_ii"]["coverage"]
    for lab, key in (("all", "all"), ("colour-quantized (probe)", "colour_quantized_probe")):
        c = cov[key]
        lt = c["low_tercile"]
        want(f"| {lab} | {c['n_splats']:,} | {c['n_zero_trace_left_out']:,} | {c['min']:.2f} | {c['p10']:.2f} | {c['p25']:.2f} | "
             f"{c['p50']:.2f} | {c['p75']:.2f} | {c['p90']:.2f} | {c['max']:.2f} | {lt['n']:,}, {lt['rank_max']:.2f}, "
             f"{lt['trace_share']:.3f} |", f"coverage {key}")
    want(f"effective rank of {cov['all']['p50']:.2f} of 16", "median rank")
    check(cov["all"]["n_zero_trace_left_out"] == M["gn"]["full"]["splats_zero_trace"], "zero trace = GN pass's")

    # OGC's job
    t19 = OG["table19"]
    check(t19["ok"] and all(s["returncode"] == 0 for s in t19["steps"]), "Table 19 ran")
    labels19 = [("full", "the full model"), ("trunc2", "truncation, degree 2"), ("ours2", "their projection, degree 2"),
                ("trunc1", "truncation, degree 1"), ("ours1", "their projection, degree 1"),
                ("trunc0", "truncation, degree 0"), ("ours0", "their projection, degree 0")]
    for k, lab in labels19:
        r = t19["rows"][k]
        check(abs(r["psnr"] - r["published_psnr"] - r["minus_published"]) < 1e-12 and round(r["psnr"], 2) == r["published_psnr"],
              f"Table 19 {k} rounds to the published value")
        want(f"| {lab} | {r['psnr']:.4f} | {r['published_psnr']:.2f} | {sg(r['minus_published'])} |", f"t19 {k}")
    want(f"Their full model reads {t19['rows']['full']['psnr']:.4f} dB in their evaluation and {f(u['PSNR_ii']):.3f} dB in protocol ii",
         "two protocols")
    dp = OG["deps"]
    check(dp["missing_modules"] == ["lpips"] and dp["installed"] == ["lpips==0.1.4"] and not dp["session_site_packages_modified"]
          and dp["shadows_session"] == [] and dp["ok"], "deps")
    want(f"(`{dp['installed'][0]}`)", "lpips")
    cmp_ = OG["comparison"]
    want(f"over the {cmp_['n_tr_A_positive']:,} splats with `tr(A_i)` > 0", "tr A > 0")
    ps_ = cmp_["per_splat_relative_error"]
    want(f"| {cmp_['relative_error']:.3f} | {ps_['median']:.3f} | {ps_['p90']:.3f} | {ps_['p99']:.3f} | "
         f"{cmp_['trace_ratio_M_over_A']:.4f} | {cmp_['n_exactly_one_trace_zero']} |", "gram")
    want(f"Summed over {cmp_['n_tr_A_positive']:,} splats and {cmp_['our_metric']['n_views']} views it does not leave our total "
         f"{(1 - cmp_['trace_ratio_M_over_A']) * 100:.1f}% short", "shortfall")
    want(f"the per-splat\nspread (median {ps_['median']:.3f})".replace("\n", " "), "spread")
    want(f"makes our metric's trace {(1 - cmp_['trace_ratio_M_over_A']) * 100:.1f}% short", "settles trace")
    fjc = OG["fork_job_cache"]
    check(fjc["present"] and not fjc["bit_identical"] and cmp_["our_metric"]["probe_seed"] == 0, "cache not bit-identical, seed 0")
    want(f"with the same probe seed\n({cmp_['our_metric']['time_s']:.1f} s), and it differs from job fork's cache by up to "
         f"{fjc['max_abs_diff']:.5f} in an entry".replace("\n", " "), "cache")

    # time and memory
    setup = tim["restore_s"] + tim["install_s"] + tim["c3dgs_build_s"]
    jf = M["timings_s"]["job"]
    want(f"restore {tim['restore_s']:.1f} s, install {tim['install_s']:.1f} s, C3DGS build {tim['c3dgs_build_s']:.1f} s", "setup")
    want(f"{jf:,.1f} s by its own clock ({tim['gn_e4p_fork_train_s']:,.1f} s in\nthe queue), {jf / 3600:.1f} h; with setup and the "
         f"queue's time, {setup + tim['gn_e4p_fork_train_s']:,.1f} s ({(setup + tim['gn_e4p_fork_train_s']) / 3600:.2f} h)"
         .replace("\n", " "), "fork job")
    want(f"{MO['timings_s']['job']:,.1f} s ({tim['gn_e4p_ogc_train_s']:,.1f} s in the queue)", "ogc job")
    stp = M["steps"]
    by = {}
    for s in stp:
        by.setdefault(s["name"], []).append(s)
    one = lambda n: by[n][0]  # noqa: E731
    rss = lambda s: gb((s.get("host_rss") or {}).get("rss_peak_bytes", 0))  # noqa: E731
    al = lambda s: gb(s["cuda_peak"]["allocated"])  # noqa: E731
    want(f"| dataset download | {one('download_dataset')['time_s']:.1f} | {al(one('download_dataset'))} | {rss(one('download_dataset'))} |",
         "download")
    want(f"| probe run (C3DGS's process) | {one('c3dgs_probe')['time_s']:.1f} | {gb(runs['probe']['wrapper']['max_memory_allocated'])} | - |",
         "probe step")
    br = one("build_runner")
    want(f"| runner (first build) | {br['time_s']:.1f} | {al(br)} | {rss(br)} |", "runner")
    gf, ge = one("gn_pass16_full"), one("gn_pass16_even")
    want(f"| GN passes 16 x 16, all / even views | {gf['time_s']:.1f} / {ge['time_s']:.1f} | {al(gf)} / {al(ge)} | {rss(gf)} / {rss(ge)} |",
         "gn passes")
    cvs = [s for s in stp if s["name"].startswith("gn_vq_cv_")]
    dms = [s for s in stp if s["name"].startswith("dmse_cv_")]
    check(len(cvs) == len(dms) == 7, "7 CV steps")
    want(f"| CV: GN-VQ, 7 runs | {min(s['time_s'] for s in cvs):.1f}-{max(s['time_s'] for s in cvs):.1f} each, "
         f"{sum(s['time_s'] for s in cvs):.1f} in all | {min(al(s) for s in cvs)}-{max(al(s) for s in cvs)} | "
         f"{min(rss(s) for s in cvs)}-{max(rss(s) for s in cvs)} |", "cv steps")
    want(f"| CV: dMSE, 7 rows | {min(s['time_s'] for s in dms):.1f}-{max(s['time_s'] for s in dms):.1f} each | "
         f"{min(al(s) for s in dms)}-{max(al(s) for s in dms)} | {min(rss(s) for s in dms)}-{max(rss(s) for s in dms)} |", "dmse steps")
    orr, cvg = one("note_ii_orbit_reference_renders"), one("note_ii_coverage")
    want(f"| note ii: orbit reference renders / coverage | {orr['time_s']:.1f} / {cvg['time_s']:.1f} | {al(orr)} / {al(cvg)} | "
         f"{rss(orr)} / {rss(cvg)} |", "note ii steps")
    pst = [one(f"c3dgs_p{p}_a0") for p in P]
    pk = {gb(W[p]["max_memory_allocated"]) for p in P}
    check(len(pk) == 1, "process peaks equal to 0.01 GB")
    want(f"| the three forked processes | {' / '.join(f'{s['time_s']:,.1f}' for s in pst)} | {pk.pop()} (C3DGS's process) | "
         f"{min(rss(s) for s in pst)}-{max(rss(s) for s in pst)} |", "processes")
    pev = one("c3dgs_probe_eval")
    want(f"| probe evaluation from its `.npz` | {pev['time_s']:.1f} | {gb(runs['probe_eval']['wrapper']['max_memory_allocated'])} "
         f"(C3DGS's process) | {rss(pev)} |", "probe eval")
    n2 = [s for s in stp if s["name"].startswith("npz2ply_")]
    e2s = [s for s in stp if s["name"].startswith("eval_ii_") and s["name"] != "eval_ii_uncompressed"]
    fis = [s for s in stp if s["name"].startswith("fidelity_")]
    check(len(n2) == len(e2s) == 28 and len(fis) == 27, "28 decoded rows, 27 fidelity")
    want(f"| per decoded row: `npz2ply.py` / protocol ii / fidelity renders | {min(s['time_s'] for s in n2):.1f}-"
         f"{max(s['time_s'] for s in n2):.1f} / {min(s['time_s'] for s in e2s):.1f}-{max(s['time_s'] for s in e2s):.1f} / "
         f"{min(s['time_s'] for s in fis):.2f}-{max(s['time_s'] for s in fis):.2f} | - / {max(al(s) for s in e2s)} / "
         f"{max(al(s) for s in fis)} | - |", "per decoded row")
    tot = sum(s["time_s"] for s in stp)
    want(f"The steps add up to {tot:,.1f} s; {jf - tot:.1f} s of the job lies outside them", "steps add up")
    fr_tot = ST["note_ii"]["fidelity_render_time_total_s"]
    check(abs(fr_tot - sum(s["time_s"] for s in fis)) < 1e-9, "fidelity total = 27 steps")
    want(f"The 27 rows' fidelity renders took {fr_tot:.1f} s in all", "fidelity total")
    cost = {p: FK[p]["cost"] for p in P}
    colour = [sum(cost[p][r]["time_s"] for r in PRIMARY) for p in P]
    saves = [sum(v["time_s"] for k, v in cost[p].items() if k.startswith("save_")) for p in P]
    evp = [sum(v["time_s"] for k, v in cost[p].items() if k.startswith("eval_") and not k.endswith("_ft")) for p in P]
    fts = [sum(v["time_s"] for k, v in cost[p].items() if k.startswith("finetune_")) for p in P]
    evf = [sum(v["time_s"] for k, v in cost[p].items() if k.startswith("eval_") and k.endswith("_ft")) for p in P]
    wall = [W[p]["wall_s"] for p in P]
    rest = [wall[p] - sum(v["time_s"] for v in cost[p].values()) for p in P]
    check(all(len([k for k in cost[p] if k.startswith("save_")]) == 5 for p in P), "five saves")
    rng = lambda xs, d=1: f"{min(xs):,.{d}f}-{max(xs):,.{d}f}"  # noqa: E731
    want(f"`compress.py`'s own wall time {rng(wall)} s", "wall")
    want(f"the six colour rows {rng(colour)} s, the five saves {rng(saves)} s, C3DGS's evaluation of the six rows {rng(evp)} s",
         "inside 1")
    want(f"the three fine-tunings with their saves {rng(fts)} s, and their evaluations {rng(evf)} s", "inside 2")
    want(f"the rest, {rng(rest)} s", "inside rest")
    for r in PRIMARY:
        c = [cost[p][r] for p in P]
        t = [x["time_s"] for x in c]
        g = {gb(x["cuda_peak_allocated"]) for x in c}
        check(len(g) == 1, f"{r} peaks equal to 0.01 GB")
        r0 = [x["host"]["rss_start_bytes"] for x in c]
        r1 = [x["host"]["rss_peak_bytes"] for x in c]
        lab = "`c3dgs` (C3DGS's own VQ)" if r == "c3dgs" else f"`{r}`"
        want(f"| {lab} | {min(t):.1f}-{max(t):.1f} | {g.pop()} | {gb(min(r0))}-{gb(max(r0))} | {gb(min(r1))}-{gb(max(r1))} |",
             f"cost {r}")
    # OGC's chunk
    chunk, K = 100_000, 4096
    four = 4 * chunk * K * 4
    inputs = chunk * 256 * 4 + chunk * 48 * 4
    cbook = K * 3 * 16 * 4 + K * 256 * 4 + K * 48 * 4
    check(FK[0]["rows"]["ogc"]["ogc"]["call"]["chunk"] == chunk and FK[0]["rows"]["ogc"]["ogc"]["call"]["K"] == K, "the call's chunk, K")
    rise = [cost[p][r]["cuda_peak_allocated"] - cost[p][r]["start_allocated"] for r in ("ogc", "ogc_lam1e6") for p in P]
    want(f"4 x 100,000 x 4,096 x 4 = {four:,} bytes", "four")
    want(f"float32: {inputs:,} bytes; the codebook's copies: {cbook:,} bytes", "inputs")
    want(f"in all {four + inputs + cbook:,} bytes, against the measured rise of OGC's rows above their start, "
         f"{min(rise):,}-{max(rise):,}\nbytes; {min(rise) - four - inputs - cbook:,}-{max(rise) - four - inputs - cbook:,} bytes are not "
         f"attributed".replace("\n", " "), "rise")
    c25 = 4 * 25_000 * K * 4 + 25_000 * (256 + 48) * 4 + cbook
    want(f"chunk 25,000 would hold {c25:,} bytes", "chunk 25k")
    check("OGC's assignment chunk (100,000 x 4,096 float32 scores and their inputs, 1.76 GB)" in re.sub(r"\s+", " ", pre), "note i 1.76")
    want("took OGC's chunk as 1.76 GB", "1.76")
    # reserved
    res_ = [W[p]["max_memory_reserved"] for p in P]
    check(all(W[p]["max_memory_reserved"] < W[p]["max_memory_allocated"] for p in P), "reserved below allocated")
    want(f"`max_memory_reserved` of {res_[0]:,},\n{res_[1]:,} and {res_[2]:,} bytes".replace("\n", " "), "reserved")
    want(f"below their allocated peaks of {gb(W[0]['max_memory_allocated'])} GB", "allocated")
    pw = runs["probe"]["wrapper"]
    check(pw["max_memory_reserved"] > pw["max_memory_allocated"], "probe reserved above allocated")
    want(f"reserved {pw['max_memory_reserved']:,} bytes against {pw['max_memory_allocated']:,} allocated", "probe reserved")
    ram = ST["session_ram"]
    want(f"the session had {gb(ram['total_bytes'])} GB, {gb(ram['available_bytes'])} GB available", "ram")
    hg = FK[0]["rows"]["ogc"]["ogc"]["host_bytes_G"]
    check(hg == nq * 1024 and all(FK[p]["rows"][r]["ogc"]["host_bytes_G"] == hg for p in P for r in ("ogc", "ogc_lam1e6")), "G bytes")
    want(f"OGC's `G` for the {nq:,} quantized splats\nis {hg:,} bytes".replace("\n", " "), "G")
    os_ = {s["name"]: s for s in ST["ogc_job_steps"]}
    want(f"Job ogc's steps peaked at {gb(max(s['host_rss_peak_bytes'] for s in os_.values()))} GB", "ogc rss")
    check(max(os_.values(), key=lambda s: s["host_rss_peak_bytes"])["name"] == "ogc_table19", "their Table 19 peak")
    ts = t19["steps"]
    want(f"their Table 19 commands {os_['ogc_table19']['time_s']:.1f} s (`gram.py` {ts[0]['time_s']:.1f} s, `shfit.py` "
         f"{ts[1]['time_s']:.1f} s, `run_exps.py` {ts[2]['time_s']:.1f} s)", "t19 steps")
    eg = OG["exact_gram"]
    want(f"their exact Gram {os_['ogc_exact_gram']['time_s']:.1f} s, whose process peaked at "
         f"{gb(eg['info']['cuda_peak_allocated'])} GB of GPU memory; our metric {os_['ogc_job_gn_pass16_full']['time_s']:.1f} s "
         f"({gb(os_['ogc_job_gn_pass16_full']['cuda_peak_allocated'])} GB); the comparison\n{os_['ogc_compare_gram']['time_s']:.1f} s"
         .replace("\n", " "), "ogc steps")
    # against the estimate (HANDOFF's text, quoted)
    ho = re.sub(r"\s+", " ", open(HANDOFF, encoding="utf-8").read())
    for q in ("So job fork is about 8,700-10,000 s.", "about 2,440 s per process", "(about 74 s each)",
              "up to about 1,260 s over the 27 rows, plus about 55 s for the reference renders", "**setup:** about 396 s",
              "so about 6.0 GB", "roughly 12-14 GB for both jobs"):
        check(q in ho, f"HANDOFF still says {q!r}")
    want(f"job fork {jf:,.1f} s against \"about 8,700-10,000 s\"; each process about {round(sum(wall) / 3, -1):,.0f} s against "
         f"\"about 2,440 s\"", "est 1")
    ot = [cost[p][r]["time_s"] for r in ("ogc", "ogc_lam1e6") for p in P]
    want(f"OGC's rows {min(ot):.1f}-{max(ot):.1f} s against \"about 74 s each\"", "est 2")
    check(3.5 < (sum(cost[p]["gnvq_cv"]["time_s"] for p in P) / 3) / (sum(ot) / 6) < 4.5, "a quarter of a GN-VQ row")
    want(f"the fidelity renders {fr_tot:.1f} s and the reference renders {orr['time_s']:.1f} s against \"up to about 1,260 s\" and "
         f"\"about 55 s\"", "est 3")
    want(f"setup {setup:.1f} s against \"about 396 s\"", "est 4")
    want(f"the process's GPU peak {gb(W[0]['max_memory_allocated'])} GB against \"about 6.0 GB\"", "est 5")
    want(f"host RSS: job fork's steps peaked at {gb(max(s.get('host_rss', {}).get('rss_peak_bytes', 0) for s in stp))} GB and job "
         f"ogc's at {gb(max(s['host_rss_peak_bytes'] for s in os_.values()))} GB", "est 6")

    # post hoc: the observer, the clip
    fr = {r: [arr(r + "_ft", p)["features_rest"]["compressed_bytes"] - arr(r, p)["features_rest"]["compressed_bytes"] for p in P]
          for r in ("ogc", "c3dgs", "gnvq_cv")}
    rel = [-fr["ogc"][p] / arr("ogc", p)["features_rest"]["compressed_bytes"] * 100 for p in P]
    want(f"shrank `ogc`'s `features_rest` by {' / '.join(f'{-x:,}' for x in fr['ogc'])} bytes ({min(rel):.1f}-{max(rel):.1f}%)",
         "ogc shrink")
    want(f"while `c3dgs`'s grew by {' / '.join(f'{x:,}' for x in fr['c3dgs'])} and `gnvq_cv`'s shrank by "
         f"{' / '.join(f'{-x:,}' for x in fr['gnvq_cv'])}", "others")
    check(all(x < 0 for x in fr["ogc"] + fr["gnvq_cv"]) and all(x > 0 for x in fr["c3dgs"]), "signs")
    oft = comp(diff("ogc_ft", "c3dgs_ft", "npz_bytes"))[0]
    want(f"`ogc_ft` is {abs(oft):,.0f} bytes below `c3dgs_ft` on average after `ogc` was "
         f"{comp(diff('ogc', 'c3dgs', 'npz_bytes'))[0]:,.0f} above `c3dgs`", "ft bytes")
    for p in P:
        check(W[p]["e4p"]["qa_at_save"] == qa[0], f"p{p} quantizer at row 1's save = at the colour step")
    q3 = json.loads(e3r["gnvq_k4096_ft5000"]["qa_at_save"])
    want(f"AC scale moving ({qa[0]['rest_scale']:.5g} to {q3['rest_scale']:.5g}, section 15)", "E3r scale")
    qs = {p: {r: FK[p]["rows"][r]["gn_vq"]["quantizer"] for r in ("gnvq_rho0", "scalar", "gnvq_cv")} for p in P}
    width = [qs[p]["gnvq_cv"]["rest"]["codebook_max"] - qs[p]["gnvq_cv"]["rest"]["codebook_min"] for p in P]
    order_w = sorted(P, key=lambda p: -width[p])
    order_s = sorted(P, key=lambda p: fr["gnvq_cv"][p])
    check(order_w == order_s, "the shrink orders as the AC width")
    want(f"AC codebook range ({', '.join(f'{w:.3f}' for w in width)})", "widths")
    for p in P:
        g = FK[p]["rows"]["gnvq_cv"]["gn_vq"]
        lo, hi = g["warm_start"]["range"]
        wa = g["warm_start"]["quantizer"]["rest"]
        iv = lambda q: f"[{q['codebook_min']:.4f}, {q['codebook_max']:.4f}]"  # noqa: E731
        want(f"| {p} | [{lo:.4f}, {hi:.4f}] | {iv(wa)} | {iv(qs[p]['gnvq_rho0']['rest'])} | {iv(qs[p]['scalar']['rest'])} | "
             f"{iv(qs[p]['gnvq_cv']['rest'])} |", f"clip {p}")
        for r in ("gnvq_rho0", "scalar"):
            check(FK[p]["rows"][r]["gn_vq"]["warm_start"]["range"] == [lo, hi], f"p{p} {r} same clip range")
    grids = {(q["rest"]["grid_min"], q["rest"]["grid_max"], q["dc"]["grid_min"], q["dc"]["grid_max"]) for p in P for q in qs[p].values()}
    check(len(grids) == 1, "one grid")
    agmin, agmax, dgmin, dgmax = grids.pop()
    want(f"The AC int8 grid is [{agmin:.4f}, {agmax:.4f}]** (scale {qa[0]['rest_scale']:.5g}, zero point {qa[0]['rest_zero_point']})",
         "AC grid")
    want(f"inside the DC grid, [{dgmin:.4f}, {dgmax:.4f}]", "DC grid")
    cvq = [qs[p]["gnvq_cv"]["rest"] for p in P]
    check(all(q["codebook_max"] > agmax for q in cvq) and [q["codebook_min"] < agmin for q in cvq] == [True, False, False],
          "gnvq_cv past the top in all, the bottom in process 0")
    check(all(max(qs[p]["gnvq_rho0"]["rest"]["codebook_max"], -qs[p]["gnvq_rho0"]["rest"]["codebook_min"])
              > max(qs[p]["gnvq_cv"]["rest"]["codebook_max"], -qs[p]["gnvq_cv"]["rest"]["codebook_min"]) for p in P), "rho0 further")
    check(all(agmin <= qs[p]["scalar"]["rest"]["codebook_min"] and qs[p]["scalar"]["rest"]["codebook_max"] <= agmax for p in P),
          "scalar inside")
    check(all(dgmin <= q["dc"]["codebook_min"] and q["dc"]["codebook_max"] <= dgmax for p in P for q in qs[p].values()), "DC inside")

    # the title, the summary, what E4p settles
    want(f"D1 is {sg(d1[0])} dB, D2 {sg(d2[0])} dB, and OGC uses all 4,096 codewords", "title")
    want(f"({gb(W[0]['max_memory_allocated'])} GB at the peak, OGC's assignment chunk)", "summary peak")
    want(f"by {sg(gc[0])} dB at +{gcb:.2f}% bytes. D1\n(`rho_cv` against rho = 0) was {sg(d1[0])} dB with `SE_noise` {d1[3]:.4f}. "
         f"D2 (against OGC's VQ) was {sg(d2[0])} dB in all three\nprocesses, against a row using all 4,096 codewords to `gnvq_cv`'s "
         f"{min(used['gnvq_cv']):,}-{max(used['gnvq_cv']):,} and {abs(comp(diff('gnvq_cv', 'ogc', 'npz_bytes'))[0]):,.0f} more bytes"
         .replace("\n", " "), "summary")
    want(f"total trace is {cmp_['trace_ratio_M_over_A']:.3f}\nof their exact Gram's".replace("\n", " "), "summary trace")
    want(f"{gb(W[0]['max_memory_allocated'])} GB at the peak (OGC's chunk)", "settles peak")
    want(f"`gnvq_cv` beats C3DGS's own VQ by {sg(gc[0])} dB at +{gcb:.2f}% bytes, `SE_noise` {gc[3]:.4f}", "settles gc")
    want(f"D1 is about 0: {sg(d1[0])} dB, `SE_noise` {d1[3]:.4f}", "settles d1")
    want(f"D2 is {sg(d2[0])} dB, negative in all three processes, but against a row that uses all 4,096 codewords to\n"
         f"`gnvq_cv`'s {min(used['gnvq_cv']):,}-{max(used['gnvq_cv']):,} and spends "
         f"{abs(comp(diff('gnvq_cv', 'ogc', 'npz_bytes'))[0]):,.0f} more bytes".replace("\n", " "), "settles d2")
    want(f"is {cmp_['trace_ratio_M_over_A']:.3f} of their exact Gram's", "settles trace")
    want(f"all {len(gv)} runs stopped at the cap", "settles cap")
    stamps = sorted(r["timestamp"] for r in R.values())
    want(f"rows timestamped\n{stamps[0][:16]} to {stamps[-1][11:16]}".replace("\n", " "), "sources stamps")
    want(f"one Kaggle session on {env['gpu']['count']}x T4", "sources session")
    want(f"gsplat commit `{env['gsplat_commit'][:8]}`", "commit")

    # this repository's cited lines, read at the commit E4p ran at
    for path, ln, needle, cite in CITED:
        check(needle in lines_of(path, ln, ln), f"{path}:{ln} holds {needle[:40]!r}")
        want(cite, f"cite {path}:{ln}")

    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | OUTSIDE_LINES | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 16: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
