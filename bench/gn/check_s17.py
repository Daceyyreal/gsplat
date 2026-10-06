"""Re-check every number in kaggle/FINDINGS.md section 17 (E4q), its summary paragraph and its Sources entry against
the committed files.

    python bench/gn/check_s17.py

It reads only committed files: the E4q bundle (``kaggle/gn_e4q/gn4q/``), E4p's rows (``kaggle/gn_e4p/gn4p/``, the post
hoc chain), Amendment 16 e's estimate and note i's feasibility (``kaggle/PREREG_GN.md``), the test of ``lad_all`` against
OGC's code (``bench/gn/test_gn.py``), ``bench/gn/g2.py``'s BD fit for the post hoc checks, and this repository's job,
hooks and stand-in for the file:line citations, read at the commit E4q ran at. OGC's line numbers (``vq.py`` at
``49ccae72``) and C3DGS's (at ``2a234af5``) were read from their sources, which are not in the repository; they are
listed constants. Each quoted number is recomputed and must appear in the text, each claim is re-asserted, and every
numeric token of the text must be a recomputed string or a listed constant. It prints each failure and exits 1 if
there is any, 0 otherwise. Built like ``check_s16.py``.
"""

import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
E4Q = os.path.join(REPO, "kaggle", "gn_e4q", "gn4q")
E4P = os.path.join(REPO, "kaggle", "gn_e4p", "gn4p")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
PREREG = os.path.join(REPO, "kaggle", "PREREG_GN.md")
TESTS = os.path.join(HERE, "test_gn.py")
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
# the amendments', notes', protocols' and codecs' own constants, the design's seeds, angles, K, chunks, caps, grids and
# threshold rule, names and dates; not results
CONSTANTS = {"0", "1", "2", "3", "4", "5", "8", "9", "11", "12", "13", "14", "15", "16", "17", "20", "4,096", "100,000",
             "25,000", "1e-3", "1e-2", "1e-1", "3e-1", "1e-4", "1e-6", "6e-7", "10^9", "-40", "-20", "-10", "+10", "+20",
             "+40", "40", "-2", "-1", "+1", "+2", "1.14", "0.0"}
# OGC's vq.py at 49ccae72 and C3DGS's compress.py, utils/loss_utils.py, compression/vq.py and arguments/__init__.py at
# 2a234af5 (their lines, and C3DGS's colour batches, batch size and decay); not in this repository
OUTSIDE_LINES = {"17", "105", "107", "45", "33", "35", "37", "58", "78", "116", "77", "79", "100", "262,144", "0.8"}
COMMITS = {"1f0b7e15", "de49bd24", "c0e34e8e", "1d986d1a", "9db4cf8b", "2a234af5", "49ccae72"}
# this repository's citations: (path, first line, last line, text the lines must contain, the citation in the text)
CITED_AT = "c0e34e8e"
CITED = [("kaggle/gn_e4q_scene.py", 632, 636, 'measured_ok = rec.get("PSNR_ii") not in ("", None)',
          "(`kaggle/gn_e4q_scene.py:632-636`)"),
         ("kaggle/gn_e4q_scene.py", 632, 636, 'rec["status"] = "alias" if r.get("alias_of") else "ok"', "`kaggle/gn_e4q_scene.py:632-636`"),
         ("kaggle/e4p_hooks.py", 359, 361, 'fr["rows"][row]["c3dgs_eval_error"]', "(`kaggle/e4p_hooks.py:359-361`)"),
         ("kaggle/gn_e4q_scene.py", 675, 675, 'row["status"] = "ok" if evii and e else "failed"', "(`:675`)"),
         ("kaggle/gn_e4q_scene.py", 538, 538, '"vq_reseeded_total": sum(h.get("reseeded", 0) for h in hist)',
          "(`kaggle/gn_e4q_scene.py:538`)"),
         ("kaggle/gn_e4q_scene.py", 506, 506, 'hist = gq.get("history_last3") or []', "its last three entries"),
         ("bench/gn/dryrun/fake_c3dgs/compress.py", 32, 36, 'return {"SSIM": 0.8, "PSNR": -10.0 * torch.log10(torch.tensor(mse)).item()',
          "(`bench/gn/dryrun/fake_c3dgs/compress.py:32`)")]
SC = ("train", "treehill")
P = (0, 1)
LAD = ["lad_reseed", "lad_init_tr", "lad_clip_part", "lad_no_clip", "lad_ridge_mean", "lad_iters15", "lad_iters50",
       "lad_no_final_int8"]
ROWS = ["c3dgs", "gnvq_cv"] + LAD + ["lad_all", "ogc"]
ANG = ["-40", "-20", "-10", "0", "10", "20", "40"]
RHO_LABEL = {0.0: "0", 0.001: "1e-3", 0.01: "1e-2", 0.1: "1e-1", 0.3: "3e-1", 1.0: "1", 3.0: "3"}
LAM_LABEL = {1e-6: "1e-6", 1e-4: "1e-4", 1e-3: "1e-3", 1e-2: "1e-2", 1e-1: "1e-1", 1.0: "1"}
DIFFS = ["ogc_minus_c3dgs", "gnvq_cv_minus_c3dgs", "gnvq_cv_minus_ogc", "lad_all_minus_ogc", "lad_reseed_minus_gnvq_cv",
         "lad_init_tr_minus_gnvq_cv"]


def section_text(txt: str) -> str:
    start = txt.index("## 17. E4q")
    nxt = txt.find("\n## ", start + 1)
    sec = txt[start:] if nxt < 0 else txt[start:nxt]
    s0 = txt.index("**E4q (section 17")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 17:")
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


def gb(b) -> str:
    return f"{float(b) / 1e9:.2f}"


def comp(v):
    d = sum(v) / len(v)
    var = sum((x - d) ** 2 for x in v) / (len(v) - 1)
    sd = math.sqrt(var)
    return d, var, sd, sd / math.sqrt(len(v))


def pooled(vals):
    return 10 * math.log10(1 / (sum(10 ** (-v / 10) for v in vals) / len(vals)))


def run():
    sys.path.insert(0, HERE)
    import g2
    import numpy as np

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

    S = json.load(open(os.path.join(E4Q, "gn4q_summary.json")))
    R = {s: {r["config"]: r for r in csv.DictReader(open(os.path.join(E4Q, f"gn4q_results_{s}.csv"), newline=""))} for s in SC}
    M = {s: json.load(open(os.path.join(E4Q, f"gn4q_meta_{s}.json"))) for s in SC}
    CV = {s: list(csv.DictReader(open(os.path.join(E4Q, f"gn4q_cv_{s}.csv"), newline=""))) for s in SC}
    LCV = {s: list(csv.DictReader(open(os.path.join(E4Q, f"gn4q_lamcv_{s}.csv"), newline=""))) for s in SC}
    env = json.load(open(os.path.join(E4Q, "gn4q_env.json")))
    tim = json.load(open(os.path.join(E4Q, "timings.json")))
    build = json.load(open(os.path.join(E4Q, "gn4q_c3dgs_build.json")))
    tails = {s: json.load(open(os.path.join(E4Q, f"gn_e4q_{s}_log_tail.json"))) for s in SC}
    ST = {s: S["scenes"][s] for s in SC}
    W = {s: {k: r["wrapper"] for k, r in M[s]["runs"].items()} for s in SC}
    FK = {s: {p: W[s][f"j0_p{p}_a0"]["e4p"]["fork"] for p in P} for s in SC}
    row = lambda s, r, p: R[s][f"p{p}_{r}"]  # noqa: E731

    def dsp(s, a, b, k="PSNR_ii"):
        return [f(row(s, a, p)[k]) - f(row(s, b, p)[k]) for p in P]

    # what ran
    check(all(t["exit_code"] == 0 and not t["skipped_by_cutoff"] for t in tails.values()), "both jobs exit 0, not skipped")
    st = {s: {} for s in SC}
    for s in SC:
        for r in R[s].values():
            st[s][r["status"]] = st[s].get(r["status"], 0) + 1
    tr, th = R["train"], R["treehill"]
    d0 = [c for c in tr if c.startswith("p") and c != "probe"]
    sw = [c for c in tr if c.startswith("j")]
    check(len(d0) == 26 and len(sw) == 20 and {"uncompressed", "probe"} <= set(tr), "train's rows")
    want(f"**train** wrote {len(tr)} rows: `uncompressed`, the `probe`, {len(d0)} rows at the default point ({len(d0) // 2} per process) "
         f"and {len(sw)} at the sweep's four other points ({len(sw) // 4} per point). {st['train']['ok']} are `ok` and "
         f"{st['train']['alias']} `alias`", "train rows")
    check(set(st["train"]) == {"ok", "alias"}, "train: only ok and alias")
    want(f"**treehill** wrote {len(th)} rows: `uncompressed`, the `probe` and {len([c for c in th if c.startswith('p') and c != 'probe'])} "
         f"at the default point. {st['treehill']['ok']} are `ok`, {st['treehill']['alias']} `alias` and {st['treehill']['failed']}", "treehill rows")
    check([c for c, r in th.items() if r["status"] == "failed"] == ["probe"], "treehill's failed row is the probe")
    for s in SC:
        t = ST[s]
        check(not t["failed_steps"] and not t["skipped_steps"] and not t["deviations"] and t["dropped"] is None
              and not t["cfg_args_mismatch"] and not M[s]["cfg_args_mismatch"], f"{s}: nothing failed")
        for k, pr in t["processes"].items():
            a = pr["attempts"]
            check(len(a) == 1 and a[0]["attempt"] == 0 and not a[0]["oom"] and not a[0]["checks_failed"], f"{s} {k} one attempt")
        check(t["scene_device"] == ("cuda" if s == "train" else "cpu")
              and all(r["data_device"] == t["scene_device"] for r in R[s].values() if r["kind"] in ("fork", "probe")), f"{s} device")
    check(M["treehill"]["done"] is False and M["treehill"]["missing_or_failed"] == ["probe"] and M["train"]["done"], "done flags")

    # inputs and checks
    check(env["gpu"]["count"] == 2 and env["gpu"]["name"] == "Tesla T4" and env["wheel_restored"], "2x T4, restored wheel")
    want(f"torch {env['torch']} (CUDA {env['torch_cuda']}), driver {env['nvidia_smi'].split(', ')[1]}, cuDNN {env['cudnn']}, "
         f"Python {env['python']}, {env['gpu']['count']}x Tesla T4; gsplat commit `{env['gsplat_commit'][:8]}`, the restored wheel "
         f"(install {tim['install_s']:.1f} s)", "session")
    check(build["ok"] and build["head"].startswith("2a234af5") and len(build["deviations"]) == 6, "build")
    want(f"{build['build_time_s']:.1f} s of builds and {build['total_time_s']:.1f} s in all", "build times")
    check(all(ST[s]["ogc_clone"]["head"].startswith("49ccae72") and ST[s]["ogc_clone"]["dest"] == f"/tmp/ogc_{s}" for s in SC), "clones")
    mt, mh = M["train"]["inria"]["members"], M["treehill"]["inria"]["members"]
    check(all(v["source"] == "present" and v["bytes"] == v["bytes_pinned"] and v["crc32"] == v["crc32_pinned"] for v in mt.values())
          and len(mt) == 3, "train members present, pins")
    check(all(v["source"] == "fetched" and v["bytes"] == v["bytes_pinned"] and v["crc32"] == v["crc32_pinned"] for v in mh.values())
          and len(mh) == 3, "treehill members fetched, pins")
    e3pm = json.load(open(os.path.join(REPO, "kaggle", "gn_e3p", "gn3p", "gn3p_meta_train.json")))
    check(mt["ply"]["sha1"] == e3pm["inria"]["members"]["ply"]["sha1"], "train ply = E3p's")
    want(f"`point_cloud.ply` SHA-1 `{mt['ply']['sha1'][:8]}`, E3p's", "train ply")
    want(f"`point_cloud.ply` SHA-1 `{mh['ply']['sha1'][:8]}` |", "treehill ply")
    ni = json.load(open(os.path.join(REPO, "kaggle", "gn_e4q_note_i", "header_read.json")))
    check(json.dumps(ni).count(mh["ply"]["crc32_pinned"]) >= 1, "treehill's pin is note i's")
    cfg = M["treehill"]["inria_cfg_args"]
    check(cfg["images"] == "images_4" and cfg["resolution"] == 1, "images_4, -r 1")
    ls = M["treehill"]["loaded_size_check"]
    check(ls["ok"] and ls["first_image_size"] == ls["loaded_size"] == ls["assumed"], "loaded size")
    want(f"first image {ls['first_image_size'][0]} x {ls['first_image_size'][1]}, loaded size {ls['loaded_size'][0]} x "
         f"{ls['loaded_size'][1]}, as note i assumed", "loaded size")
    cf = {s: M[s]["camera_frame_check"] for s in SC}
    check(all(c["pass"] for c in cf.values()), "camera frames")
    want(f"pass: {cf['train']['n_matched']} of {cf['train']['n_runner']}, largest differences {cf['train']['max_position_diff']:.2e} "
         f"(position), {cf['train']['max_rotation_diff']:.2e} (rotation) | pass: {cf['treehill']['n_matched']} of "
         f"{cf['treehill']['n_runner']}, {cf['treehill']['max_position_diff']:.2e}, {cf['treehill']['max_rotation_diff']:.2e} |", "frames")
    check(all(M[s]["split_check"]["equals_cameras_json_head"] for s in SC), "splits")
    want(f"| equal, {M['train']['split_check']['n_test']} test views | equal, {M['treehill']['split_check']['n_test']} test views |", "split")
    want(f"| {M['train']['n_even_views']} / {M['train']['n_odd_views']} | {M['treehill']['n_even_views']} / {M['treehill']['n_odd_views']} |", "views")
    e4pr = {r["config"]: r for r in csv.DictReader(open(os.path.join(E4P, "gn4p_results_train.csv"), newline=""))}
    uu = {s: R[s]["uncompressed"] for s in SC}
    check(all(uu["train"][k] == e4pr["uncompressed"][k] for k in ("PSNR_ii", "SSIM_ii", "LPIPS_ii")), "train uncompressed = E4p's")

    def ustr(u):
        res = json.loads(u["resolution_ii"])[0]
        return f"{f(u['PSNR_ii']):.3f} / {f(u['SSIM_ii']):.4f} / {f(u['LPIPS_ii']):.4f} at {res[0]}x{res[1]}"

    want(ustr(uu["train"]) + ", E4p's row in every digit the CSVs hold", "uncompressed train")
    want(ustr(uu["treehill"]) + " |", "uncompressed treehill")
    cnt = {}
    for s in SC:
        c = {(r["n_ckpt"], r["n_pruned"], r["n_kept_colour"], r["n_colour_quantized"]) for r in R[s].values()
             if r["kind"] == "fork" and r["j"] == "0"}
        check(len(c) == 1, f"{s} one set of counts")
        cnt[s] = [int(x) for x in c.pop()]
    check(cnt["train"] == [int(e4pr["p0_c3dgs"][k]) for k in ("n_ckpt", "n_pruned", "n_kept_colour", "n_colour_quantized")], "E4p's counts")
    want(f"| {', '.join(f'{x:,}' for x in cnt['train'])} (E4p's) | {', '.join(f'{x:,}' for x in cnt['treehill'])} |", "counts")
    fork_rows = [r for s in SC for r in R[s].values() if r["kind"] == "fork"]
    check(all(r["checks_ok"] == "True" for r in fork_rows), "all checks ok")
    want(f"(`checks_ok` in all {len(fork_rows)} fork rows)", "fork rows")
    for s in SC:
        for k, w in W[s].items():
            if "fork" not in w.get("e4p", {}):
                continue
            pre_ = k[3:5] + "_" if k.startswith("j0_") else k.split("_a")[0] + "_"
            rs = [r for r in R[s].values() if r["kind"] == "fork" and r["config"].startswith(pre_)]
            check(len(rs) >= 5 and len({r["geometry_sha1"] for r in rs}) == 1, f"{s} {k} one SHA-1")
    shas = {s: [row(s, "c3dgs", p)["geometry_sha1"][:8] for p in P] + [R[s]["probe"]["geometry_sha1"][:8]] for s in SC}
    check(all(len(set(v)) == 3 for v in shas.values()), "SHA-1s differ between processes")
    want(f"train `{shas['train'][0]}` and `{shas['train'][1]}` at the default point, the probe `{shas['train'][2]}`; treehill "
         f"`{shas['treehill'][0]}` and `{shas['treehill'][1]}`, the probe `{shas['treehill'][2]}`", "shas")

    # C3DGS's evaluation on the CPU
    msg = "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same"
    errs = [FK["treehill"][p]["rows"][r].get("c3dgs_eval_error", "") for p in P for r in FK["treehill"][p]["rows"]
            if "c3dgs_eval" in FK["treehill"][p]["rows"][r] or "c3dgs_eval_error" in FK["treehill"][p]["rows"][r]]
    evaluated = [r for p in P for r in FK["treehill"][p]["rows"] if f"eval_{r}" in FK["treehill"][p]["cost"]]
    check(len(errs) == len(evaluated) == 24 and all(e.startswith("RuntimeError") and msg in e for e in errs), "24 of 24 raised")
    want(f"{len(errs)} of {len(evaluated)} in the two processes ({len(evaluated) // 2} evaluated rows each; `ogc_lamcv` is", "24 of 24")
    ec = {f"{v['time_s']:.1f}" for p in P for k, v in FK["treehill"][p]["cost"].items() if k.startswith("eval_")}
    check(ec == {"0.1"}, "each 0.1 s")
    want(f"after {ec.pop()} s", "0.1 s")
    pe = W["treehill"]["probe_eval"]
    check(msg in pe["error"] and M["treehill"]["runs"]["probe_eval"]["ok"] is False, "probe eval raised")
    check(all(FK["train"][p]["rows"][r].get("c3dgs_eval") for p in P for r in FK["train"][p]["rows"] if f"eval_{r}" in FK["train"][p]["cost"])
          and all(r["c3dgs_PSNR"] for r in tr.values() if r["kind"] in ("fork", "probe")), "train has C3DGS's evaluation")
    check(all(not r["c3dgs_PSNR"] for r in th.values() if r["kind"] in ("fork", "probe")), "treehill none")
    want(f"of all {len([r for r in th.values() if r['kind'] == 'fork'])} treehill rows and of the probe", "26 rows")
    pb = th["probe"]
    check(pb["status"] == "failed" and pb["PSNR_ii"] != "", "probe failed with protocol ii")
    want(f"present\n  ({f(pb['PSNR_ii']):.3f} dB)".replace("\n  ", " "), "probe psnr")
    pre = re.sub(r"\s+", " ", open(PREREG, encoding="utf-8").read())
    check("C3DGS's evaluation, protocol ii and the fidelity renders scaled by test-view pixels" in pre
          and "without C3DGS's evaluation of the 12 rows after row 1" in pre, "Amendment 16 e counted the evaluation")
    want("of the 12 rows after row 1", "12 rows")

    # rho_cv and lam_cv
    for s in SC:
        rs = sorted(CV[s], key=lambda r: f(r["rho"]))
        sc = {RHO_LABEL[f(r["rho"])]: f(r["measured_odd_clamped"]) for r in rs}
        check(sc == ST[s]["cv_odd_scores"], f"{s} rho scores = summary's")
        best = min(sc, key=sc.get)
        check(RHO_LABEL[ST[s]["rho_cv"]] == best, f"{s} rho_cv")
        want(f"| {s} | " + " | ".join(f"**{sc[k]:.4e}**" if k == best else f"{sc[k]:.4e}" for k in RHO_LABEL.values()) + " |", f"rho {s}")
        second = sorted(sc, key=sc.get)[1]
        want(f"ahead of {second} by {(sc[second] - sc[best]) / sc[best] * 100:.2f}%", f"rho margin {s}")
        check(all(r["vq_iterations"] == "20" for r in rs), f"{s} CV at the cap")
    check(ST["train"]["rho_cv_equals_e4p"]["equal"] and ST["train"]["rho_cv"] == 0.01, "train = E4p's")
    want(f"train's `rho_cv` is {RHO_LABEL[ST['train']['rho_cv']]}, as E4p's", "train rho")
    want(f"treehill's is\n  {RHO_LABEL[ST['treehill']['rho_cv']]}".replace("\n  ", " "), "treehill rho")
    want(f"All {sum(len(CV[s]) for s in SC)} cross-validation runs", "14 runs")
    ot = {}
    for s in SC:
        ls_ = {LAM_LABEL[f(r["lam"])]: f(r["measured_odd_clamped"]) for r in LCV[s]}
        check(ls_ == ST[s]["lam_cv_scores"] and all(r["status"] == "ok" and r["codewords_used"] == "4096" for r in LCV[s]), f"{s} lam")
        best = min(ls_, key=ls_.get)
        check(best == "1e-3" and ST[s]["lam_cv"] == 1e-3, f"{s} lam_cv 1e-3")
        want(f"| {s} | " + " | ".join(f"**{ls_[k]:.4e}**" if k == best else f"{ls_[k]:.4e}" for k in LAM_LABEL.values()) + " |", f"lam {s}")
        second = sorted(ls_, key=ls_.get)[1]
        ot[s] = (second, (ls_[second] - ls_[best]) / ls_[best] * 100, ls_, best)
        order = sorted(ls_, key=ls_.get)
        check(order.index("1e-6") == (5 if s == "train" else 4), f"{s} 1e-6 worst / second worst")
    want(f"the runner-up is {ot['train'][1]:.2f}% above on train ({ot['train'][0]}) and {ot['treehill'][1]:.2f}% on treehill "
         f"({ot['treehill'][0]})", "lam margins")
    tt = {s: [f(r["ogc_time_s"]) for r in LCV[s]] for s in SC}
    want(f"each call took {min(tt['train']):.1f}-{max(tt['train']):.1f} s on train and {min(tt['treehill']):.1f}-"
         f"{max(tt['treehill']):.1f} s on treehill", "lam times")
    pc = {s: (ot[s][2]["1e-6"] - ot[s][2][ot[s][3]]) / ot[s][2][ot[s][3]] * 100 for s in SC}
    want(f"{pc['train']:.1f}% and {pc['treehill']:.1f}% above the\n  minimum".replace("\n  ", " "), "1e-6")
    aliases = [r for s in SC for r in R[s].values() if r["status"] == "alias"]
    check(all(r["row"] == "ogc_lamcv" and r["alias_of"] == "ogc" for r in aliases), "aliases are ogc_lamcv")
    want(f"({len(aliases)} alias rows)", "8 aliases")

    # iterations
    def it(s, r):
        return [FK[s][p]["rows"][r]["gn_vq"]["iterations"] for p in P], [FK[s][p]["rows"][r]["gn_vq"]["stopped_because"] for p in P]

    for s in SC:
        for r in ("gnvq_cv", "lad_clip_part", "lad_no_clip", "lad_ridge_mean", "lad_no_final_int8"):
            check(it(s, r) == ([20, 20], ["max_iters"] * 2), f"{s} {r} at the cap")
        for r in ("lad_iters15", "lad_all"):
            check(it(s, r) == ([15, 15], ["max_iters"] * 2), f"{s} {r} 15")
        for r in ("lad_reseed", "lad_init_tr", "lad_iters50"):
            check(it(s, r)[1] == ["rel_tol"] * 2, f"{s} {r} rel_tol")

    def rng_i(s, r):
        a = it(s, r)[0]
        return f"{min(a)}" if min(a) == max(a) else f"{min(a)}-{max(a)}"

    want(f"`lad_reseed` at {rng_i('train', 'lad_reseed')} (train) and\n{rng_i('treehill', 'lad_reseed')} (treehill), `lad_init_tr` at "
         f"{rng_i('train', 'lad_init_tr')} and {rng_i('treehill', 'lad_init_tr')}, and `lad_iters50` at {rng_i('train', 'lad_iters50')} "
         f"and {rng_i('treehill', 'lad_iters50')}".replace("\n", " "), "iterations")
    for s in SC:
        for p in P:
            for r in ("lad_reseed", "lad_all"):
                h = FK[s][p]["rows"][r]["gn_vq"]["history_last3"]
                check(int(row(s, r, p)["vq_reseeded_total"]) == sum(x.get("reseeded", 0) for x in h) and len(h) == 3, f"{s} {r} last three")

    # the rows
    for s in SC:
        for r in ROWS:
            v = [row(s, r, p) for p in P]
            ps = [f(x["PSNR_ii"]) for x in v]
            cells = [f"`{r}`", f"{sum(ps) / 2:.3f} ({min(ps):.3f}-{max(ps):.3f})", f"{sum(f(x['SSIM_ii']) for x in v) / 2:.4f}",
                     f"{sum(f(x['LPIPS_ii']) for x in v) / 2:.4f}"]
            if s == "train":
                cells.append(f"{sum(f(x['c3dgs_PSNR']) for x in v) / 2:.3f}")
            cells += [f"{sum(f(x['npz_bytes']) for x in v) / 2:,.0f}", " / ".join(f"{int(x['codebook_distinct']):,}" for x in v),
                      " / ".join(f"{f(x['index_entropy_bits']):.2f}" for x in v)]
            want("| " + " | ".join(cells) + " |", f"row {s} {r}")
            check(all(int(x["codebook_distinct"]) == int(x["distinct_indices"]) - int(x["n_kept_colour"]) for x in v), f"{s} {r} used")
    pt, ph = tr["probe"], th["probe"]
    want(f"reads {f(pt['PSNR_ii']):.3f} / {f(pt['SSIM_ii']):.4f} / {f(pt['LPIPS_ii']):.4f} at {int(pt['npz_bytes']):,} bytes on train\n"
         f"(C3DGS's evaluation {f(pt['c3dgs_PSNR']):.3f}), and {f(ph['PSNR_ii']):.3f} / {f(ph['SSIM_ii']):.4f} / {f(ph['LPIPS_ii']):.4f} "
         f"at {int(ph['npz_bytes']):,} bytes on treehill".replace("\n", " "), "probes")
    spr = {s: abs(dsp(s, "c3dgs", "c3dgs")[0] + f(row(s, "c3dgs", 0)["PSNR_ii"]) - f(row(s, "c3dgs", 1)["PSNR_ii"])) for s in SC}
    want(f"by {spr['train']:.4f} dB (train) and {spr['treehill']:.4f} dB (treehill)", "c3dgs spread")
    used = {s: {r: [int(row(s, r, p)["codebook_distinct"]) for p in P] for r in ROWS} for s in SC}
    em = {s: [4096 - x for x in used[s]["c3dgs"]] for s in SC}
    want(f"leaves {min(em['train']):,}-{max(em['train']):,} of its 4,096 entries without a splat on train and\n"
         f"{min(em['treehill']):,}-{max(em['treehill']):,} on treehill".replace("\n", " "), "empty")
    check(all(used[s][r] == [4096, 4096] for s in SC for r in ("lad_reseed", "lad_all", "ogc")), "reseeding rows use all")
    check(all(used[s][r][0] < 4096 for s in SC for r in ROWS if r not in ("lad_reseed", "lad_all", "ogc")), "only they")
    want(f"uses {used['train']['lad_init_tr'][0]:,} and {used['treehill']['lad_init_tr'][0]:,}", "init_tr used")
    oth = [x for s in SC for r in ["gnvq_cv"] + LAD for x in used[s][r] if r not in ("lad_reseed", "lad_init_tr")]
    want(f"Every other GN-VQ row stays within {min(oth):,}-{max(oth):,}", "others used")
    for s in SC:
        mb = {r: sum(f(row(s, r, p)["npz_bytes"]) for p in P) for r in ROWS}
        check(set(sorted(mb, key=mb.get)[-4:]) == {"lad_reseed", "lad_init_tr", "lad_all", "ogc"}, f"{s} four largest")
        me = {r: sum(f(row(s, r, p)["index_entropy_bits"]) for p in P) for r in ROWS}
        check(set(sorted(me, key=me.get)[-4:]) == {"lad_reseed", "lad_init_tr", "lad_all", "ogc"}, f"{s} entropy order")

    # the ladder
    def dline(s, a, b, k="PSNR_ii"):
        v = dsp(s, a, b, k)
        d, _, _, se = comp(v)
        return v, d, se

    sums = {}
    for s in SC:
        sums[s] = sum(dline(s, r, "gnvq_cv")[1] for r in LAD)
    for r in LAD + ["lad_all"]:
        cells = [f"`{r}`"]
        for s in SC:
            v, d, se = dline(s, r, "gnvq_cv")
            b = comp(dsp(s, r, "gnvq_cv", "npz_bytes"))[0]
            key = f"{r}_minus_gnvq_cv"
            check(abs(ST[s]["differences"][key]["PSNR_ii"]["D_bar"] - d) < 1e-12 and abs(ST[s]["differences"][key]["PSNR_ii"]["SE_noise"] - se) < 1e-12,
                  f"{s} {key} = summary's")
            cells += [" / ".join(sg(x) for x in v), sg(d), f"{se:.4f}", f"{b:+,.0f}"]
        want("| " + " | ".join(cells) + " |", f"ladder {r}")
    OT = [("`lad_all` - `ogc`", "lad_all", "ogc"), ("`gnvq_cv` - `c3dgs`", "gnvq_cv", "c3dgs"), ("`ogc` - `c3dgs`", "ogc", "c3dgs"),
          ("`gnvq_cv` - `ogc`", "gnvq_cv", "ogc")]
    for lab, a, b in OT:
        cells = [lab]
        for s in SC:
            v, d, se = dline(s, a, b)
            cells += [" / ".join(sg(x) for x in v), sg(d), f"{se:.4f}", f"{comp(dsp(s, a, b, 'npz_bytes'))[0]:+,.0f}"]
            if s == "train":
                cells.append(sg(comp(dsp(s, a, b, "c3dgs_PSNR"))[0]))
        want("| " + " | ".join(cells) + " |", f"other {lab}")
    for s in SC:
        dd = ST[s]["differences"]
        check(dd["gnvq_cv_minus_ogc_lamcv"] == dd["gnvq_cv_minus_ogc"] or
              all(dd["gnvq_cv_minus_ogc_lamcv"][k]["D_bar"] == dd["gnvq_cv_minus_ogc"][k]["D_bar"] for k in dd["gnvq_cv_minus_ogc"]), "alias equal")
        check(all(v["D_bar"] == 0 for v in dd["ogc_lamcv_minus_ogc"].values() if isinstance(v, dict) and "D_bar" in v), "zero")
    L = {s: {r: dline(s, r, "gnvq_cv")[1] for r in LAD + ["lad_all"]} for s in SC}
    want(f"reseeding empty clusters (`lad_reseed`, {sg(L['train']['lad_reseed'])} dB on\n  train) and OGC's trace-weighted draw as the "
         f"start (`lad_init_tr`, {sg(L['treehill']['lad_init_tr'])} dB on treehill)".replace("\n  ", " "), "two largest")
    want(f"on treehill `lad_reseed` alone is {sg(L['treehill']['lad_reseed'])} dB", "reseed treehill")
    check(max(abs(L[s][r]) for s in SC for r in ("lad_clip_part", "lad_no_clip")) == abs(L["train"]["lad_clip_part"]), "clip max")
    want(f"(`lad_ridge_mean`) {sg(L['train']['lad_ridge_mean'])} and {sg(L['treehill']['lad_ridge_mean'])}\n  dB, the per-part clip and "
         f"no clip within {abs(L['train']['lad_clip_part']):.4f} dB, the iteration counts within "
         f"{max(abs(L[s][r]) for s in SC for r in ('lad_iters15', 'lad_iters50')):.4f} dB, the final int8 assignment\n  within "
         f"{max(abs(L[s]['lad_no_final_int8']) for s in SC):.4f} dB".replace("\n  ", " "), "regularization")
    want(f"Their sum is {sg(sums['train'])} dB on train against {sg(L['train']['lad_all'])} for `lad_all`,\n  and "
         f"{sg(sums['treehill'])} dB on treehill against {sg(L['treehill']['lad_all'])}".replace("\n  ", " "), "sums")
    for s in SC:
        bl = ST[s]["best_ladder"]
        means = {r: sum(f(row(s, r, p)["PSNR_ii"]) for p in P) / 2 for r in LAD}
        best = max(means, key=lambda r: round(means[r], 9))
        check(bl["row"] == best and all(abs(bl["means_psnr_ii"][r] - means[r]) < 1e-12 for r in LAD), f"{s} best ladder")
    bt, bh = ST["train"]["best_ladder"], ST["treehill"]["best_ladder"]
    want(f"is `{bt['row']}` on train ({bt['means_psnr_ii'][bt['row']]:.6f} dB) and `{bh['row']}` on treehill "
         f"({bh['means_psnr_ii'][bh['row']]:.6f} dB)", "best")
    check(set(S["scenes"]["train"]["sweep"]) >= {"lad_reseed"}, "the sweep used lad_reseed")
    ao = {s: ST[s]["lad_all_vs_ogc"] for s in SC}
    check(all(ao[s]["j0_p0"] == ao[s]["j0_p1"] for s in SC), "agreement identical across processes")
    a0 = {s: ao[s]["j0_p0"] for s in SC}
    check(all(a0[s]["used_a"] == a0[s]["used_b"] == a0[s]["used_both"] == 4096 for s in SC), "both use all")
    want(f"The difference is {sg(dline('train', 'lad_all', 'ogc')[1])} dB (train) and {sg(dline('treehill', 'lad_all', 'ogc')[1])} dB "
         f"(treehill),** with {comp(dsp('train', 'lad_all', 'ogc', 'npz_bytes'))[0]:,.0f} and "
         f"{comp(dsp('treehill', 'lad_all', 'ogc', 'npz_bytes'))[0]:+,.0f} bytes, against the {sg(L['train']['lad_all'])} and\n  "
         f"{sg(L['treehill']['lad_all'])} dB `lad_all` gains".replace("\n  ", " "), "lad_all - ogc")
    want(f"The labels agree on {a0['train']['n_equal']:,} of {a0['train']['n']:,} splats ({a0['train']['fraction_equal']:.3f}) and "
         f"{a0['treehill']['n_equal']:,} of {a0['treehill']['n']:,} ({a0['treehill']['fraction_equal']:.3f}).", "agreement")
    want(f"largest absolute difference is {a0['train']['codebook_max_abs_diff']:.4f} and {a0['treehill']['codebook_max_abs_diff']:.4f}", "cb diff")
    tests = open(TESTS, encoding="utf-8").read()
    check("def test_e4q_lad_all_with_ogcs_arithmetic_reproduces_their_gram_kmeans" in tests
          and 'res["reseeded_total"] == 4' in tests and "_e4q_case(n=400, K=24, seed=2)" in tests, "the test: 400 splats, 4 reseedings")
    want("on 400 splats with 4\n  reseedings".replace("\n  ", " "), "test case")
    q = {s: [W[s][f"j0_p{p}_a0"]["e4p"]["qa_at_colour_vq"] for p in P] for s in SC}
    check(all(q[s][0] == q[s][1] for s in SC), "same quantizer in both processes")
    check(all(FK[s][0]["n_colour_quantized"] == FK[s][1]["n_colour_quantized"] for s in SC), "same counts")
    qt = q["train"][0]
    want(f"(DC scale {qt['dc_scale']:.6f}, zero point\n  {qt['dc_zero_point']}, AC scale {qt['rest_scale']:.5g}, zero point "
         f"{qt['rest_zero_point']}, on train)".replace("\n  ", " "), "quantizer")
    for s in SC:
        tre = {r: [json.dumps(FK[s][p]["rows"][r]["table_range"], sort_keys=True) for p in P] for r in ROWS}
        check(all((tre[r][0] == tre[r][1]) == (r in ("lad_all", "ogc")) for r in ROWS), f"{s} table ranges")
        check(all(FK[s][p]["rows"]["ogc"]["ogc"]["call"]["seed"] == 0 for p in P), f"{s} ogc seed 0")
        check(row(s, "c3dgs", 0)["geometry_sha1"] != row(s, "c3dgs", 1)["geometry_sha1"], f"{s} geometry differs")
    want(f"(-0.0013 and\n  {sg(dsp('train', 'lad_all', 'ogc')[1])} dB on train)".replace("\n  ", " "), "ogc p1")
    check(f"{dsp('train', 'lad_all', 'ogc')[0]:+.4f}" == "-0.0013", "p0 -0.0013")

    # the sweep and BD
    sw_ = S["scenes"]["train"]["sweep"]
    for i in range(5):
        j = sw_["c3dgs"][i]["j"]
        check(abs(sw_["c3dgs"][i]["threshold"] - 6e-7 * 3 ** j) < 1e-15, "threshold rule")
        js = "0" if j == 0 else f"{j:+d}"
        want(f"| {js} | {sw_['c3dgs'][i]['threshold']:.3g} | " + " | ".join(
            f"{int(sw_[r][i]['npz_bytes']):,} / {sw_[r][i]['PSNR_ii']:.4f}" for r in ("c3dgs", "gnvq_cv", "lad_reseed", "ogc")) + " |", f"sweep {j}")
        if j == 0:
            check(all(sw_[r][i]["npz_bytes"] == f(row("train", r, 0)["npz_bytes"]) for r in ("c3dgs", "gnvq_cv", "lad_reseed", "ogc")), "j = 0 is p0")
    bd = S["scenes"]["train"]["bd"]
    C = {r: (np.array([p["npz_bytes"] for p in v]), np.array([p["PSNR_ii"] for p in v])) for r, v in sw_.items()}
    for a, b in (("ogc", "gnvq_cv"), ("lad_reseed", "gnvq_cv"), ("c3dgs", "gnvq_cv"), ("ogc", "c3dgs")):
        r_ = bd[f"{a}_vs_{b}"]
        (ba, pa), (bb, pb_) = C[a], C[b]
        check(abs(g2.bd_rate_scaled(bb, pb_, ba, pa) - r_["bd_rate_percent"]) < 1e-9 and abs(g2.bd_psnr_scaled(bb, pb_, ba, pa) - r_["bd_psnr_db"]) < 1e-12,
              f"BD {a} {b} recomputed")
        want(f"| `{a}` against `{b}` | {r_['bd_rate_percent']:+.2f}% | {r_['bd_psnr_db']:+.4f} |", f"bd {a} {b}")
    check(bd["ogc_lamcv_vs_gnvq_cv"] == bd["ogc_vs_gnvq_cv"] and bd["ogc_lamcv_vs_c3dgs"] == bd["ogc_vs_c3dgs"]
          and bd["ogc_lamcv_vs_ogc"]["bd_rate_percent"] == 0 and bd["ogc_lamcv_vs_ogc"]["bd_psnr_db"] == 0, "alias BD")
    want(f"OGC reaches `gnvq_cv`'s PSNR with {-bd['ogc_vs_gnvq_cv']['bd_rate_percent']:.2f}% fewer bytes, and\n  `lad_reseed` with "
         f"{-bd['lad_reseed_vs_gnvq_cv']['bd_rate_percent']:.2f}% fewer. GN-VQ still beats C3DGS's own VQ (C3DGS needs "
         f"{bd['c3dgs_vs_gnvq_cv']['bd_rate_percent']:.2f}% more bytes)".replace("\n  ", " "), "equal bytes")
    spans = [C[r][1].max() - C[r][1].min() for r in ("c3dgs", "gnvq_cv", "lad_reseed", "ogc")]
    ratios = [C[r][0].max() / C[r][0].min() for r in ("c3dgs", "gnvq_cv", "lad_reseed", "ogc")]
    want(f"each spans {min(spans):.3f}-{max(spans):.3f} dB in PSNR over a {min(ratios):.1f}-{max(ratios):.1f}x range of bytes", "spans")

    def pl_rate(bb, pb_, ba, pa):
        lo, hi = max(pa.min(), pb_.min()), min(pa.max(), pb_.max())
        x = np.linspace(lo, hi, 20001)
        fa = np.interp(x, np.sort(pa), np.log10(ba)[np.argsort(pa)])
        fb = np.interp(x, np.sort(pb_), np.log10(bb)[np.argsort(pb_)])
        return (10 ** (np.trapezoid(fa - fb, x) / (hi - lo)) - 1) * 100

    rng_w = {}
    for a, b in (("ogc", "gnvq_cv"), ("lad_reseed", "gnvq_cv"), ("ogc", "c3dgs"), ("c3dgs", "gnvq_cv")):
        (ba, pa), (bb, pb_) = C[a], C[b]
        plo, phi = max(pa.min(), pb_.min()), min(pa.max(), pb_.max())
        rlo, rhi = max(ba.min(), bb.min()), min(ba.max(), bb.max())
        ins = lambda p: int(((p >= plo - 1e-12) & (p <= phi + 1e-12)).sum())  # noqa: E731
        shr = lambda p: (phi - plo) / (p.max() - p.min()) * 100  # noqa: E731
        shb = lambda bx: math.log(rhi / rlo) / math.log(bx.max() / bx.min()) * 100  # noqa: E731
        degs = [g2.bd_rate_scaled(bb, pb_, ba, pa, degree=d) for d in (3, 2, 1)]
        loo = [(g2.bd_rate_scaled(bb[m], pb_[m], ba[m], pa[m]), g2.bd_psnr_scaled(bb[m], pb_[m], ba[m], pa[m]))
               for m in (np.arange(5) != i for i in range(5))]
        plr = pl_rate(bb, pb_, ba, pa)
        rr = [x for x, _ in loo]
        ps_ = [y for _, y in loo]
        rng_w[(a, b)] = max(rr) - min(rr)
        want(f"| `{a}` against `{b}` | {phi - plo:.3f} dB | {shr(pa):.0f}% / {shr(pb_):.0f}% | {ins(pa)} / {ins(pb_)} | "
             f"{shb(ba):.0f}% / {shb(bb):.0f}% | {' / '.join(f'{d:+.2f}' for d in degs)} / {plr:+.2f}% | {min(rr):+.1f} to {max(rr):+.1f}% | "
             f"{min(ps_):+.4f} to {max(ps_):+.4f} |", f"overlap {a} {b}")
        check(all(np.sign(x) == np.sign(degs[0]) for x in degs + [plr] + rr) and all(np.sign(y) == np.sign(bd.get(f"{a}_vs_{b}", {"bd_psnr_db": degs[0] * -1 if a == 'c3dgs' else 1})["bd_psnr_db"]) for y in ps_),
              f"{a} {b} signs hold")
        for nm, (bx, px) in ((a, (ba, pa)), (b, (bb, pb_))):
            Pf = np.polynomial.Polynomial.fit(px, np.log10(bx), 3)
            xx = np.linspace(plo, phi, 2001)
            Qf = np.polynomial.Polynomial.fit(np.log10(bx), px, 3)
            yy = np.linspace(math.log10(rlo), math.log10(rhi), 2001)
            check((np.diff(Pf(xx)) > 0).all() and (np.diff(Qf(yy)) > 0).all(), f"{nm} monotone in the overlap with {a}/{b}")
    others = [v for k, v in rng_w.items() if k != ("c3dgs", "gnvq_cv")]
    want(f"BD-rate moves by {min(others):.1f}-{max(others):.1f} percentage points with one point, and by "
         f"{rng_w[('c3dgs', 'gnvq_cv')]:.1f} for `c3dgs` against `gnvq_cv`", "drop-one widths")
    pd = {r: abs(f(row("train", r, 0)["PSNR_ii"]) - f(row("train", r, 1)["PSNR_ii"])) for r in ("c3dgs", "gnvq_cv")}
    want(f"differ by {pd['c3dgs']:.4f} dB (`c3dgs`) and {pd['gnvq_cv']:.4f} dB (`gnvq_cv`)", "j0 spread")
    for b in ("ogc", "gnvq_cv"):
        (bb, pb_), (bc, pc_) = C[b], C["c3dgs"]
        plo, phi = max(pc_.min(), pb_.min()), min(pc_.max(), pb_.max())
        check(int(((pc_ >= plo - 1e-12) & (pc_ <= phi + 1e-12)).sum()) == 2, f"c3dgs 2 points against {b}")

    # fidelity
    for s in SC:
        fp = ST[s]["fidelity_per_angle"]
        for a in ANG:
            al = a if a == "0" or a.startswith("-") else "+" + a
            cells = []
            for d in DIFFS:
                x, y = d.split("_minus_")
                mv = [json.loads(row(s, x, p)["fidelity_psnr"])[a] - json.loads(row(s, y, p)["fidelity_psnr"])[a] for p in P]
                pv = [json.loads(row(s, x, p)["fidelity_pooled_psnr"])[a] - json.loads(row(s, y, p)["fidelity_pooled_psnr"])[a] for p in P]
                check(abs(fp[d]["fidelity_psnr"][a]["D_bar"] - sum(mv) / 2) < 1e-12
                      and abs(fp[d]["fidelity_pooled_psnr"][a]["D_bar"] - sum(pv) / 2) < 1e-12, f"{s} {d} {a} = summary's")
                cells.append(f"{sum(mv) / 2:+.3f} / {sum(pv) / 2:+.3f}")
            want(f"| {al} | " + " | ".join(cells) + " |", f"fidelity {s} {a}")
        want(f"**{s}** ({len(M[s]['split_check']['test_names'])} cameras)", f"{s} cameras")
    fh = ST["treehill"]["fidelity_per_angle"]
    flat = LAD + ["lad_all"]
    flat_keys = [f"{r}_minus_gnvq_cv" for r in flat] + ["lad_all_minus_ogc", "gnvq_cv_minus_ogc"]
    rngs = [max(fh[k][m][a]["D_bar"] for a in ANG) - min(fh[k][m][a]["D_bar"] for a in ANG)
            for k in flat_keys for m in ("fidelity_psnr", "fidelity_pooled_psnr")]
    want(f"(11 in all) moves by {max(rngs):.3f} dB or less", "treehill flat")
    check(len(flat_keys) == 11, "11 differences")
    cr = [max(fh[k][m][a]["D_bar"] for a in ANG) - min(fh[k][m][a]["D_bar"] for a in ANG)
          for k in ("gnvq_cv_minus_c3dgs", "ogc_minus_c3dgs") for m in ("fidelity_psnr", "fidelity_pooled_psnr")]
    want(f"The differences against `c3dgs` move by {min(cr):.3f}-{max(cr):.3f} dB, lowest at 0 degrees", "treehill c3dgs")
    check(all(min(ANG, key=lambda a: fh[k][m][a]["D_bar"]) == "0" for k in ("gnvq_cv_minus_c3dgs", "ogc_minus_c3dgs")
              for m in ("fidelity_psnr", "fidelity_pooled_psnr")), "lowest at 0")
    seh = max(fh[k][m][a]["SE_noise"] for k in fh for m in ("fidelity_psnr", "fidelity_pooled_psnr") for a in ANG)
    want(f"`SE_noise` is at most {seh:.3f} dB in either measure", "treehill SE")
    ft = ST["train"]["fidelity_per_angle"]
    check(max(ANG, key=lambda a: ft["ogc_minus_c3dgs"]["fidelity_psnr"][a]["D_bar"]) == "0"
          and max(ANG, key=lambda a: ft["gnvq_cv_minus_c3dgs"]["fidelity_psnr"][a]["D_bar"]) == "0", "train peaks at 0")
    want(f"(`ogc` {ft['ogc_minus_c3dgs']['fidelity_psnr']['0']['D_bar']:+.3f} dB, `gnvq_cv` "
         f"{ft['gnvq_cv_minus_c3dgs']['fidelity_psnr']['0']['D_bar']:+.3f} dB) and\n  falls toward +40 degrees "
         f"({ft['ogc_minus_c3dgs']['fidelity_psnr']['40']['D_bar']:+.3f} and {ft['gnvq_cv_minus_c3dgs']['fidelity_psnr']['40']['D_bar']:+.3f} dB)"
         .replace("\n  ", " "), "train peak")
    check(all(ft[k]["fidelity_psnr"]["40"]["D_bar"] < ft[k]["fidelity_psnr"]["-40"]["D_bar"] for k in ("ogc_minus_c3dgs", "gnvq_cv_minus_c3dgs")),
          "steeper on the positive side")
    sep = max(ft[k]["fidelity_pooled_psnr"][a]["SE_noise"] for k in ft for a in ANG)
    sem = max(ft[k]["fidelity_psnr"][a]["SE_noise"] for k in ft for a in ANG)
    want(f"its `SE_noise` reaches {sep:.3f} dB ({sem:.3f} dB for the mean PSNR)", "train SE")
    check(all(ft[k]["fidelity_pooled_psnr"][a]["D_bar"] < 0 for k in ("ogc_minus_c3dgs", "gnvq_cv_minus_c3dgs") for a in ("20", "40")), "negative")
    names = M["train"]["split_check"]["test_names"]
    meas = {p: M["train"]["processes"][f"j0_p{p}"]["measured"]["0"] for p in P}
    i = names.index("00113")
    va = [meas[p]["lad_all"]["fidelity"]["per_view"]["-20"] for p in P]
    vb = [meas[p]["ogc"]["fidelity"]["per_view"]["-20"] for p in P]
    check(all(max(range(38), key=lambda j: abs(va[p][j] - vb[p][j])) == i for p in P), "00113 is the largest difference")
    want(f"At camera `{names[i]}` there", "00113")
    want(f"`lad_all` - `ogc` reads {ft['lad_all_minus_ogc']['fidelity_pooled_psnr']['-20']['D_bar']:+.3f} dB pooled at -20 degrees "
         f"against\n  {ft['lad_all_minus_ogc']['fidelity_psnr']['-20']['D_bar']:+.3f} dB mean".replace("\n  ", " "), "-20 effect")
    want(f"`lad_all`'s render reads {va[0][i]:.2f} and {va[1][i]:.2f} dB against `ogc`'s {vb[0][i]:.2f} and {vb[1][i]:.2f}\n  dB "
         f"in processes 0 and 1; without that camera the pooled difference is "
         f"{pooled([v for j, v in enumerate(va[0]) if j != i]) - pooled([v for j, v in enumerate(vb[0]) if j != i]):+.3f} and "
         f"{pooled([v for j, v in enumerate(va[1]) if j != i]) - pooled([v for j, v in enumerate(vb[1]) if j != i]):+.3f} dB"
         .replace("\n  ", " "), "00113")
    check(all(abs((pooled(va[p]) - pooled(vb[p])) - json.loads(row("train", "lad_all", p)["fidelity_pooled_psnr"])["-20"]
                  + json.loads(row("train", "ogc", p)["fidelity_pooled_psnr"])["-20"]) < 1e-9 for p in P), "pooled formula")
    lr = [fh["lad_reseed_minus_gnvq_cv"]["fidelity_psnr"][a]["D_bar"] for a in ANG]
    want(f"{min(lr):+.3f} to {max(lr):+.3f} dB closer to the\n  uncompressed model than `gnvq_cv`, but "
         f"{sg(L['treehill']['lad_reseed'])} dB in protocol ii".replace("\n  ", " "), "reseed references")

    # the chunk check
    cc = ST["train"]["chunk_check"]
    check(cc["labels_equal"] and cc["n_labels_differ"] == 0 and cc["codebook_max_abs_diff"] == 0.0 and cc["chunks"] == [25000, 100000]
          and cc["process"] == "j0_p0", "chunk check")
    want(f"chunk 25,000 in all {FK['train'][0]['n_colour_quantized']:,} splats", "chunk splats")
    want(f"It took {FK['train'][0]['cost']['ogc_chunk_check']['time_s']:.1f} s\nagainst {FK['train'][0]['cost']['ogc']['time_s']:.1f} s"
         .replace("\n", " "), "chunk time")

    # time and memory
    setup = tim["restore_s"] + tim["install_s"] + tim["c3dgs_build_s"]
    jt, jh = ST["train"]["timings_s"]["job"], ST["treehill"]["timings_s"]["job"]
    want(f"restore {tim['restore_s']:.1f} s, install {tim['install_s']:.1f} s, C3DGS build {tim['c3dgs_build_s']:.1f} s", "setup")
    want(f"{jt:,.1f} s by its own clock ({tim['gn_e4q_train_s']:,.1f} s in\nthe queue), {jt / 3600:.1f} h".replace("\n", " "), "train job")
    want(f"{jh:,.1f} s ({tim['gn_e4q_treehill_s']:,.1f} s in the queue), {jh / 3600:.1f} h", "treehill job")
    sess = setup + max(tim["gn_e4q_train_s"], tim["gn_e4q_treehill_s"])
    want(f"setup\nplus the longer job, {sess:,.1f} s ({sess / 3600:.2f} h)".replace("\n", " "), "session")
    steps = {s: {x["name"]: x for x in ST[s]["steps"]} for s in SC}
    one = lambda s, n: steps[s][n]["time_s"]  # noqa: E731
    want(f"| {one('train', 'fetch_inria'):.1f} / {one('train', 'download_dataset'):.1f} | - | - | {one('treehill', 'fetch_inria'):.1f} / "
         f"{one('treehill', 'download_dataset'):.1f} |", "fetch")

    def gpu(s, k):
        e = W[s][k]["e4p"]
        return f"{gb(e['cuda_peak_allocated_process'])} / {gb(e['cuda_peak_reserved_process'])}"

    rssw = lambda s, k: gb(W[s][k]["host_rss"]["rss_peak_bytes"])  # noqa: E731
    want(f"| probe run (C3DGS's process) | {one('train', 'c3dgs_probe'):.1f} | {gpu('train', 'probe')} | {rssw('train', 'probe')} | "
         f"{one('treehill', 'c3dgs_probe'):.1f} | {gpu('treehill', 'probe')} | {rssw('treehill', 'probe')} |", "probe step")
    for p in P:
        want(f"| default process {p} | {one('train', f'c3dgs_j0_p{p}_a0'):,.1f} | {gpu('train', f'j0_p{p}_a0')} | {rssw('train', f'j0_p{p}_a0')} | "
             f"{one('treehill', f'c3dgs_j0_p{p}_a0'):,.1f} | {gpu('treehill', f'j0_p{p}_a0')} | {rssw('treehill', f'j0_p{p}_a0')} |", f"process {p}")
    sk = ["j-2_p0_a0", "j-1_p0_a0", "j+1_p0_a0", "j+2_p0_a0"]
    sa = [W["train"][k]["e4p"]["cuda_peak_allocated_process"] for k in sk]
    sr = [W["train"][k]["e4p"]["cuda_peak_reserved_process"] for k in sk]
    sh = [W["train"][k]["host_rss"]["rss_peak_bytes"] for k in sk]
    want(f"| sweep processes, j = -2 / -1 / +1 / +2 | {' / '.join(f'{one('train', 'c3dgs_' + k):.1f}' for k in sk)} | "
         f"{gb(min(sa))}-{gb(max(sa))} / {gb(min(sr))}-{gb(max(sr))} | {gb(min(sh))}-{gb(max(sh))} |", "sweep processes")
    want(f"| GN passes 16 x 16, all / even views | {one('train', 'gn_pass16_full'):.1f} / {one('train', 'gn_pass16_even'):.1f} | - | - | "
         f"{one('treehill', 'gn_pass16_full'):.1f} / {one('treehill', 'gn_pass16_even'):.1f} |", "gn passes")
    tot = lambda s, pre: sum(x["time_s"] for n, x in steps[s].items() if n.startswith(pre))  # noqa: E731
    check(all(len([n for n in steps[s] if n.startswith("gn_vq_cv_")]) == 7 and len([n for n in steps[s] if n.startswith("ogc_cv_")]) == 6
              for s in SC), "7 and 6 CV steps")
    want(f"| `rho` CV: GN-VQ, 7 runs | {tot('train', 'gn_vq_cv_'):,.1f} in all | - | - | {tot('treehill', 'gn_vq_cv_'):,.1f} in all |", "rho cv")
    want(f"| `lam` CV: OGC, 6 runs | {tot('train', 'ogc_cv_'):,.1f} in all | - | - | {tot('treehill', 'ogc_cv_'):,.1f} in all |", "lam cv")

    def per(s):
        n2 = [x["time_s"] for n, x in steps[s].items() if n.startswith("npz2ply_")]
        e2 = [x["time_s"] for n, x in steps[s].items() if n.startswith("eval_ii_") and n != "eval_ii_uncompressed"]
        fi = [x["time_s"] for n, x in steps[s].items() if n.startswith("fidelity_")]
        return f"{min(n2):.1f}-{max(n2):.1f} / {min(e2):.1f}-{max(e2):.1f} / {min(fi):.1f}-{max(fi):.1f}"

    want(f"| per decoded row: `npz2ply.py` / protocol ii / fidelity renders | {per('train')} | - | - | {per('treehill')} |", "per row")
    cost = {s: {p: FK[s][p]["cost"] for p in P} for s in SC}

    def cpair(s, k):
        return " / ".join(f"{cost[s][p][k]['time_s']:.1f}" for p in P)

    lt = {s: [cost[s][p][r]["time_s"] for p in P for r in LAD] for s in SC}
    ev = [v["time_s"] for p in P for k, v in cost["train"][p].items() if k.startswith("eval_")]
    rest = {s: [W[s][f"j0_p{p}_a0"]["wall_s"] - sum(v["time_s"] for v in cost[s][p].values()) for p in P] for s in SC}
    check(max(LAD, key=lambda r: cost["train"][0][r]["time_s"]) == "lad_iters50", "iters50 the longest")
    want(f"C3DGS's own VQ {cpair('train', 'c3dgs')}; `gnvq_cv` {cpair('train', 'gnvq_cv')}; the eight ladder rows "
         f"{min(lt['train']):.1f}-{max(lt['train']):.1f} (`lad_iters50` the longest),\n  `lad_all` {cpair('train', 'lad_all')}; `ogc` "
         f"{cpair('train', 'ogc')}; C3DGS's evaluation of the 12 rows {min(ev):.1f}-{max(ev):.1f} each; the rest, "
         f"{' / '.join(f'{x:.1f}' for x in rest['train'])}".replace("\n  ", " "), "inside train")
    check(len([k for k in cost["train"][0] if k.startswith("eval_")]) == 12, "12 evaluations")
    want(f"C3DGS's own VQ {cpair('treehill', 'c3dgs')}; `gnvq_cv` {cpair('treehill', 'gnvq_cv')}; the eight ladder rows "
         f"{min(lt['treehill']):.1f}-{max(lt['treehill']):.1f} (`lad_iters50`\n  {cpair('treehill', 'lad_iters50')}), `lad_all` "
         f"{cpair('treehill', 'lad_all')}; `ogc` {cpair('treehill', 'ogc')}; C3DGS's evaluation 0.1 (it failed at once); the rest, "
         f"{' / '.join(f'{x:.1f}' for x in rest['treehill'])}".replace("\n  ", " "), "inside treehill")
    # reserved
    for s in SC:
        for k, w in W[s].items():
            e = w["e4p"]
            check(e["cuda_peak_reserved_process"] >= e["cuda_peak_allocated_process"], f"{s} {k} reserved >= allocated")
        for r in R[s].values():
            if r["row_cuda_peak_reserved"]:
                check(f(r["row_cuda_peak_reserved"]) >= f(r["row_cuda_peak_allocated"]), f"{s} {r['config']} row reserved")
                check(f(r["process_peak_reserved"]) >= f(r["row_cuda_peak_reserved"]), f"{s} {r['config']} process >= row")
        for x in ST[s]["steps"]:
            if x.get("cuda_peak_reserved") is not None:
                check(x["cuda_peak_reserved"] >= x["cuda_peak_allocated"], f"{s} step {x['name']}")
    pk = W["train"]["j0_p0_a0"]["e4p"]
    want(f"The largest GPU peak is train's process 0, {gb(pk['cuda_peak_allocated_process'])} GB allocated and "
         f"{gb(pk['cuda_peak_reserved_process'])} GB reserved", "largest")
    check(all(W["train"][k]["e4p"]["cuda_peak_allocated_process"] < pk["cuda_peak_allocated_process"] for k in W["train"] if k != "j0_p0_a0")
          and all(W[s][k]["e4p"]["cuda_peak_allocated_process"] <= pk["cuda_peak_allocated_process"] for s in SC for k in W[s]), "the largest")
    want(f"a train process peaks at {gb(max(W['train'][k]['e4p']['cuda_peak_allocated_process'] for k in W['train'] if k != 'j0_p0_a0'))} GB",
         "without the check")
    ph_ = W["treehill"]["j0_p0_a0"]["e4p"]
    check(all(W["treehill"][k]["e4p"]["cuda_peak_allocated_process"] <= ph_["cuda_peak_allocated_process"] for k in W["treehill"]), "treehill p0 largest")
    want(f"treehill with the images on the CPU peaked at {gb(max(W['treehill'][k]['e4p']['cuda_peak_allocated_process'] for k in W['treehill']))} "
         f"GB allocated, {gb(max(W['treehill'][k]['e4p']['cuda_peak_reserved_process'] for k in W['treehill']))} GB reserved", "treehill peak")
    feas = json.load(open(os.path.join(REPO, "kaggle", "gn_e4q_note_i", "feas.json")))
    fr0 = feas["rows"][0]
    want(f"(7.47-9.64 GB) and below its {fr0['cpu_colour_reserved']:.2f} GB upper end", "note i")
    check(f"{fr0['cpu_lo']:.2f}" == "7.47" and f"{fr0['cpu_hi']:.2f}" == "9.64", "note i CPU range")
    ram = ST["train"]["session_ram"]
    want(f"the session had {gb(ram['total_bytes'])} GB", "ram")
    hw = [W["treehill"][f"j0_p{p}_a0"]["host_rss"]["rss_peak_bytes"] for p in P]
    want(f"treehill's C3DGS processes peaked at {gb(hw[0])} and {gb(hw[1])} GB (the wrapper's\n  figure); the job's own steps, which "
         f"count the job's process too, at {gb(max(x['host_rss_peak_bytes'] or 0 for x in ST['treehill']['steps']))} GB. train's peaked at "
         f"{gb(max(W['train'][k]['host_rss']['rss_peak_bytes'] for k in W['train']))} GB and "
         f"{gb(max(x['host_rss_peak_bytes'] or 0 for x in ST['train']['steps']))} GB".replace("\n  ", " "), "host")
    for q_ in ("| the scene | 10,432-11,263 s | 6,575-24,247 s |", "10,861-24,676 s, 3.0-6.9 h",
               "| one default process (13 rows) | 2,468-2,618 s (the seed-0 one, with the chunk check) | 2,390-9,362 s |"):
        check(q_ in pre, f"PREREG still says {q_!r}")
    want(f"{jt:,.1f} s against 10,432-11,263 s, {(10432 - jt) / 10432 * 100:.1f}% below the lower end; a default process "
         f"{W['train']['j0_p0_a0']['wall_s']:,.1f} / {W['train']['j0_p1_a0']['wall_s']:,.1f} s against\n  2,468-2,618 s".replace("\n  ", " "), "est train")
    want(f"{jh:,.1f} s, inside 6,575-24,247 s; a default process {W['treehill']['j0_p0_a0']['wall_s']:,.1f} / "
         f"{W['treehill']['j0_p1_a0']['wall_s']:,.1f} s against 2,390-9,362 s", "est treehill")
    want(f"{sess:,.1f} s against 10,861-24,676 s", "est session")

    # the post hoc chain
    P4 = (0, 1, 2)
    e4 = lambda a, b: [f(e4pr[f"p{p}_{a}"]["PSNR_ii"]) - f(e4pr[f"p{p}_{b}"]["PSNR_ii"]) for p in P4]  # noqa: E731
    chain = [("(iii) GN-VQ's update on the isotropic metric, from C3DGS's codebook", "`scalar` - `c3dgs`", "E4p", comp(e4("scalar", "c3dgs"))),
             ("(i) the 16 x 16 metric against its trace", "`gnvq_rho0` - `scalar`", "E4p", comp(e4("gnvq_rho0", "scalar"))),
             ("the floor", "`gnvq_cv` - `gnvq_rho0`", "E4p", comp(e4("gnvq_cv", "gnvq_rho0"))),
             ("(ii) OGC's choices in GN-VQ's code", "`lad_all` - `gnvq_cv`", "E4q", comp(dsp("train", "lad_all", "gnvq_cv"))),
             ("arithmetic", "`ogc` - `lad_all`", "E4q", comp(dsp("train", "ogc", "lad_all")))]
    for lab, rows_, run_, c in chain:
        want(f"| {lab} | {rows_} | {run_} | {3 if run_ == 'E4p' else 2} | {sg(c[0])} | {c[3]:.4f} |", f"chain {rows_}")
    tot_ = sum(c[0] for *_, c in chain)
    want(f"| sum | | | | {sg(tot_)} | |", "chain sum")
    oc = comp(dsp("train", "ogc", "c3dgs"))
    want(f"| `ogc` - `c3dgs` | | E4q | 2 | {sg(oc[0])} | {oc[3]:.4f} |", "chain target")
    sh_ = [c[0] / tot_ * 100 for *_, c in chain]
    want(f"the metric {sh_[1]:.1f}%, OGC's choices {sh_[3]:.1f}%, step (iii) {sh_[0]:.1f}%, the arithmetic {sh_[4]:.1f}%, the floor "
         f"{sh_[2]:.1f}%", "shares")
    gc4, gcq = comp(e4("gnvq_cv", "c3dgs"))[0], comp(dsp("train", "gnvq_cv", "c3dgs"))[0]
    want(f"`gnvq_cv` - `c3dgs`: {sg(gc4)} dB in\n  E4p and {sg(gcq)} dB in E4q".replace("\n  ", " "), "the join")
    want(f"reads {f(e4pr['p0_c3dgs']['PSNR_ii']):.4f} in E4p and {f(row('train', 'c3dgs', 0)['PSNR_ii']):.4f} in E4q", "c3dgs p0")
    check(e4pr["p0_c3dgs"]["geometry_sha1"] != row("train", "c3dgs", 0)["geometry_sha1"], "different runs")
    want(f"They agree within {abs(gc4 - gcq):.4f}\n  dB; `ogc` - `c3dgs` reads {sg(comp(e4('ogc', 'c3dgs'))[0])} in E4p and {sg(oc[0])} in E4q"
         .replace("\n  ", " "), "ogc join")
    lrs = dline("train", "lad_reseed", "gnvq_cv")[1]
    alt = chain[0][3][0] + chain[1][3][0] + chain[2][3][0] + lrs
    want(f"taking `lad_reseed` alone for (ii), {sg(lrs)}, closes the sum to\n  within {abs(oc[0] - alt):.4f} dB on train".replace("\n  ", " "),
         "reseed alone")
    want(f"on treehill `lad_reseed` is {sg(L['treehill']['lad_reseed'])} dB and `lad_init_tr` {sg(L['treehill']['lad_init_tr'])}", "treehill pair")

    # what E4q settles
    want(f"recover all but {abs(dline('train', 'lad_all', 'ogc')[1]):.4f} dB (train) and {abs(dline('treehill', 'lad_all', 'ogc')[1]):.4f} "
         f"dB (treehill) of it", "settles 1")
    want(f"(reseeding, {sg(L['train']['lad_reseed'])} dB on train; OGC's draw, {sg(L['treehill']['lad_init_tr'])} dB on treehill); the "
         f"regularization choices are each {max(abs(L[s][r]) for s in SC for r in ('lad_ridge_mean', 'lad_clip_part', 'lad_no_clip')):.4f} dB\n  "
         f"or less, GN-VQ's floor {chain[2][3][0]:.4f} dB (E4p)".replace("\n  ", " "), "settles 2")
    want(f"the 16 x 16 metric against its trace is\n  {sh_[1]:.1f}% of the chain's sum, codebook choices {sh_[3]:.1f}%".replace("\n  ", " "), "settles 3")
    want(f"BD-rate {bd['ogc_vs_gnvq_cv']['bd_rate_percent']:.2f}% and BD-PSNR {bd['ogc_vs_gnvq_cv']['bd_psnr_db']:+.4f} dB for `ogc` "
         f"against `gnvq_cv`", "settles 4")
    want(f"GN-VQ reads {sg(dline('train', 'gnvq_cv', 'ogc')[1])} dB (train) and {sg(dline('treehill', 'gnvq_cv', 'ogc')[1])} dB\n  "
         f"(treehill) below OGC".replace("\n  ", " "), "settles 5")
    check(all(comp(dsp(s, "gnvq_cv", "ogc", "npz_bytes"))[0] < 0 for s in SC), "with fewer bytes")
    # summary paragraph
    want(f"come within {abs(dline('train', 'lad_all', 'ogc')[1]):.4f} dB (train)\n  and {abs(dline('treehill', 'lad_all', 'ogc')[1]):.4f} dB "
         f"(treehill) of OGC's VQ. Reseeding empty clusters ({sg(L['train']['lad_reseed'])} dB, train) and OGC's start "
         f"({sg(L['treehill']['lad_init_tr'])} dB,\n  treehill)".replace("\n  ", " "), "summary 1")
    want(f"OGC needs {-bd['ogc_vs_gnvq_cv']['bd_rate_percent']:.2f}% fewer bytes than GN-VQ (BD-PSNR {bd['ogc_vs_gnvq_cv']['bd_psnr_db']:+.4f} dB)",
         "summary 2")
    stamps = sorted(r["timestamp"] for s in SC for r in R[s].values())
    want(f"rows timestamped\n  {stamps[0][:16]} to {stamps[-1][11:16]}".replace("\n  ", " "), "stamps")
    want(f"gsplat commit `{env['gsplat_commit'][:8]}`", "commit")

    # this repository's cited lines, read at the commit E4q ran at
    for path, a, b, needle, cite in CITED:
        check(needle in lines_of(path, a, b), f"{path}:{a}-{b} holds {needle[:40]!r}")
        want(cite, f"cite {path}:{a}")

    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | OUTSIDE_LINES | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 17: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
