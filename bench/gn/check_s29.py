"""FINDINGS section 29 (report only, no new run, no new BD value): the ten scenes with an OGC row, gate and development,
side by side. Every number is recomputed here from committed files and checked against the text:

- per scene, from every committed meta file of E5 (``kaggle/gn_e5/S*/gn5/``, the gate scenes) and E6 part (a)
  (``kaggle/gn_e6/S4-S5/gn6/``, the development scenes): the dataset, the scene set, the data factor, the numbers of
  training and test views, and the checkpoint's splat count (all copies must agree);
- per scene, P1, P2 and P_FT under the registered parabola and PCHIP, from ``check_s28.facts()`` (section 28's own
  computation; E5 and E6 part (b) for the gate, E6 part (a) for the development scenes), and their means per set;
- the selection record: the PREREG amendments that made each scene a development or gate scene, their first commits
  and dates (from git), and that the first OGC-against-C3DGS value on any of the ten scenes came after them.

Indoor or outdoor is not in any committed row: it is the datasets' own description (Mip-NeRF 360's four indoor scenes
are the four at data factor 2 in ``examples/benchmarks/compression/mcmc.sh``; Tanks and Temples' truck and train are
outdoor; Deep Blending's drjohnson and playroom are indoor). It is reported as a label, and nothing is inferred from it.

    .venv/Scripts/python.exe bench/gn/check_s29.py
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import check_s21 as c21  # noqa: E402
import check_s28 as c28  # noqa: E402

FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
NUM = c21.NUM
GATE = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom")
DEV = ("flowers", "treehill", "train")
DATASET = {"mipnerf360": "Mip-NeRF 360", "tandt": "Tanks and Temples", "db": "Deep Blending"}
INDOOR = {"bonsai", "counter", "kitchen", "room", "drjohnson", "playroom"}  # the datasets' description, not a row
# the amendment that made each set, and the first OGC-against-C3DGS value per development scene (FINDINGS section)
SELECTION = [("7", "garden and bicycle are development scenes"),
             ("9", "treehill, flowers and train are development scenes"),
             ("10", "treehill, flowers and stump are development scenes"),
             ("11", "Bonsai, counter, kitchen, room and truck, all five gate scenes"),
             ("12", "Deep Blending (drjohnson, playroom) and the five scenes no tuning decision has used"),
             ("15", "INRIA's official 30k checkpoints of bonsai, counter, kitchen, room, truck, drjohnson and playroom")]
FIRST_OGC = [("train", "16"), ("treehill", "17"), ("flowers", "23")]  # the first section with an OGC row on the scene
AM9_RATIOS = ("-1.06", "0.32", "0.47", "0.52-0.90", "0.52", "0.90")  # quoted from Amendment 9 b (FINDINGS section 10, post hoc)
CONSTANTS = {"0", "1", "2", "3", "4", "7", "9", "10", "11", "12", "15", "16", "17", "19", "23", "28", "29", "30"}


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def first_commit(path, needle):
    out = git("log", "--reverse", "--format=%h %ad", "--date=short", "-S", needle, "--", path)
    return out.split("\n")[0].split() if out else None


def _find(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key and not isinstance(v, (dict, list)):
                yield v
            yield from _find(v, key)
    elif isinstance(d, list):
        for v in d:
            yield from _find(v, key)


def meta():
    paths = glob.glob(os.path.join(REPO, "kaggle", "gn_e5", "S*", "gn5", "gn5_meta_*.json"))
    paths += glob.glob(os.path.join(REPO, "kaggle", "gn_e6", "S4", "gn6", "gn5_meta_*.json"))
    paths += glob.glob(os.path.join(REPO, "kaggle", "gn_e6", "S5", "gn6", "gn5_meta_*.json"))
    M, agree = {}, []
    for p in sorted(paths):
        d = json.load(open(p, encoding="utf-8"))
        n_train, n_test = set(_find(d, "n_train")), set(_find(d, "n_test"))
        rec = (d["dataset"], d["scene_set"], d["data_factor"], d["n_splats"], tuple(sorted(n_train)), tuple(sorted(n_test)))
        agree.append(len(n_train) == 1 and len(n_test) == 1 and (d["scene"] not in M or M[d["scene"]] == rec))
        M.setdefault(d["scene"], rec)
    return M, agree


def f4(x):
    return f"{x:+.4f}"


def facts():
    M, agree = meta()
    _, F = c28.facts()
    V = F["V"]
    key = {s: (("E5", "P1"), ("E5", "P2"), ("E6 part (b)", "P_FT")) for s in GATE}
    key.update({s: (("E6 part (a)", "P1"), ("E6 part (a)", "P2"), ("E6 part (a)", "P_FT")) for s in DEV})
    rows, vals = [], {}
    for s in GATE + DEV:
        ds, sset, df, nspl, ntr, nte = M[s]
        v = [(V[k][s]["reg"], V[k][s]["pchip"]) for k in key[s]]
        vals[s] = v
        rows.append(f"| {s} | {'gate' if s in GATE else 'development'} | {DATASET[ds]}, {df} | "
                    f"{'indoor' if s in INDOOR else 'outdoor'} | {ntr[0]} / {nte[0]} | {nspl:,} | "
                    + " | ".join(f"{f4(a)} / {f4(b)}" for a, b in v) + " |")
    means = {}
    for name, ss in (("gate", GATE), ("development", DEV)):
        means[name] = [(sum(vals[s][i][0] for s in ss) / len(ss), sum(vals[s][i][1] for s in ss) / len(ss))
                       for i in range(3)]
    sel = {a: first_commit("kaggle/PREREG_GN.md", f"## Amendment {a} (") for a, _ in SELECTION}
    first = {s: first_commit("kaggle/FINDINGS.md", f"## {n}. ") for s, n in FIRST_OGC}
    sets_ok = all(M[s][1] == "gate" for s in GATE) and all(M[s][1] == "E6 part (a)" for s in DEV)
    return {"rows": rows, "means": means, "sel": sel, "first": first, "agree": agree, "sets_ok": sets_ok, "M": M, "vals": vals}


def strings(F):
    out = {"rows": F["rows"]}
    for name, m in F["means"].items():
        out[f"mean_{name}"] = f"| mean, {name} | " + " | ".join(f"{f4(a)} / {f4(b)}" for a, b in m) + " |"
    g, d = F["means"]["gate"], F["means"]["development"]
    out["headline"] = (f"P1 {g[0][0]:.4f} dB on the gate scenes against {d[0][0]:.4f} dB on the development scenes, P_FT "
                       f"{g[2][0]:.4f} dB against {d[2][0]:.4f} dB")
    out["sel"] = [f"Amendment {a} (`{F['sel'][a][0]}`, {F['sel'][a][1]})" for a, _ in SELECTION]
    out["first"] = [f"{s}: section {n} (`{F['first'][s][0]}`, {F['first'][s][1]})" for s, n in FIRST_OGC]
    return out


def section_text(txt):
    out = [c21.section(txt, 29, "Scene heterogeneity")]
    k = "**Scene heterogeneity (section 29"
    if k in txt:
        s0 = txt.index(k)
        out.append(txt[s0:txt.index("\n\n", s0)])
    if "- Section 29:" in txt:
        s1 = txt.index("- Section 29:")
        end = txt.find("\n- Section", s1 + 1)
        out.append(txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)])
    return re.sub(r"\s+", " ", " ".join(out))


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    prereg = open(os.path.join(REPO, "kaggle", "PREREG_GN.md"), encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    F = facts()
    exp, fails = set(), []

    def want(s, why=""):
        exp.add(s)
        if re.sub(r"\s+", " ", s) not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    for k, v in strings(F).items():
        for s in (v if isinstance(v, list) else [v]):
            want(s, k)
    check(all(F["agree"]), "every committed copy of a scene's meta file agrees")
    check(F["sets_ok"], "the meta files label the seven gate scenes 'gate' and the three development scenes 'E6 part (a)'")
    check(all(F["sel"].values()) and all(F["first"].values()), "every amendment and section has a first commit")
    flat = re.sub(r"\s+", " ", prereg)
    check(all(re.sub(r"\s+", " ", q) in flat for _, q in SELECTION), "each quoted selection phrase is in PREREG")
    check(all(r in flat for r in AM9_RATIOS[:4]), "Amendment 9 b's ratios are PREREG's")
    last_sel = max(F["sel"][a][1] for a, _ in SELECTION)
    check(all(F["first"][s][1] >= last_sel for s, _ in FIRST_OGC),
          "the first OGC-against-C3DGS value on a development scene is dated after the last selecting amendment")
    p1 = {s: v[0][0] for s, v in F["vals"].items()}
    check(max(p1[s] for s in DEV) < min(p1[s] for s in GATE), "every development scene's P1 is below every gate scene's")
    check("no new run" in text and "report only" in text, "the section says it is report only, no new run")
    stripped = re.sub(r"\d{4}-\d{2}-\d{2}|`[0-9a-f]{7,40}`", " ", text)
    tokens = set(NUM.findall(stripped))
    unmatched = sorted(t for t in tokens if t not in CONSTANTS and t not in AM9_RATIOS and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    if "--print" in sys.argv:
        for k, v in strings(facts()).items():
            for s in (v if isinstance(v, list) else [v]):
                print(s)
        return 0
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 29: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens checked, "
          f"{len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
