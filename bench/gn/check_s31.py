"""FINDINGS section 31 (post hoc, report only, no new run, no new BD value): the ten scenes with an OGC row split into
indoor and outdoor. Every number is recomputed from section 28's values (through ``check_s29.facts()``) and checked
against the text; the labels are checked against ``paper/QUOTES.md``'s label rows, and the confounds against the
committed meta files (dataset, data factor) and section 30's sets.

    .venv/Scripts/python.exe bench/gn/check_s31.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import check_s21 as c21  # noqa: E402
import check_s29 as c29  # noqa: E402

FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
QUOTES = os.path.join(REPO, "paper", "QUOTES.md")
NUM = c21.NUM
INDOOR = ("bonsai", "counter", "kitchen", "room", "drjohnson", "playroom")
OUTDOOR = ("truck", "flowers", "treehill", "train")
# each label row: (bib key, its Number cell, the scenes it labels, the label)
LABELS = [("barron2022mipnerf360", "Indoor: room, counter, kitchen, bonsai", ("room", "counter", "kitchen", "bonsai"), "indoor"),
          ("barron2022mipnerf360", "Outdoor: bicycle, flowers, garden, stump, treehill", ("flowers", "treehill"), "outdoor"),
          ("hedman2016scalable", "indoor: PLAYROOM", ("playroom",), "indoor"),
          ("hedman2016scalable", "indoor: Dr Johnson's house", ("drjohnson",), "indoor"),
          ("hanson2025speedy", "outdoor: train, truck", ("train", "truck"), "outdoor"),
          ("hanson2025speedy", "indoor: drjohnson, playroom", ("drjohnson", "playroom"), "indoor")]
OWN_PAPER = {"barron2022mipnerf360", "hedman2016scalable"}  # a dataset's own paper (Inside-Out: playroom's and drjohnson's source)
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "10", "17", "28", "29", "30", "31", "2025", "2016", "5.1", "5.2", "360"}


def f4(x):
    return f"{x:+.4f}"


def facts():
    F = c29.facts()
    V, M = F["vals"], F["M"]
    means = {}
    for name, ss in (("indoor", INDOOR), ("outdoor", OUTDOOR)):
        means[name] = [(sum(V[s][i][0] for s in ss) / len(ss), sum(V[s][i][1] for s in ss) / len(ss)) for i in range(3)]
    return {"V": V, "M": M, "means": means}


def quote_rows():
    t = open(QUOTES, encoding="utf-8").read().replace("\r\n", "\n")
    return [(m.group(1), m.group(2)) for m in re.finditer(r"^\| ([A-Za-z][\w-]*) \| ([^|]+?) \|", t, re.M)]


def section_text(txt):
    out = [c21.section(txt, 31, "Indoor and outdoor scenes")]
    k = "**Indoor and outdoor (section 31"
    out.append(txt[txt.index(k):txt.index("\n\n", txt.index(k))])
    s1 = txt.index("- Section 31:")
    end = txt.find("\n- Section", s1 + 1)
    out.append(txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)])
    return re.sub(r"\s+", " ", " ".join(out))


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    F = facts()
    V, M, means = F["V"], F["M"], F["means"]
    fails, exp = [], set()

    def want(s, why=""):
        exp.add(s)
        if re.sub(r"\s+", " ", s) not in text:
            fails.append(f"not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    # the means table
    for name, ss in (("indoor", INDOOR), ("outdoor", OUTDOOR)):
        want(f"| {name} | {', '.join(ss)} | " + " | ".join(f"{f4(a)} / {f4(b)}" for a, b in means[name]) + " |", name)
    g, o = means["indoor"], means["outdoor"]
    want(f"P1 {g[0][0]:.4f} dB and P_FT {g[2][0]:.4f} dB against outdoor (four) {o[0][0]:.4f} and {o[2][0]:.4f} dB", "summary")
    # the ordering, both fits
    for f in (0, 1):
        check(min(V[s][0][f] for s in INDOOR) > max(V[s][0][f] for s in OUTDOOR), f"every indoor P1 above every outdoor P1 (fit {f})")
    lo_in = min(INDOOR, key=lambda s: V[s][0][0])
    hi_out = max(OUTDOOR, key=lambda s: V[s][0][0])
    check(lo_in == "drjohnson" and hi_out == "truck", "the lowest indoor P1 is drjohnson's, the highest outdoor truck's")
    want(f"({f4(V['drjohnson'][0][0])} / {f4(V['drjohnson'][0][1])})", "drjohnson P1")
    want(f"({f4(V['truck'][0][0])} / {f4(V['truck'][0][1])})", "truck P1")
    lo_ft = min(INDOOR, key=lambda s: V[s][2][0])
    hi_ft = max(OUTDOOR, key=lambda s: V[s][2][0])
    check(lo_ft == "drjohnson" and hi_ft == "flowers" and V[lo_ft][2][0] < V[hi_ft][2][0], "P_FT is not ordered")
    want(f"({f4(V['drjohnson'][2][0])} / {f4(V['drjohnson'][2][1])})", "drjohnson P_FT")
    want(f"({f4(V['flowers'][2][0])} / {f4(V['flowers'][2][1])})", "flowers P_FT")
    own = {s for k, _, ss, _ in LABELS if k in OWN_PAPER for s in ss}
    check(own == set(INDOOR) | {"flowers", "treehill"}, "eight labels come from a dataset's own paper")
    check(min(V[s][0][f] for s in INDOOR for f in (0, 1)) > max(V[s][0][f] for s in ("truck", "train") for f in (0, 1)),
          "truck and train are below every indoor scene")
    # the labels: each row is in QUOTES.md, every scene is labelled, and no row contradicts the split
    rows = quote_rows()
    for k, cell, ss, lab in LABELS:
        check((k, cell) in rows, f"QUOTES.md has the row {k} | {cell}")
        check(all((s in INDOOR) == (lab == "indoor") for s in ss), f"{k}'s row agrees with the split")
    check({s for _, _, ss, _ in LABELS for s in ss} == set(INDOOR) | set(OUTDOOR), "every scene has a label row")
    # the confounds
    check(set(INDOOR) <= set(c29.GATE) and set(OUTDOOR) & set(c29.GATE) == {"truck"} and set(OUTDOOR) - {"truck"} == set(c29.DEV),
          "apart from truck the split is the gate/development split")
    check(min(V[s][0][0] for s in c29.GATE) == V["truck"][0][0], "truck's P1 is the gate's lowest")
    ds = {s: (M[s][0], M[s][2]) for s in INDOOR + OUTDOOR}
    check(sorted(ds[s] for s in INDOOR) == sorted([("mipnerf360", 2)] * 4 + [("db", 1)] * 2), "indoor: four Mip-NeRF 360 at 2, two Deep Blending at 1")
    check(sorted(ds[s] for s in OUTDOOR) == sorted([("mipnerf360", 4)] * 2 + [("tandt", 1)] * 2), "outdoor: two Mip-NeRF 360 at 4, two Tanks and Temples at 1")
    check("no cause is claimed" in text and "not a test" in text, "the section claims no cause and is no test")
    stripped = re.sub(r"\d{4}-\d{2}-\d{2}|`[^`]*`|arXiv \d{4}\.\d{5}v\d", " ", text)
    tokens = set(NUM.findall(stripped))
    unmatched = sorted(t for t in tokens if t not in CONSTANTS and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails, "means": means}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 31: {res['n_numbers']} recomputed values, {res['n_tokens']} numeric tokens checked, "
          f"{len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
