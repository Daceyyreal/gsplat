"""FINDINGS section 30 (report only, post hoc, no new run, no new value): which outcome placed each of INRIA's thirteen
scenes, and how GN-VQ's metric relates to the metric OGC's VQ ran on in E5-E7. Every quotation, line number, date and
number in the section is checked against the committed files:

- each quotation is in ``kaggle/PREREG_GN.md`` (or ``bench/gn/gn_metric.py``) inside the cited lines;
- each amendment's first commit and date, and the first OGC-against-C3DGS section's (16), from git;
- each scene's ratio is section 10's table value (GN-VQ's test/train PSNR gain over ``lloyd_trace`` at K = 65,536);
- the scenes with a GN-VQ row are read from the committed E0-E4q bundles' file names: drjohnson and playroom have none,
  and no CSV in those bundles names them;
- the selecting three of Amendment 9 are the three lowest ratios, and the five kept gate scenes are the rest of E2's
  held-out scenes outside Amendment 10's three;
- every numeric token in the section is accounted for.

    .venv/Scripts/python.exe bench/gn/check_s30.py
"""
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import check_s21 as c21  # noqa: E402

FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
PREREG = os.path.join(REPO, "kaggle", "PREREG_GN.md")
GNM = os.path.join(HERE, "gn_metric.py")
NUM = c21.NUM
GATE = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom")
DEV = ("flowers", "treehill", "train")
GNVQ_DEV = ("bicycle", "garden", "stump")
KEPT = ("bonsai", "counter", "kitchen", "room", "truck")
UNPLACED = ("drjohnson", "playroom")
# (file, first line, last line, quotation) as the section cites them
QUOTES = [
    (PREREG, 868, 872, "The symptom in E2 is a train-view gain that does not transfer to test views: at K = 65,536, GN-VQ's "
                       "test PSNR gain over `lloyd_trace` divided by its train PSNR gain is lowest on **treehill (-1.06), "
                       "flowers (0.32) and train (0.47)**, against 0.52-0.90 on the other eight scenes (FINDINGS section 10, "
                       "post hoc)."),
    (PREREG, 876, 881, "Treehill, flowers and train, the three lowest ratios above"),
    (PREREG, 876, 881, "**From this amendment on, treehill, flowers and train are development scenes**"),
    (PREREG, 1015, 1020, "**Treehill, flowers and stump**, the three held-out scenes where the dMSE ratio grows most from "
                         "train to test"),
    (PREREG, 1007, 1007, "room (1.624) ranks above flowers (1.596)"),
    (PREREG, 1082, 1084, "**Bonsai, counter, kitchen, room and truck, all five gate scenes.** They are the only scenes with a "
                         "pinned checkpoint that no tuning decision has used: garden and bicycle became development scenes in "
                         "Amendment 7, treehill, flowers and train in Amendment 9, and stump in Amendment 10."),
    (PREREG, 1233, 1233, "Deep Blending (drjohnson, playroom) and the five scenes no tuning decision has used"),
    (PREREG, 2599, 2600, "as written"),
    (PREREG, 2558, 2558, "`G` our full-train-view 16 x 16 metric of the same splats"),
    (PREREG, 2531, 2531, "What GN-VQ contributes beyond OGC is the metric, which OGC's VQ already uses."),
    (GNM, 5, 6, "M_i = sum_v s_iv * y(d_iv) y(d_iv)^T"),
    (GNM, 14, 16, "the same render, probes and ``s_iv``, a 16 x 16 metric (136 packed values) whose lower-right 15 x 15 "
                  "block is the one above, entry for entry"),
]
LINE_CITES = ["868-872, 876-881", "1015-1020", "1082-1084", "1233", "1893", "2599-2600", "2558", "2531",
              "gn_metric.py:5-6", "14-16"]
AMENDMENTS = {"9": "1ae0c05b", "10": "e580b731", "11": "14145093", "12": "aa67e21e", "15": "573142ac", "17": "b3923bfb"}
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "9", "10", "11", "12", "13", "15", "16", "17", "29", "30",
             "65,536", "136", "8"}


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def first_commit(path, needle):
    out = git("log", "--reverse", "--format=%h %ad", "--date=short", "-S", needle, "--", path)
    return out.split("\n")[0].split() if out else None


def ratios(txt):
    """Section 10's table: scene -> its test/train ratio string, as printed."""
    sec = c21.section(txt, 10, "E2")
    t = sec[sec.index("### Train and test views against `lloyd_trace`"):]
    out = {}
    for m in re.finditer(r"^\| ([a-z]+)(?: \(dev\))? \| \d+ \| \d \| [-+]\d\.\d{4} \| [-+]\d\.\d{4} \| (-?\d\.\d\d) \|$", t, re.M):
        out[m.group(1)] = m.group(2)
    return out


def gnvq_scenes():
    """Scenes with a GN-VQ row, from the committed E0-E4q bundles' file names, and the CSVs naming Deep Blending."""
    files = [p for d in glob.glob(os.path.join(REPO, "kaggle", "gn_e[0-4]*")) for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)]
    scenes = set()
    for p in files:
        m = re.search(r"gn_vq.*_([a-z]+)\.(?:json|csv)$", os.path.basename(p))
        if m:
            scenes.add(m.group(1))
    db_csv = [p for p in files if p.endswith(".csv") and re.search(r"drjohnson|playroom", open(p, encoding="utf-8", errors="ignore").read())]
    return scenes, db_csv


def lines(path, a, b):
    L = open(path, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    return re.sub(r"\s+", " ", " ".join(L[a - 1:b]))


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    sec = c21.section(txt, 30, "Scene selection and its direction")
    k = "**Scene selection (section 30"
    summ = txt[txt.index(k):txt.index("\n\n", txt.index(k))]
    s1 = txt.index("- Section 30:")
    src = txt[s1:txt.find("\n- Section", s1 + 1) + 1]
    text = re.sub(r"\s+", " ", sec + " " + summ + " " + src)
    fails, exp = [], set()

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    def want(s, why=""):
        exp.add(s)
        if re.sub(r"\s+", " ", s) not in text:
            fails.append(f"not in the text: {s!r} ({why})")

    for path, a, b, q in QUOTES:
        qn = re.sub(r"\s+", " ", q)
        src_lines = lines(path, a, b)
        check(qn.replace("**", "") in src_lines.replace("**", ""), f"{os.path.basename(path)}:{a}-{b} holds {q[:50]!r}")
        want(q, "quotation")
    for c in LINE_CITES:
        want(c, "line citation")
    # amendments: first commit and date
    for n, h in AMENDMENTS.items():
        fc = first_commit("kaggle/PREREG_GN.md", f"## Amendment {n} (")
        check(fc is not None and fc[0] == h, f"Amendment {n}'s first commit is {h}")
        if fc:
            want(f"`{h}`, {fc[1]}" if n != "17" else f"`{h}`, {fc[1]}", f"Amendment {n}")
    s16 = first_commit("kaggle/FINDINGS.md", "## 16. ")
    a15 = first_commit("kaggle/PREREG_GN.md", "## Amendment 15 (")
    check(s16 and a15 and s16[1] > a15[1], "section 16 (the first OGC row) is dated after Amendment 15")
    want(f"section 16, {s16[1]}", "first OGC row")
    want(f"Amendment 15 ({a15[1]})", "Amendment 15's date")
    # section 10's ratios
    R = ratios(txt)
    check(set(R) == set(GATE[:5] + DEV + GNVQ_DEV), "section 10's table has the eleven E2 scenes")
    for s in GATE[:5] + DEV + GNVQ_DEV:
        role = "gate" if s in GATE else "development" if s in DEV else "development (GN-VQ)"
        want(f"| {s} | {role} | {R[s]} |", f"{s}'s ratio")
    for s in UNPLACED:
        want(f"| {s} | gate | none |", f"{s} has no ratio")
    order = sorted(R, key=lambda s: float(R[s]))
    check(set(order[:3]) == set(DEV), "Amendment 9's three are the three lowest ratios")
    check(min(float(R[s]) for s in KEPT) == float(R["truck"]), "truck has the lowest ratio of the five kept scenes")
    check(set(KEPT) == set(R) - set(DEV) - set(GNVQ_DEV), "the five kept gate scenes are E2's other held-out scenes")
    # GN-VQ rows
    scenes, db_csv = gnvq_scenes()
    check(scenes == set(R), "the scenes with a GN-VQ row are exactly section 10's eleven")
    check(not (scenes & set(UNPLACED)) and not db_csv, "drjohnson and playroom have no GN-VQ row and no E0-E4q CSV names them")
    check("no reading is drawn" in text and "not a test" in text, "the section says it draws no reading and is no test")
    stripped = re.sub(r"\d{4}-\d{2}-\d{2}|`[0-9a-f]{7,40}`|`[^`]*`", " ", text)
    tokens = set(NUM.findall(stripped))
    allowed = set(CONSTANTS) | set(R.values()) | {"-1.06", "0.32", "0.47", "0.52-0.90", "0.52", "0.90", "1.624", "1.596"}
    allowed |= {x for c in LINE_CITES for x in re.findall(r"\d+", c)}
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not accounted for: {t!r}" for t in unmatched]
    return {"n_checked": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 30: {res['n_checked']} quotations and values checked, {res['n_tokens']} numeric tokens, "
          f"{len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
