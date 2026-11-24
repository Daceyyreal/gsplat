"""FINDINGS section 32 (post hoc, report only, declared before computing these): GN-VQ's train-to-test transfer ratio
(section 10) against OGC's P1, P2 and P_FT (section 29), per scene. Recomputes, from committed files only:

- the ratio per scene from section 10's table, as test PSNR gain / train PSNR gain (the 4-decimal gains);
- P1, P2 and P_FT per scene under the registered fit and PCHIP (``check_s29.facts()``, i.e. section 28's computation);
- Spearman's rho of the ratio with each, on the n = 8 scenes with both and on n = 7 without treehill, with the exact
  two-sided p over all n! orderings (|rho| >= the observed |rho| - 1e-12; mid-ranks if ties);
- and checks every number in the section's text.

    .venv/Scripts/python.exe bench/gn/check_s32.py [--print]
"""
import itertools
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import check_s21 as c21  # noqa: E402
import check_s29 as c29  # noqa: E402

FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
NUM = c21.NUM
SCENES = ("bonsai", "counter", "kitchen", "room", "truck", "flowers", "treehill", "train")
TARGETS = ("P1", "P2", "P_FT")


def gains(txt):
    """Section 10's table: scene -> (test gain, train gain, printed ratio)."""
    sec = c21.section(txt, 10, "E2")
    t = sec[sec.index("### Train and test views against `lloyd_trace`"):]
    out = {}
    for m in re.finditer(r"^\| ([a-z]+)(?: \(dev\))? \| \d+ \| \d \| ([-+]\d\.\d{4}) \| ([-+]\d\.\d{4}) \| (-?\d\.\d\d) \|$", t, re.M):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)), m.group(4))
    return out


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    return sab / (saa * sbb) ** 0.5


def spearman_exact(x, y):
    """Spearman's rho and its exact two-sided p over all orderings of y's ranks."""
    rx, ry = ranks(x), ranks(y)
    rho = pearson(rx, ry)
    hits = tot = 0
    for perm in itertools.permutations(ry):
        tot += 1
        if abs(pearson(rx, list(perm))) >= abs(rho) - 1e-12:
            hits += 1
    return rho, hits / tot, hits, tot, any(rx.count(v) > 1 for v in rx) or any(ry.count(v) > 1 for v in ry)


def facts():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    G = gains(txt)
    V = c29.facts()["vals"]
    ratio = {s: G[s][0] / G[s][1] for s in SCENES}
    res = {}
    for drop in (None, "treehill"):
        ss = [s for s in SCENES if s != drop]
        for ti, tname in enumerate(TARGETS):
            for fi, fname in enumerate(("reg", "pchip")):
                res[(drop, tname, fname)] = (len(ss),) + spearman_exact([ratio[s] for s in ss], [V[s][ti][fi] for s in ss])
    return {"G": G, "V": V, "ratio": ratio, "res": res}


def strings(F):
    rows = []
    for s in SCENES:
        v = F["V"][s]
        rows.append(f"| {s} | {'gate' if s in c29.GATE else 'development'} | {F['ratio'][s]:+.4f} | "
                    + " | ".join(f"{v[i][0]:+.4f} / {v[i][1]:+.4f}" for i in range(3)) + " |")
    corr = []
    for drop, label in ((None, "all eight"), ("treehill", "without treehill")):
        for t in TARGETS:
            n, rho, p, _, _, ties = F["res"][(drop, t, "reg")]
            _, rho2, p2, _, _, _ = F["res"][(drop, t, "pchip")]
            corr.append(f"| {label} | {n} | {t} | {rho:+.4f} | {p:.4f} | {rho2:+.4f} | {p2:.4f} |")
    return {"rows": rows, "corr": corr}


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    sec = re.sub(r"\s+", " ", c21.section(txt, 32, "GN-VQ's train-to-test transfer against OGC's lead"))
    F = facts()
    fails, exp = [], set()
    for k, v in strings(F).items():
        for s in v:
            exp.add(s)
            if s not in sec:
                fails.append(f"not in the text: {s!r}")
    if any(r[5] for r in F["res"].values()):
        fails.append("ties occurred: the text must say mid-ranks were used")
    for s in SCENES:
        pr = F["G"][s][2]
        if f"{F['ratio'][s]:.2f}" != pr and not (pr == "-1.06" and round(F["ratio"][s], 2) == -1.06):
            fails.append(f"{s}: the recomputed ratio does not round to section 10's {pr}")
    return {"n_numbers": len(exp), "fails": fails, "F": F}


def main() -> int:
    if "--print" in sys.argv:
        for k, v in strings(facts()).items():
            print("\n".join(v))
        return 0
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 32: {res['n_numbers']} recomputed rows, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
