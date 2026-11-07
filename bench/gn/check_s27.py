"""Re-check every number in kaggle/FINDINGS.md section 27 (the perceptual BD secondaries, declared at ``cf26643f`` before
any of them was computed), its summary paragraph and its Sources entry against the committed files.

    python bench/gn/check_s27.py            # check
    python bench/gn/check_s27.py --print    # print the computed strings (for drafting)

It reads only committed files: E5's three bundles (``kaggle/gn_e5/S1-S3/gn5/``) and E6 part (b)'s three
(``kaggle/gn_e6/S1-S3/gn6/``), each scene from the session that ran it, through ``check_s21.py``'s loaders, and E7's
eleven (``kaggle/gn_e7/S*/gn7/``) through ``check_s26.py``'s. As the declaration fixes it:

- **The measures:** BD-SSIM and BD-LPIPS are E5's BD (``bench/gn/e5.py`` ``bd``, Amendment 9 a's domain-scaled fit of
  degree 2 through the three seed-0 points, averaged over the two curves' shared log10-byte range) with ``SSIM_ii`` or
  ``LPIPS_ii`` in place of ``PSNR_ii``; the first row's fitted mean minus the second's. Every value is computed twice:
  with ``e5.bd`` (the tool) and with this file's own ``numpy.polyfit`` fit (``ind_bd``, as ``check_s21.py`` builds
  its own), and the two must agree below 1e-9.
- **The values:** P1 (``ogc_gram`` against ``c3dgs``) and P2 (``ogc_gram`` against ``ogc_scalar``) from E5; P_FT
  (``ogc_gram_ft`` against ``c3dgs_ft``) from E6 part (b); per E7 arm X, BD_X (``ogc_gram_ft_X`` against ``c3dgs_ft_X``),
  then Delta_refit = BD_G - BD_R and Delta_cb = BD_CG - BD_G; per scene and the mean over the scenes that have one.
- **Agreement:** per scene and for the mean, BD-SSIM agrees with the PSNR value when sign(r9(BD-SSIM)) equals
  sign(r9(PSNR value)), BD-LPIPS when sign(r9(-BD-LPIPS)) does; an exact 0 after rounding agrees with nothing. The PSNR
  values are recomputed the same way and must equal the registered ones (E5's P1 and P2, E6's P_FT, E7's per-scene
  Delta) to 1e-9.
- **Already known:** E5's P1 and P2 perceptual BDs must equal section 20's (``check_s20.py``'s ``g6_bd``).
- **Bending:** as section 25's later note, the perceptual BD values whose fit (on either curve) has its vertex strictly
  inside the shared log10-byte range.

Each quoted number is recomputed and must appear in the text, each claim is re-asserted, and every numeric token of the
text must be a recomputed string or a listed constant. It prints each failure and exits 1 if there is any, 0
otherwise. Built like ``check_s21.py`` to ``check_s26.py``.
"""

import math
import os
import re
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(REPO, "kaggle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import check_s21 as c21  # noqa: E402
import check_s25 as c25  # noqa: E402
import check_s26 as c26  # noqa: E402
import e5  # noqa: E402

FINDINGS = c21.FINDINGS
NUM = c21.NUM
SCENES = c21.SCENES
POINTS = c21.POINTS
METRICS = ("SSIM_ii", "LPIPS_ii")
LABEL = {"SSIM_ii": "BD-SSIM", "LPIPS_ii": "BD-LPIPS", "PSNR_ii": "BD-PSNR"}
ORIENT = {"SSIM_ii": 1, "LPIPS_ii": -1, "PSNR_ii": 1}  # the sign that favours the first row
E7_ARMS = {"R": "", "G": "_g", "CG": "_cg"}
DECLARATION = "cf26643f"
CONSTANTS = {"-1", "+1", "0", "1", "2", "3", "5", "6", "7", "9", "11", "16", "20", "21", "22", "25", "26", "27", "2026",
             "1e-9", "0.07"}
COMMITS = {DECLARATION, "9a90dabb", "1710cc8a", "88b5ebed", "87cc6a59", "94551278", "13a95aa1"}


def r9(x):
    return round(float(x), 9)


def sgn(x):
    x = r9(x)
    return (x > 0) - (x < 0)


# ------------------------------------------------------------------ the curves and the two fits
def curve(rows, row, metric, seed=0):
    """(bytes, metric) at j = -1, 0, +1 for ``seed``; None if a point is missing or not ``ok``."""
    xs, ys = [], []
    for j in POINTS:
        r = rows.get(c21.cfg(j, seed, row))
        if r is None or r.get("status", "ok") != "ok" or r.get("deadline_cut") or r.get("npz_bytes") in (None, "") \
                or r.get(metric) in (None, ""):
            return None
        xs.append(float(r["npz_bytes"]))
        ys.append(float(r[metric]))
    return xs, ys


def tool_bd(ca, cb):
    """``e5.bd`` of a against b (the tool): its value, or None when not computed."""
    res = e5.bd({"a": ca, "b": cb}, [("a", "b")])["a_vs_b"]
    return res.get("bd_psnr_db") if res.get("computed") else None


def ind_bd(ca, cb):
    """This file's own fit: each curve's parabola in log10 bytes (``numpy.polyfit``), averaged over the shared range."""
    xa, ya = np.log10(np.asarray(ca[0])), np.asarray(ca[1])
    xb, yb = np.log10(np.asarray(cb[0])), np.asarray(cb[1])
    lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
    if not hi > lo:
        return None
    ia, ib = np.polyint(np.polyfit(xa, ya, 2)), np.polyint(np.polyfit(xb, yb, 2))
    return float(((np.polyval(ia, hi) - np.polyval(ia, lo)) - (np.polyval(ib, hi) - np.polyval(ib, lo))) / (hi - lo))


def bent(ca, cb):
    """Section 25's later note: either fit's vertex strictly inside the shared log10-byte range."""
    xa, xb = [math.log10(x) for x in ca[0]], [math.log10(x) for x in cb[0]]
    lo, hi = max(min(xa), min(xb)), min(max(xa), max(xb))
    return c25.bends(xa, ca[1], lo, hi) or c25.bends(xb, cb[1], lo, hi)


def one(rows, a, b, metric, agree):
    """(tool value, independent value, bent) of a against b, or None when either curve is incomplete."""
    ca, cb = curve(rows, a, metric), curve(rows, b, metric)
    if ca is None or cb is None:
        return None
    t, i = tool_bd(ca, cb), ind_bd(ca, cb)
    if t is None or i is None:
        return None
    agree.append(abs(t - i) < 1e-9)
    return t, bent(ca, cb)


# ------------------------------------------------------------------ the values
def facts():
    R, _, R5, carried = c21.load()
    R7 = c26.load(c26.REAL_ROOT)[0]
    F = {"carried": carried, "agree_fit": [], "psnr_match": []}
    V = {}  # (primary, metric) -> {scene: value or None}
    B = {}  # (primary, metric) -> number of scenes with a bending fit
    for name, Rs, a, b in (("P1", R5, "ogc_gram", "c3dgs"), ("P2", R5, "ogc_gram", "ogc_scalar"),
                           ("P_FT", R, "ogc_gram_ft", "c3dgs_ft")):
        for m in ("PSNR_ii",) + METRICS:
            res = {s: one(Rs[s], a, b, m, F["agree_fit"]) for s in SCENES}
            V[(name, m)] = {s: (v[0] if v else None) for s, v in res.items()}
            B[(name, m)] = sum(1 for v in res.values() if v and v[1])
    for m in ("PSNR_ii",) + METRICS:
        bd = {}
        for x, suf in E7_ARMS.items():
            res = {s: one(R7[s], "ogc_gram_ft" + suf, "c3dgs_ft" + suf, m, F["agree_fit"]) for s in SCENES}
            bd[x] = {s: (v[0] if v else None) for s, v in res.items()}
            B[(f"BD_{x}", m)] = sum(1 for v in res.values() if v and v[1])
        for name, (p, q) in (("Delta_refit", ("G", "R")), ("Delta_cb", ("CG", "G"))):
            V[(name, m)] = {s: (None if bd[p][s] is None or bd[q][s] is None else bd[p][s] - bd[q][s]) for s in SCENES}
    # the PSNR values must equal the registered ones
    P_reg = {"P1": {s: c21.bd(R5[s], "ogc_gram", "c3dgs")[0] for s in SCENES},
             "P2": {s: c21.bd(R5[s], "ogc_gram", "ogc_scalar")[0] for s in SCENES},
             "P_FT": {s: c21.bd(R[s], "ogc_gram_ft", "c3dgs_ft")[0] for s in SCENES}}
    for k, d in P_reg.items():
        F["psnr_match"] += [abs(V[(k, "PSNR_ii")][s] - d[s]) < 1e-9 for s in SCENES]
    for k, (p, q) in (("Delta_refit", ("G", "R")), ("Delta_cb", ("CG", "G"))):
        reg = {s: c26.ind_bd(R7[s], "ogc_gram_ft" + E7_ARMS[p], "c3dgs_ft" + E7_ARMS[p])
               - c26.ind_bd(R7[s], "ogc_gram_ft" + E7_ARMS[q], "c3dgs_ft" + E7_ARMS[q]) for s in SCENES}
        F["psnr_match"] += [abs(V[(k, "PSNR_ii")][s] - reg[s]) < 1e-9 for s in SCENES]
    # E5's perceptual BDs equal section 20's
    import check_s20 as c20

    g6 = c20.facts()[1]["g6_bd"]
    F["s20_equal"] = all(abs(V[(p, m)][s] - g6[(s, "ogc_gram", b, m)]) < 1e-12
                         for p, b in (("P1", "c3dgs"), ("P2", "ogc_scalar")) for m in METRICS for s in SCENES)
    F["V"], F["B"] = V, B
    out = strings(V, B)
    out["n_fit"] = f"{len(F['agree_fit'])} BD values"
    return out, F


PRIMARIES = ("P1", "P2", "P_FT", "Delta_refit", "Delta_cb")
SOURCE = {"P1": "E5", "P2": "E5", "P_FT": "E6 part (b)", "Delta_refit": "E7", "Delta_cb": "E7"}


def agrees(v, p, m):
    return v is not None and p is not None and sgn(ORIENT[m] * v) != 0 and sgn(ORIENT[m] * v) == sgn(p)


def strings(V, B):
    out = {}
    rows, summ = [], []
    for k in PRIMARIES:
        P = V[(k, "PSNR_ii")]
        for s in SCENES:
            cells = [f"{P[s]:+.4f}"]
            for m in METRICS:
                v = V[(k, m)][s]
                cells.append("missing" if v is None else f"{v:+.6f} ({'agrees' if agrees(v, P[s], m) else 'differs'})")
            rows.append(f"| {k} | {s} | " + " | ".join(cells) + " |")
        have = {m: [s for s in SCENES if V[(k, m)][s] is not None] for m in METRICS}
        pm = st.mean(P[s] for s in SCENES)
        cells = [f"{pm:+.4f}"]
        for m in METRICS:
            mm = st.mean(V[(k, m)][s] for s in have[m])
            n_ag = sum(1 for s in have[m] if agrees(V[(k, m)][s], P[s], m))
            cells.append(f"{mm:+.6f} ({'agrees' if agrees(mm, pm, m) else 'differs'}; {n_ag} of {len(have[m])} scenes agree)")
            out[f"{k}_{m}_n"] = f"{n_ag} of {len(have[m])}"
        rows.append(f"| {k} | mean | " + " | ".join(cells) + " |")
    out["rows"] = rows
    bend = []
    for k in ("P1", "P2", "P_FT", "BD_R", "BD_G", "BD_CG"):
        bend.append(f"| {k} | {B[(k, 'SSIM_ii')]} of 7 | {B[(k, 'LPIPS_ii')]} of 7 |")
    out["bend_rows"] = bend
    return out


# ------------------------------------------------------------------ the text
def section_text(txt):
    sec = c21.section(txt, 27, "Perceptual BD secondaries")
    out = [sec]
    k = "**Perceptual BD secondaries (section 27"
    if k in txt:
        s0 = txt.index(k)
        out.append(txt[s0:txt.index("\n\n", s0)])
    if "- Section 27:" in txt:
        s1 = txt.index("- Section 27:")
        end = txt.find("\n- Section", s1 + 1)
        out.append(txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)])
    return re.sub(r"\s+", " ", " ".join(out))


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    out, F = facts()
    exp, fails = set(), []

    def want(s, why=""):
        exp.add(s)
        if re.sub(r"\s+", " ", s) not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    for k, v in out.items():
        for s in (v if isinstance(v, (list, tuple)) else [v]):
            want(s, k)
    check(all(F["carried"]), "later part (b) bundles carry each scene byte-identical")
    check(F["agree_fit"] and all(F["agree_fit"]), "the tool's BD and the independent polyfit agree below 1e-9")
    check(F["psnr_match"] and all(F["psnr_match"]), "the PSNR values equal the registered P1, P2, P_FT and E7 Deltas")
    check(F["s20_equal"], "E5's perceptual BDs equal section 20's")
    check(all(v is not None for (k, m), d in F["V"].items() for v in d.values()), "no value is missing")
    check("declared after the primary unblinding, before computing these" in text and DECLARATION in text,
          "the declaration's label and commit are in the section")
    check("no verdict" in text.lower(), "the section gives no verdict")
    V = F["V"]
    differ = {m: sorted(s for s in SCENES if not agrees(V[("P_FT", m)][s], V[("P_FT", "PSNR_ii")][s], m)) for m in METRICS}
    check(differ["SSIM_ii"] == ["kitchen", "playroom"] and differ["LPIPS_ii"] == ["drjohnson"],
          "P_FT's disagreeing scenes are the ones the text names")
    check("kitchen and playroom read slightly below 0" in text and "drjohnson, whose P_FT in PSNR is below 0" in text,
          "the named scenes are in the text")
    check(all(agrees(st.mean(V[(k, m)].values()), st.mean(V[(k, "PSNR_ii")].values()), m) for k in PRIMARIES
              for m in METRICS), "every mean agrees in sign with its PSNR value")
    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    if "--print" in sys.argv:
        out, F = facts()
        for k, v in out.items():
            print(f"[{k}]")
            for s in (v if isinstance(v, (list, tuple)) else [v]):
                print(s)
        print("fits agree:", all(F["agree_fit"]), len(F["agree_fit"]), "psnr match:", all(F["psnr_match"]),
              "s20 equal:", F["s20_equal"])
        return 0
    res = run()
    for x in res["fails"]:
        print("FAIL", x)
    print(f"FINDINGS section 27: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
