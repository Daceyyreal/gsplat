"""Shared reading, BD fit and style for the paper's figure and table scripts (``paper/figs/*.py``).

It reads only committed bundles: E5's three session bundles (``kaggle/gn_e5/S*/gn5/``) and E6's five
(``kaggle/gn_e6/S*/gn6/``), each scene from the session that ran it (later bundles carry it byte-identical; FINDINGS
sections 19-23 and their checkers rely on the same reading). Nothing here imports ``bench/gn/e5.py``, ``e6.py`` or
``g2.py``.

BD over three points is the parabola through them (``numpy.polyfit`` of degree 2, the construction FINDINGS sections
19-23 use): BD-PSNR averages the difference of the two parabolas over the shared range of log10 bytes; BD-rate fits
log10 bytes against PSNR and averages over the shared PSNR range.

Outputs go to ``paper/figs/out/``: tables as ``.csv`` (full precision) and ``.tex`` (booktabs, at FINDINGS' rounding),
figures as ``.pdf`` and ``.png`` with their plotted data as ``.csv``. The figures use the Okabe-Ito palette, and every
method also has its own marker and line style, so colour is never the only cue.
"""

import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
E5 = os.path.join(REPO, "kaggle", "gn_e5")
E6 = os.path.join(REPO, "kaggle", "gn_e6")

GATE = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom")
PART_A = ("flowers", "treehill", "train")
E5_SESSION = {"counter": "S1", "room": "S1", "kitchen": "S1", "truck": "S2", "drjohnson": "S2", "playroom": "S3",
              "bonsai": "S3"}
E6_SESSION = {"playroom": "S1", "counter": "S1", "truck": "S1", "drjohnson": "S2", "room": "S2", "kitchen": "S3",
              "bonsai": "S3", "flowers": "S4", "train": "S4", "treehill": "S5"}
POINTS = (-1, 0, 1)
# the four processes of E5 and E6: (j, seed)
PROCS = ((-1, 0), (0, 0), (0, 1), (1, 0))
SMALL_OVERLAP = 0.1  # dB: FINDINGS section 21's convention for an interpretable BD-rate


def cfg(j, seed, row):
    return f"j{'+' if j > 0 else ''}{j}_p{seed}_{row}" if j else f"p{seed}_{row}"


def read_rows(path, scene):
    with open(path, newline="") as f:
        return {r["config"]: r for r in csv.DictReader(f) if r["scene"] == scene}


def e5_rows(scene):
    return read_rows(os.path.join(E5, E5_SESSION[scene], "gn5", f"gn5_results_{scene}.csv"), scene)


def e6_rows(scene):
    return read_rows(os.path.join(E6, E6_SESSION[scene], "gn6", f"gn5_results_{scene}.csv"), scene)


def val(rows, c, k="PSNR_ii"):
    return float(rows[c][k])


def curve(rows, row):
    """(log10 npz bytes, protocol ii PSNR) at j = -1, 0, +1, seed 0."""
    x = np.array([math.log10(val(rows, cfg(j, 0, row), "npz_bytes")) for j in POINTS])
    y = np.array([val(rows, cfg(j, 0, row)) for j in POINTS])
    return x, y


def bd_psnr(rows, a, b):
    (xa, ya), (xb, yb) = curve(rows, a), curve(rows, b)
    lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
    ia, ib = np.polyint(np.polyfit(xa, ya, 2)), np.polyint(np.polyfit(xb, yb, 2))
    return float(((np.polyval(ia, hi) - np.polyval(ia, lo)) - (np.polyval(ib, hi) - np.polyval(ib, lo))) / (hi - lo))


def d_sp(rows, a, b):
    """The paired protocol ii difference ``a`` - ``b`` at j = 0, seeds 0 and 1."""
    return [val(rows, cfg(0, p, a)) - val(rows, cfg(0, p, b)) for p in (0, 1)]


def r9(x):
    return round(float(x), 9)


def criteria(P, D):
    """E5's three criteria (Amendment 17 d, Note 2 C7) over scenes: ``P`` per scene, ``D`` the two j = 0 differences."""
    n = len(P)
    pbar = sum(P.values()) / n
    vs = {s: sum((x - sum(d) / 2) ** 2 for x in d) for s, d in D.items()}
    sd = math.sqrt(sum(vs.values()) / n)
    se = sd / math.sqrt(2 * n)
    pos = sum(1 for v in P.values() if r9(v) > 0)
    need = (7 * n + 9) // 10
    return {"P_bar": pbar, "SD_pool": sd, "SE_noise": se, "n_pos": pos, "need": need, "n": n,
            "c1": r9(pbar) > 0, "c2": pos >= need, "c3": r9(pbar) > 2 * r9(se)}


def qa(rows, c):
    return json.loads(rows[c]["qa_at_save"])


# ------------------------------------------------------------------------------------------------ output helpers


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def write_csv(name, header, rows):
    with open(out(name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([repr(x) if isinstance(x, float) else x for x in r])


def tex_escape(s):
    """Escape ``_`` and ``%`` outside ``$...$`` math."""
    parts = str(s).split("$")
    return "$".join(q if i % 2 else q.replace("_", r"\_").replace("%", r"\%") for i, q in enumerate(parts))


def write_tex(name, source, colspec, header, body, footer=(), script=None):
    """A booktabs tabular, to ``\\input`` inside a table environment. ``body`` and ``footer`` are lists of cell lists,
    already formatted; ``footer`` rows follow a midrule. ``script`` names the generating script when it is not
    ``name``'s stem."""
    script = script or name.rsplit(".", 1)[0]
    lines = [f"% {source}", f"% generated by paper/figs/{script}.py from committed bundles; do not edit",
             f"\\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in body]
    if footer:
        lines.append(r"\midrule")
        lines += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in footer]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(out(name), "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------------------------------------------ figure style

# Okabe-Ito; each method also has its own marker and line style
STYLE = {
    "c3dgs": dict(color="#E69F00", marker="o", ls="-", label="C3DGS VQ"),
    "ogc_plain": dict(color="#56B4E9", marker="^", ls=":", label="OGC Lloyd, identity"),
    "ogc_scalar": dict(color="#009E73", marker="s", ls="-.", label="OGC Lloyd, trace"),
    "ogc_gram": dict(color="#0072B2", marker="D", ls="--", label="OGC Lloyd, Gram"),
}
STYLE["c3dgs_ft"] = dict(STYLE["c3dgs"], label="C3DGS VQ, fine-tuned")
STYLE["ogc_gram_ft"] = dict(STYLE["ogc_gram"], label="OGC Lloyd, Gram, fine-tuned")


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 200, "svg.hashsalt": "paper-figs"})
    return plt


def save(fig, stem):
    """``stem``.pdf and ``stem``.png, without creation dates or software stamps, so a rebuild is byte-stable."""
    fig.savefig(out(stem + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None,
                                                                   "Producer": None, "Creator": None})
    fig.savefig(out(stem + ".png"), bbox_inches="tight", metadata={"Software": None})
