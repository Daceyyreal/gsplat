"""The appendix's four tables, generated so that no cell is typed by hand:

    .venv/Scripts/python.exe paper/figs/appendix_tables.py

- ``app_prereg``: every heading of Amendments 17-20 in ``kaggle/PREREG_GN.md`` (the amendments, all their notes and
  errata): the date and what existed then, in the heading's own words; what changed, the heading's own subject or, where it
  has none, a verbatim quote from that section (``QUOTED_SUBJECT``); and the commit that first added the heading.
- ``app_sessions``: E5's, E6's and E7's sessions from their committed bundles: the scenes each ran (its log tails; later E7
  bundles carry earlier scenes, which it did not run), the seeds or the half, the commit that first added the bundle, and
  the Kaggle image (its ``gn5_env_S#.json``).
- ``app_checkers``: each ``bench/gn/check_s##.py``, its FINDINGS section's title, and whether the paper cites that section.
- ``app_bd_robust``: FINDINGS section 28 per scene: every BD value section 25's later note counts and E7's two primaries,
  registered (the parabola) and PCHIP, their absolute difference, the rate-matched configs and the pointwise mean ΔPSNR
  (``bench/gn/check_s28.py``'s ``per_scene_rows``, the tools' values).

Each table is written as ``.csv`` (the cells) and ``.tex`` (a ``longtable``, \\input by ``paper/tex/appendix.tex``).
``bench/gn/test_gn.py`` recomputes every cell by a different route and compares a fresh run with the committed outputs.
"""

import csv
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
PREREG = "kaggle/PREREG_GN.md"
FINDINGS = "kaggle/FINDINGS.md"
AMENDMENTS = (17, 18, 19, 20)
EXPERIMENTS = (("E5", "gn_e5", "gn5", 3), ("E6", "gn_e6", "gn6", 5), ("E7", "gn_e7", "gn7", 11))
# A heading with no subject of its own: a verbatim quote from its section (whitespace normalized), checked on every run.
QUOTED_SUBJECT = {
    "Amendment 17": "retires GN-VQ from any gate (a), registers a replication and dissection of OGC (arXiv 2609.28997) "
                    "inside C3DGS",
    "Amendment 17 Note 4, erratum": "S2's session-hours read 4.32-7.71 h; the correct high figure is 7.70 h",
    "Amendment 18": "the Kaggle image change (b), and where OGC's code comes from and how it is verified (c)",
    "Amendment 19 note 1, erratum": "Note 1's pairing bullet for part (a) summed flowers' host peak at `images` with "
                                    "`-r -1` (21.06 GB), a size note 1 does not cover",
}
SPECIALS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def first_commit(*args):
    """The oldest commit a ``git log`` selection returns, at the 8 digits the project's records use."""
    return git("log", "--reverse", "--format=%H", *args).split("\n")[0][:8]


def parse_heading(line):
    """(label, amendment, date, what existed then, subject) of an amendment's heading; the parenthetical may nest."""
    m = re.match(r"#{2,3} (Amendment (\d+)[^(]*?) \(", line)
    if not m:
        return None
    start, depth = m.end() - 1, 0
    for i in range(start, len(line)):
        depth += {"(": 1, ")": -1}.get(line[i], 0)
        if depth == 0:
            break
    inner, rest = line[start + 1:i], line[i + 1:]
    return m.group(1), int(m.group(2)), inner[:10], inner[10:].lstrip(", "), rest[2:] if rest.startswith(": ") else ""


def prereg_rows():
    lines = read(PREREG).split("\n")
    starts = [i for i, s in enumerate(lines) if s.startswith("## ") or s.startswith("### Amendment")] + [len(lines)]
    rows = []
    for a, b in zip(starts, starts[1:]):
        h = parse_heading(lines[a])
        if not h or h[1] not in AMENDMENTS:
            continue
        label, amendment, date, state, subject = h
        if not subject:
            subject = QUOTED_SUBJECT[label]
            if subject not in " ".join(" ".join(lines[a:b]).split()):
                raise ValueError(f"{label}: the quoted subject is not in its section")
        prefix = lines[a].split(" (")[0] + f" ({date}"
        commit = first_commit(f"-S{prefix}", "--", PREREG)
        rows.append((amendment, {"item": label, "date": date, "commit": commit, "what_changed": subject, "state": state}))
    rows.sort(key=lambda r: r[0])  # stable: within an amendment, the file's order
    return [r[1] for r in rows]


def image_of(env):
    driver = env["nvidia_smi"].splitlines()[0].split(", ")[1]
    return f"py{env['python']}|torch{env['torch']}|cuda{env['torch_cuda']}|cudnn{env['cudnn']}|driver{driver}"


def session_rows():
    rows, labels = [], {}
    for exp, top, sub, n in EXPERIMENTS:
        for i in range(1, n + 1):
            d = f"kaggle/{top}/S{i}/{sub}"
            ran = [re.match(rf"gn_e\d_(.+?)(?:_h(\d))?_log_tail\.json$", os.path.basename(p)).groups()
                   for p in glob.glob(os.path.join(REPO, d, "*_log_tail.json"))]
            halves = {h for _, h in ran}
            if exp == "E7":
                if len(halves) != 1:
                    raise ValueError(f"{d}: a session of more than one half: {halves}")
                seed = f"seed {halves.pop()} half"
            else:
                seed = "seeds 0 and 1"
            with open(os.path.join(REPO, d, "gn5_attempt.json"), encoding="utf-8") as f:
                part = json.load(f).get("part")
            with open(os.path.join(REPO, d, f"gn5_env_S{i}.json"), encoding="utf-8") as f:
                image = image_of(json.load(f))
            label = labels.setdefault(image, "ABCDEFGH"[len(labels)])
            rows.append({"experiment": f"{exp} ({part})" if part else exp, "session": f"S{i}",
                         "scenes": ", ".join(sorted(s for s, _ in ran)), "seed_or_half": seed,
                         "bundle_commit": first_commit("--diff-filter=A", "--", f"{d}/gn5_attempt.json"),
                         "image_label": label, "image": image})
    return rows


def checker_rows():
    headings = dict(re.findall(r"^## (\d+)\. (.*)$", read(FINDINGS), flags=re.M))
    tex = "".join(read(os.path.relpath(p, REPO)) for p in sorted(glob.glob(os.path.join(REPO, "paper", "tex", "*.tex")))
                  if os.path.basename(p) != "appendix.tex")  # the paper's body: the appendix lists the checkers itself
    rows = []
    for p in sorted(glob.glob(os.path.join(REPO, "bench", "gn", "check_s*.py"))):
        n = re.search(r"check_s(\d+)\.py$", p).group(1)
        title = headings[n].split(" — ")[0]
        title = re.sub(r"\s*\((?:[^()]|\([^()]*\))*\)\s*$", "", title)  # the status parenthetical, which may nest
        rows.append({"checker": f"bench/gn/check_s{n}.py", "section": n, "title": title,
                     "cited_by_paper": "yes" if re.search(rf"\bs{n}\b", tex) else "no"})
    return rows


DELTA_TEX = {"Delta_refit": r"$\Delta_{\mathrm{refit}}$", "Delta_cb": r"$\Delta_{\mathrm{cb}}$"}


def esc(s):
    return "".join(SPECIALS.get(c, c) for c in s)


def tex(s):
    """Backticked spans as ``\\code``, the rest escaped."""
    return "".join(rf"\code{{{esc(p)}}}" if i % 2 else esc(p) for i, p in enumerate(s.split("`")))


def longtable(widths, header, rows, caption, label, foot=""):
    spec = "".join(rf">{{\raggedright\arraybackslash}}p{{{w}in}}" for w in widths)
    head = " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\"
    out = ["% Generated by paper/figs/appendix_tables.py; do not edit.",
           r"\begingroup\footnotesize\setlength{\tabcolsep}{4pt}\setlength{\LTcapwidth}{\textwidth}",
           rf"\begin{{longtable}}{{@{{}}{spec}@{{}}}}", rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule", head,
           r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    out += [" & ".join(r) + r" \\" for r in rows]
    if foot:
        out += [r"\midrule", rf"\multicolumn{{{len(widths)}}}{{@{{}}p{{{sum(widths) + 0.4}in}}@{{}}}}{{\raggedright {foot}}} \\"]
    return "\n".join(out + [r"\end{longtable}", r"\endgroup", ""])


def write(out, name, rows, table):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{name}.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, f"{name}.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(table)


def main(out=OUT):
    prereg = prereg_rows()
    write(out, "app_prereg", prereg, longtable(
        (1.0, 0.6, 0.55, 2.1, 1.75), ("Item", "Date", "Commit", "What changed", "What existed then"),
        [(esc(r["item"]), r["date"], rf"\code{{{r['commit']}}}", tex(r["what_changed"]), tex(r["state"])) for r in prereg],
        r"The registration history from Amendment 17 on: each amendment, note and erratum, with its date and what existed "
        r"then in its heading's own words, what it changed, and the commit that first added it. Generated from "
        r"\code{kaggle/PREREG\_GN.md} and the repository's history.", "tab:app-prereg"))
    sessions = session_rows()
    images = {r["image_label"]: r["image"] for r in sessions}
    legend = "; ".join(rf"image {k}: \code{{{esc(v).replace('|', '|' + chr(92) + 'allowbreak ')}}}" for k, v in images.items())
    write(out, "app_sessions", sessions, longtable(
        (0.75, 0.5, 2.1, 1.0, 0.75, 0.5), ("Experiment", "Session", "Scenes it ran", "Seeds or half", "Bundle commit", "Image"),
        [(esc(r["experiment"]), r["session"], esc(r["scenes"]), r["seed_or_half"], rf"\code{{{r['bundle_commit']}}}",
          r["image_label"]) for r in sessions],
        r"Every session of the three pre-registered experiments: the scenes it ran (later E7 bundles also carry earlier "
        r"scenes), the seeds or the half, the commit that first added its bundle, and its Kaggle image. Generated from the "
        r"committed bundles and the repository's history.", "tab:app-sessions", foot=legend + "."))
    checkers = checker_rows()
    write(out, "app_checkers", checkers, longtable(
        (1.75, 0.55, 3.2, 0.5), ("Checker", "Section", "FINDINGS section it re-checks", "Cited"),
        [(rf"\code{{{esc(r['checker'])}}}", r["section"], tex(r["title"]), r["cited_by_paper"]) for r in checkers],
        r"Each section checker and the FINDINGS section whose numbers it re-checks. ``Cited'' says whether this paper's "
        r"body cites that section. Generated from \code{bench/gn/} and \code{kaggle/FINDINGS.md}.",
        "tab:app-checkers"))
    sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
    import check_s28

    bd = check_s28.per_scene_rows()
    write(out, "app_bd_robust", bd, longtable(
        (0.8, 1.35, 0.75, 0.65, 0.6, 0.55, 0.55, 0.65), ("Experiment", "Value", "Scene", "Registered", "PCHIP", "Difference",
                                                        "Rate-matched", "Pointwise"),
        [(esc(r["experiment"]), DELTA_TEX.get(r["value"], esc(r["value"])), r["scene"], r["registered"], r["pchip"], r["abs_diff"],
          r["rate_matched"], r["pointwise"] or "--") for r in bd],
        r"BD robustness per scene (FINDINGS section 28, post hoc, declared before it was computed): each BD value (dB, "
        r"OGC's row against the other) with the registered fit (a parabola through the three points) and with PCHIP, their "
        r"absolute difference, the configs of three whose byte gap is within 1.0\%, and the mean $\Delta$PSNR there. "
        r"Generated from \code{bench/gn/check\_s28.py}.", "tab:app-bd-robust"))
    return prereg, sessions, checkers


if __name__ == "__main__":
    p, s, c = main(sys.argv[1] if len(sys.argv) > 1 else OUT)
    print(f"appendix tables: {len(p)} registration rows, {len(s)} sessions, {len(c)} checkers")
