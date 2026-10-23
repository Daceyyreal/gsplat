"""Check every number in the paper's LaTeX sources (``paper/tex/*.tex``) against the files it cites.

    .venv/Scripts/python.exe paper/tex/check_tex.py

The sources are written one sentence per line, each result sentence with a trailing comment naming its FINDINGS
sections (``s19``, ``s21 secondary 1``) and its ``paper/CLAIMS.md`` rows (``A2``, ``F1``). Amendments are cited as
``Am17 b`` so that they cannot be read as CLAIMS rows. For each line:

1. **Numbers.** Every number in the line's text (comments, ``\\label``, ``\\ref``, ``\\cite``, ``\\input`` and
   ``\\includegraphics`` arguments and lengths removed; a leading sign ignored; thousands separators kept, as FINDINGS
   prints them) must be in ``ALLOW`` or appear, at the same rounding, as a number in the line's cited sources: the
   cited FINDINGS sections and CLAIMS rows. A line that cites neither is checked against all of FINDINGS and CLAIMS.
2. **Citations.** A line that states a result (a number outside ``ALLOW``, or one of ``RESULT_WORDS``) must cite at
   least one CLAIMS row in its trailing comment. Every cited CLAIMS row and FINDINGS section must exist.

Lines holding a ``\\TODO`` placeholder are skipped. The tables and figures come from ``paper/figs/`` (checked by ``paper/figs/check_figs.py``) and are not read here.
Integers 0-10 are allowed without a source (counts such as "5 of 7", j, seeds, criteria), so they are not checked.

It prints each failure and exits 1 if there is any, 0 otherwise.
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
CLAIMS = os.path.join(REPO, "paper", "CLAIMS.md")

# structural constants: integers 0-10, the codebook size K, fine-tuning's iterations, OGC's iterations and chunk,
# the metric's width, the decimals of the criteria, the year, the dataset name "Mip-NeRF 360"
ALLOW = {str(i) for i in range(11)} | {"4,096", "5,000", "15", "25,000", "16", "2026", "360"}
RESULT_WORDS = re.compile(r"\b(passed|failed|positive on|held(?!-)|led|negative|reproduced|equalled|agreed|rose|fell|stayed|"
                          r"smaller|larger)\b")
NUM = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w,])|(?<![\w.,])\d+(?:\.\d+)?(?![\w]|,\d)")
STRIP = [
    re.compile(r"\\(?:label|ref|eqref|cite[pt]?|input|includegraphics|graphicspath|usepackage|documentclass|"
               r"bibliography|bibliographystyle|setlength)(?:\[[^\]]*\])?\{[^}]*\}(?:\{[^}]*\})?"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:\\linewidth|\\textwidth|\\columnwidth|pt|em|cm|mm|in)\b"),
]
CLAIM_ID = re.compile(r"\b([A-G]\d{1,2})\b")
SECTION = re.compile(r"\bs(\d{1,2})\b")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def numbers(text):
    return NUM.findall(text)


def split_comment(line):
    m = re.search(r"(?<!\\)%", line)
    return (line[:m.start()], line[m.start() + 1:]) if m else (line, "")


def sources():
    f = read(FINDINGS)
    secs = {}
    for m in re.finditer(r"^## (\d+)\. ", f, re.M):
        secs[int(m.group(1))] = m.start()
    starts = sorted(secs.items(), key=lambda x: x[1])
    sec_text = {n: f[s:(starts[i + 1][1] if i + 1 < len(starts) else len(f))] for i, (n, s) in enumerate(starts)}
    c = read(CLAIMS)
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\| ([A-G]\d{1,2}) \|.*$", c, re.M)}
    return f, sec_text, c, rows


def tokens(text):
    # FINDINGS and CLAIMS write ratios as "2.49x"; read them as the number
    return set(numbers(re.sub(r"(\d)x\b", r"\1 x", text)))


def run(tex_dir=HERE):
    f_all, sec_text, c_all, rows = sources()
    all_tokens = tokens(f_all) | tokens(c_all)
    fails, n_numbers, n_lines, n_result = [], 0, 0, 0
    for path in sorted(glob.glob(os.path.join(tex_dir, "*.tex"))):
        name = os.path.basename(path)
        for i, line in enumerate(read(path).splitlines(), 1):
            text, comment = split_comment(line)
            for rx in STRIP:
                text = rx.sub(" ", text)
            if not text.strip() or r"\TODO" in text:
                continue  # a placeholder states nothing
            n_lines += 1
            ids = CLAIM_ID.findall(comment)
            secs = [int(s) for s in SECTION.findall(comment)]
            for cid in ids:
                if cid not in rows:
                    fails.append(f"{name}:{i}: cites CLAIMS row {cid}, which does not exist")
            for s in secs:
                if s not in sec_text:
                    fails.append(f"{name}:{i}: cites FINDINGS section {s}, which does not exist")
            cited = set()
            for s in secs:
                cited |= tokens(sec_text.get(s, ""))
            for cid in ids:
                cited |= tokens(rows.get(cid, ""))
            pool = cited if (secs or ids) else all_tokens
            checked = [x for x in numbers(text) if x not in ALLOW]
            for x in checked:
                n_numbers += 1
                if x not in pool:
                    where = "its cited sources " + ", ".join([f"s{s}" for s in secs] + ids) if (secs or ids) else \
                        "FINDINGS or CLAIMS"
                    fails.append(f"{name}:{i}: {x} is not in {where}")
            if checked or RESULT_WORDS.search(text):
                n_result += 1
                if not ids:
                    fails.append(f"{name}:{i}: states a result without a CLAIMS row in its trailing comment: "
                                 f"{text.strip()[:90]!r}")
    return {"fails": fails, "n_numbers": n_numbers, "n_lines": n_lines, "n_result_lines": n_result}


def main():
    res = run()
    for f in res["fails"]:
        print("FAIL:", f)
    print(f"paper/tex: {res['n_lines']} lines, {res['n_result_lines']} result lines, {res['n_numbers']} numbers checked, "
          f"{len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
