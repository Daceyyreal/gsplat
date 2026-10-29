"""Check every number in the paper's LaTeX sources (``paper/tex/*.tex``) against the files it cites.

    .venv/Scripts/python.exe paper/tex/check_tex.py

The sources are written one sentence per line, each result sentence with a trailing comment naming its FINDINGS
sections (``s19``, ``s21 secondary 1``) and its ``paper/CLAIMS.md`` rows (``A2``, ``F1``). Amendments are cited as
``Am17 b`` so that they cannot be read as CLAIMS rows. For each line:

1. **Numbers.** Every number in the line's text (comments, ``\\label``, ``\\ref``, ``\\cite``, ``\\input`` and
   ``\\includegraphics`` arguments and lengths removed; a leading sign ignored; thousands separators kept, as FINDINGS
   prints them) must be in ``ALLOW`` or appear, at the same rounding, as a number in the line's cited sources: the
   cited FINDINGS sections and CLAIMS rows. A line that cites neither is checked against all of FINDINGS and CLAIMS.
2. **Counts.** Every "k of n" (also "k of the n") and "k/n" in the line's text must appear as written in its cited
   sources (or, with no citation, in FINDINGS or CLAIMS), in either form: so a count is checked even where k and n are
   integers that ``ALLOW`` lets pass as numbers.
3. **Citations.** A line that states a result (a number outside ``ALLOW``, a count, or one of ``RESULT_WORDS``) must
   cite at least one CLAIMS row in its trailing comment. Every cited CLAIMS row and FINDINGS section must exist. A
   figure named in a comment as ``Fig.N`` is not a CLAIMS row and is ignored for the lookup. A row is cited only as its
   own item (``cited_ids``): a ``;``-separated segment whose ``,``-separated parts are all IDs, each optionally with
   one parenthetical. Any other ID-shaped token in a comment outside quotes and parentheses fails as ambiguous.
4. **Amendment items (design lines only, in ``DESIGN_FILES`` only).** In Section 5.2's file (``sec5_finetuning.tex``;
   in any other file an amendment item is an annotation only and the line is checked as in 1-3), a comment may cite an
   item of an amendment in ``DESIGN_AMENDMENTS`` (E7's, 20) as ``Am20 f`` (item f) or ``Am20 note 1``, each as its
   own ``;``-separated item, optionally with the exact lines of ``kaggle/PREREG_GN.md`` it quotes: ``Am20 f (PREREG_GN.md:4064, 4066)``. An item that does not exist, or a cited
   line outside its item, fails. A **design line** cites amendment items and no FINDINGS section or CLAIMS row and has
   none of ``RESULT_WORDS``: its numbers and counts must appear in the cited items' text (the cited lines when given,
   else the whole item, from its heading to the next), and it needs no CLAIMS row. **Every other line is checked as
   in 1-3, with amendment items ignored as sources:** an item never verifies a number on a result line, and a result
   line whose only citations are amendment items fails.

Lines holding a ``\\TODO`` placeholder are skipped. The tables and figures come from ``paper/figs/`` (checked by
``paper/figs/check_figs.py``) and are not read here. Other integers 0-10 (j, seeds, criteria, "8 bits") are allowed
without a source.

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
PREREG = os.path.join(REPO, "kaggle", "PREREG_GN.md")

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
FIG = re.compile(r"\bFig\.\s*\d+")
QUOTED = re.compile(r'"[^"]*"')
ITEM_ID = re.compile(r"^([A-G]\d{1,2})(?:\s*\([^()]*\))?$")


def cited_ids(comment):
    """CLAIMS rows cited by a trailing comment, and ID-shaped tokens that are not citations.

    A ';'-separated segment cites rows only if every ','-separated part of it is an ID, optionally followed by one
    parenthetical ("A1 (the common limits block)"). Quoted text (FINDINGS section titles) and parentheticals are
    ignored. Any other ID-shaped token ("common E5 limits", "Am17 Note 2 C7", "as A4 and A5 state it") is returned
    as ambiguous, so it can never resolve silently to a row."""
    c = re.sub(r"\([^()]*\)", " ", QUOTED.sub(" ", FIG.sub(" ", comment)))
    ids, ambiguous = [], []
    for seg in c.split(";"):
        parts = [p.strip() for p in seg.split(",")]
        found = [CLAIM_ID.findall(p) for p in parts]
        if not any(found):
            continue
        if all(ITEM_ID.match(p) for p in parts):
            ids += [ITEM_ID.match(p).group(1) for p in parts]
        else:
            ambiguous += [x for f in found for x in f]
    return ids, ambiguous
COUNT = re.compile(r"(?<![\w.,/])(\d+)(?:\s+of\s+(?:the\s+)?|/)(\d+)(?![\w.,/])")
SECTION = re.compile(r"\bs(\d{1,2})\b")
# amendments whose items may verify a design line's numbers (E7's design, Section 5.2); a cite of any other amendment
# ("Am18 c", "Am19 c") stays an annotation, as before items were sources, so facts such as OGC's licence version still
# need their CLAIMS row
DESIGN_AMENDMENTS = {"20"}
# the files whose lines may be design lines: Section 5.2 (E7's design) lives in sec5_finetuning.tex; in any other file
# an amendment item is an annotation only and the line is checked as if items were never sources
DESIGN_FILES = {"sec5_finetuning.tex"}
AM_ITEM = re.compile(r"^\s*Am(\d{1,2}) (note \d{1,2}|[a-z])(?:\s*\(PREREG_GN\.md:([\d, ]+)\))?\s*$")


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


def amendment_items():
    """``kaggle/PREREG_GN.md``'s items by (amendment, item): each ``### x.`` under ``## Amendment N`` and each ``###
    Amendment N note K``, as its lines from its heading to the next heading, keyed by line number (1-based)."""
    lines = read(PREREG).splitlines()
    heads = [i for i, ln in enumerate(lines) if re.match(r"#{2,3} ", ln)]
    out, amend = {}, None
    for n, i in enumerate(heads):
        line = lines[i]
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        body = {j + 1: lines[j] for j in range(i, end)}
        if line.startswith("## "):
            m = re.match(r"## Amendment (\d+) \(", line)
            amend = m.group(1) if m else None
            continue
        m = re.match(r"### Amendment (\d+) note (\d+) \(", line)
        if m:
            out[(m.group(1), f"note {m.group(2)}")] = body
        elif amend and re.match(r"### [a-z]\. ", line):
            out[(amend, line[4])] = body
    return out


def cited_items(comment):
    """Amendment items cited by a trailing comment, each its own ';'-separated item ("Am20 f", "Am20 note 1",
    "Am20 f (PREREG_GN.md:4064, 4066)"), as (amendment, item, [cited line numbers])."""
    out = []
    for seg in comment.split(";"):
        m = AM_ITEM.match(seg)
        if m:
            out.append((m.group(1), m.group(2), [int(x) for x in re.findall(r"\d+", m.group(3) or "")]))
    return out


def counts(text):
    """The "k of n" / "k of the n" / "k/n" phrases of ``text`` as (k, n) pairs."""
    return COUNT.findall(text)


def tokens(text):
    # FINDINGS and CLAIMS write ratios as "2.49x"; read them as the number
    return set(numbers(re.sub(r"(\d)x\b", r"\1 x", text)))


def run(tex_dir=HERE):
    f_all, sec_text, c_all, rows = sources()
    items = amendment_items()
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
            ids, ambiguous = cited_ids(comment)
            for x in ambiguous:
                fails.append(f"{name}:{i}: '{x}' in the comment is not a citation item; write it as its own "
                             f"';'-separated item, or reword (ambiguous ID)")
            secs = [int(s) for s in SECTION.findall(comment)]
            ams = [x for x in cited_items(comment) if x[0] in DESIGN_AMENDMENTS] if name in DESIGN_FILES else []
            am_text = []
            for a, k, refs in ams:
                body = items.get((a, k))
                if body is None:
                    fails.append(f"{name}:{i}: cites Amendment {a} item {k!r}, which does not exist")
                    continue
                for r in refs:
                    if r not in body:
                        fails.append(f"{name}:{i}: PREREG_GN.md:{r} is outside Amendment {a} item {k!r} (lines "
                                     f"{min(body)}-{max(body)})")
                am_text += [body[r] for r in refs if r in body] if refs else list(body.values())
            am_names = ", ".join(f"Am{a} {k}" + (f" (PREREG_GN.md:{', '.join(map(str, refs))})" if refs else "")
                                 for a, k, refs in ams)
            # a design line: amendment items its only citations, no result word; only there may an item verify a number
            design = bool(ams) and not secs and not ids and not RESULT_WORDS.search(text)
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
            if design:
                pool, src = tokens(" ".join(am_text)), " ".join(am_text)
                where = f"its cited amendment items {am_names}"
            else:  # amendment items are not sources here
                pool = cited if (secs or ids) else all_tokens
                src = " ".join([sec_text.get(s, "") for s in secs] + [rows.get(c, "") for c in ids]) if (secs or ids) \
                    else f_all + " " + c_all
                where = "its cited sources " + ", ".join([f"s{s}" for s in secs] + ids) if (secs or ids) else \
                    "FINDINGS or CLAIMS"
            src_counts = set(counts(" ".join(src.split())))
            line_counts = counts(" ".join(text.split()))
            for k, n in line_counts:
                n_numbers += 1
                if (k, n) not in src_counts:
                    fails.append(f"{name}:{i}: the count {k} of {n} is not in {where}")
            checked = [x for x in numbers(text) if x not in ALLOW]
            for x in checked:
                n_numbers += 1
                if x not in pool:
                    fails.append(f"{name}:{i}: {x} is not in {where}")
            if (checked or line_counts or RESULT_WORDS.search(text)) and not design:
                n_result += 1
                if not ids and ams and not secs:
                    fails.append(f"{name}:{i}: states a result citing only amendment items ({am_names}); an amendment "
                                 f"item verifies design lines only, a result needs a CLAIMS row: {text.strip()[:70]!r}")
                elif not ids:
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
