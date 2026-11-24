"""Build the arXiv source package of the paper from one commit (prepare only: it never uploads, submits or tags).

    .venv/Scripts/python.exe paper/make_arxiv.py --commit <sha> --out F:/arxiv_pkg/gsplat_ogc_vN

Recreated 2026-11-24 from HANDOFF's records of the 2026-11-08 package builds (v1-v3), whose script lived in a session
scratchpad and was never committed. Steps:

1. **Export** ``paper/tex/*.tex`` and ``paper/tex/references.bib`` at ``--commit`` (``git show``, so the working tree does
   not matter), and every ``paper/figs/out/`` file the sources use (``\\input{../figs/out/...}`` and
   ``\\includegraphics`` under ``\\graphicspath{{../figs/out/}}``), into ``<out>/src`` with ``../figs/out/`` rewritten to
   ``figs/``. Nothing else is packaged: no ``.py``, no CLAIMS, FINDINGS or PREREG file, no OGC file.
2. **Build the uncleaned source** with ``latexmk -pdf`` in a copy (``<out>/build_unclean``), keep its ``main.bbl`` in
   ``src``, and record pages, errors, undefined references and overfull boxes from the log.
3. **Clean a copy** (``src`` -> ``src_clean``): comment-only lines removed, trailing comments cut after their ``%`` (kept,
   so line joins do not change), escaped ``\\%`` kept, ``verbatim`` environments untouched, ``references.bib`` dropped
   (arXiv uses ``main.bbl``). Build it with ``pdflatex`` x3 from the ``.bbl`` (``<out>/build_clean``).
4. **Compare** the two PDFs page by page with ``pdftotext`` (whitespace-normalized): every page's text must be equal.
5. **Tar** each source tree (``arxiv_src.tar.gz``, ``arxiv_src_clean.tar.gz``; sorted members, fixed mtime, uid/gid 0,
   no ``./`` prefix) and write their sha256 to ``manifest.json`` / ``manifest_clean.json`` with the checks:
   e-mail addresses (only ``ALLOWED_EMAILS``), the sensitive terms of the 2026-11-08 redaction (none), OGC code (no
   ``.py`` and none of ``OGC_MARKERS``), ``../`` paths (none), the PolyForm licence note (present), comment text left in
   the cleaned sources (none).
6. **Write** ``INSTRUCTIONS.txt``: the five preview steps (upload the cleaned tarball after checking its sha256, compare
   arXiv's compiled PDF with ``build_clean/main.pdf``, note any endorsement prompt, fill the metadata, save unsubmitted).

It exits 1 if any check fails. ``clean_tex`` and ``check_text`` are pure functions, tested in ``bench/gn/test_gn.py``.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ALLOWED_EMAILS = {"taifurrahman0606@gmail.com"}  # the author's address, as the paper prints it
SENSITIVE = ("patent", "correspondence", "private", "redact", "commercial use", "by email", "the author states",
             "stated that the method")
OGC_MARKERS = ("def gram_kmeans", "Required Notice", "import torch")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def git(*a, text=True):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=text, check=True).stdout


def clean_tex(s: str) -> str:
    """Remove comment-only lines and cut trailing comments after their ``%``; ``\\%`` and verbatim are kept."""
    out, verb = [], False
    for line in s.replace("\r\n", "\n").split("\n"):
        if re.match(r"\s*\\begin\{verbatim\*?\}", line):
            verb = True
        if verb:
            out.append(line)
            if re.match(r"\s*\\end\{verbatim\*?\}", line):
                verb = False
            continue
        m = re.search(r"(?<!\\)(?:\\\\)*%", line)
        if m is None:
            out.append(line)
            continue
        cut = m.end() - 1  # the index of the unescaped %
        head = line[:cut]
        if head.strip() == "":
            continue  # a comment-only line
        out.append(head + "%")
    return "\n".join(out)


def comment_text_left(s: str) -> int:
    """The number of characters after an unescaped % (outside verbatim): 0 for a cleaned file."""
    n = 0
    for line in clean_tex_keep(s):
        m = re.search(r"(?<!\\)(?:\\\\)*%", line)
        if m:
            n += len(line) - m.end()
    return n


def clean_tex_keep(s):
    verb = False
    for line in s.replace("\r\n", "\n").split("\n"):
        if re.match(r"\s*\\begin\{verbatim\*?\}", line):
            verb = True
        if not verb:
            yield line
        if verb and re.match(r"\s*\\end\{verbatim\*?\}", line):
            verb = False


def check_text(files: dict) -> dict:
    """The content checks over ``{path: text}``."""
    emails = sorted({e for t in files.values() for e in EMAIL.findall(t)})
    low = {p: t.lower() for p, t in files.items()}
    return {
        "emails": emails,
        "emails_ok": set(emails) <= ALLOWED_EMAILS,
        "sensitive_hits": sorted({(p, w) for p, t in low.items() for w in SENSITIVE if w in t}),
        "ogc_code_files": sorted(p for p, t in files.items() if p.endswith(".py") or any(m in t for m in OGC_MARKERS)),
        "parent_paths": sum(t.count("../") for t in files.values()),
        "licence_note": any("PolyForm" in t for t in files.values()),
    }


def export(commit, src):
    names = git("ls-tree", "-r", "--name-only", commit, "paper/tex/", "paper/figs/out/").split()
    tex = [n for n in names if re.match(r"paper/tex/[^/]+\.(tex|bib)$", n)]
    os.makedirs(os.path.join(src, "figs"), exist_ok=True)
    used = set()
    for n in tex:
        body = git("show", f"{commit}:{n}")
        used |= {f"{m}.tex" if not os.path.splitext(m)[1] else m for m in re.findall(r"\\input\{\.\./figs/out/([^}]+)\}", body)}
        used |= {m for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)}
        body = body.replace("{../figs/out/}", "{figs/}").replace("../figs/out/", "figs/")
        open(os.path.join(src, os.path.basename(n)), "w", encoding="utf-8", newline="\n").write(body.replace("\r\n", "\n"))
    out_files = {os.path.basename(n): n for n in names if n.startswith("paper/figs/out/")}
    missing = []
    for u in sorted(used):
        cands = [u] if os.path.splitext(u)[1] else [u + ".pdf", u + ".png", u + ".tex"]
        hit = next((c for c in cands if c in out_files), None)
        if hit is None:
            missing.append(u)
            continue
        data = subprocess.run(["git", "-C", REPO, "show", f"{commit}:{out_files[hit]}"], capture_output=True, check=True).stdout
        if hit.endswith(".tex"):
            data = data.replace(b"\r\n", b"\n")
        open(os.path.join(src, "figs", hit), "wb").write(data)
    return missing


def run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", env=env)


def log_checks(logpath):
    t = open(logpath, encoding="utf-8", errors="replace").read()
    pages = re.findall(r"Output written on \S+ \((\d+) pages?", t)
    return {"pages": int(pages[-1]) if pages else 0, "errors": len(re.findall(r"^! ", t, re.M)),
            "undefined": len(re.findall(r"undefined", t, re.I)), "overfull": len(re.findall(r"Overfull \\hbox", t))}


def build_unclean(src, build):
    shutil.copytree(src, build)
    r = run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], build)
    shutil.copy(os.path.join(build, "main.bbl"), os.path.join(src, "main.bbl"))
    return r.returncode, log_checks(os.path.join(build, "main.log"))


def build_clean(src_clean, build):
    shutil.copytree(src_clean, build)
    rc = 0
    for _ in range(3):
        rc |= run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], build).returncode
    return rc, log_checks(os.path.join(build, "main.log"))


def page_texts(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", run(["pdfinfo", pdf], os.path.dirname(pdf)).stdout).group(1))
    out = []
    for i in range(1, n + 1):
        t = run(["pdftotext", "-f", str(i), "-l", str(i), pdf, "-"], os.path.dirname(pdf)).stdout
        out.append(re.sub(r"\s+", " ", t).strip())
    return out


def tar_tree(root, dest):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for dirpath, dirs, files in sorted(os.walk(root)):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(dirpath, f)
                arc = os.path.relpath(p, root).replace(os.sep, "/")
                ti = tf.gettarinfo(p, arc)
                ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = 0, 0, 0, "", "", 0o644
                with open(p, "rb") as fh:
                    tf.addfile(ti, fh)
    raw = buf.getvalue()
    with open(dest, "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(raw)
    return hashlib.sha256(open(dest, "rb").read()).hexdigest()


def tree_texts(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith((".tex", ".bbl", ".bib", ".py")):
                p = os.path.join(dirpath, f)
                out[os.path.relpath(p, root).replace(os.sep, "/")] = open(p, encoding="utf-8", errors="replace").read()
    return out


INSTRUCTIONS = """arXiv upload of the cleaned package (prepare and preview only; do not submit). Source commit: {commit}.
1. On arxiv.org start a new submission and upload arxiv_src_clean.tar.gz from this folder (check its sha256 first: sha256sum arxiv_src_clean.tar.gz must print {sha}).
2. After arXiv's own TeX Live run, open the compiled PDF: {pages} pages, every figure and table present, no "??" reference, references listed; compare it with build_clean/main.pdf.
3. Note whether arXiv shows an endorsement prompt for the chosen primary category; if it does, stop there and record it.
4. Fill the metadata (title, author, abstract, licence) as in the PDF, but do not press Submit.
5. Save the submission unsubmitted (it stays in "incomplete"), and report the preview's result back before any tag.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=REPO, help="the repository (default: this file's parent's parent)")
    a = ap.parse_args(argv)
    globals()["REPO"] = os.path.abspath(a.repo)
    commit = git("rev-parse", a.commit).strip()
    if os.path.exists(a.out):
        print(f"refusing: {a.out} exists")
        return 2
    src, src_clean = os.path.join(a.out, "src"), os.path.join(a.out, "src_clean")
    missing = export(commit, src)
    rc_u, chk_u = build_unclean(src, os.path.join(a.out, "build_unclean"))
    shutil.copytree(src, src_clean)
    removed_lines = removed_bytes = 0
    for dirpath, _, files in os.walk(src_clean):
        for f in files:
            p = os.path.join(dirpath, f)
            if f == "references.bib":
                os.remove(p)
                continue
            if f.endswith(".tex"):
                before = open(p, encoding="utf-8").read()
                after = clean_tex(before)
                removed_lines += len(before.split("\n")) - len(after.split("\n"))
                removed_bytes += len(before.encode()) - len(after.encode())
                open(p, "w", encoding="utf-8", newline="\n").write(after)
    rc_c, chk_c = build_clean(src_clean, os.path.join(a.out, "build_clean"))
    pu = page_texts(os.path.join(a.out, "build_unclean", "main.pdf"))
    pc = page_texts(os.path.join(a.out, "build_clean", "main.pdf"))
    same = [i + 1 for i in range(min(len(pu), len(pc))) if pu[i] == pc[i]]
    sha_u = tar_tree(src, os.path.join(a.out, "arxiv_src.tar.gz"))
    sha_c = tar_tree(src_clean, os.path.join(a.out, "arxiv_src_clean.tar.gz"))
    tu, tc = tree_texts(src), tree_texts(src_clean)
    cu, cc = check_text(tu), check_text(tc)
    left = sum(comment_text_left(t) for p, t in tc.items() if p.endswith(".tex"))
    n_files = lambda r: sum(len(f) for _, _, f in os.walk(r))
    man_u = {"commit": commit, "sha256": sha_u, "files": n_files(src), "missing_files": missing, "latexmk_rc": rc_u,
             **chk_u, **{k: v for k, v in cu.items() if k != "sensitive_hits"}, "sensitive_hits": [list(x) for x in cu["sensitive_hits"]]}
    man_c = {"commit": commit, "sha256": sha_c, "files": n_files(src_clean), "pdflatex_rc": rc_c, **chk_c,
             "lines_removed": removed_lines, "bytes_removed": removed_bytes, "comment_text_left": left,
             "pages_text_identical": len(same), "pages_unclean": len(pu), "pages_clean": len(pc),
             **{k: v for k, v in cc.items() if k != "sensitive_hits"}, "sensitive_hits": [list(x) for x in cc["sensitive_hits"]]}
    json.dump(man_u, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    json.dump(man_c, open(os.path.join(a.out, "manifest_clean.json"), "w"), indent=1)
    open(os.path.join(a.out, "INSTRUCTIONS.txt"), "w", newline="\n").write(
        INSTRUCTIONS.format(commit=commit, sha=sha_c, pages=chk_c["pages"]))
    ok = (not missing and rc_u == 0 and rc_c == 0 and chk_u["errors"] == chk_c["errors"] == 0
          and chk_c["undefined"] == 0 and chk_c["overfull"] == 0 and chk_u["pages"] == chk_c["pages"] > 0
          and len(same) == len(pu) == len(pc) and cu["emails_ok"] and cc["emails_ok"] and not cc["sensitive_hits"]
          and not cc["ogc_code_files"] and cc["parent_paths"] == 0 and cc["licence_note"] and left == 0)
    print(json.dumps({"commit": commit, "sha256_clean": sha_c, "sha256_unclean": sha_u, "pages": chk_c["pages"],
                      "pages_text_identical": len(same), "ok": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
