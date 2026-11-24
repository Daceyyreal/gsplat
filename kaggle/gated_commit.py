"""Commit the staged changes only after the pytest gate passes (kaggle/HANDOFF.md, "Conventions and gotchas").

    .venv/Scripts/python.exe kaggle/gated_commit.py <label> (-F <message file> | -m <message>)

Every commit goes through this script, HANDOFF-only commits included. It refuses to commit unless:

1. something is staged, and nothing else differs from HEAD: no unstaged change to a tracked file and no untracked file
   (the gate tests the working tree, the commit records the index, so the two must be the same content);
2. ``kaggle/gate.sh <label>`` exits 0, run with ``TMP``/``TEMP`` and pytest's ``--basetemp`` (through
   ``PYTEST_ADDOPTS``) on F: (C: is full; a full C: once made ``tmp_path`` setup fail);
3. the gate's log ends as ``gate.sh`` writes it: the pytest summary line has passes and no "failed" or "error", the
   ``tests: N; skipped: M`` line is there with one ``skipped:`` line per skipped test, and ``exit: 0``;
4. the index and the working tree are the same after the gate as before it.

The message's ``GATE_LOG`` (or, if absent, a new last line) becomes ``Gate: N passed, M skipped (<log name>)``.
``GATE_CMD`` and ``GATE_TMP`` (environment) replace the gate command and the temp root; the test uses them.
Exit codes: 0 committed; 2 refused before the gate; 3 the gate failed; 4 the log did not show a clean pass;
5 the tree changed during the gate; 6 ``git commit`` failed.
"""
import os
import re
import shlex
import subprocess
import sys
import time


def git(*a, check=True):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=check).stdout


def tree_state():
    """The staged tree id and the porcelain status (everything that could differ between index, tree and HEAD)."""
    return git("write-tree").strip(), git("status", "--porcelain", "--untracked-files=all")


def refuse_reason(status: str) -> str:
    """Why the working tree is not committable as staged, or '' if it is."""
    lines = [l for l in status.split("\n") if l]
    if not lines:
        return "nothing is staged"
    bad = [l for l in lines if l.startswith("??") or l[1] != " "]
    if bad:
        return "unstaged or untracked changes (the gate would test what the commit does not record): " + "; ".join(bad[:8])
    return ""


def parse_log(text: str):
    """(ok, passed, skipped, problems) from a gate.sh log."""
    problems = []
    summ = [l for l in text.split("\n") if re.match(r"^=*\s*\d+ (passed|failed)|^=*\s*\d+ error", l.strip())]
    s = summ[-1] if summ else ""
    m = re.search(r"(\d+) passed", s)
    passed = int(m.group(1)) if m else 0
    if not summ:
        problems.append("no pytest summary line")
    if re.search(r"\d+ failed", s):
        problems.append(f"summary has failures: {s.strip()}")
    if re.search(r"\d+ errors?\b", s):
        problems.append(f"summary has errors: {s.strip()}")
    if passed == 0:
        problems.append("no test passed")
    t = re.search(r"^tests: (\d+); skipped: (\d+)$", text, re.M)
    skipped = int(t.group(2)) if t else -1
    if not t:
        problems.append("no 'tests: N; skipped: M' line (the skip list was not printed)")
    elif len(re.findall(r"^  skipped: ", text, re.M)) != skipped:
        problems.append("the skip list does not name every skipped test")
    if not re.search(r"^exit: 0$", text, re.M):
        problems.append("the log does not end with 'exit: 0'")
    return (not problems), passed, skipped, problems


def message_with_gate(msg: str, passed: int, skipped: int, logname: str) -> str:
    g = f"Gate: {passed} passed, {skipped} skipped ({logname})"
    return msg.replace("GATE_LOG", g) if "GATE_LOG" in msg else msg.rstrip("\n") + "\n\n" + g + "\n"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[1] not in ("-F", "-m"):
        print(__doc__)
        return 2
    label, how, val = argv
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        print("refused: the label must be [A-Za-z0-9_.-]+")
        return 2
    msg = open(val, encoding="utf-8").read() if how == "-F" else val
    repo = git("rev-parse", "--show-toplevel").strip()
    os.chdir(repo)
    before = tree_state()
    why = refuse_reason(before[1])
    if why:
        print(f"refused: {why}")
        return 2
    tmp_root = os.environ.get("GATE_TMP", "F:/pytest_tmp").replace("\\", "/").rstrip("/")
    # forward slashes only: pytest splits PYTEST_ADDOPTS shell-style, and a backslash there is an escape (it was lost once)
    base = f"{tmp_root}/{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(tmp_root, exist_ok=True)
    env = dict(os.environ, TMP=tmp_root, TEMP=tmp_root,
               PYTEST_ADDOPTS=(os.environ.get("PYTEST_ADDOPTS", "") + f" --basetemp={base}").strip())
    cmd = shlex.split(os.environ.get("GATE_CMD", "bash kaggle/gate.sh")) + [label]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    m = re.findall(r"^log: (.+)$", out, re.M)
    if r.returncode != 0:
        print(f"gate failed (exit {r.returncode}); NOT committed. {out[-600:]}")
        return 3
    if not m or not os.path.exists(m[-1].strip()):
        print("refused: the gate printed no log path")
        return 4
    log = m[-1].strip()
    ok, passed, skipped, problems = parse_log(open(log, encoding="utf-8", errors="replace").read())
    if not ok:
        print(f"refused: the log does not show a clean pass ({os.path.basename(log)}): " + "; ".join(problems))
        return 4
    after = tree_state()
    if after != before:
        print("refused: the index or the working tree changed while the gate ran")
        return 5
    full = message_with_gate(msg, passed, skipped, os.path.basename(log))
    c = subprocess.run(["git", "commit", "-q", "-F", "-"], input=full, text=True, capture_output=True)
    if c.returncode != 0:
        print(f"git commit failed: {c.stderr}")
        return 6
    print(f"committed {git('rev-parse', '--short', 'HEAD').strip()}: {passed} passed, {skipped} skipped ({os.path.basename(log)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
