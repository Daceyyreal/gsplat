#!/usr/bin/env bash
# The pytest gate before every commit (kaggle/HANDOFF.md, "Conventions and gotchas"): pytest's own exit code decides
# (pipefail), the full output is kept in a log outside the repository, GN_OGC_SRC and GN_C3DGS_SRC are set, and the log
# ends with the skip count and the name of every skipped test.
#   usage (Git Bash, from anywhere): kaggle/gate.sh <label>
#   env overrides: GN_OGC_SRC, GN_C3DGS_SRC, GATE_LOG_DIR
set -o pipefail
label="${1:?usage: kaggle/gate.sh <label>}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo" || exit 2
export GN_OGC_SRC="${GN_OGC_SRC:-F:/ogc_restore/ogc-3dgs}"
export GN_C3DGS_SRC="${GN_C3DGS_SRC:-F:/c3dgs_src_2a234af5}"
export PYTHONIOENCODING=utf-8
logdir="${GATE_LOG_DIR:-F:/pytest_logs}"
stamp="$(date +%Y%m%d_%H%M%S)"
log="$logdir/gate_${stamp}_${label}.log"
xml="$logdir/gate_${stamp}_${label}.junit.xml"
py="$repo/.venv/Scripts/python.exe"
{
  echo "gate: $label"
  echo "HEAD: $(git rev-parse HEAD)"
  echo "status:"; git status --short
  echo "GN_OGC_SRC=$GN_OGC_SRC (HEAD $(git -C "$GN_OGC_SRC" rev-parse HEAD 2>/dev/null || echo 'not a git checkout'))"
  echo "GN_C3DGS_SRC=$GN_C3DGS_SRC (HEAD $(git -C "$GN_C3DGS_SRC" rev-parse HEAD 2>/dev/null || echo 'not a git checkout'))"
  echo "----"
} > "$log"
"$py" -m pytest bench/gn/test_gn.py -q -rs --junitxml="$xml" 2>&1 | tee -a "$log"
rc=$?
"$py" -I - "$xml" >> "$log" 2>&1 <<'EOF'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
cases = list(root.iter("testcase"))
skipped = [c for c in cases if c.find("skipped") is not None]
print("----")
print(f"tests: {len(cases)}; skipped: {len(skipped)}")
for c in skipped:
    print(f"  skipped: {c.get('classname')}::{c.get('name')}: {c.find('skipped').get('message')}")
EOF
echo "exit: $rc" >> "$log"
tail -n 4 "$log"
echo "log: $log"
exit $rc
