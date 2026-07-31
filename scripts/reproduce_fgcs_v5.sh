#!/usr/bin/env bash
# Reproduce the FGCS v5 SpMM sweep and assert the expected roster shapes.
#
#   deployed roster    : 8 custom kernel/precision paths + cuSPARSE = 9 per config
#                        -> 192 x 9 = 1728 outcome rows
#   paper-audit roster : deployed + RODE_ENHANCED = 9 custom + cuSPARSE = 10
#                        -> 192 x 10 = 1920 correctness rows
#   non-cuSPARSE checks : 8 deployed custom kernels x 192 = 1536 kernel rows
#
# Usage: scripts/reproduce_fgcs_v5.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
OUT="${OUT:-fgcs_results/revision/tf32/reproduce}"
mkdir -p "$OUT"

echo "== deployed roster (8 paths + cuSPARSE) =="
$PY ra_real_graph_eval.py --portfolio deployed \
    --output "$OUT/deployed_sweep.csv" "$@"

echo "== paper-audit roster (deployed + RODE_ENHANCED) =="
$PY ra_real_graph_eval.py --portfolio paper-audit \
    --output "$OUT/paper_audit_sweep.csv" "$@"

echo "== assert roster shapes =="
$PY - "$OUT/deployed_sweep.csv" "$OUT/paper_audit_sweep.csv" <<'PY'
import csv, sys
dep, aud = sys.argv[1], sys.argv[2]
def rows(p): return list(csv.DictReader(open(p)))
d, a = rows(dep), rows(aud)
def configs(rr): return {(r["dataset"], r["N"]) for r in rr}
def kernels(rr): return {r["kernel"] for r in rr}
n_cfg = len(configs(d))
assert n_cfg == 192, f"expected 192 configs, got {n_cfg}"
# deployed: 8 custom + cuSPARSE = 9 outcomes per config
dep_k = kernels(d)
assert len([k for k in dep_k if k != "CUSPARSE"]) == 8, dep_k
assert len(d) == 192 * 9, f"deployed rows {len(d)} != 1728"
# paper-audit: 9 custom + cuSPARSE = 10 correctness rows per config
aud_k = kernels(a)
assert "RODE_ENHANCED" in aud_k and len([k for k in aud_k if k != "CUSPARSE"]) == 9, aud_k
assert len(a) == 192 * 10, f"paper-audit rows {len(a)} != 1920"
# non-cuSPARSE kernel correctness checks in deployed: 8 x 192
noncus = [r for r in d if r["kernel"] != "CUSPARSE"]
assert len(noncus) == 192 * 8, f"non-cuSPARSE kernel rows {len(noncus)} != 1536"
bad = [r for r in noncus if str(r.get("correct", "")).lower() not in ("true", "1")]
print(f"deployed 192x9={len(d)} OK | paper-audit 192x10={len(a)} OK | "
      f"non-cuSPARSE 192x8={len(noncus)} OK | correctness failures={len(bad)}")
PY
echo "== reproduce_fgcs_v5 assertions passed =="
