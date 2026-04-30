"""
Parity test for the round-2 router recalibration. Asserts that the Python
simple_router() and the C++ make_router_plan() produce identical kernel
selections for every (graph, N) pair in paper_combined_datasets.json.

PASS criterion: PARITY OK 192/192 across all real + synthetic graphs.
Any disagreement is reported with the feature triple so the offending
rule can be located.

Usage:
    python ra_router_parity_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch  # noqa: F401  -- loads libtorch before ra_spmm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ra_spmm  # noqa: E402

from ra_router_eval import simple_router  # noqa: E402
from ra_real_graph_eval import load_dataset  # noqa: E402


N_VALUES = [64, 128, 256, 512]


def main():
    manifest_path = REPO_ROOT / "fgcs_results" / "paper_combined_datasets.json"
    if not manifest_path.exists():
        manifest_path = REPO_ROOT / "paper_datasets.json"
    manifest = json.loads(manifest_path.read_text())["datasets"]

    n_total = 0
    n_match = 0
    mismatches = []

    for entry in manifest:
        path = REPO_ROOT / entry["path"]
        if not path.exists():
            continue
        try:
            ds = load_dataset(entry)
        except Exception as ex:
            print(f"[skip] {entry['name']}: {ex}")
            continue
        if ds is None:
            continue
        rowptr = ds["rowptr"]
        colind = ds["colind"]
        vals = ds["vals"]
        M = ds["M"]
        nnz = int(rowptr[-1].item())
        deg = (rowptr[1:] - rowptr[:-1]).float()
        d_bar = nnz / max(1, M)
        cv_d = float((deg.std() / deg.mean()).item()) if d_bar > 0 else 0.0

        # Restrict to entry-allowed N values
        Ns = [int(n) for n in entry.get("Ns", N_VALUES)
              if int(n) <= int(entry.get("max_N", 512))]

        for N in Ns:
            py_pick = simple_router(d_bar, cv_d, M, N, nnz)
            plan = ra_spmm.make_router_plan(
                rowptr.cpu(), colind.cpu(), vals.cpu(), M, M, int(N), "MAIN")
            cpp_pick = str(plan["chosen_path"])
            n_total += 1
            if py_pick == cpp_pick:
                n_match += 1
            else:
                mismatches.append({
                    "dataset": entry["name"], "category": entry.get("category", "?"),
                    "M": M, "d_bar": round(d_bar, 3), "cv_d": round(cv_d, 3),
                    "N": N, "py": py_pick, "cpp": cpp_pick,
                })

    print(f"PARITY {'OK' if not mismatches else 'FAIL'} "
          f"{n_match}/{n_total}")
    if mismatches:
        print("\nMismatches:")
        for m in mismatches:
            print(f"  {m['dataset']:<32s} N={m['N']:>4d}  cat={m['category']:<22s}  "
                  f"M={m['M']:>9d}  d={m['d_bar']:>6.2f}  CV={m['cv_d']:>5.2f}  "
                  f"py={m['py']:<18s} cpp={m['cpp']:<18s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
