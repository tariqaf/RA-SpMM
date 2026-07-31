"""
Captures the packed-format checksums (RA_PLAN_CHECKSUM=1) for the graphs used in
the byte-identity check. The FNV-1a checksum is printed by the C++ plan builders
to stderr and covers every host-side format array (tiles, offsets, permutation,
reordered CSR, fp32 rows, community ids).

Usage:  RA_PLAN_CHECKSUM=1 python experiments/verify_format_checksums.py [--tag label]
Output: one line per (graph, kernel): "<tag> <graph> <KERNEL> ... <checksum>"
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
import ra_spmm  # noqa
from ra_real_graph_eval import load_dataset  # noqa

GRAPHS = ["CiteSeer", "com-DBLP", "web-Google", "Reddit", "ogbn-products"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--graphs", default=",".join(GRAPHS))
    args = ap.parse_args()

    man = {d["name"]: d for d in json.loads((R / "paper_datasets.json").read_text())["datasets"]}
    for name in args.graphs.split(","):
        mat = load_dataset(man[name])
        M = mat["M"]
        rp = mat["rowptr"].contiguous().int()
        ci = mat["colind"].contiguous().int()
        v = mat["vals"].contiguous().float()
        print(f"### {args.tag} {name} TC_DIRECT", file=sys.stderr, flush=True)
        ra_spmm.make_tc_direct_plan(rp, ci, v, M, M, 128)
        print(f"### {args.tag} {name} COMMUNITY_TC", file=sys.stderr, flush=True)
        ra_spmm.make_community_tc_plan(rp, ci, v, M, M, 128)
    print("checksum pass complete", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
