#!/usr/bin/env python3
"""Post-staging ncu capture: master table, roofline CSV, and drift check.

1. ncu_master_final.csv  - same schema as ncu_master.csv (profile_v2_parse),
   captured on the frozen deployed binaries (staged tile path active).
2. ncu_roofline_final.csv - roofline-ready rows: duration, DRAM GB/s,
   CUDA-core FLOPs, HMMA FLOPs (2048 FLOP per SASS HMMA inst - calibrated:
   fp16 and tf32 mma.m16n8k8 both produce identical HMMA counts), achieved
   GFLOP/s, arithmetic intensity (hardware and algorithmic 2*nnz*N).
3. Drift report vs the committed ncu_master.csv: any percent-point metric
   moving > 2 points is flagged (tab:profiling gate).
"""
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path

R = Path("/mnt/shared/development/tariq/RA-SpMM")
TF = R / "fgcs_results/revision/tf32"
NEW_DIR = TF / "ncu_final"
MASTER_OLD = TF / "ncu_master.csv"
MASTER_NEW = TF / "ncu_master_final.csv"
ROOFLINE = TF / "ncu_roofline_final.csv"

HMMA_FLOP_PER_INST = 2048.0  # mma.m16n8k8: 16*8*8 MACs * 2

# 1) master via the same parser that made ncu_master.csv
subprocess.run([sys.executable, str(R / "experiments/profile_v2_parse.py"),
                "--profdir", str(NEW_DIR), "--out", str(MASTER_NEW)], check=True)

# 2) roofline CSV from raw ncu csvs
FLOPC = {
    "ffma": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "fadd": "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul": "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "hfma": "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "hadd": "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "hmul": "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "hmma": "sm__inst_executed_pipe_tensor_op_hmma.sum",
    "dram_bytes": "dram__bytes.sum",
    "dur": "gpu__time_duration.sum",
}


def num(row, units, key):
    v = (row.get(key) or "").replace(",", "")
    if not v:
        return 0.0
    x = float(v)
    u = units.get(key, "")
    if key == FLOPC["dur"]:
        return {"nsecond": x / 1e3, "usecond": x, "msecond": x * 1e3,
                "second": x * 1e6}.get(u, x)
    if key == FLOPC["dram_bytes"]:
        return {"byte": x, "Kbyte": x * 1e3, "Mbyte": x * 1e6, "Gbyte": x * 1e9}.get(u, x)
    return x


roof = []
for meta_path in sorted(NEW_DIR.glob("*.meta.json")):
    meta = json.loads(meta_path.read_text())
    raw = NEW_DIR / meta["raw_csv"]
    if not raw.exists():
        continue
    rows = list(csv.DictReader(raw.open()))
    if not rows:
        continue
    units = rows[0]  # ncu raw csv: first row is the units row
    for r in rows[1:]:
        kname = (r.get("Kernel Name") or "").strip()
        if not kname or "elementwise" in kname or "convert_b_to_half" in kname:
            continue
        dur_us = num(r, units, FLOPC["dur"])
        if dur_us <= 0:
            continue
        dram_b = num(r, units, FLOPC["dram_bytes"])
        cuda_flops = (2 * num(r, units, FLOPC["ffma"]) + num(r, units, FLOPC["fadd"])
                      + num(r, units, FLOPC["fmul"]) + 2 * num(r, units, FLOPC["hfma"])
                      + num(r, units, FLOPC["hadd"]) + num(r, units, FLOPC["hmul"]))
        hmma_inst = num(r, units, FLOPC["hmma"])
        hmma_flops = hmma_inst * HMMA_FLOP_PER_INST
        tot = cuda_flops + hmma_flops
        algo = 2.0 * meta["nnz"] * meta["N"]
        roof.append({
            "dataset": meta["dataset"], "category": meta["category"],
            "kernel": meta["kernel"], "N": meta["N"], "M": meta["M"],
            "nnz": meta["nnz"], "launch": kname.split("(")[0].replace("<unnamed>::", ""),
            "duration_us": round(dur_us, 3),
            "dram_bytes": int(dram_b),
            "dram_gbps": round(dram_b / dur_us / 1e3, 2),
            "cuda_core_flops": int(cuda_flops),
            "hmma_inst": int(hmma_inst),
            "hmma_flops": int(hmma_flops),
            "total_flops": int(tot),
            "achieved_gflops": round(tot / dur_us / 1e3, 2),
            "algo_flops_2nnzN": int(algo),
            "algo_gflops": round(algo / dur_us / 1e3, 2),
            "ai_hw_flop_per_byte": round(tot / dram_b, 4) if dram_b else "",
            "ai_algo_flop_per_byte": round(algo / dram_b, 4) if dram_b else "",
        })

with ROOFLINE.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(roof[0].keys()))
    w.writeheader(); w.writerows(roof)
print(f"wrote {ROOFLINE} ({len(roof)} launch rows)")

# 3) drift vs committed ncu_master.csv (percent-point metrics, >2pt flag)
old = {(r["dataset"], r["kernel"], r["N"], r["role"]): r
       for r in csv.DictReader(MASTER_OLD.open())}
new = {(r["dataset"], r["kernel"], r["N"], r["role"]): r
       for r in csv.DictReader(MASTER_NEW.open())}
PP = ["sm_pct", "dram_pct", "l1_hit", "l2_hit", "tensor_pct", "occ_ach", "issue_pct"]
flags, hmma_mismatch = [], []
for k in sorted(set(old) & set(new)):
    o, n = old[k], new[k]
    for m in PP:
        try:
            d = float(n[m] or 0) - float(o[m] or 0)
        except ValueError:
            continue
        if abs(d) > 2.0:
            flags.append((k, m, round(float(o[m]), 1), round(float(n[m]), 1), round(d, 1)))
    ho, hn = float(o["hmma"] or 0), float(n["hmma"] or 0)
    if ho != hn:
        hmma_mismatch.append((k, ho, hn))
print(f"\nDRIFT >2pt on shared (dataset,kernel,N,role) rows: {len(flags)}")
for k, m, ov, nv, d in flags:
    print(f"  {'/'.join(k):55s} {m:11s} {ov:>7} -> {nv:>7}  ({d:+.1f})")
print(f"\nHMMA count mismatches: {len(hmma_mismatch)}")
for k, ho, hn in hmma_mismatch[:10]:
    print(f"  {'/'.join(k):55s} {ho:.0f} -> {hn:.0f}")
print(f"\nshared rows: {len(set(old) & set(new))} | old-only: {len(set(old)-set(new))} | new-only: {len(set(new)-set(old))}")
# launch-name changes (staged kernels expected)
for k in sorted(set(old) & set(new)):
    if old[k]["launch"] != new[k]["launch"]:
        print(f"  launch change {'/'.join(k)}: {old[k]['launch']} -> {new[k]['launch']}")
