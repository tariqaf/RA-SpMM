"""
HC-SpMM baseline driver.

Runs one isolated subprocess per graph (bench_hcspmm_worker.py) so a kernel crash
on one graph does not corrupt the CUDA context for the others. Benchmarks HC-SpMM
at its stable native GNN embedding dim N=64 (forward_fixed64), vs cuSPARSE and vs
our router's chosen kernel. Graphs where HC-SpMM crashes/times out are recorded in
a BUILD_NOTE.

Outputs:
  fgcs_results/revision/baselines/hcspmm.csv
  fgcs_results/revision/baselines/hcspmm_preproc.csv
  fgcs_results/revision/baselines/hcspmm_BUILD_NOTE.txt
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
WORKER = R / "experiments" / "bench_hcspmm_worker.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(R / "paper_datasets.json"))
    ap.add_argument("--out", default=str(R / "fgcs_results/revision/baselines/hcspmm.csv"))
    ap.add_argument("--preproc-out", default=str(R / "fgcs_results/revision/baselines/hcspmm_preproc.csv"))
    ap.add_argument("--note-out", default=str(R / "fgcs_results/revision/baselines/hcspmm_BUILD_NOTE.txt"))
    ap.add_argument("--Ns", default="64")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-nodes", type=int, default=3_000_000)
    args = ap.parse_args()

    manifest = json.loads(Path(args.datasets_file).read_text())["datasets"]
    rows, preproc, notes = [], [], []
    import torch
    notes.append(f"HC-SpMM built OK against torch {torch.__version__} / CUDA {torch.version.cuda} on RTX 3090 (SM 86).")
    notes.append("Benchmarked at HC-SpMM's stable native GNN dim N=64 (forward_fixed64). Its "
                 "arbitrary-dim `forward` (N>=128) raises illegal-memory-access and is not used.")

    for entry in manifest:
        if not entry.get("enabled", True):
            continue
        name = entry["name"]
        if int(entry.get("M", 0)) > args.max_nodes:
            notes.append(f"{name}: SKIP (M={entry.get('M')} > {args.max_nodes}; TC preprocessing memory).")
            continue
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = args.gpu
        cmd = [sys.executable, str(WORKER), "--dataset", name,
               "--datasets-file", args.datasets_file, "--Ns", args.Ns]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            notes.append(f"{name}: TIMEOUT after {args.timeout}s."); print(f"  [timeout] {name}"); continue
        got = False
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in d:
                notes.append(f"{name}: {d['error']}"); continue
            pms = d.pop("preproc_ms", None)
            rows.append(d); got = True
            if pms is not None:
                preproc.append({"dataset": name, "M": d["M"], "nnz": d["nnz"], "preproc_ms": pms})
            print(f"  {name:<20s} N={d['N']} HC={d['ms']}ms cus={d['ms_cusparse']}ms "
                  f"({d['speedup_vs_cusparse']}x) correct={d['correct']}")
        if r.returncode != 0 and not got:
            tail = (r.stderr.strip().splitlines() or ["<no stderr>"])[-1]
            notes.append(f"{name}: kernel CRASH (rc={r.returncode}): {tail[:160]}")
            print(f"  [crash] {name}: {tail[:80]}")
        elif r.returncode != 0 and got:
            notes.append(f"{name}: partial — crashed after emitting some N (rc={r.returncode}).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out} ({len(rows)} rows)")
    if preproc:
        with open(args.preproc_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(preproc[0].keys())); w.writeheader(); w.writerows(preproc)
        print(f"Wrote {args.preproc_out} ({len(preproc)} rows)")
    Path(args.note_out).write_text("\n".join(notes) + "\n")
    print(f"Wrote {args.note_out} ({len(notes)} notes)")


if __name__ == "__main__":
    main()
