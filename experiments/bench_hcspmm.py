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
    ap.add_argument("--Ns", default="64,128,256,512")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-nodes", type=int, default=3_000_000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--cold-iters", type=int, default=10)
    ap.add_argument("--datasets", default="",
                    help="Optional comma-separated dataset names")
    args = ap.parse_args()

    manifest = json.loads(Path(args.datasets_file).read_text())["datasets"]
    selected = {name.strip() for name in args.datasets.split(",") if name.strip()}
    requested_Ns = [int(value) for value in args.Ns.split(",")]
    rows, preproc, notes = [], [], []
    import torch
    notes.append(f"HC-SpMM built OK against torch {torch.__version__} / CUDA {torch.version.cuda} on RTX 3090 (SM 86).")
    notes.append("Benchmarked at HC-SpMM's stable native GNN dim N=64 (forward_fixed64). Its "
                 "arbitrary-dim `forward` (N>=128) raises illegal-memory-access and is not used.")

    def status_row(entry, N, status, error):
        return {
            "dataset": entry["name"], "category": entry.get("category", "?"),
            "M": int(entry.get("M", 0)), "nnz": int(entry.get("nnz", 0)),
            "N": N, "kernel": "HC-SpMM", "status": status,
            "ms_warm": None, "preprocess_ms": None, "cold_exec_ms": None,
            "ms_cold": None, "ms_cusparse_warm": None,
            "ms_cusparse_cold": None, "speedup_vs_cusparse_warm": None,
            "speedup_vs_cusparse_cold": None, "correct": False,
            "soft_fail": False, "hard_fail": status == "INCORRECT",
            "max_error": None, "tolerance": None, "warmup": args.warmup,
            "timed_iters": args.timed, "cold_iters": args.cold_iters,
            "error": error,
        }

    for entry in manifest:
        if not entry.get("enabled", True):
            continue
        if selected and entry.get("name") not in selected:
            continue
        name = entry["name"]
        entry_Ns = {int(value) for value in entry.get("Ns", requested_Ns)}
        active_Ns = [N for N in requested_Ns if N in entry_Ns]
        for N in active_Ns:
            if N != 64:
                rows.append(status_row(
                    entry, N, "UNSUPPORTED_FEATURE_DIM",
                    "HC-SpMM artifact exposes stable fixed kernels only for N=32 and N=64"))
        if int(entry.get("M", 0)) > args.max_nodes:
            notes.append(f"{name}: SKIP (M={entry.get('M')} > {args.max_nodes}; TC preprocessing memory).")
            for N in active_Ns:
                if N == 64:
                    rows.append(status_row(entry, N, "SKIPPED_MEMORY", "M exceeds --max-nodes"))
            continue
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = args.gpu
        cmd = [sys.executable, str(WORKER), "--dataset", name,
               "--datasets-file", args.datasets_file, "--Ns", args.Ns,
               "--warmup", str(args.warmup), "--timed", str(args.timed),
               "--cold-iters", str(args.cold_iters)]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            notes.append(f"{name}: TIMEOUT after {args.timeout}s."); print(f"  [timeout] {name}"); continue
        got = False
        emitted_Ns = set()
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("error"):
                notes.append(f"{name}: {d['error']}"); continue
            d.setdefault("status", "OK" if d.get("correct") else "INCORRECT")
            d.setdefault("error", "")
            pms = d.get("preprocess_ms")
            rows.append(d); got = True; emitted_Ns.add(int(d["N"]))
            if pms is not None:
                preproc.append({"dataset": name, "M": d["M"], "nnz": d["nnz"], "preproc_ms": pms})
            print(f"  {name:<20s} N={d['N']} HC(warm)={d['ms_warm']}ms "
                  f"cus(warm)={d['ms_cusparse_warm']}ms "
                  f"({d['speedup_vs_cusparse_warm']}x) correct={d['correct']}")
        if r.returncode != 0 and not got:
            tail = (r.stderr.strip().splitlines() or ["<no stderr>"])[-1]
            notes.append(f"{name}: kernel CRASH (rc={r.returncode}): {tail[:160]}")
            for N in active_Ns:
                if N == 64 and N not in emitted_Ns:
                    rows.append(status_row(entry, N, "CRASH", tail[:320]))
            print(f"  [crash] {name}: {tail[:80]}")
        elif r.returncode != 0 and got:
            notes.append(f"{name}: partial — crashed after emitting some N (rc={r.returncode}).")

    expected = {
        (entry["name"], N)
        for entry in manifest if entry.get("enabled", True)
        if not selected or entry.get("name") in selected
        for N in requested_Ns if N in {int(value) for value in entry.get("Ns", requested_Ns)}
    }
    observed = {(row["dataset"], int(row["N"])) for row in rows}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise SystemExit(
            f"HC-SpMM status table incomplete: expected={len(expected)}, "
            f"observed={len(observed)}, missing={missing[:10]}, extra={extra[:10]}")

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
