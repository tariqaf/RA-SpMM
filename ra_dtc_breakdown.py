#!/usr/bin/env python3
"""
ra_dtc_breakdown.py - Sweep reordered DTC over the paper's real graph set.

Per graph:
  1. run reorder once
  2. record reorder_ms
  3. save reordered CSR

Per (graph, N):
  1. preprocess on reordered CSR -> preprocess_ms
  2. variant selection / tuning -> selection_variant_ms
  3. measure best kernel time with warmup 3, timed 20 -> mean_kernel_ms
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dtc_reorder_utils import REORDER_METHOD, REORDER_METHOD_NOTE, DEFAULT_CACHE_DIR, reorder_once
from ra_real_graph_eval import load_dataset


REPO_ROOT = Path(__file__).resolve().parent
CHILD = REPO_ROOT / "ra_dtc_single.py"


def load_manifest(path: str) -> List[Dict[str, object]]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "datasets" in raw:
        return raw["datasets"]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Unexpected manifest shape: {type(raw).__name__}")


def active_ns(entry: Dict[str, object], requested: List[int]) -> List[int]:
    entry_ns = [int(x) for x in entry.get("Ns", requested)]
    max_n = int(entry.get("max_N", max(requested) if requested else 0))
    return [n for n in requested if n in entry_ns and n <= max_n]


def extract_json_payload(stdout: str) -> Tuple[Optional[Dict[str, object]], str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload, ""
    detail = lines[-1] if lines else "empty stdout"
    return None, detail


def run_child(
    entry_json: str,
    reorder_info: Dict[str, object],
    entry: Dict[str, object],
    n: int,
    timeout_s: int,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
    atol: float,
) -> Tuple[Optional[Dict[str, object]], Optional[str], str]:
    cmd = [
        sys.executable,
        str(CHILD),
        "--dataset_json_entry", entry_json,
        "--reordered_npz", str(reorder_info["reordered_npz"]),
        "--reorder_perm_npz", str(reorder_info["reorder_perm_npz"]),
        "--dataset_name", str(entry.get("name", "")),
        "--category", str(entry.get("category", "")),
        "--reorder_method", str(reorder_info["reorder_method"]),
        "--reorder_version", str(reorder_info["reorder_version"]),
        "--reorder_ms", str(float(reorder_info["reorder_ms"])),
        "--N", str(int(n)),
        "--warmup_iters", str(int(warmup_iters)),
        "--timed_iters", str(int(timed_iters)),
        "--seed", str(int(seed)),
        "--atol", str(float(atol)),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)) if timeout_s > 0 else None,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return None, "timeout", f"timed out after {timeout_s}s"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr.splitlines()[-1] if stderr else (stdout.splitlines()[-1] if stdout else "")
        return None, "nonzero_returncode", f"returncode={proc.returncode} detail={detail}"

    payload, detail = extract_json_payload(stdout)
    if payload is None:
        return None, "json_parse_failed", detail
    if "error" in payload:
        return None, "dtc_reported_error", str(payload["error"])
    return payload, None, ""


def failure_row(entry: Dict[str, object], n: int, reorder_info: Dict[str, object], failure_class: str, detail: str) -> Dict[str, object]:
    return {
        "dataset": str(entry.get("name", "")),
        "category": str(entry.get("category", "")),
        "M": int(reorder_info.get("M", entry.get("M", 0))),
        "nnz": int(reorder_info.get("nnz", entry.get("nnz", 0))),
        "N": int(n),
        "reorder_method": str(reorder_info.get("reorder_method", REORDER_METHOD)),
        "reorder_version": str(reorder_info.get("reorder_version", "unknown")),
        "reorder_ms": round(float(reorder_info.get("reorder_ms", 0.0)), 3) if reorder_info else "",
        "preprocess_ms": "",
        "selection_variant_ms": "",
        "mean_kernel_ms": "",
        "std_kernel_ms": "",
        "end_to_end_ms": "",
        "best_variant": "",
        "variant_count": "",
        "correct": "",
        "max_error": "",
        "failure_class": failure_class,
        "failure_detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_json", default="paper_datasets.json")
    parser.add_argument("--n_values", default="64,128,256,512")
    parser.add_argument("--output", default="ra_dtc_breakdown_reordered.csv")
    parser.add_argument("--category", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--reorder_threshold", type=int, default=16)
    parser.add_argument("--max_rows", type=int, default=500000)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--reorder_timeout", type=int, default=0,
                        help="Seconds for one graph reorder; 0 means no timeout")
    parser.add_argument("--per_point_timeout", type=int, default=600,
                        help="Seconds for one (graph,N) preprocess+tuning point; 0 means no timeout")
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--timed_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    datasets = load_manifest(args.datasets_json)
    requested_ns = [int(x) for x in args.n_values.split(",") if x.strip()]
    if args.category:
        datasets = [d for d in datasets if d.get("category") == args.category]
    if args.dataset:
        datasets = [d for d in datasets if d.get("name") == args.dataset]
    datasets = [d for d in datasets if d.get("enabled", True)]
    if args.max_rows > 0:
        datasets = [d for d in datasets if int(d.get("M", 0)) <= args.max_rows]

    rows: List[Dict[str, object]] = []
    print(f"Datasets: {len(datasets)}")
    print(f"Requested Ns: {requested_ns}")
    print(f"Reorder method: {REORDER_METHOD_NOTE}")
    print(f"Reorder threshold: {args.reorder_threshold}")
    print(f"Timed iters: {args.timed_iters}")
    if args.max_rows > 0:
        print(f"Max rows filter: M <= {args.max_rows}")
    print("-" * 100)

    with tempfile.TemporaryDirectory(prefix="dtc_reorder_entry_") as entry_tmp:
        for entry in datasets:
            ns = active_ns(entry, requested_ns)
            name = str(entry["name"])
            print(f"[{name}] category={entry.get('category', '?')} active_Ns={ns}")
            if not ns:
                continue

            entry_json = str(Path(entry_tmp) / f"{name.replace('/', '_').replace(' ', '_')}.json")
            with open(entry_json, "w") as f:
                json.dump(entry, f)

            try:
                data = load_dataset(entry)
                if data is None:
                    raise RuntimeError("dataset_load_failed")
                reorder_info = reorder_once(
                    entry,
                    data,
                    args.reorder_threshold,
                    cache_dir=args.cache_dir,
                    python_exe=sys.executable,
                    timeout_s=args.reorder_timeout if args.reorder_timeout > 0 else None,
                )
                cache_note = "cache-hit" if reorder_info.get("cache_hit") else "new"
                print(
                    f"  reorder: {float(reorder_info['reorder_ms']):.3f}ms "
                    f"({cache_note}) -> {Path(str(reorder_info['reordered_npz'])).name}"
                )
            except Exception as e:
                detail = str(e)
                print(f"  reorder FAILED: {detail}")
                for n in ns:
                    rows.append(failure_row(entry, n, {}, "reorder_failed", detail))
                continue

            for n in ns:
                payload, failure_class, detail = run_child(
                    entry_json, reorder_info, entry, n, args.per_point_timeout,
                    args.warmup_iters, args.timed_iters, args.seed, args.atol
                )
                if payload is None:
                    row = failure_row(entry, n, reorder_info, failure_class or "unknown_failure", detail)
                    print(f"  N={n}: FAILED ({failure_class}) {detail}")
                else:
                    row = {
                        "dataset": payload["dataset"],
                        "category": payload["category"],
                        "M": payload["M"],
                        "nnz": payload["nnz"],
                        "N": payload["N"],
                        "reorder_method": payload["reorder_method"],
                        "reorder_version": payload["reorder_version"],
                        "reorder_ms": round(float(payload["reorder_ms"]), 3),
                        "preprocess_ms": round(float(payload["preprocess_ms"]), 3),
                        "selection_variant_ms": round(float(payload["selection_variant_ms"]), 3),
                        "mean_kernel_ms": round(float(payload["mean_kernel_ms"]), 3),
                        "std_kernel_ms": round(float(payload["std_kernel_ms"]), 3),
                        "end_to_end_ms": round(float(payload["end_to_end_ms"]), 3),
                        "best_variant": payload["dtc_variant"],
                        "variant_count": payload["variant_count"],
                        "correct": payload["correct"],
                        "max_error": payload["max_error"],
                        "failure_class": "",
                        "failure_detail": "",
                    }
                    print(
                        f"  N={n}: preprocess={row['preprocess_ms']}ms  "
                        f"select={row['selection_variant_ms']}ms  "
                        f"kernel={row['mean_kernel_ms']}ms±{row['std_kernel_ms']}  "
                        f"e2e={row['end_to_end_ms']}ms  "
                        f"correct={row['correct']}  "
                        f"variant={row['best_variant']}"
                    )
                rows.append(row)

    if not rows:
        print("No rows produced.")
        return

    fieldnames = [
        "dataset", "category", "M", "nnz", "N", "reorder_method", "reorder_version",
        "reorder_ms", "preprocess_ms", "selection_variant_ms", "mean_kernel_ms",
        "std_kernel_ms", "end_to_end_ms", "best_variant", "variant_count",
        "correct", "max_error", "failure_class", "failure_detail",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    note_path = str(Path(args.output).with_suffix(".methodology.txt"))
    with open(note_path, "w") as f:
        f.write(REORDER_METHOD_NOTE + "\n")
        f.write(f"Reorder version: auto-detected from external repo git SHA when available.\n")
        if args.max_rows > 0:
            f.write(
                f"This sweep restricted reordered DTC to datasets with M <= {args.max_rows}, "
                "which is 18 enabled paper graphs in the current manifest.\n"
            )
        f.write("Per graph: reorder once, record reorder_ms, save reordered CSR.\n")
        f.write(
            f"Per (graph, N): preprocess on reordered CSR, run variant selection/tuning, "
            f"then measure best mean kernel time with warmup={args.warmup_iters} and timed_iters={args.timed_iters}.\n"
        )
        f.write(
            "Correctness uses permute-in / inverse-permute-out against cuSPARSE on the original CSR.\n"
        )
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
