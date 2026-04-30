"""
ra_external_baselines.py — Run reordered DTC-SpMM and PyG torch_sparse as
external baselines on the same (graph, N) pairs as
router_real_results_after_tightening.csv.
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dtc_reorder_utils import DEFAULT_CACHE_DIR, REORDER_METHOD, REORDER_METHOD_NOTE, reorder_once
from pyg_baseline import is_pyg_available, build_pyg_sparse, time_pyg_spmm
from ra_real_graph_eval import DATASET_FILE, load_dataset, measure_ms

try:
    import ra_spmm
except ImportError:
    print("ERROR: ra_spmm not found. Build first.", file=sys.stderr)
    sys.exit(1)

try:
    from dtc_baseline import is_dtc_available
    DTC_LOADED = True
except Exception as e:
    print(f"WARNING: dtc_baseline import failed: {e}", file=sys.stderr)
    DTC_LOADED = False


WARMUP = 20
TIMED = 50
_DTC_CHILD = REPO_ROOT / "ra_dtc_single.py"


def bench_cusparse(rowptr, colind, vals, B) -> float:
    return measure_ms(lambda: ra_spmm.spmm_cusparse(rowptr, colind, vals, B), warmup=WARMUP, iters=TIMED)


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


def bench_dtc_best_subprocess(
    entry_json: str,
    reorder_info: Dict[str, object],
    entry: Dict[str, object],
    N: int,
    timeout_s: int,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
    atol: float,
) -> Tuple[Optional[Dict[str, object]], Optional[str], str]:
    if not DTC_LOADED or not is_dtc_available():
        return None, "dtc_unavailable", "dtc unavailable in environment"

    cmd = [
        sys.executable,
        str(_DTC_CHILD),
        "--dataset_json_entry", entry_json,
        "--reordered_npz", str(reorder_info["reordered_npz"]),
        "--reorder_perm_npz", str(reorder_info["reorder_perm_npz"]),
        "--dataset_name", str(entry.get("name", "")),
        "--category", str(entry.get("category", "")),
        "--reorder_method", str(reorder_info["reorder_method"]),
        "--reorder_version", str(reorder_info["reorder_version"]),
        "--reorder_ms", str(float(reorder_info["reorder_ms"])),
        "--N", str(int(N)),
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

    try:
        return {
            "dtc_ms": float(payload["dtc_ms"]),
            "dtc_variant": str(payload.get("dtc_variant", "")),
            "dtc_reorder_method": str(payload["reorder_method"]),
            "dtc_reorder_version": str(payload["reorder_version"]),
            "dtc_reorder_ms": float(payload["reorder_ms"]),
            "dtc_preprocess_ms": float(payload["preprocess_ms"]),
            "dtc_selection_variant_ms": float(payload["selection_variant_ms"]),
            "dtc_mean_kernel_ms": float(payload["mean_kernel_ms"]),
            "dtc_std_kernel_ms": float(payload["std_kernel_ms"]),
            "dtc_end_to_end_ms": float(payload["end_to_end_ms"]),
            "dtc_variant_count": int(payload.get("variant_count", 0)),
            "dtc_correct": payload["correct"],
            "dtc_max_error": float(payload["max_error"]),
        }, None, ""
    except Exception:
        return None, "json_parse_failed", f"unexpected payload: {payload!r}"


def bench_pyg(rowptr, colind, vals, M, B) -> Optional[float]:
    if not is_pyg_available():
        return None
    try:
        st = build_pyg_sparse(rowptr, colind, vals, M)
        return time_pyg_spmm(st, B, warmup_iters=3, timed_iters=TIMED)
    except Exception as e:
        print(f"  PyG timing failed: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_json", default=DATASET_FILE)
    parser.add_argument("--output", default="ra_external_baselines_reordered.csv")
    parser.add_argument("--n_values", default="64,128,256,512")
    parser.add_argument("--category", default="")
    parser.add_argument("--skip_dtc", action="store_true")
    parser.add_argument("--skip_pyg", action="store_true")
    parser.add_argument("--dtc_reorder_threshold", type=int, default=16)
    parser.add_argument("--dtc_max_rows", type=int, default=500000)
    parser.add_argument("--dtc_cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--reorder_timeout", type=int, default=0,
                        help="Seconds for one graph reorder; 0 means no timeout")
    parser.add_argument("--per_point_timeout", type=int, default=600,
                        help="Seconds for one (graph,N) preprocess+tuning point; 0 means no timeout")
    parser.add_argument("--dtc_warmup_iters", type=int, default=3)
    parser.add_argument("--dtc_timed_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtc_atol", type=float, default=1e-3)
    args = parser.parse_args()

    with open(args.datasets_json) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "datasets" in raw:
        datasets = raw["datasets"]
    elif isinstance(raw, list):
        datasets = raw
    else:
        raise ValueError(f"Unexpected manifest shape: {type(raw).__name__}")

    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    rows: List[Dict[str, object]] = []

    print(f"DTC available: {DTC_LOADED and is_dtc_available()}")
    print(f"PyG available: {is_pyg_available()}")
    print(f"Datasets: {sum(1 for d in datasets if d.get('enabled', True))}")
    print(f"N values: {n_values}")
    if not args.skip_dtc:
        print(f"DTC reorder method: {REORDER_METHOD_NOTE}")
        print(f"DTC reorder threshold: {args.dtc_reorder_threshold}")
        print(f"DTC timed iters: {args.dtc_timed_iters}")
        if args.dtc_max_rows > 0:
            print(f"DTC max rows filter: M <= {args.dtc_max_rows}")
    print("-" * 80)

    with tempfile.TemporaryDirectory(prefix="dtc_reorder_entry_") as entry_tmp:
        for entry in datasets:
            if not entry.get("enabled", True):
                continue
            if args.category and entry.get("category") != args.category:
                continue

            name = entry["name"]
            print(f"[{name}] category={entry.get('category', '?')}")
            try:
                data = load_dataset(entry)
                if data is None:
                    print("  SKIP (load failed)")
                    continue
            except Exception as e:
                print(f"  SKIP ({e})")
                continue

            rowptr = data["rowptr"].cuda()
            colind = data["colind"].cuda()
            vals = data["vals"].cuda()
            M = int(data["M"])
            nnz = int(colind.shape[0]) if "nnz" not in data else int(data["nnz"])
            if nnz == 0:
                nnz = int(entry.get("nnz", 0))

            entry_ns = entry.get("Ns", n_values)
            entry_max_n = int(entry.get("max_N", max(n_values) if n_values else 0))
            active_ns = [N for N in n_values if N in entry_ns and N <= entry_max_n]
            if not active_ns:
                print("  SKIP (no matching N values for this dataset)")
                continue
            print(f"  M={M}, nnz={nnz}, active_Ns={active_ns}")

            entry_json = str(Path(entry_tmp) / f"{name.replace('/', '_').replace(' ', '_')}.json")
            with open(entry_json, "w") as f:
                json.dump(entry, f)

            reorder_info = None
            reorder_failed = ""
            dtc_eligible = (not args.skip_dtc) and (args.dtc_max_rows <= 0 or M <= args.dtc_max_rows)
            if dtc_eligible:
                try:
                    reorder_info = reorder_once(
                        entry,
                        data,
                        args.dtc_reorder_threshold,
                        cache_dir=args.dtc_cache_dir,
                        python_exe=sys.executable,
                        timeout_s=args.reorder_timeout if args.reorder_timeout > 0 else None,
                    )
                    cache_note = "cache-hit" if reorder_info.get("cache_hit") else "new"
                    print(f"  DTC reorder={float(reorder_info['reorder_ms']):.3f}ms ({cache_note})")
                except Exception as e:
                    reorder_failed = str(e)
                    print(f"  DTC reorder failed: {reorder_failed}")

            for N in active_ns:
                torch.manual_seed(args.seed + N)
                B = torch.randn((M, N), device="cuda", dtype=torch.float32)
                row: Dict[str, object] = {
                    "dataset": name,
                    "category": entry.get("category", ""),
                    "M": M,
                    "nnz": nnz,
                    "N": N,
                }
                try:
                    row["cusparse_ms"] = bench_cusparse(rowptr, colind, vals, B)
                except Exception as e:
                    row["cusparse_ms"] = float("nan")
                    row["cusparse_error"] = str(e)

                if not args.skip_dtc:
                    if args.dtc_max_rows > 0 and M > args.dtc_max_rows:
                        row["dtc_ms"] = float("nan")
                        row["dtc_variant"] = ""
                        row["dtc_failure_class"] = "skipped_reorder_filter"
                    elif reorder_failed:
                        row["dtc_ms"] = float("nan")
                        row["dtc_variant"] = ""
                        row["dtc_failure_class"] = "reorder_failed"
                        row["dtc_failure_detail"] = reorder_failed
                    else:
                        dtc, failure_class, failure_detail = bench_dtc_best_subprocess(
                            entry_json, reorder_info, entry, N, args.per_point_timeout,
                            args.dtc_warmup_iters, args.dtc_timed_iters, args.seed, args.dtc_atol
                        )
                        if dtc:
                            row.update(dtc)
                            row["dtc_speedup_vs_cusparse"] = (
                                row["cusparse_ms"] / dtc["dtc_ms"] if dtc["dtc_ms"] > 0 else float("nan")
                            )
                        else:
                            row["dtc_ms"] = float("nan")
                            row["dtc_variant"] = ""
                            if failure_class:
                                row["dtc_failure_class"] = failure_class
                                row["dtc_failure_detail"] = failure_detail
                                if reorder_info is not None:
                                    row["dtc_reorder_method"] = reorder_info["reorder_method"]
                                    row["dtc_reorder_version"] = reorder_info["reorder_version"]
                                    row["dtc_reorder_ms"] = float(reorder_info["reorder_ms"])
                                print(f"  [{name}] N={N} DTC failed ({failure_class}): {failure_detail}")

                if not args.skip_pyg:
                    pyg_ms = bench_pyg(rowptr, colind, vals, M, B)
                    if pyg_ms is not None:
                        row["pyg_ms"] = pyg_ms
                        row["pyg_speedup_vs_cusparse"] = (
                            row["cusparse_ms"] / pyg_ms if pyg_ms > 0 else float("nan")
                        )
                    else:
                        row["pyg_ms"] = float("nan")

                rows.append(row)
                if row.get("dtc_failure_class"):
                    dtc_str = f"DTC={row['dtc_failure_class']}"
                else:
                    dtc_str = f"DTC={row.get('dtc_ms', float('nan')):.3f}ms" if "dtc_ms" in row else "DTC=--"
                pyg_str = f"PyG={row.get('pyg_ms', float('nan')):.3f}ms" if "pyg_ms" in row else "PyG=--"
                print(f"  N={N}: cuSPARSE={row['cusparse_ms']:.3f}ms  {dtc_str}  {pyg_str}")
                del B
                torch.cuda.empty_cache()

    if not rows:
        print("No rows produced.")
        return

    fieldnames = sorted({k for r in rows for k in r})
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    if not args.skip_dtc:
        note_path = str(Path(args.output).with_suffix(".methodology.txt"))
        with open(note_path, "w") as f:
            f.write(REORDER_METHOD_NOTE + "\n")
            if args.dtc_max_rows > 0:
                f.write(
                    f"Reordered DTC was run only for graphs with M <= {args.dtc_max_rows}; "
                    "other rows retain cuSPARSE/PyG measurements and mark DTC as skipped.\n"
                )
            f.write(
                f"Per graph: reorder once, record dtc_reorder_ms, save reordered CSR in {args.dtc_cache_dir}.\n"
            )
            f.write(
                f"Per (graph, N): preprocess on reordered CSR, run variant selection/tuning, "
                f"then measure best mean kernel time with warmup={args.dtc_warmup_iters} and timed_iters={args.dtc_timed_iters}.\n"
            )
            f.write("Correctness uses permute-in / inverse-permute-out against cuSPARSE on the original CSR.\n")
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
