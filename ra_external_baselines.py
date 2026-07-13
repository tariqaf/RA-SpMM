"""Run fair PyG and DTC-SpMM lifecycles on manifest (graph, N) pairs."""
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dtc_reorder_utils import (DEFAULT_CACHE_DIR, IDENTITY_METHOD, REORDER_METHOD,
                               REORDER_METHOD_NOTE, reorder_once)
from pyg_baseline import is_pyg_available, build_pyg_sparse
from ra_real_graph_eval import (BASE_ATOL, DATASET_FILE,
                                load_dataset, measure_ms, measure_one_ms)

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


WARMUP = 50
TIMED = 200
COLD_ITERS = 10
_DTC_CHILD = REPO_ROOT / "ra_dtc_single.py"


def bench_cusparse(rowptr, colind, vals, B) -> Dict[str, float]:
    warm = ra_spmm.benchmark_cusparse(
        rowptr, colind, vals, B, warmup=WARMUP, iters=TIMED)
    cold = ra_spmm.benchmark_cusparse_cold(
        rowptr, colind, vals, B, COLD_ITERS)
    return {
        "ms_warm": float(warm["exec_ms"]),
        "preprocess_ms": float(cold["plan_ms"]),
        "cold_exec_ms": float(cold["exec_ms"]),
        "ms_cold": float(cold["total_ms"]),
    }


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
    selection_warmup_iters: int,
    selection_timed_iters: int,
    cold_iters: int,
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
        "--selection_warmup_iters", str(int(selection_warmup_iters)),
        "--selection_timed_iters", str(int(selection_timed_iters)),
        "--cold_iters", str(int(cold_iters)),
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
            "dtc_cold_exec_ms": float(payload["cold_exec_ms"]),
            "dtc_variant_count": int(payload.get("variant_count", 0)),
            "dtc_correct": payload["correct"],
            "dtc_max_error": float(payload["max_error"]),
        }, None, ""
    except Exception:
        return None, "json_parse_failed", f"unexpected payload: {payload!r}"


def bench_pyg(rowptr, colind, vals, M, B) -> Optional[Dict[str, object]]:
    if not is_pyg_available():
        return None
    try:
        st = build_pyg_sparse(rowptr, colind, vals, M)
        output = st @ B
        ms_warm = measure_ms(lambda: st @ B, WARMUP, TIMED)
        setup_total = 0.0
        exec_total = 0.0
        for _ in range(COLD_ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            cold_state = build_pyg_sparse(rowptr, colind, vals, M)
            torch.cuda.synchronize()
            setup_total += (time.perf_counter() - start) * 1e3
            cold_output, exec_ms = measure_one_ms(lambda: cold_state @ B)
            exec_total += exec_ms
            del cold_output, cold_state
            torch.cuda.synchronize()
        preprocess_ms = setup_total / COLD_ITERS
        cold_exec_ms = exec_total / COLD_ITERS
        return {
            "output": output,
            "ms_warm": ms_warm,
            "preprocess_ms": preprocess_ms,
            "cold_exec_ms": cold_exec_ms,
            "ms_cold": preprocess_ms + cold_exec_ms,
        }
    except Exception as e:
        print(f"  PyG timing failed: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_json", default=DATASET_FILE)
    parser.add_argument("--output", default="ra_external_baselines_reordered.csv")
    parser.add_argument("--n_values", default="64,128,256,512")
    parser.add_argument("--category", default="")
    parser.add_argument("--datasets", default="",
                        help="Optional comma-separated dataset names")
    parser.add_argument("--skip_dtc", action="store_true")
    parser.add_argument("--skip_pyg", action="store_true")
    parser.add_argument("--dtc_reorder_threshold", type=int, default=16)
    parser.add_argument("--dtc-reorder", choices=("identity", "tca"), default="identity",
                        help="identity is reproducible locally; tca requires the optional upstream stack")
    parser.add_argument("--dtc_max_rows", type=int, default=0,
                        help="Optional explicit row-count skip limit; 0 attempts every graph")
    parser.add_argument("--dtc_cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--reorder_timeout", type=int, default=0,
                        help="Seconds for one graph reorder; 0 means no timeout")
    parser.add_argument("--per_point_timeout", type=int, default=600,
                        help="Seconds for one (graph,N) preprocess+tuning point; 0 means no timeout")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--cold-iters", type=int, default=10)
    parser.add_argument("--dtc_warmup_iters", type=int, default=50)
    parser.add_argument("--dtc_timed_iters", type=int, default=200)
    parser.add_argument("--dtc_selection_warmup_iters", type=int, default=3)
    parser.add_argument("--dtc_selection_timed_iters", type=int, default=20)
    parser.add_argument("--dtc-cold-iters", type=int, default=1,
                        help="Independent DTC preprocess+autotune+first-execute trials")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtc_atol", type=float, default=1e-3)
    args = parser.parse_args()
    global WARMUP, TIMED, COLD_ITERS
    WARMUP, TIMED, COLD_ITERS = args.warmup, args.timed, args.cold_iters

    with open(args.datasets_json) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "datasets" in raw:
        datasets = raw["datasets"]
    elif isinstance(raw, list):
        datasets = raw
    else:
        raise ValueError(f"Unexpected manifest shape: {type(raw).__name__}")

    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    selected_datasets = {name.strip() for name in args.datasets.split(",") if name.strip()}
    rows: List[Dict[str, object]] = []

    print(f"DTC available: {DTC_LOADED and is_dtc_available()}")
    print(f"PyG available: {is_pyg_available()}")
    print(f"Datasets: {sum(1 for d in datasets if d.get('enabled', True))}")
    print(f"N values: {n_values}")
    if not args.skip_dtc:
        if args.dtc_reorder == "tca":
            print(f"DTC reorder method: {REORDER_METHOD_NOTE}")
        else:
            print("DTC reorder method: identity CSR order (no TCA claim)")
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
            if selected_datasets and entry.get("name") not in selected_datasets:
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
            max_row_nnz = max(1, int((rowptr[1:] - rowptr[:-1]).max().item()))
            dtc_tolerance = BASE_ATOL * max(1.0, math.sqrt(max_row_nnz)) * 10.0

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
                        method=REORDER_METHOD if args.dtc_reorder == "tca" else IDENTITY_METHOD,
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
                    cusparse = bench_cusparse(rowptr, colind, vals, B)
                    row.update({f"cusparse_{key}": value for key, value in cusparse.items()})
                except Exception as e:
                    cusparse = None
                    row["cusparse_ms_warm"] = float("nan")
                    row["cusparse_ms_cold"] = float("nan")
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
                            args.dtc_warmup_iters, args.dtc_timed_iters,
                            args.dtc_selection_warmup_iters,
                            args.dtc_selection_timed_iters, args.dtc_cold_iters,
                            args.seed, dtc_tolerance
                        )
                        if dtc:
                            row.update(dtc)
                            dtc_warm = float(dtc["dtc_end_to_end_ms"])
                            dtc_setup = (float(dtc["dtc_reorder_ms"]) +
                                         float(dtc["dtc_preprocess_ms"]) +
                                         float(dtc["dtc_selection_variant_ms"]))
                            dtc_cold_exec = float(dtc["dtc_cold_exec_ms"])
                            dtc_cold = dtc_setup + dtc_cold_exec
                            dtc_correct = bool(dtc["dtc_correct"])
                            row["dtc_tolerance"] = dtc_tolerance
                            row["dtc_soft_fail"] = (dtc_tolerance < dtc["dtc_max_error"] < 1.0)
                            row["dtc_hard_fail"] = dtc["dtc_max_error"] >= 1.0
                            row["dtc_ms_warm"] = dtc_warm if dtc_correct else float("nan")
                            row["dtc_preprocess_ms"] = dtc_setup if dtc_correct else float("nan")
                            row["dtc_cold_exec_ms"] = dtc_cold_exec if dtc_correct else float("nan")
                            row["dtc_ms_cold"] = dtc_cold if dtc_correct else float("nan")
                            if dtc_correct and cusparse:
                                row["dtc_speedup_vs_cusparse_warm"] = cusparse["ms_warm"] / dtc_warm
                                row["dtc_speedup_vs_cusparse_cold"] = cusparse["ms_cold"] / dtc_cold
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
                    pyg = bench_pyg(rowptr, colind, vals, M, B)
                    if pyg is not None:
                        reference = ra_spmm.spmm_cusparse(rowptr, colind, vals, B)
                        max_row_nnz = int((rowptr[1:] - rowptr[:-1]).max().item())
                        tolerance = BASE_ATOL * max(1.0, math.sqrt(max_row_nnz))
                        max_error = float((pyg["output"] - reference).abs().max().item())
                        correct = max_error <= tolerance and max_error < 1.0
                        row.update({
                            "pyg_ms_warm": pyg["ms_warm"] if correct else float("nan"),
                            "pyg_preprocess_ms": pyg["preprocess_ms"] if correct else float("nan"),
                            "pyg_cold_exec_ms": pyg["cold_exec_ms"] if correct else float("nan"),
                            "pyg_ms_cold": pyg["ms_cold"] if correct else float("nan"),
                            "pyg_correct": correct,
                            "pyg_soft_fail": tolerance < max_error < 1.0,
                            "pyg_hard_fail": max_error >= 1.0,
                            "pyg_max_error": max_error,
                            "pyg_tolerance": tolerance,
                        })
                        if correct and cusparse:
                            row["pyg_speedup_vs_cusparse_warm"] = cusparse["ms_warm"] / float(pyg["ms_warm"])
                            row["pyg_speedup_vs_cusparse_cold"] = cusparse["ms_cold"] / float(pyg["ms_cold"])
                        del pyg["output"], reference
                    else:
                        row["pyg_ms_warm"] = float("nan")

                rows.append(row)
                if row.get("dtc_failure_class"):
                    dtc_str = f"DTC={row['dtc_failure_class']}"
                else:
                    dtc_str = f"DTC(warm)={row.get('dtc_ms_warm', float('nan')):.3f}ms" if "dtc_ms_warm" in row else "DTC=--"
                pyg_str = f"PyG(warm)={row.get('pyg_ms_warm', float('nan')):.3f}ms" if "pyg_ms_warm" in row else "PyG=--"
                print(f"  N={N}: cuSPARSE(warm)={row['cusparse_ms_warm']:.3f}ms  {dtc_str}  {pyg_str}")
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
            f.write((REORDER_METHOD_NOTE if args.dtc_reorder == "tca" else
                     "DTC ran on the original CSR order; no TCA reordering claim is made.") + "\n")
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
            f.write(
                f"Variant selection uses warmup={args.dtc_selection_warmup_iters}, "
                f"timed_iters={args.dtc_selection_timed_iters}; cold setup/first-execute "
                f"is averaged over {args.dtc_cold_iters} independent state builds.\n")
            f.write("Correctness uses permute-in / inverse-permute-out against cuSPARSE on the original CSR.\n")
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
