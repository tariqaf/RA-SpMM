"""
MP-SpMM baseline (SC'25, 2:4 Sparse Tensor Cores).
Code: Zenodo 10.5281/zenodo.16933452 (CGCL-codes/MP-SpMM).

MP-SpMM accelerates unstructured SpMM with 2:4 Structured Sparse Tensor Cores. It
PADS (does not prune) so the result is exact. It is preprocessing-heavy: a
"match-and-pad" step converts CSR into 2:4 block metadata, which we time SEPARATELY
from the kernel. Its SpMM kernel supports N in {32,128}; N=128 is
the comparison point in the paper's N set.

Pipeline per graph:
  1. export our CSR to Matrix Market (.mtx) in a scratch dataset/ dir,
  2. run the match-and-pad preprocessing binary (wall-clock timed = preprocessing),
  3. run the MP-SpMM kernel binary at N=128 (prints kernel ms + GFLOP/s),
  4. compare kernel-only time vs cuSPARSE (measured here, same protocol) and vs our
     router's chosen kernel.
Binaries were built for SM 86 (impl_cu_sm86, spmm_sm86). Graphs whose .mtx/preproc
is too heavy are skipped with a BUILD_NOTE.

Outputs:
  fgcs_results/revision/baselines/mp_spmm.csv
  fgcs_results/revision/baselines/mp_spmm_preproc.csv
  fgcs_results/revision/baselines/mpspmm_BUILD_NOTE.txt
"""
from __future__ import annotations
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
import ra_spmm  # noqa
from ra_real_graph_eval import (  # noqa
    benchmark_custom_cold,
    build_kernel_plan,
    load_dataset,
    measure_ms,
    population_cv,
    run_planned_kernel,
)
from ra_router_eval import simple_router  # noqa

MP = R / "baselines/MP-SpMM_code/MP-SpMM_SC25/mpspmm"
PREPROC_BIN = MP / "preprocessing" / "impl_cu_sm86"
SPMM_BIN = MP / "SpMM" / "spmm_sm86"
SPMM_VERIFY = MP / "SpMM" / "spmm_verify"
N_MP = 128
REQUESTED_NS = (64, 128, 256, 512)
WARMUP, TIMED = 50, 200
COLD_ITERS = 10


def parse_spmm_output(output: str) -> dict[str, float]:
    line = output.strip().splitlines()[-1] if output.strip() else ""
    values = {key: float(value) for key, value in
              re.findall(r"(warm_ms|setup_ms|first_exec_ms|gflops)=([0-9.eE+-]+)", line)}
    required = {"warm_ms", "setup_ms", "first_exec_ms", "gflops"}
    if set(values) != required:
        raise ValueError(f"unexpected MP-SpMM output: {line!r}")
    return values


def run_spmm(data_bin: Path, warmup: int, timed: int, timeout: int = 600):
    completed = subprocess.run(
        [str(SPMM_BIN), str(data_bin), str(N_MP), str(warmup), str(timed)],
        cwd=str(SPMM_BIN.parent), capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip()[-500:])
    return parse_spmm_output(completed.stdout)


def verify_spmm(data_bin: Path, rowptr: torch.Tensor, output: Path):
    if not SPMM_VERIFY.exists():
        raise RuntimeError(f"strict correctness binary missing: {SPMM_VERIFY}")
    env = dict(os.environ)
    env["VERIFY_OUT"] = str(output)
    completed = subprocess.run(
        [str(SPMM_VERIFY), str(data_bin), str(N_MP)], cwd=str(SPMM_VERIFY.parent),
        capture_output=True, text=True, timeout=600, env=env)
    if completed.returncode or not output.exists():
        raise RuntimeError((completed.stderr or completed.stdout).strip()[-500:])
    actual = np.atleast_1d(np.loadtxt(output))[:rowptr.numel() - 1]
    expected = (rowptr[1:] - rowptr[:-1]).numpy().astype(np.float64)
    max_error = float(np.max(np.abs(actual - expected))) if expected.size else 0.0
    max_degree = int(expected.max()) if expected.size else 0
    tolerance = 1.0e-3 * max(1.0, np.sqrt(max_degree)) * 10.0
    correct = max_error <= tolerance and max_error < 1.0
    return correct, max_error, tolerance


def write_mtx(path: Path, rowptr, colind, M, K):
    """Write CSR adjacency (unit values) as a Matrix Market coordinate-real general file."""
    nnz = int(rowptr[-1])
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{M} {K} {nnz}\n")
        rp = rowptr.tolist()
        ci = colind.tolist()
        lines = []
        for i in range(M):
            for p in range(rp[i], rp[i + 1]):
                lines.append(f"{i+1} {ci[p]+1} 1\n")
                if len(lines) >= 65_536:
                    f.writelines(lines)
                    lines.clear()
        if lines:
            f.writelines(lines)


def setup_workdir(work: Path):
    (work / "dataset").mkdir(parents=True, exist_ok=True)
    (work / "result" / "time_and_tb_num").mkdir(parents=True, exist_ok=True)
    # path.txt is read relative to the preprocessing binary's CWD as ../../path.txt,
    # i.e. mpspmm/path.txt. We point both paths at the scratch workdir.
    (MP.parent / "path.txt").write_text(f"project_path={work}\ndataset_path={work}\n")


def status_row(entry: dict, N: int, status: str, error: str,
               warmup: int, timed: int, cold_iters: int) -> dict:
    return {
        "dataset": entry["name"], "category": entry.get("category", "?"),
        "M": int(entry.get("M", 0)), "nnz": int(entry.get("nnz", 0)),
        "N": N, "kernel": "MP-SpMM", "status": status,
        "ms_warm": None, "preprocess_ms": None, "cold_exec_ms": None,
        "ms_cold": None, "gflops_true_nnz": None, "gflops_padded_work": None,
        "correct": False, "soft_fail": False, "hard_fail": status == "INCORRECT",
        "max_error": None, "tolerance": None,
        "ms_cusparse_warm": None, "ms_cusparse_cold": None,
        "speedup_warm_vs_cusparse": None, "speedup_cold_vs_cusparse": None,
        "router_kernel": None, "ms_router_warm": None, "ms_router_cold": None,
        "speedup_router_vs_mpspmm_warm": None,
        "speedup_router_vs_mpspmm_cold": None,
        "warmup": warmup, "timed_iters": timed, "cold_iters": cold_iters,
        "error": error,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-file", default=str(R / "paper_datasets.json"))
    ap.add_argument("--out", default=str(R / "fgcs_results/revision/baselines/mp_spmm.csv"))
    ap.add_argument("--preproc-out", default=str(R / "fgcs_results/revision/baselines/mp_spmm_preproc.csv"))
    ap.add_argument("--note-out", default=str(R / "fgcs_results/revision/baselines/mpspmm_BUILD_NOTE.txt"))
    ap.add_argument("--work", default=str(R / "fgcs_results/revision/baselines/_mpspmm_work"))
    ap.add_argument("--max-nnz", type=int, default=0,
                    help="Optional explicit nnz skip limit; 0 attempts every graph")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--timed", type=int, default=TIMED)
    ap.add_argument("--cold-iters", type=int, default=COLD_ITERS)
    ap.add_argument("--datasets", default="",
                    help="Optional comma-separated dataset names")
    args = ap.parse_args()

    assert PREPROC_BIN.exists() and SPMM_BIN.exists(), "MP-SpMM binaries not built"
    work = Path(args.work); setup_workdir(work)
    manifest = json.loads(Path(args.datasets_file).read_text())["datasets"]
    selected = {name.strip() for name in args.datasets.split(",") if name.strip()}
    rows, preproc_rows, notes = [], [], []
    notes.append(f"MP-SpMM (2:4 SpTC) built for SM 86 (CUDA {torch.version.cuda}); kernel supports N in {{32,128}}, "
                 f"compared at N=128. PADS not prunes (exact). Preprocessing = match-and-pad, timed separately.")

    for entry in manifest:
        if not entry.get("enabled", True):
            continue
        if selected and entry.get("name") not in selected:
            continue
        entry_Ns = [int(n) for n in entry.get("Ns", REQUESTED_NS)]
        for N in entry_Ns:
            if N != N_MP:
                rows.append(status_row(
                    entry, N, "UNSUPPORTED_FEATURE_DIM",
                    "MP-SpMM artifact supports N=32 and N=128",
                    args.warmup, args.timed, args.cold_iters))
        if N_MP not in entry_Ns:
            continue
        name = entry["name"]
        mat = load_dataset(entry)
        if mat is None:
            rows.append(status_row(
                entry, N_MP, "DATASET_NOT_FOUND", "dataset could not be loaded",
                args.warmup, args.timed, args.cold_iters))
            continue
        M, K = mat["M"], mat["K"]
        rp = mat["rowptr"].contiguous().int(); ci = mat["colind"].contiguous().int()
        nnz = int(rp[-1].item())
        if args.max_nnz > 0 and nnz > args.max_nnz:
            notes.append(f"{name}: SKIP (nnz={nnz} > {args.max_nnz}; .mtx/preproc too heavy)")
            rows.append(status_row(
                entry, N_MP, "SKIPPED_RESOURCE",
                f"nnz={nnz} exceeds explicit --max-nnz={args.max_nnz}",
                args.warmup, args.timed, args.cold_iters))
            print(f"  [skip] {name}: nnz={nnz}")
            continue
        mtx_path = work / "dataset" / f"{name}.mtx"
        write_mtx(mtx_path, rp.numpy(), ci.numpy(), M, K)

        data_bin = work / "dataset_mp_processed" / "2-4-cu" / "0.Adjacent-matching" / f"{name}_data.bin"
        conversion_samples, runtime_setup_samples, first_exec_samples = [], [], []
        failed = None
        for _ in range(max(1, args.cold_iters)):
            try:
                t0 = time.perf_counter()
                pr = subprocess.run([str(PREPROC_BIN), name], cwd=str(PREPROC_BIN.parent),
                                    capture_output=True, text=True, timeout=1800)
                conversion_samples.append((time.perf_counter() - t0) * 1e3)
                if pr.returncode != 0 or not data_bin.exists():
                    raise RuntimeError((pr.stderr or pr.stdout).strip()[-500:])
                cold_run = run_spmm(data_bin, 0, 1)
                runtime_setup_samples.append(cold_run["setup_ms"])
                first_exec_samples.append(cold_run["first_exec_ms"])
            except (subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
                failed = str(exc)
                break
        if failed:
            notes.append(f"{name}: cold measurement FAILED: {failed[:300]}")
            rows.append(status_row(
                entry, N_MP, "COLD_ERROR", failed[:500],
                args.warmup, args.timed, args.cold_iters))
            print(f"  [cold fail] {name}: {failed[:120]}")
            continue
        try:
            warm_run = run_spmm(data_bin, args.warmup, args.timed)
        except (subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
            notes.append(f"{name}: warm measurement FAILED: {str(exc)[:300]}")
            rows.append(status_row(
                entry, N_MP, "WARM_ERROR", str(exc)[:500],
                args.warmup, args.timed, args.cold_iters))
            continue
        try:
            correct, max_error, tolerance = verify_spmm(
                data_bin, rp, work / f"{name}_Ccol0.txt")
        except (subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
            notes.append(f"{name}: strict correctness FAILED to run: {str(exc)[:300]}")
            rows.append(status_row(
                entry, N_MP, "CORRECTNESS_ERROR", str(exc)[:500],
                args.warmup, args.timed, args.cold_iters))
            continue
        if not correct:
            notes.append(
                f"{name}: strict correctness FAIL max_error={max_error} tolerance={tolerance}")
            print(f"  [incorrect] {name}: max_error={max_error:.6g} tol={tolerance:.6g}")
            failed_row = status_row(
                entry, N_MP, "INCORRECT", "strict adaptive tolerance exceeded",
                args.warmup, args.timed, args.cold_iters)
            failed_row["max_error"] = max_error
            failed_row["tolerance"] = tolerance
            failed_row["soft_fail"] = tolerance < max_error < 1.0
            failed_row["hard_fail"] = max_error >= 1.0
            rows.append(failed_row)
            continue
        conversion_ms = float(np.mean(conversion_samples))
        runtime_setup_ms = float(np.mean(runtime_setup_samples))
        first_exec_ms = float(np.mean(first_exec_samples))
        preprocess_ms = conversion_ms + runtime_setup_ms
        ms_mp = warm_run["warm_ms"]
        ms_mp_cold = preprocess_ms + first_exec_ms
        gflops = warm_run["gflops"]
        preproc_rows.append({
            "dataset": name, "M": M, "nnz": nnz,
            "conversion_ms": round(conversion_ms, 6),
            "runtime_setup_ms": round(runtime_setup_ms, 6),
            "preprocess_ms": round(preprocess_ms, 6),
            "cold_exec_ms": round(first_exec_ms, 6),
            "ms_cold": round(ms_mp_cold, 6),
            "cold_iters": args.cold_iters,
        })

        # (4) cuSPARSE + router reference at N=128 on our stack
        rp_g = rp.cuda(); ci_g = ci.cuda(); v_g = mat["vals"].cuda().float()
        B = torch.randn(M, N_MP, device="cuda")
        cus_warm = ra_spmm.benchmark_cusparse(
            rp_g, ci_g, v_g, B, args.warmup, args.timed)
        cus_cold = ra_spmm.benchmark_cusparse_cold(
            rp_g, ci_g, v_g, B, args.cold_iters)
        ms_cus = float(cus_warm["exec_ms"])
        ms_cus_cold = float(cus_cold["total_ms"])
        d_bar = nnz / max(1, M); deg = (rp_g[1:] - rp_g[:-1]).float()
        cv_d = population_cv(deg)
        rk = simple_router(d_bar, cv_d, M, N_MP, nnz)
        try:
            plan = build_kernel_plan(rk, rp, ci, mat["vals"].float(), M, K, N_MP)
            ms_router = measure_ms(
                lambda: run_planned_kernel(rk, plan, rp_g, ci_g, v_g, B),
                args.warmup, args.timed)
            router_cold = benchmark_custom_cold(
                rk, rp, ci, mat["vals"].float(), rp_g, ci_g, v_g, B,
                args.cold_iters)
            ms_router_cold = float(router_cold["ms_cold"])
        except Exception:
            ms_router = float("nan")
            ms_router_cold = float("nan")
        rows.append({
            "dataset": name, "category": entry.get("category", "?"),
            "M": M, "nnz": nnz, "N": N_MP, "kernel": "MP-SpMM",
            "status": "OK",
            "ms_warm": round(ms_mp, 6), "preprocess_ms": round(preprocess_ms, 6),
            "cold_exec_ms": round(first_exec_ms, 6), "ms_cold": round(ms_mp_cold, 6),
            "gflops_true_nnz": round((2.0 * nnz * N_MP) / (ms_mp * 1.0e6), 3),
            "gflops_padded_work": round(gflops, 3),
            "correct": correct, "soft_fail": tolerance < max_error < 1.0,
            "hard_fail": max_error >= 1.0, "max_error": max_error,
            "tolerance": tolerance,
            "ms_cusparse_warm": round(ms_cus, 6),
            "ms_cusparse_cold": round(ms_cus_cold, 6),
            "speedup_warm_vs_cusparse": round(ms_cus / ms_mp, 6) if ms_mp > 0 else 0,
            "speedup_cold_vs_cusparse": round(ms_cus_cold / ms_mp_cold, 6) if ms_mp_cold > 0 else 0,
            "router_kernel": rk, "ms_router_warm": round(ms_router, 6),
            "ms_router_cold": round(ms_router_cold, 6),
            "speedup_router_vs_mpspmm_warm": round(ms_mp / ms_router, 6)
                if ms_router == ms_router and ms_router > 0 else 0,
            "speedup_router_vs_mpspmm_cold": round(ms_mp_cold / ms_router_cold, 6)
                if ms_router_cold == ms_router_cold and ms_router_cold > 0 else 0,
            "warmup": args.warmup, "timed_iters": args.timed,
            "cold_iters": args.cold_iters,
            "error": "",
        })
        print(f"  {name:<20s} MP warm={ms_mp:.5f}ms cold={ms_mp_cold:.3f}ms "
              f"cuSPARSE warm={ms_cus:.5f}ms cold={ms_cus_cold:.3f}ms")
        del B, rp_g, ci_g, v_g
        torch.cuda.empty_cache()

    expected = {
        (entry["name"], int(N))
        for entry in manifest if entry.get("enabled", True)
        if not selected or entry.get("name") in selected
        for N in entry.get("Ns", REQUESTED_NS)
    }
    observed = {(row["dataset"], int(row["N"])) for row in rows}
    if observed != expected:
        raise SystemExit(
            f"MP-SpMM status table incomplete: expected={len(expected)}, "
            f"observed={len(observed)}, missing={sorted(expected-observed)[:10]}, "
            f"extra={sorted(observed-expected)[:10]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out} ({len(rows)} rows)")
    if preproc_rows:
        with open(args.preproc_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(preproc_rows[0].keys())); w.writeheader(); w.writerows(preproc_rows)
        print(f"Wrote {args.preproc_out} ({len(preproc_rows)} rows)")
    Path(args.note_out).write_text("\n".join(notes) + "\n")
    print(f"Wrote {args.note_out} ({len(notes)} notes)")


if __name__ == "__main__":
    main()
