"""Collect warm-kernel Nsight Compute reports for the six paper kernels."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "bench" / "profile_case_ra.py"
DEFAULT_MANIFEST = REPO_ROOT / "fgcs_results" / "paper_combined_datasets.json"
DEFAULT_OUT = REPO_ROOT / "fgcs_results" / "revision" / "fair" / "profile"
NCU = os.environ.get("NCU", "ncu")
KERNELS = [
    "CSR_DIRECT", "RODE_ENHANCED", "ZERO_OVERHEAD_CSR",
    "TC_DIRECT", "COMMUNITY_TC", "SEGMENT_HYBRID",
]
FLOP_METRICS = [
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "sm__inst_executed_pipe_tensor_op_hmma.sum",
]

# Two structural representatives per regime where synthetic data exists.
# Dense Large-Scale has no synthetic generator, so it uses two real graphs.
REPRESENTATIVE_DATASETS = [
    "roadNet-PA", "synth_sparse_uniform_d8",
    "twitter-combined", "synth_sparse_skewed_cv2p5",
    "amazon-photo", "synth_dense_small_d70",
    "gplus-combined", "ogbn-proteins",
    "com-DBLP", "synth_community_nc100",
    "Flickr", "synth_mixed_v3",
]


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_label(dataset: str, kernel: str, N: int) -> str:
    clean = dataset.replace("/", "_").replace(" ", "_")
    return f"{clean}__{kernel}__N{N}"


def collect_one(dataset: str, category: str, synthetic: bool, kernel: str,
                N: int, manifest: Path, outdir: Path, gpu: int,
                force: bool) -> bool:
    label = safe_label(dataset, kernel, N)
    report = outdir / f"{label}.ncu-rep"
    raw_csv = outdir / f"{label}.ncu.csv"
    meta_path = outdir / f"{label}.meta.json"
    if report.exists() and raw_csv.exists() and not force:
        print(f"[skip] {label}", flush=True)
        return True

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        NCU, "--nvtx", "--nvtx-include", "SPMM_PROFILE/",
        "--set", "full", "--metrics", ",".join(FLOP_METRICS),
        "--force-overwrite", "-o", str(report.with_suffix("")),
        sys.executable, str(HARNESS),
        "--dataset", dataset, "--kernel", kernel, "--N", str(N),
        "--datasets-file", str(manifest), "--warmup", "10",
    ]
    print(f"[ncu] gpu={gpu} {label}", flush=True)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"[fail] {label}\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
        return False

    exported = subprocess.run(
        [NCU, "--import", str(report), "--csv", "--page", "raw"],
        capture_output=True, text=True)
    if exported.returncode != 0 or not exported.stdout.strip():
        print(f"[fail] export {label}: {exported.stderr[-1000:]}")
        return False
    raw_csv.write_text(exported.stdout)
    meta_path.write_text(json.dumps({
        "dataset": dataset, "category": category, "synthetic": synthetic,
        "kernel": kernel, "N": N, "gpu": gpu,
        "timing_regime": "warm execute-only; plan built before NVTX range",
        "report": report.name, "raw_csv": raw_csv.name,
    }, indent=2) + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-file", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--outdir", default=str(DEFAULT_OUT))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--datasets", default=",".join(REPRESENTATIVE_DATASETS))
    parser.add_argument("--kernels", default=",".join(KERNELS))
    parser.add_argument("--Ns", default="128,512")
    parser.add_argument("--all-real", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.datasets_file)
    manifest = json.loads(manifest_path.read_text())["datasets"]
    by_name = {entry["name"]: entry for entry in manifest}
    selected = ([entry["name"] for entry in manifest if not entry.get("synthetic", False)]
                if args.all_real else csv_list(args.datasets))
    kernels = csv_list(args.kernels)
    invalid = sorted(set(kernels) - set(KERNELS))
    if invalid:
        raise SystemExit(f"Unsupported kernels: {invalid}")
    Ns = [int(value) for value in csv_list(args.Ns)]

    pairs: list[tuple[str, str, bool, str, int]] = []
    for dataset in selected:
        if dataset not in by_name:
            raise SystemExit(f"Dataset not in manifest: {dataset}")
        entry = by_name[dataset]
        for kernel in kernels:
            for N in Ns:
                pairs.append((dataset, entry.get("category", "?"),
                              bool(entry.get("synthetic", False)), kernel, N))
    pairs = [pair for index, pair in enumerate(pairs)
             if index % args.shard_count == args.shard_index]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for pair in pairs:
        if not collect_one(*pair, manifest_path, outdir, args.gpu, args.force):
            failures += 1
    print(f"Profiles complete: {len(pairs) - failures}/{len(pairs)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
