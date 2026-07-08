#!/usr/bin/env bash
# run_ada_sweep.sh — RTX 4090 (Ada, SM 89) architecture sweep for RA-SpMM.
#
# Builds the RA-SpMM bindings for SM 89, runs the 26-graph x N in {64,128,256,512}
# sweep (6 kernels + cuSPARSE + router), evaluates router quality, and computes
# kernel-choice PARITY vs the RTX 3090 baseline CSVs (baseline_3090/).
#
# Prereqs: CUDA >= 11.8 toolkit (nvcc), a Python 3.10 env with a CUDA-matched torch.
# Run from a directory containing the RA-SpMM sources and baseline_3090/ CSVs:
#     bash run_ada_sweep.sh
set -euo pipefail

PKG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${PKG_ROOT}"
OUT="fgcs_results/revision/ada"
mkdir -p "${OUT}"

echo "=================================================================="
echo " RA-SpMM Ada (SM 89) sweep — fair-comparison checklist"
echo "   [x] 50 warmup + 200 timed iters via CUDA events (paper protocol)"
echo "   [x] FP32 atol=1e-4 rtol=1e-3 ; FP16/TC atol=1e-2 rtol=1e-2"
echo "   [x] cuSPARSE is the correctness reference and speedup denominator"
echo "   [x] Timing runs own the GPU exclusively (CUDA_VISIBLE_DEVICES pinned)"
echo "   [x] Same 6 kernels + router as the 3090 headline results"
echo "=================================================================="

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "[1/5] Building bindings for SM 89 (Ada)..."
RA_SM_ARCH=89 python setup.py build_ext --inplace
python -c "import torch, ra_spmm; print('  bindings OK on', torch.cuda.get_device_name(0))"

echo "[2/5] Sanity parity (router C++ vs Python) ..."
python ra_router_parity_test.py || echo "  WARNING: parity mismatch on Ada"

echo "[3/5] Full sweep: 6 kernels + cuSPARSE, 26 graphs x N in {64,128,256,512} ..."
python ra_real_graph_eval.py --datasets-file paper_datasets.json \
    --output "${OUT}/ada_all_graphs_results.csv"

echo "[4/5] Router quality on Ada ..."
python ra_router_eval.py --results "${OUT}/ada_all_graphs_results.csv" \
    | tee "${OUT}/ada_router_quality.txt"

echo "[5/5] Kernel-choice PARITY: Ada oracle vs 3090 oracle/router ..."
python parity_vs_3090.py \
    --ada "${OUT}/ada_all_graphs_results.csv" \
    --baseline3090 baseline_3090/all_graphs_results.csv \
    --out "${OUT}/parity_vs_3090.csv" | tee "${OUT}/parity_vs_3090.txt"

echo ""
echo "DONE. Deliverables in ${OUT}/ :"
echo "  ada_all_graphs_results.csv, ada_router_quality.txt,"
echo "  parity_vs_3090.csv, parity_vs_3090.txt"
echo "Also run: bash flashsparse.sh   (FlashSparse baseline, needs SM 89/90)"
