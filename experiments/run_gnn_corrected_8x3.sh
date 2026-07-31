#!/usr/bin/env bash
# Corrected warm+cold 8x3 end-to-end GNN rerun on the Phase-1 backend
# (faithful per-model operators + resolved TF32 tile labels/execution).
#   warm: router,cusparse,tc_direct,tc_direct_tf32,pyg  (PyG is warm-only)
#   cold: router,cusparse only                          (PyG warm-only, no --include-pyg)
# 20 warmup + 100 timed steps warm; 25 trials cold; seed 123.
# Single RTX 3090 -> archs run sequentially to avoid OOM (Reddit TC plan ~2.2 GB).
set -euo pipefail
cd /mnt/shared/development/tariq/RA-SpMM
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
DS="Reddit,ogbn-proteins,ogbn-arxiv,PPI,amazon-photo,amazon-computers,Cora,CiteSeer"
BK="router,cusparse,tc_direct,tc_direct_tf32,pyg"
OUT_REL="fgcs_results/revision/tf32/gnn_corrected"
OUT_ABS="/mnt/shared/development/tariq/RA-SpMM/${OUT_REL}"
mkdir -p "${OUT_ABS}"

echo "===== WARM GCN ====="
$PY gnn_bench/router_vs_baselines_gcn.py --datasets "$DS" --backends "$BK" \
    --results_dir "${OUT_REL}/gcn" --warmup_steps 20 --timed_steps 100 --seed 123 \
    2>&1 | tee "${OUT_ABS}/warm_gcn.log"
echo "===== WARM SAGE ====="
$PY gnn_bench/router_vs_baselines_sage.py --datasets "$DS" --backends "$BK" \
    --results_dir "${OUT_REL}/sage" --warmup_steps 20 --timed_steps 100 --seed 123 \
    2>&1 | tee "${OUT_ABS}/warm_sage.log"
echo "===== WARM GIN ====="
$PY gnn_bench/router_vs_baselines_gin.py --datasets "$DS" --backends "$BK" \
    --results_dir "${OUT_REL}/gin8" --warmup_steps 20 --timed_steps 100 --seed 123 \
    2>&1 | tee "${OUT_ABS}/warm_gin.log"

echo "===== COLD GCN ====="
$PY gnn_bench/cold_first_step.py --arch gcn --datasets "$DS" \
    --out "${OUT_ABS}/cold_gcn.csv" --trials 25 --seed 123 \
    2>&1 | tee "${OUT_ABS}/cold_gcn.log"
echo "===== COLD SAGE ====="
$PY gnn_bench/cold_first_step.py --arch graphsage --datasets "$DS" \
    --out "${OUT_ABS}/cold_graphsage.csv" --trials 25 --seed 123 \
    2>&1 | tee "${OUT_ABS}/cold_graphsage.log"
echo "===== COLD GIN ====="
$PY gnn_bench/cold_first_step.py --arch gin --datasets "$DS" \
    --out "${OUT_ABS}/cold_gin.csv" --trials 25 --seed 123 \
    2>&1 | tee "${OUT_ABS}/cold_gin.log"

echo "===== EMIT CSVs + MANIFEST ====="
GNN_OUT="${OUT_ABS}" $PY experiments/emit_gnn_warmcold_csvs.py \
    2>&1 | tee "${OUT_ABS}/emit.log"
echo "===== DONE ====="
