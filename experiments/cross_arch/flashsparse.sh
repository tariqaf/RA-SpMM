#!/usr/bin/env bash
# flashsparse.sh — FlashSparse baseline on the RTX 4090 (Ada, SM 89) for the
# 26-graph real suite. FlashSparse needs SM 89/90 tensor cores, which is why it
# runs here and not on the 3090.
#
# Records warm execute-only and cold setup-plus-first-execute separately.
#
# Run AFTER run_ada_sweep.sh, from the same root:  bash flashsparse.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -f "${SCRIPT_DIR}/../../setup.py" ]; then
    PKG_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
else
    PKG_ROOT="${SCRIPT_DIR}"
fi
cd "${PKG_ROOT}"
OUT="fgcs_results/revision/ada"
mkdir -p "${OUT}" baselines
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "=================================================================="
echo " FlashSparse (SM 89) fair-comparison checklist"
echo "   [x] warm and cold lifecycle costs recorded separately"
echo "   [x] same 26-graph suite, same N set, 50+200 protocol"
echo "   [x] correctness checked vs cuSPARSE reference"
echo "   [x] speedups use matching warm/warm and cold/cold denominators"
echo "=================================================================="

echo "[1/3] Cloning + building FlashSparse (SM 89)..."
if [ ! -d baselines/FlashSparse ]; then
    git clone https://github.com/ParCIS/FlashSparse.git baselines/FlashSparse
fi
pushd baselines/FlashSparse >/dev/null
# FlashSparse build: follow its README. Most versions expose a pip-installable
# CUDA extension; target Ada explicitly.
export TORCH_CUDA_ARCH_LIST="8.9"
if [ -f setup.py ]; then
    python setup.py install 2>&1 | tail -20 || { echo "BUILD_NOTE: FlashSparse setup.py failed on SM 89 — see log"; }
fi
popd >/dev/null

echo "[2/3] Running FlashSparse on the 26-graph suite..."
: "${FLASHSARSE_ADAPTER:?Set FLASHSARSE_ADAPTER to module:function for the installed commit}"
python "${SCRIPT_DIR}/fs_bench.py" \
    --datasets-file paper_datasets.json \
    --adapter "${FLASHSARSE_ADAPTER}" \
    --out "${OUT}/flashsparse.csv" \
    --preproc-out "${OUT}/flashsparse_preproc.csv" \
    || echo "BUILD_NOTE: FlashSparse run failed — record blocker in ${OUT}/FLASHSPARSE_BUILD_NOTE.txt"

echo "[3/3] Done. Deliverables:"
echo "  ${OUT}/flashsparse.csv           (warm/cold timings and strict correctness)"
echo "  ${OUT}/flashsparse_preproc.csv   (format-conversion timing)"
