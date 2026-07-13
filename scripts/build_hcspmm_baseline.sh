#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/baselines/HC-SpMM"
commit=3484cf74b0591e44bf656978d90ddaf9f86e00a5

if [[ ! -d "${source_dir}/.git" ]]; then
    mkdir -p "${repo_root}/baselines"
    git clone https://github.com/ZJU-DAILY/HC-SpMM.git "${source_dir}"
fi

git -C "${source_dir}" fetch origin "${commit}"
git -C "${source_dir}" checkout --detach "${commit}"

cd "${source_dir}/hybrid_kernel"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" \
    "${repo_root}/.venv/bin/python" setup.py build_ext --inplace
printf 'Built HC-SpMM at %s\n' "${commit}"
