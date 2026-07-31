#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
record=16933452
archive_name=MP-SpMM_SC25.zip
archive_url="https://zenodo.org/api/records/${record}/files/${archive_name}/content"
archive_md5=7aacfbc60cdc0c535bf666538cbe2046
parent="${repo_root}/baselines/MP-SpMM_code"
source_dir="${parent}/MP-SpMM_SC25"
archive="${parent}/${archive_name}"
arch="${CUDA_ARCH:-86}"

mkdir -p "${parent}"
if [[ ! -d "${source_dir}" ]]; then
    if [[ ! -f "${archive}" ]]; then
        curl -fL "${archive_url}" -o "${archive}"
    fi
    printf '%s  %s\n' "${archive_md5}" "${archive}" | md5sum -c -
    unzip -q "${archive}" -d "${parent}"
fi

apply_once() {
    local patch_file="$1"
    if patch --dry-run -p1 -d "${source_dir}" < "${patch_file}" >/dev/null 2>&1; then
        patch -p1 -d "${source_dir}" < "${patch_file}"
    elif patch --dry-run -R -p1 -d "${source_dir}" < "${patch_file}" >/dev/null 2>&1; then
        printf 'Already applied: %s\n' "${patch_file}"
    else
        printf 'Patch does not match the pinned source: %s\n' "${patch_file}" >&2
        return 1
    fi
}

apply_once "${repo_root}/scripts/patches/mp_spmm_fair_timing.patch"
apply_once "${repo_root}/scripts/patches/mp_spmm_verify.patch"

preprocess="${source_dir}/mpspmm/preprocessing"
nvcc -std=c++17 -diag-suppress 1650 -diag-suppress 2464 -diag-suppress 550 \
    -arch="sm_${arch}" -O3 -o "${preprocess}/impl_cu_sm${arch}" \
    "${preprocess}/impl-adjacent-matching-2-4.cu"

spmm="${source_dir}/mpspmm/SpMM"
nvcc -std=c++11 -O3 -gencode "arch=compute_${arch},code=sm_${arch}" \
    "${spmm}/spmm_sp_new.cu" "${spmm}/data_reader.cpp" --use_fast_math \
    -Xptxas "-v -dlcm=ca" -o "${spmm}/spmm_sm${arch}"
nvcc -std=c++11 -O3 -gencode "arch=compute_${arch},code=sm_${arch}" \
    "${spmm}/spmm_verify.cu" "${spmm}/data_reader.cpp" --use_fast_math \
    -Xptxas "-v -dlcm=ca" -o "${spmm}/spmm_verify"

printf 'Built MP-SpMM fair adapter from Zenodo %s for SM%s\n' "${record}" "${arch}"
