#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="${repo_root}/external/DTC-SpMM_ASPLOS24"
arch="${CUDA_ARCH:-86}"

if [[ ! -f "${source_root}/DTC-SpMM/setup.py" ]]; then
  echo "DTC-SpMM source not found at ${source_root}" >&2
  echo "Clone https://github.com/HPMLL/DTC-SpMM_ASPLOS24.git --recursive there first." >&2
  exit 1
fi

glog="${source_root}/third_party/glog"
sputnik="${source_root}/third_party/sputnik"
cmake -S "${glog}" -B "${glog}/build" \
  -DCMAKE_INSTALL_PREFIX="${glog}/build" -DWITH_GFLAGS=OFF -DBUILD_TESTING=OFF
cmake --build "${glog}/build" -j"${BUILD_JOBS:-8}"
cmake --install "${glog}/build"

cmake -S "${sputnik}" -B "${sputnik}/build" \
  -DGLOG_INCLUDE_DIR="${glog}/build/include" \
  -DGLOG_LIBRARY="${glog}/build/lib/libglog.so" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF \
  -DCUDA_ARCHS="${arch}" -DCMAKE_CUDA_FLAGS="-I${glog}/build/include"
cmake --build "${sputnik}/build" -j"${BUILD_JOBS:-8}"

pushd "${source_root}/DTC-SpMM" >/dev/null
SPUTNIK_PATH="${sputnik}" GLOG_PATH="${glog}" \
TORCH_CUDA_ARCH_LIST="${arch:0:1}.${arch:1}" \
CPLUS_INCLUDE_PATH="${glog}/build/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}" \
LD_LIBRARY_PATH="${glog}/build/lib:${sputnik}/build/sputnik${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${repo_root}/.venv/bin/python" setup.py build_ext --inplace
popd >/dev/null

echo "Built DTC-SpMM for SM ${arch}."
