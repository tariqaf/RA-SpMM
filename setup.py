"""
setup.py - Build script for ra_spmm CUDA extension
RA-SpMM: Regime-Aware Sparse Matrix-Matrix Multiplication on GPUs
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

def normalize_sm_arch(value):
    value = str(value).strip().lower()
    value = value.replace("sm_", "").replace("compute_", "").replace(".", "")
    if not value.isdigit():
        raise ValueError(f"Invalid RA_SM_ARCH value: {value!r}")
    return value

SM_ARCH = normalize_sm_arch(os.environ.get("RA_SM_ARCH", "86"))

# Source files
sources = [
    # --- Baseline and ablation kernels ---
    'csr/csr_direct.cu',
    'csr/csr_adaptive.cu',
    'csr/row_split.cu',
    'staged/staged_reuse.cu',
    'tc/tc_features.cu',
    'tc/tc_sparse.cu',
    'tc/tc_reordered.cu',
    'tc/hybrid_tc_cuda.cu',
    # --- Paper portfolio: CUDA-core kernels ---
    'csr/ra_zero_overhead.cu',       # R6: Overhead-sensitive
    'csr/ra_vectorized_coarse.cu',   # R2: Road-network
    'csr/ra_rode_enhanced.cu',       # R1: Power-law
    # --- Paper portfolio: Tensor Core kernels ---
    'tc/ra_tc_direct.cu',             # R4: TC-friendly
    'tc/ra_locality_tiled.cu',       # R3: Reordered locality
    # --- Paper portfolio: structured hybrid kernels ---
    'tc/ra_community_tc.cu',         # R5: Community
    'tc/ra_segment_hybrid.cu',       # R7: Hybrid/mixed
    # --- Infrastructure ---
    'router/router_features.cpp',
    'router/router_features_cuda.cu',
    'router/router_scores.cpp',
    'router/router_dispatch.cpp',
    'graph/generators.cpp',
    'bench/benchmark.cpp',
    'bindings/ra_bindings.cpp',
]

# Build flags
nvcc_flags = [
    '-O3',
    '--use_fast_math',
    f'-arch=sm_{SM_ARCH}',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-lineinfo',
    '-Xcompiler', '-fPIC',
    '-Xcompiler', '-fopenmp',   # OpenMP for host-side plan construction
    '--generate-code', f'arch=compute_{SM_ARCH},code=sm_{SM_ARCH}',
]

cpp_flags = [
    '-O3',
    '-fPIC',
    '-std=c++17',
    '-fopenmp',                 # OpenMP for host-side plan construction
]

setup(
    name='ra_spmm',
    ext_modules=[
        CUDAExtension(
            name='ra_spmm',
            sources=sources,
            libraries=['cusparse'],
            extra_compile_args={
                'cxx': cpp_flags,
                'nvcc': nvcc_flags,
            },
            extra_link_args=['-fopenmp'],   # link libgomp

            include_dirs=[
                os.path.dirname(os.path.abspath(__file__)),
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
