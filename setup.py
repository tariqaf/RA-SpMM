"""
setup.py - Build script for ra_spmm CUDA extension
RA-SpMM: Regime-Aware Sparse Matrix-Matrix Multiplication on GPUs
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Source files
sources = [
    # --- Existing kernels ---
    'csr/csr_direct.cu',
    'csr/csr_adaptive.cu',
    'csr/row_split.cu',
    'staged/staged_reuse.cu',
    'tc/tc_features.cu',
    'tc/tc_sparse.cu',
    'tc/tc_reordered.cu',
    'tc/hybrid_tc_cuda.cu',
    # --- New regime-specific kernels (Wave 1: CUDA-core) ---
    'csr/ra_zero_overhead.cu',       # R6: Overhead-sensitive
    'csr/ra_vectorized_coarse.cu',   # R2: Road-network
    'csr/ra_rode_enhanced.cu',       # R1: Power-law
    # --- New regime-specific kernels (Wave 2: TC) ---
    'tc/ra_tc_direct.cu',             # R4: TC-friendly
    'tc/ra_locality_tiled.cu',       # R3: Reordered locality
    # --- New regime-specific kernels (Wave 3: Complex) ---
    'tc/ra_community_tc.cu',         # R5: Community
    'tc/ra_segment_hybrid.cu',       # R7: Hybrid/mixed
    # --- Infrastructure ---
    'router/router_features.cpp',
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
    '-arch=sm_86',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-lineinfo',
    '-Xcompiler', '-fPIC',
    '--generate-code', 'arch=compute_86,code=sm_86',
]

cpp_flags = [
    '-O3',
    '-fPIC',
    '-std=c++17',
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
            include_dirs=[
                os.path.dirname(os.path.abspath(__file__)),
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
