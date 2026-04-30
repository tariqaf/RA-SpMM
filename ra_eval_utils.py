import csv
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse as sp

import ra_spmm
import test_next as harness


MAIN_PATHS = list(harness.MAIN_PATHS)
FULL_PATHS = list(harness.FULL_PATHS)
EXTERNAL_BASELINE_PATHS = list(harness.EXTERNAL_BASELINE_PATHS)
CATEGORY_ORDER = (
    "hub-dominated power-law",
    "ordered sparse / road-network",
    "reordered locality",
    "dense block-local / TC-friendly",
    "sparse modular community",
    "dense co-purchase / overhead-sensitive",
    "hybrid/mixed",
)
ALLOWED_CATEGORIES = set(CATEGORY_ORDER)
STATUS_OK = "OK"
STATUS_OOM = "OOM"
STATUS_SKIPPED_MEMORY = "SKIPPED_MEMORY"
STATUS_SKIPPED_BY_MANIFEST = "SKIPPED_BY_MANIFEST"
STATUS_ERROR = "ERROR"
STATUS_VALUES = (
    STATUS_OK,
    STATUS_OOM,
    STATUS_SKIPPED_MEMORY,
    STATUS_SKIPPED_BY_MANIFEST,
    STATUS_ERROR,
)
HEAVY_MEMORY_PATHS = {"TC_REORDERED", "HYBRID_TC_CUDA", "STAGED_REUSE", "TC_SPARSE"}
GIB = float(1024 ** 3)
MEMORY_POLICIES = ("optimistic", "conservative", "manifest_only")


@dataclass(frozen=True)
class EvalMatrixCase:
    name: str
    source: str
    category: str
    group: str
    M: int
    K: int
    Ns: Sequence[int]
    loader: Callable[[], Dict[str, object]]
    tags: Tuple[str, ...] = ()
    notes: str = ""
    disable_paths: Tuple[str, ...] = ()
    memory_sensitive: bool = False
    metadata: Optional[Mapping[str, object]] = None
    data_path: str = ""
    max_N: Optional[int] = None

    @property
    def size_tag(self) -> str:
        return f"{self.M}x{self.K}"

    @property
    def case_key(self) -> Tuple[str, str]:
        return self.name, self.size_tag


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if abs(float(den)) > 1e-12 else 0.0


def bytes_to_gb(num_bytes: float) -> float:
    return float(num_bytes) / GIB


def is_ok_status(status: object) -> bool:
    return str(status) == STATUS_OK


def status_counts(rows: Sequence[Mapping[str, object]], status_key: str = "status") -> Dict[str, int]:
    counts = {status: 0 for status in STATUS_VALUES}
    for row in rows:
        status = str(row.get(status_key, STATUS_ERROR))
        counts.setdefault(status, 0)
        counts[status] += 1
    return counts


def first_non_ok_status(counts: Mapping[str, int]) -> str:
    for status in (STATUS_SKIPPED_BY_MANIFEST, STATUS_SKIPPED_MEMORY, STATUS_OOM, STATUS_ERROR):
        if int(counts.get(status, 0)) > 0:
            return status
    return STATUS_ERROR


def is_oom_exception(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(token in message for token in [
        "out of memory",
        "cudaerrormemoryallocation",
        "memory allocation",
        "cublas_status_alloc_failed",
        "cuda out of memory",
    ])


def cleanup_cuda_state() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def geomean(values: Sequence[float]) -> float:
    positives = [float(v) for v in values if float(v) > 0.0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = clamp01(q) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def ensure_results_tree(root: str) -> Dict[str, str]:
    paths = {
        "root": root,
        "tables": os.path.join(root, "tables"),
        "csv": os.path.join(root, "csv"),
        "json": os.path.join(root, "json"),
        "plots": os.path.join(root, "plots"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def write_csv_rows(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: object) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def write_latex_table(
    path: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    caption: str,
    label: str,
    alignment: Optional[str] = None,
) -> None:
    alignment = alignment or ("l" + "r" * max(0, len(headers) - 1))
    with open(path, "w") as fh:
        fh.write("\\begin{table}[t]\n")
        fh.write("\\centering\n")
        fh.write(f"\\caption{{{caption}}}\n")
        fh.write(f"\\label{{{label}}}\n")
        fh.write(f"\\begin{{tabular}}{{{alignment}}}\n")
        fh.write("\\toprule\n")
        fh.write(" & ".join(headers) + " \\\\\n")
        fh.write("\\midrule\n")
        for row in rows:
            fh.write(" & ".join(str(value) for value in row) + " \\\\\n")
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")
        fh.write("\\end{table}\n")


def _compute_row_stats(rowptr: Sequence[int]) -> Tuple[float, float, float]:
    m = max(0, len(rowptr) - 1)
    if m == 0:
        return 0.0, 0.0, 0.0
    lengths = [rowptr[r + 1] - rowptr[r] for r in range(m)]
    avg = sum(lengths) / float(m)
    var = sum((length - avg) ** 2 for length in lengths) / float(max(1, m))
    density = safe_div(rowptr[-1], max(1, m))
    return avg, math.sqrt(var), density


def _coo_to_csr(
    rows: Sequence[int],
    cols: Sequence[int],
    vals: Optional[Sequence[float]],
    M: int,
    K: int,
    unit_values: bool = False,
) -> Dict[str, object]:
    row_buckets: List[Dict[int, float]] = [dict() for _ in range(M)]
    input_vals = list(vals) if vals is not None else []
    for idx, (r, c) in enumerate(zip(rows, cols)):
        if r < 0 or r >= M or c < 0 or c >= K:
            continue
        value = 1.0 if unit_values or not input_vals else float(input_vals[idx])
        if c not in row_buckets[r]:
            row_buckets[r][c] = value

    rowptr = [0]
    colind: List[int] = []
    out_vals: List[float] = []
    for cols_to_vals in row_buckets:
        for c in sorted(cols_to_vals.keys()):
            colind.append(c)
            out_vals.append(cols_to_vals[c])
        rowptr.append(len(colind))

    avg_nnz_per_row, std_nnz_per_row, _ = _compute_row_stats(rowptr)
    nnz = len(colind)
    density = safe_div(nnz, max(1, M * K))
    return {
        "rowptr": torch.tensor(rowptr, dtype=torch.int32),
        "colind": torch.tensor(colind, dtype=torch.int32),
        "vals": torch.tensor(out_vals, dtype=torch.float32),
        "M": M,
        "K": K,
        "nnz": nnz,
        "avg_nnz_per_row": avg_nnz_per_row,
        "std_nnz_per_row": std_nnz_per_row,
        "density": density,
    }


def load_matrix_market(
    path: str,
    symmetrize: bool = True,
    unit_values: bool = True,
) -> Dict[str, object]:
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    nrows = ncols = nnz = 0
    symmetric = False
    header_seen = False

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("%"):
                if not header_seen and line.lower().startswith("%%matrixmarket"):
                    symmetric = "symmetric" in line.lower()
                    header_seen = True
                continue
            if nrows == 0:
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid MatrixMarket size line in {path}")
                nrows, ncols, nnz = map(int, parts[:3])
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            v = 1.0 if unit_values or len(parts) < 3 else float(parts[2])
            rows.append(r)
            cols.append(c)
            vals.append(v)
            if symmetrize and symmetric and r != c:
                rows.append(c)
                cols.append(r)
                vals.append(v)

    if nrows <= 0 or ncols <= 0:
        raise ValueError(f"Failed to parse MatrixMarket file {path}")
    return _coo_to_csr(rows, cols, vals, nrows, ncols, unit_values=unit_values)


def load_edge_list(
    path: str,
    num_nodes: Optional[int] = None,
    delimiter: Optional[str] = None,
    directed: bool = False,
    symmetrize: bool = True,
    one_indexed: bool = False,
    unit_values: bool = True,
) -> Dict[str, object]:
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    max_node = -1

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split(delimiter)
            parts = [token for token in parts if token]
            if len(parts) < 2:
                continue
            r = int(parts[0]) - (1 if one_indexed else 0)
            c = int(parts[1]) - (1 if one_indexed else 0)
            if r < 0 or c < 0:
                continue
            max_node = max(max_node, r, c)
            rows.append(r)
            cols.append(c)
            vals.append(1.0)
            if symmetrize and not directed and r != c:
                rows.append(c)
                cols.append(r)
                vals.append(1.0)

    dim = max(max_node + 1, num_nodes or 0)
    return _coo_to_csr(rows, cols, vals, dim, dim, unit_values=unit_values)


def load_npz_graph(
    path: str,
    symmetrize: bool = False,
    unit_values: bool = True,
) -> Dict[str, object]:
    csr = None

    try:
        loaded = np.load(path, allow_pickle=False)
        keys = set(loaded.files)
        if {"rowptr", "colind"}.issubset(keys):
            rowptr = np.asarray(loaded["rowptr"], dtype=np.int32)
            colind = np.asarray(loaded["colind"], dtype=np.int32)
            vals = np.asarray(loaded["vals"], dtype=np.float32) if "vals" in keys else np.ones_like(colind, dtype=np.float32)
            if "shape" in keys:
                shape = tuple(int(v) for v in np.asarray(loaded["shape"]).tolist())
                M, K = shape[:2]
            else:
                M = int(loaded["M"]) if "M" in keys else max(0, rowptr.size - 1)
                K = int(loaded["K"]) if "K" in keys else max(0, int(colind.max()) + 1 if colind.size else 0)
            csr = sp.csr_matrix((vals, colind, rowptr), shape=(M, K), dtype=np.float32)
        elif {"indptr", "indices"}.issubset(keys):
            indptr = np.asarray(loaded["indptr"], dtype=np.int32)
            indices = np.asarray(loaded["indices"], dtype=np.int32)
            vals = np.asarray(loaded["data"], dtype=np.float32) if "data" in keys else np.ones_like(indices, dtype=np.float32)
            shape = tuple(int(v) for v in np.asarray(loaded["shape"]).tolist())
            csr = sp.csr_matrix((vals, indices, indptr), shape=shape[:2], dtype=np.float32)
        elif "edge_index" in keys:
            edge_index = np.asarray(loaded["edge_index"], dtype=np.int64)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"edge_index in {path} must have shape [2, nnz]")
            num_nodes = int(loaded["num_nodes"]) if "num_nodes" in keys else int(edge_index.max()) + 1
            vals = np.ones(edge_index.shape[1], dtype=np.float32)
            csr = sp.coo_matrix((vals, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    except Exception:
        csr = None

    if csr is None:
        csr = sp.load_npz(path).tocsr()

    csr.sum_duplicates()
    csr.eliminate_zeros()
    if unit_values:
        csr.data = np.ones_like(csr.data, dtype=np.float32)
    else:
        csr.data = np.asarray(csr.data, dtype=np.float32)

    if symmetrize:
        csr = (csr + csr.transpose()).tocsr()
        csr.sum_duplicates()
        csr.eliminate_zeros()
        if unit_values:
            csr.data = np.ones_like(csr.data, dtype=np.float32)

    rowptr = np.asarray(csr.indptr, dtype=np.int32)
    colind = np.asarray(csr.indices, dtype=np.int32)
    vals = np.asarray(csr.data, dtype=np.float32)
    avg_nnz_per_row, std_nnz_per_row, _ = _compute_row_stats(rowptr.tolist())
    nnz = int(csr.nnz)
    density = safe_div(nnz, max(1, int(csr.shape[0]) * int(csr.shape[1])))
    return {
        "rowptr": torch.from_numpy(rowptr.copy()),
        "colind": torch.from_numpy(colind.copy()),
        "vals": torch.from_numpy(vals.copy()),
        "M": int(csr.shape[0]),
        "K": int(csr.shape[1]),
        "nnz": nnz,
        "avg_nnz_per_row": avg_nnz_per_row,
        "std_nnz_per_row": std_nnz_per_row,
        "density": density,
    }


def _make_real_loader(entry: Mapping[str, object]) -> Callable[[], Dict[str, object]]:
    path = str(entry["path"])
    fmt = str(entry.get("format", "mtx")).lower()
    if fmt == "mtx":
        symmetrize = bool(entry.get("symmetrize", True))
        unit_values = bool(entry.get("unit_values", True))
        return lambda: load_matrix_market(path, symmetrize=symmetrize, unit_values=unit_values)
    if fmt == "npz":
        return lambda: load_npz_graph(
            path,
            symmetrize=bool(entry.get("symmetrize", False)),
            unit_values=bool(entry.get("unit_values", True)),
        )
    if fmt in {"edge", "edgelist", "csv", "tsv", "txt"}:
        return lambda: load_edge_list(
            path,
            num_nodes=entry.get("num_nodes"),
            delimiter=entry.get("delimiter"),
            directed=bool(entry.get("directed", False)),
            symmetrize=bool(entry.get("symmetrize", True)),
            one_indexed=bool(entry.get("one_indexed", False)),
            unit_values=bool(entry.get("unit_values", True)),
        )
    raise ValueError(f"Unsupported dataset format {fmt} for {path}")


def load_real_cases(manifest_path: Optional[str], default_ns: Sequence[int]) -> List[EvalMatrixCase]:
    if not manifest_path or not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r") as fh:
        payload = json.load(fh)

    entries = payload.get("datasets", payload)
    cases: List[EvalMatrixCase] = []
    for entry in entries:
        if not bool(entry.get("enabled", True)):
            continue
        path = str(entry.get("path", ""))
        if not path:
            continue
        if not os.path.exists(path):
            if bool(entry.get("skip_if_missing", False)):
                continue
            raise FileNotFoundError(f"Dataset path not found: {path}")
        category = str(entry["category"])
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Dataset {entry['name']} has invalid category {category}")
        loader = _make_real_loader(entry)
        group = str(entry.get("group", _category_to_group(category)))
        tags = tuple(entry.get("tags", _group_to_tags(group)))
        notes = str(entry.get("notes", ""))
        disable_paths = tuple(str(path) for path in entry.get("disable_paths", ()))
        max_n = entry.get("max_N")
        ns = tuple(int(n) for n in entry.get("Ns", default_ns))
        if max_n is not None:
            max_n = int(max_n)
            ns = tuple(n for n in ns if n <= max_n)
        manifest_metadata = {
            key: entry.get(key)
            for key in ["M", "K", "nnz", "avg_nnz_per_row", "std_nnz_per_row", "density"]
        }
        if manifest_metadata["M"] is None or manifest_metadata["K"] is None or manifest_metadata["nnz"] is None:
            mat = loader()
            metadata = {
                "M": int(mat["M"]),
                "K": int(mat["K"]),
                "nnz": int(mat["nnz"]),
                "avg_nnz_per_row": float(mat.get("avg_nnz_per_row", 0.0)),
                "std_nnz_per_row": float(mat.get("std_nnz_per_row", 0.0)),
                "density": float(mat.get("density", 0.0)),
            }
        else:
            metadata = {
                "M": int(manifest_metadata["M"]),
                "K": int(manifest_metadata["K"]),
                "nnz": int(manifest_metadata["nnz"]),
                "avg_nnz_per_row": float(manifest_metadata.get("avg_nnz_per_row") or 0.0),
                "std_nnz_per_row": float(manifest_metadata.get("std_nnz_per_row") or 0.0),
                "density": float(manifest_metadata.get("density") or 0.0),
            }
        cases.append(EvalMatrixCase(
            name=str(entry["name"]),
            source="real",
            category=category,
            group=group,
            M=int(metadata["M"]),
            K=int(metadata["K"]),
            Ns=ns,
            loader=loader,
            tags=tags,
            notes=notes,
            disable_paths=disable_paths,
            memory_sensitive=bool(entry.get("memory_sensitive", False)),
            metadata=metadata,
            data_path=path,
            max_N=max_n if max_n is not None else None,
        ))
    return cases


def _category_to_group(category: str) -> str:
    if category in {"hub-dominated power-law", "reordered locality"}:
        return "row_split_targets"
    if category == "ordered sparse / road-network":
        return "baseline_reference"
    if category in {
        "dense block-local / TC-friendly",
        "sparse modular community",
        "dense co-purchase / overhead-sensitive",
    }:
        return "tc_locality_targets"
    return "hybrid_mixed_targets"


def _group_to_tags(group: str) -> Tuple[str, ...]:
    if group == "row_split_targets":
        return ("RoDe",)
    if group == "tc_locality_targets":
        return ("FlashSparse", "Acc-SpMM", "DTC")
    if group == "hybrid_mixed_targets":
        return ("Libra", "RSH")
    return ()


def build_synthetic_cases() -> List[EvalMatrixCase]:
    cases: List[EvalMatrixCase] = []
    for graph_case in harness.build_graph_cases():
        tags = _group_to_tags(graph_case.group)
        for M, K in graph_case.sizes:
            loader = (lambda gc=graph_case, m=M, k=K: gc.builder(m, k, gc.seed))
            cases.append(EvalMatrixCase(
                name=graph_case.name,
                source="synthetic",
                category=graph_case.category,
                group=graph_case.group,
                M=M,
                K=K,
                Ns=tuple(graph_case.Ns),
                loader=loader,
                tags=tags,
                notes="Synthetic regime-targeted case aligned with paper methodology.",
                metadata={"M": M, "K": K},
            ))
    return cases


def collect_cases(
    manifest_path: Optional[str],
    default_ns: Sequence[int],
    include_synthetic: bool = True,
    include_real: bool = True,
) -> List[EvalMatrixCase]:
    cases: List[EvalMatrixCase] = []
    if include_synthetic:
        cases.extend(build_synthetic_cases())
    if include_real:
        cases.extend(load_real_cases(manifest_path, default_ns))
    return cases


def prepare_gpu_matrix(mat: Mapping[str, object]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rowptr = mat["rowptr"].cuda().int()
    colind = mat["colind"].cuda().int()
    vals = mat["vals"].cuda().float()
    return rowptr, colind, vals


def make_aligned_B(K: int, N: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return torch.randn(K, N, generator=gen, device="cuda", dtype=torch.float32)


def make_misaligned_B(K: int, N: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    backing = torch.empty(K * N + 1, device="cuda", dtype=torch.float32)
    view = backing[1:].view(K, N)
    view.normal_(generator=gen)
    if not view.is_contiguous():
        raise RuntimeError("Failed to create a contiguous misaligned B tensor")
    return view


def build_torch_sparse_matrix(
    rowptr: torch.Tensor,
    colind: torch.Tensor,
    vals: torch.Tensor,
    M: int,
    K: int,
    prefer_csr: bool = True,
) -> Tuple[torch.Tensor, str]:
    if prefer_csr:
        try:
            return (
                torch.sparse_csr_tensor(
                    rowptr,
                    colind,
                    vals,
                    size=(M, K),
                    dtype=torch.float32,
                    device=rowptr.device,
                ),
                "csr",
            )
        except Exception:
            pass

    rows = torch.repeat_interleave(
        torch.arange(M, device=rowptr.device, dtype=torch.int64),
        (rowptr[1:] - rowptr[:-1]).to(torch.int64),
    )
    sparse = torch.sparse_coo_tensor(
        torch.stack([rows, colind.to(torch.int64)]),
        vals,
        (M, K),
        dtype=torch.float32,
        device=rowptr.device,
    ).coalesce()
    return sparse, "coo"


def estimate_spmm_bytes(nnz: int, M: int, N: int) -> float:
    csr_bytes = 12.0 * nnz
    dense_read = 4.0 * nnz * N
    dense_write = 4.0 * M * N
    return csr_bytes + dense_read + dense_write


def bandwidth_gbps(nnz: int, M: int, N: int, exec_ms: float) -> float:
    if exec_ms <= 1e-12:
        return 0.0
    return estimate_spmm_bytes(nnz, M, N) / (exec_ms * 1e6)


def resolve_memory_budget_gb(explicit_budget_gb: Optional[float]) -> float:
    if explicit_budget_gb is not None:
        return float(explicit_budget_gb)
    if not torch.cuda.is_available():
        return 0.0
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return bytes_to_gb(float(free_bytes)) * 0.80
    except Exception:
        device = torch.cuda.current_device()
        total_bytes = float(torch.cuda.get_device_properties(device).total_memory)
        return bytes_to_gb(total_bytes) * 0.60


def estimate_path_memory_bytes(
    path: str,
    M: int,
    K: int,
    N: int,
    nnz: int,
    memory_sensitive: bool = False,
) -> int:
    # Conservative pre-launch heuristic used only when the harness is in
    # conservative memory mode for crowded/shared nodes. Dedicated paper runs
    # default to optimistic execution and treat OOM recovery, not free-memory
    # snapshots on a busy machine, as the primary control path.
    csr_bytes = 4.0 * (M + 1) + 8.0 * nnz
    output_bytes = 4.0 * M * N
    b_bytes = 4.0 * K * N
    base_bytes = csr_bytes + output_bytes + b_bytes

    extra_bytes = {
        "CSR_DIRECT": 64.0 * 1024 ** 2 + 0.05 * output_bytes + 0.02 * b_bytes,
        "CSR_ADAPTIVE": 96.0 * 1024 ** 2 + 0.10 * output_bytes + 10.0 * nnz + 24.0 * M,
        "STAGED_REUSE": 192.0 * 1024 ** 2 + 0.25 * output_bytes + 0.25 * b_bytes + 18.0 * nnz,
        "TC_SPARSE": 384.0 * 1024 ** 2 + 0.75 * output_bytes + 0.50 * b_bytes + 20.0 * nnz,
        "ROW_SPLIT_CUDA": 128.0 * 1024 ** 2 + 0.20 * output_bytes + 14.0 * nnz + 32.0 * M,
        "TC_REORDERED": 512.0 * 1024 ** 2 + 1.25 * output_bytes + 0.75 * b_bytes + 28.0 * nnz + 32.0 * M,
        "HYBRID_TC_CUDA": 384.0 * 1024 ** 2 + 0.90 * output_bytes + 0.50 * b_bytes + 22.0 * nnz + 32.0 * M,
        "CUSPARSE": 96.0 * 1024 ** 2 + 0.10 * output_bytes + 0.05 * b_bytes + 12.0 * nnz,
        "TORCH_SPARSE": 160.0 * 1024 ** 2 + 0.20 * output_bytes + 0.10 * b_bytes + 16.0 * nnz,
    }.get(path, 96.0 * 1024 ** 2)

    fragmentation_factor = 1.18 if not memory_sensitive else 1.30
    return int((base_bytes + extra_bytes) * fragmentation_factor)


class ExperimentRunner:
    def __init__(
        self,
        warmup: int,
        iters: int,
        portfolio: str = "MAIN",
        seed: int = 123,
        memory_budget_gb: Optional[float] = None,
        memory_policy: str = "optimistic",
        skip_memory_heavy_paths: bool = False,
        continue_on_error: bool = True,
    ):
        self.warmup = warmup
        self.iters = iters
        self.portfolio = portfolio
        self.seed = seed
        if memory_policy not in MEMORY_POLICIES:
            raise ValueError(f"Unsupported memory policy {memory_policy}")
        # Optimistic is the default paper-run mode for dedicated GPUs: try to
        # execute, recover from OOM, and keep going. Conservative mode exists
        # for crowded/shared nodes where current free-memory observations are
        # useful for debugging but should not define the scientific baseline.
        self.memory_policy = memory_policy
        self.explicit_memory_budget_gb = memory_budget_gb
        self.memory_budget_gb = resolve_memory_budget_gb(memory_budget_gb) if memory_policy == "conservative" else None
        self.skip_memory_heavy_paths = skip_memory_heavy_paths
        self.continue_on_error = continue_on_error
        self._mat_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._warm_oracle_cache: Dict[Tuple[str, str, int], Dict[str, object]] = {}
        self._cold_oracle_cache: Dict[Tuple[str, str, int], Dict[str, object]] = {}
        self._warm_router_cache: Dict[Tuple[str, str, int], Dict[str, object]] = {}
        self._router_plan_cache: Dict[Tuple[str, str, int], Dict[str, object]] = {}
        self._warm_external_cache: Dict[Tuple[str, str, int], Dict[str, Dict[str, object]]] = {}
        self._feature_cache: Dict[Tuple[str, str, int], Dict[str, float]] = {}
        self._row_split_plan_cache: Dict[Tuple[str, str], object] = {}
        self._row_split_no_long_cache: Dict[Tuple[str, str], object] = {}
        self._torch_sparse_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, str]] = {}

    def load_matrix(self, case: EvalMatrixCase) -> Dict[str, object]:
        key = case.case_key
        if key not in self._mat_cache:
            self._mat_cache[key] = case.loader()
        return self._mat_cache[key]

    def _make_B(self, case: EvalMatrixCase, N: int, seed_offset: int = 0) -> torch.Tensor:
        return make_aligned_B(case.K, N, self.seed + seed_offset + N)

    def _portfolio_paths(self) -> Sequence[str]:
        return FULL_PATHS if self.portfolio == "FULL" else MAIN_PATHS

    def _case_metadata(self, case: EvalMatrixCase, mat: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
        metadata = dict(case.metadata or {})
        source = mat if mat is not None else None
        if source is None and ("nnz" not in metadata or "M" not in metadata or "K" not in metadata):
            source = self.load_matrix(case)
        if source is not None:
            metadata.setdefault("M", int(source["M"]))
            metadata.setdefault("K", int(source["K"]))
            metadata.setdefault("nnz", int(source["nnz"]))
            metadata.setdefault("avg_nnz_per_row", float(source.get("avg_nnz_per_row", 0.0)))
            metadata.setdefault("std_nnz_per_row", float(source.get("std_nnz_per_row", 0.0)))
            metadata.setdefault("density", float(source.get("density", 0.0)))
        return metadata

    def _memory_limit_bytes(self) -> float:
        if self.memory_policy != "conservative":
            self.memory_budget_gb = None
            return 0.0
        self.memory_budget_gb = resolve_memory_budget_gb(self.explicit_memory_budget_gb)
        return float(self.memory_budget_gb or 0.0) * GIB

    def _retry_enabled(self) -> bool:
        return self.memory_policy == "optimistic"

    def _run_with_oom_retry(
        self,
        fn: Callable[[], object],
        context: str,
    ) -> Tuple[Optional[object], Optional[Tuple[str, str]], Dict[str, bool]]:
        oom_retry_attempted = False
        retry_budget = 1 if self._retry_enabled() else 0
        for attempt in range(retry_budget + 1):
            try:
                value = fn()
                return value, None, {
                    "oom_retry_attempted": oom_retry_attempted,
                    "oom_retry_succeeded": oom_retry_attempted,
                }
            except Exception as exc:
                status = STATUS_OOM if is_oom_exception(exc) else STATUS_ERROR
                cleanup_cuda_state()
                if status == STATUS_OOM and attempt < retry_budget:
                    oom_retry_attempted = True
                    continue
                if status == STATUS_ERROR and not self.continue_on_error:
                    raise
                return None, (status, f"{context}: {exc}"), {
                    "oom_retry_attempted": oom_retry_attempted,
                    "oom_retry_succeeded": False,
                }
        return None, (STATUS_ERROR, f"{context}: unexpected_retry_exhaustion"), {
            "oom_retry_attempted": oom_retry_attempted,
            "oom_retry_succeeded": False,
        }

    def _preflight_status(
        self,
        case: EvalMatrixCase,
        path: str,
        N: int,
        metadata: Mapping[str, object],
    ) -> Tuple[Optional[str], str, int]:
        estimate_bytes = estimate_path_memory_bytes(
            path, int(metadata["M"]), int(metadata["K"]), N, int(metadata["nnz"]), case.memory_sensitive)
        if path in set(case.disable_paths):
            return STATUS_SKIPPED_BY_MANIFEST, "manifest_disable_paths", estimate_bytes
        if self.memory_policy == "conservative":
            if self.skip_memory_heavy_paths and case.source == "real" and path in HEAVY_MEMORY_PATHS:
                return STATUS_SKIPPED_MEMORY, "skip_memory_heavy_paths_flag", estimate_bytes
            limit_bytes = self._memory_limit_bytes()
            if limit_bytes > 0.0:
                threshold = 0.92 if not case.memory_sensitive else 0.85
                if float(estimate_bytes) > limit_bytes * threshold:
                    return STATUS_SKIPPED_MEMORY, "memory_estimate_exceeds_budget", estimate_bytes
        return None, "", estimate_bytes

    def _make_result(
        self,
        path: str,
        estimate_bytes: int,
        status: str,
        status_reason: str,
        nnz: int,
        N: int,
        plan_ms: Optional[float] = None,
        exec_ms: Optional[float] = None,
        total_ms: Optional[float] = None,
        attempted: bool = False,
        oom_retry_attempted: bool = False,
        oom_retry_succeeded: bool = False,
    ) -> Dict[str, object]:
        timed = status == STATUS_OK and total_ms is not None
        gflops = 0.0
        if timed and exec_ms is not None and exec_ms > 1e-12:
            gflops = (2.0 * float(nnz) * float(N)) / (float(exec_ms) * 1e6)
        return {
            "path": path,
            "status": status,
            "status_reason": status_reason,
            "attempted": attempted,
            "timed": timed,
            "memory_policy": self.memory_policy,
            "oom_retry_attempted": bool(oom_retry_attempted),
            "oom_retry_succeeded": bool(oom_retry_succeeded),
            "plan_ms": plan_ms if timed else None,
            "exec_ms": exec_ms if timed else None,
            "total_ms": total_ms if timed else None,
            "gflops": gflops if timed else 0.0,
            "memory_estimate_bytes": int(estimate_bytes),
            "memory_estimate_gb": bytes_to_gb(estimate_bytes),
            "memory_limit_gb": self.memory_budget_gb,
            "in_main_portfolio": path in MAIN_PATHS,
            "legacy_baseline": path not in MAIN_PATHS and path not in EXTERNAL_BASELINE_PATHS,
            "external_baseline": path in EXTERNAL_BASELINE_PATHS,
        }

    def _prepare_gpu_inputs(
        self,
        case: EvalMatrixCase,
        mat: Mapping[str, object],
        N: int,
        seed_offset: int,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Tuple[str, str]], Dict[str, bool]]:
        def allocate_inputs() -> Dict[str, torch.Tensor]:
            rowptr, colind, vals = prepare_gpu_matrix(mat)
            B = self._make_B(case, N, seed_offset=seed_offset)
            return {
                "rowptr": rowptr,
                "colind": colind,
                "vals": vals,
                "B": B,
            }
        value, failure, retry_info = self._run_with_oom_retry(allocate_inputs, "gpu_input_allocation_failed")
        return value, failure, retry_info

    def _build_plan(self, path: str, mat: Mapping[str, object], N: int):
        M = int(mat["M"])
        K = int(mat["K"])
        if path == "CSR_DIRECT":
            return None
        if path == "CSR_ADAPTIVE":
            return ra_spmm.make_csr_adaptive_plan(mat["rowptr"], mat["colind"], M, K)
        if path == "STAGED_REUSE":
            return ra_spmm.make_staged_reuse_plan(mat["rowptr"], mat["colind"], mat["vals"], M, K)
        if path == "TC_SPARSE":
            return ra_spmm.make_tc_sparse_plan(mat["rowptr"], mat["colind"], mat["vals"], M, K, True)
        if path == "ROW_SPLIT_CUDA":
            return ra_spmm.make_row_split_plan(mat["rowptr"], M, K)
        if path == "TC_REORDERED":
            return ra_spmm.make_tc_reordered_plan(mat["rowptr"], mat["colind"], mat["vals"], M, K, N)
        if path == "HYBRID_TC_CUDA":
            return ra_spmm.make_hybrid_tc_cuda_plan(mat["rowptr"], mat["colind"], mat["vals"], M, K, N, 0.45)
        if path == "CUSPARSE":
            return None  # cuSPARSE timing is handled by dedicated benchmark helpers
        raise ValueError(f"Unknown path {path}")

    def _plan_valid(self, path: str, plan: object) -> bool:
        if path == "CSR_DIRECT" or path == "CUSPARSE":
            return True
        if path == "TC_REORDERED":
            return bool(getattr(plan, "active", False))
        return bool(getattr(plan, "valid", True))

    def _run_path_once(self, path: str, plan: object, gpu_inputs: Mapping[str, torch.Tensor]) -> None:
        rowptr = gpu_inputs["rowptr"]
        colind = gpu_inputs["colind"]
        vals = gpu_inputs["vals"]
        B = gpu_inputs["B"]
        out = None
        if path == "CSR_DIRECT":
            out = ra_spmm.spmm_csr_direct(rowptr, colind, vals, B)
        elif path == "CSR_ADAPTIVE":
            out = ra_spmm.run_csr_adaptive_plan(plan, rowptr, colind, vals, B)
        elif path == "STAGED_REUSE":
            out = ra_spmm.run_staged_reuse_plan(plan, B)
        elif path == "TC_SPARSE":
            out = ra_spmm.run_tc_sparse_plan(plan, B)
        elif path == "ROW_SPLIT_CUDA":
            out = ra_spmm.run_row_split_plan(plan, colind, vals, B)
        elif path == "TC_REORDERED":
            out = ra_spmm.run_tc_reordered_plan(plan, B)
        elif path == "HYBRID_TC_CUDA":
            out = ra_spmm.run_hybrid_tc_cuda_plan(plan, B)
        elif path == "CUSPARSE":
            out = ra_spmm.spmm_cusparse(rowptr, colind, vals, B)
        else:
            raise ValueError(f"Unknown path {path}")
        del out

    def _torch_sparse_matrix(
        self,
        case: EvalMatrixCase,
        mat: Mapping[str, object],
        gpu_inputs: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, str]:
        key = case.case_key
        if key not in self._torch_sparse_cache:
            self._torch_sparse_cache[key] = build_torch_sparse_matrix(
                gpu_inputs["rowptr"],
                gpu_inputs["colind"],
                gpu_inputs["vals"],
                int(mat["M"]),
                int(mat["K"]),
                prefer_csr=True,
            )
        return self._torch_sparse_cache[key]

    def _measure_external_path_with_gpu_inputs(
        self,
        case: EvalMatrixCase,
        path: str,
        mat: Mapping[str, object],
        gpu_inputs: Mapping[str, torch.Tensor],
        N: int,
        estimate_bytes: int,
    ) -> Dict[str, object]:
        nnz = int(mat["nnz"])

        def measure_impl() -> Dict[str, object]:
            if path == "TORCH_SPARSE":
                sparse_a, layout = self._torch_sparse_matrix(case, mat, gpu_inputs)

                def run_sparse_mm() -> None:
                    out = torch.sparse.mm(sparse_a, gpu_inputs["B"])
                    del out

                try:
                    exec_ms = measure_cuda_ms(run_sparse_mm, self.warmup, max(1, self.iters))
                except Exception:
                    if layout != "csr":
                        raise
                    sparse_a, layout = build_torch_sparse_matrix(
                        gpu_inputs["rowptr"],
                        gpu_inputs["colind"],
                        gpu_inputs["vals"],
                        int(mat["M"]),
                        int(mat["K"]),
                        prefer_csr=False,
                    )
                    self._torch_sparse_cache[case.case_key] = (sparse_a, layout)

                    def run_sparse_mm_fallback() -> None:
                        out = torch.sparse.mm(sparse_a, gpu_inputs["B"])
                        del out

                    exec_ms = measure_cuda_ms(run_sparse_mm_fallback, self.warmup, max(1, self.iters))

                return {
                    "plan_ms": 0.0,
                    "exec_ms": float(exec_ms),
                    "total_ms": float(exec_ms),
                    "torch_layout": layout,
                }

            raise ValueError(f"Unknown external baseline path {path}")

        measured, failure, retry_info = self._run_with_oom_retry(measure_impl, f"{path}_measurement_failed")
        if failure is not None:
            status, reason = failure
            return self._make_result(
                path, estimate_bytes, status, reason, nnz, N,
                attempted=True,
                oom_retry_attempted=retry_info["oom_retry_attempted"],
                oom_retry_succeeded=retry_info["oom_retry_succeeded"],
            )

        result = self._make_result(
            path, estimate_bytes, STATUS_OK, "", nnz, N,
            float(measured["plan_ms"]),
            float(measured["exec_ms"]),
            float(measured["total_ms"]),
            attempted=True,
            oom_retry_attempted=retry_info["oom_retry_attempted"],
            oom_retry_succeeded=retry_info["oom_retry_succeeded"],
        )
        if path == "CUSPARSE":
            result["cusparse_algorithm"] = measured.get("cusparse_algorithm", "")
        if path == "TORCH_SPARSE":
            result["torch_layout"] = measured.get("torch_layout", "")
        return result

    def _measure_path_with_gpu_inputs(
        self,
        case: EvalMatrixCase,
        path: str,
        mat: Mapping[str, object],
        gpu_inputs: Mapping[str, torch.Tensor],
        N: int,
        estimate_bytes: int,
        cold_mode: bool,
    ) -> Dict[str, object]:
        nnz = int(mat["nnz"])
        def measure_impl() -> Tuple[float, float, float]:
            if path == "CUSPARSE":
                if cold_mode:
                    measured = ra_spmm.benchmark_cusparse_cold(
                        gpu_inputs["rowptr"],
                        gpu_inputs["colind"],
                        gpu_inputs["vals"],
                        gpu_inputs["B"],
                        max(1, self.iters),
                    )
                else:
                    measured = ra_spmm.benchmark_cusparse(
                        gpu_inputs["rowptr"],
                        gpu_inputs["colind"],
                        gpu_inputs["vals"],
                        gpu_inputs["B"],
                        warmup=self.warmup,
                        iters=max(1, self.iters),
                    )
                return (
                    float(measured["plan_ms"]),
                    float(measured["exec_ms"]),
                    float(measured["total_ms"]),
                )

            if path == "CSR_DIRECT":
                exec_ms = measure_cuda_ms(
                    lambda: self._run_path_once(path, None, gpu_inputs),
                    self.warmup,
                    max(1, self.iters),
                )
                return 0.0, exec_ms, exec_ms

            if not cold_mode:
                plan = None
                try:
                    plan = self._build_plan(path, mat, N)
                    if not self._plan_valid(path, plan):
                        raise RuntimeError("inactive_or_invalid_plan")
                    exec_ms = measure_cuda_ms(
                        lambda: self._run_path_once(path, plan, gpu_inputs),
                        self.warmup,
                        max(1, self.iters),
                    )
                    return 0.0, exec_ms, exec_ms
                finally:
                    del plan
                    cleanup_cuda_state()

            plan_sum = 0.0
            exec_sum = 0.0
            iterations = max(1, self.iters)
            for _ in range(iterations):
                plan = None
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    plan = self._build_plan(path, mat, N)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    if not self._plan_valid(path, plan):
                        raise RuntimeError("inactive_or_invalid_plan")
                    exec_ms = measure_cuda_ms(
                        lambda: self._run_path_once(path, plan, gpu_inputs),
                        0,
                        1,
                    )
                    plan_sum += (t1 - t0) * 1000.0
                    exec_sum += exec_ms
                finally:
                    del plan
                    cleanup_cuda_state()

            avg_plan_ms = plan_sum / float(iterations)
            avg_exec_ms = exec_sum / float(iterations)
            return avg_plan_ms, avg_exec_ms, avg_plan_ms + avg_exec_ms

        measured, failure, retry_info = self._run_with_oom_retry(measure_impl, f"{path}_measurement_failed")
        if failure is not None:
            status, reason = failure
            return self._make_result(
                path, estimate_bytes, status, reason, nnz, N,
                attempted=True,
                oom_retry_attempted=retry_info["oom_retry_attempted"],
                oom_retry_succeeded=retry_info["oom_retry_succeeded"],
            )

        plan_ms, exec_ms, total_ms = measured
        return self._make_result(
            path, estimate_bytes, STATUS_OK, "", nnz, N,
            plan_ms, exec_ms, total_ms,
            attempted=True,
            oom_retry_attempted=retry_info["oom_retry_attempted"],
            oom_retry_succeeded=retry_info["oom_retry_succeeded"],
        )

    def _evaluate_paths(
        self,
        case: EvalMatrixCase,
        N: int,
        paths: Sequence[str],
        cold_mode: bool,
        seed_offset: int,
    ) -> Dict[str, Dict[str, object]]:
        mat = self.load_matrix(case)
        metadata = self._case_metadata(case, mat)
        results: Dict[str, Dict[str, object]] = {}
        runnable: List[Tuple[str, int]] = []
        for path in paths:
            preflight_status, reason, estimate_bytes = self._preflight_status(case, path, N, metadata)
            if preflight_status is not None:
                results[path] = self._make_result(
                    path, estimate_bytes, preflight_status, reason, int(metadata["nnz"]), N, attempted=False)
            else:
                runnable.append((path, estimate_bytes))

        if runnable:
            gpu_inputs, base_failure, retry_info = self._prepare_gpu_inputs(case, mat, N, seed_offset)
            if gpu_inputs is None:
                failure_status, failure_reason = base_failure or (STATUS_ERROR, "gpu_input_allocation_failed")
                for path, estimate_bytes in runnable:
                    results[path] = self._make_result(
                        path, estimate_bytes, failure_status, failure_reason, int(metadata["nnz"]), N,
                        attempted=True,
                        oom_retry_attempted=retry_info["oom_retry_attempted"],
                        oom_retry_succeeded=retry_info["oom_retry_succeeded"],
                    )
            else:
                try:
                    for path, estimate_bytes in runnable:
                        results[path] = self._measure_path_with_gpu_inputs(
                            case, path, mat, gpu_inputs, N, estimate_bytes, cold_mode)
                finally:
                    gpu_inputs.clear()
                    cleanup_cuda_state()

        return {path: results[path] for path in paths}

    def _assemble_oracle_result(
        self,
        case: EvalMatrixCase,
        N: int,
        path_results: Mapping[str, Mapping[str, object]],
        mode: str,
    ) -> Dict[str, object]:
        mat = self.load_matrix(case)
        ok_paths = [
            result for result in path_results.values()
            if is_ok_status(result["status"]) and result["total_ms"] is not None
        ]
        if ok_paths:
            oracle = min(ok_paths, key=lambda row: float(row["total_ms"]))
            oracle_path = str(oracle["path"])
            oracle_time_ms = float(oracle["total_ms"])
            oracle_plan_ms = float(oracle["plan_ms"] or 0.0)
            oracle_exec_ms = float(oracle["exec_ms"] or 0.0)
        else:
            oracle_path = "NONE"
            oracle_time_ms = math.inf
            oracle_plan_ms = math.inf
            oracle_exec_ms = math.inf
        counts = status_counts(list(path_results.values()))
        dataset_status = "OK" if counts[STATUS_OK] == len(path_results) else (
            "FAILED" if counts[STATUS_OK] == 0 else "PARTIAL")
        return {
            "mode": mode,
            "portfolio": self.portfolio,
            "path_results": dict(path_results),
            "path_times": {
                path: (float(result["total_ms"]) if is_ok_status(result["status"]) and result["total_ms"] is not None else math.inf)
                for path, result in path_results.items()
            },
            "oracle_path": oracle_path,
            "oracle_time_ms": oracle_time_ms,
            "oracle_plan_ms": oracle_plan_ms,
            "oracle_exec_ms": oracle_exec_ms,
            "M": int(mat["M"]),
            "K": int(mat["K"]),
            "N": N,
            "nnz": int(mat["nnz"]),
            "status_counts": counts,
            "dataset_status": dataset_status,
            "memory_limit_gb": self.memory_budget_gb,
        }

    def warm_oracle(self, case: EvalMatrixCase, N: int) -> Dict[str, object]:
        key = (case.name, case.size_tag, N)
        if key not in self._warm_oracle_cache:
            path_results = self._evaluate_paths(case, N, self._portfolio_paths(), False, seed_offset=1)
            self._warm_oracle_cache[key] = self._assemble_oracle_result(case, N, path_results, "warm")
        return self._warm_oracle_cache[key]

    def cold_oracle(self, case: EvalMatrixCase, N: int) -> Dict[str, object]:
        key = (case.name, case.size_tag, N)
        if key not in self._cold_oracle_cache:
            path_results = self._evaluate_paths(case, N, self._portfolio_paths(), True, seed_offset=17)
            self._cold_oracle_cache[key] = self._assemble_oracle_result(case, N, path_results, "cold")
        return self._cold_oracle_cache[key]

    def warm_router(self, case: EvalMatrixCase, N: int) -> Dict[str, object]:
        key = (case.name, case.size_tag, N)
        if key not in self._warm_router_cache:
            mat = self.load_matrix(case)
            plan = self.router_plan(case, N)
            chosen_path = str(plan["chosen_path"])
            chosen_result = self._evaluate_paths(case, N, [chosen_path], False, seed_offset=33)[chosen_path]
            self._warm_router_cache[key] = {
                "mode": "warm",
                "portfolio": self.portfolio,
                "router_path": chosen_path,
                "timing": {
                    "plan_ms": chosen_result["plan_ms"],
                    "exec_ms": chosen_result["exec_ms"],
                    "total_ms": chosen_result["total_ms"],
                    "gflops": chosen_result["gflops"],
                },
                "status": chosen_result["status"],
                "status_reason": chosen_result["status_reason"],
                "path_result": chosen_result,
                "plan": plan,
                "M": int(mat["M"]),
                "K": int(mat["K"]),
                "N": N,
                "nnz": int(mat["nnz"]),
                "memory_limit_gb": self.memory_budget_gb,
            }
        return self._warm_router_cache[key]

    def router_plan(self, case: EvalMatrixCase, N: int) -> Dict[str, object]:
        key = (case.name, case.size_tag, N)
        if key not in self._router_plan_cache:
            mat = self.load_matrix(case)
            self._router_plan_cache[key] = ra_spmm.make_router_plan(
                mat["rowptr"], mat["colind"], mat["vals"], mat["M"], mat["K"], N, self.portfolio)
        return dict(self._router_plan_cache[key])

    def warm_external_baselines(self, case: EvalMatrixCase, N: int) -> Dict[str, Dict[str, object]]:
        key = (case.name, case.size_tag, N)
        if key not in self._warm_external_cache:
            mat = self.load_matrix(case)
            metadata = self._case_metadata(case, mat)
            results: Dict[str, Dict[str, object]] = {}
            runnable: List[Tuple[str, int]] = []
            for path in EXTERNAL_BASELINE_PATHS:
                preflight_status, reason, estimate_bytes = self._preflight_status(case, path, N, metadata)
                if preflight_status is not None:
                    results[path] = self._make_result(
                        path, estimate_bytes, preflight_status, reason, int(metadata["nnz"]), N, attempted=False)
                else:
                    runnable.append((path, estimate_bytes))

            if runnable:
                gpu_inputs, base_failure, retry_info = self._prepare_gpu_inputs(case, mat, N, seed_offset=49)
                if gpu_inputs is None:
                    failure_status, failure_reason = base_failure or (STATUS_ERROR, "gpu_input_allocation_failed")
                    for path, estimate_bytes in runnable:
                        results[path] = self._make_result(
                            path, estimate_bytes, failure_status, failure_reason, int(metadata["nnz"]), N,
                            attempted=True,
                            oom_retry_attempted=retry_info["oom_retry_attempted"],
                            oom_retry_succeeded=retry_info["oom_retry_succeeded"],
                        )
                else:
                    try:
                        for path, estimate_bytes in runnable:
                            results[path] = self._measure_external_path_with_gpu_inputs(
                                case, path, mat, gpu_inputs, N, estimate_bytes)
                    finally:
                        gpu_inputs.clear()
                        cleanup_cuda_state()

            self._warm_external_cache[key] = {
                path: results[path]
                for path in EXTERNAL_BASELINE_PATHS
            }
        return self._warm_external_cache[key]

    def features(self, case: EvalMatrixCase, N: int) -> Dict[str, float]:
        key = (case.name, case.size_tag, N)
        if key not in self._feature_cache:
            mat = self.load_matrix(case)
            self._feature_cache[key] = ra_spmm.analyze_matrix(
                mat["rowptr"], mat["colind"], mat["M"], mat["K"], N)
        return dict(self._feature_cache[key])

    def row_split_plan(self, case: EvalMatrixCase):
        key = (case.name, case.size_tag)
        if key not in self._row_split_plan_cache:
            mat = self.load_matrix(case)
            self._row_split_plan_cache[key] = ra_spmm.make_row_split_plan(
                mat["rowptr"], mat["M"], mat["K"])
        return self._row_split_plan_cache[key]

    def row_split_plan_no_long_rows(self, case: EvalMatrixCase):
        key = (case.name, case.size_tag)
        if key not in self._row_split_no_long_cache:
            mat = self.load_matrix(case)
            self._row_split_no_long_cache[key] = ra_spmm.make_row_split_plan_no_long_rows(
                mat["rowptr"], mat["M"], mat["K"])
        return self._row_split_no_long_cache[key]


def _zero_fields(features: Dict[str, float], names: Iterable[str]) -> None:
    for name in names:
        if name in features:
            features[name] = 0.0


def apply_feature_ablation(features: Mapping[str, float], ablation: Optional[str]) -> Dict[str, float]:
    f = dict(features)
    if not ablation or ablation == "none":
        return f
    if ablation == "no_skew":
        _zero_fields(f, [
            "degree_cv",
            "skew_ratio",
            "long_row_fraction",
            "long_row_nnz_fraction",
            "top_1_row_nnz_fraction",
            "top_5_row_nnz_fraction",
            "row_split_affinity_proxy",
        ])
        return f
    if ablation == "no_locality":
        _zero_fields(f, [
            "row_window_colspan_compactness",
            "local_row_similarity_proxy",
            "reordered_locality_proxy",
            "locality_gain_proxy",
            "locality_selectivity_proxy",
            "road_likeness_proxy",
        ])
        return f
    if ablation == "no_mixedness":
        _zero_fields(f, [
            "mixedness_proxy",
            "estimated_tc_partition_ratio",
            "estimated_cuda_partition_ratio",
            "irregular_window_fraction",
        ])
        return f
    raise ValueError(f"Unknown feature ablation {ablation}")


def _direct_suitability(f: Mapping[str, float]) -> float:
    ordered_locality = clamp01(
        float(f["local_row_similarity_proxy"]) *
        clamp01(float(f["row_window_colspan_compactness"]) * 1.5))
    return clamp01(
        0.34 * ordered_locality +
        0.24 * float(f["road_likeness_proxy"]) +
        0.18 * clamp01(1.0 - float(f["long_row_nnz_fraction"])) +
        0.14 * clamp01(1.0 - float(f["degree_cv"]) / 1.5) +
        0.10 * clamp01(1.0 - float(f["irregular_window_fraction"])))


def _row_split_suitability(f: Mapping[str, float], N: int) -> float:
    n_scale = clamp01((float(N) - 128.0) / 384.0)
    dense_regular_signal = (
        clamp01((float(f["avg_nnz_per_row"]) - 96.0) / 96.0) *
        clamp01(float(f["row_window_colspan_compactness"]) / 0.04) *
        clamp01(float(f["row_split_affinity_proxy"]) / 0.30) *
        clamp01((0.08 - float(f["locality_selectivity_proxy"])) / 0.08) *
        clamp01((0.08 - float(f["long_row_nnz_fraction"])) / 0.08) *
        clamp01((0.05 - float(f["mixedness_proxy"])) / 0.05)
    )
    return clamp01(
        0.55 * float(f["row_split_affinity_proxy"]) +
        0.20 * float(f["long_row_nnz_fraction"]) +
        0.15 * clamp01(float(f["top_5_row_nnz_fraction"]) / 0.06) +
        0.10 * n_scale +
        0.45 * dense_regular_signal)


def _tc_suitability(f: Mapping[str, float], N: int) -> float:
    uniformity = clamp01(1.0 - float(f["degree_cv"]) / 1.2)
    reorder_signal = clamp01(float(f["reordered_locality_proxy"]) / 0.45)
    compact_signal = clamp01(float(f["row_window_colspan_compactness"]) / 0.03)
    selective_signal = clamp01(
        max(float(f["locality_selectivity_proxy"]), float(f["locality_gain_proxy"])) / 0.18)
    tc_part_signal = clamp01(float(f["estimated_tc_partition_ratio"]) / 0.20)
    low_row_split_bias = clamp01(1.0 - float(f["row_split_affinity_proxy"]))
    n_scale = clamp01((float(N) - 64.0) / 448.0)
    roadnet_locality_signal = (
        clamp01((float(f["reordered_locality_proxy"]) - 0.70) / 0.12) *
        clamp01((0.05 - float(f["locality_selectivity_proxy"])) / 0.05) *
        clamp01((0.01 - float(f["row_window_colspan_compactness"])) / 0.01) *
        clamp01((5.0 - float(f["avg_nnz_per_row"])) / 3.0) *
        clamp01((0.05 - float(f["long_row_nnz_fraction"])) / 0.05) *
        clamp01((0.10 - float(f["mixedness_proxy"])) / 0.10) *
        clamp01((0.20 - float(f["road_likeness_proxy"])) / 0.20)
    )
    dense_regular_penalty = (
        clamp01((float(f["avg_nnz_per_row"]) - 96.0) / 96.0) *
        clamp01(float(f["row_window_colspan_compactness"]) / 0.04) *
        clamp01((0.05 - float(f["locality_selectivity_proxy"])) / 0.05) *
        clamp01((0.08 - float(f["long_row_nnz_fraction"])) / 0.08) *
        clamp01((0.05 - float(f["mixedness_proxy"])) / 0.05)
    )
    return clamp01(
        0.30 * reorder_signal +
        0.24 * compact_signal +
        0.20 * selective_signal +
        0.12 * tc_part_signal +
        0.08 * uniformity +
        0.04 * low_row_split_bias +
        0.02 * n_scale +
        0.30 * roadnet_locality_signal -
        0.22 * dense_regular_penalty)


def _hybrid_suitability(f: Mapping[str, float], N: int) -> float:
    n_scale = clamp01((float(N) - 256.0) / 256.0)
    tc_part = float(f["estimated_tc_partition_ratio"])
    cuda_part = float(f["estimated_cuda_partition_ratio"])
    balance = safe_div(2.0 * min(tc_part, cuda_part), tc_part + cuda_part + 1e-6)
    # Boost for high-skew graphs where HYBRID's ROW_SPLIT-style CUDA
    # partition with row-swizzle scheduling dominates.
    skew_boost = (clamp01(float(f["row_split_affinity_proxy"]) / 0.55) *
                  clamp01(float(f["long_row_nnz_fraction"]) / 0.04))
    return clamp01(
        0.20 * float(f["mixedness_proxy"]) +
        0.12 * balance +
        0.10 * float(f["irregular_window_fraction"]) +
        0.10 * float(f["locality_selectivity_proxy"]) +
        0.10 * clamp01(tc_part + cuda_part) +
        0.16 * float(f["row_split_affinity_proxy"]) +
        0.14 * cuda_part +
        0.08 * n_scale +
        0.10 * skew_boost)


def _cusparse_suitability(f: Mapping[str, float], N: int) -> float:
    n_scale = clamp01((float(N) - 64.0) / 448.0)
    # cuSPARSE acts as the general-purpose fallback on the fair rerun.
    # It loses mainly on strongly reorder-friendly block-local structure,
    # while many sparse, irregular, and skewed graphs still favor it.
    locality_penalty = (clamp01(float(f["reordered_locality_proxy"]) / 0.45) *
                        clamp01(float(f["row_window_colspan_compactness"]) / 0.03))
    road_penalty = clamp01(float(f["road_likeness_proxy"]) / 0.75)
    density_signal = clamp01(float(f["avg_nnz_per_row"]) / 20.0)
    low_degree_boost = clamp01((10.0 - float(f["avg_nnz_per_row"])) / 10.0)
    irregular_boost = clamp01(float(f["degree_cv"]) / 2.0)
    skew_boost = clamp01(float(f["row_split_affinity_proxy"]) / 0.95)
    selectivity_penalty = clamp01(float(f["locality_selectivity_proxy"]) / 0.30)
    mixed_penalty = (
        clamp01(float(f["mixedness_proxy"]) / 0.85) *
        clamp01(float(f["estimated_tc_partition_ratio"]) / 0.35) *
        clamp01(float(f["estimated_cuda_partition_ratio"]) / 0.35)
    )
    return clamp01(
        0.34
        - 0.30 * locality_penalty
        - 0.08 * road_penalty
        - 0.12 * selectivity_penalty
        - 0.10 * mixed_penalty
        + 0.16 * density_signal
        + 0.14 * low_degree_boost
        + 0.12 * irregular_boost
        + 0.12 * n_scale
        + 0.10 * skew_boost)


def choose_policy_path(
    features: Mapping[str, float],
    N: int,
    allowed_paths: Optional[Sequence[str]] = None,
    feature_ablation: Optional[str] = None,
) -> Tuple[str, str]:
    allowed = set(allowed_paths or MAIN_PATHS)
    f = apply_feature_ablation(features, feature_ablation)
    direct_strength = _direct_suitability(f)
    best_path = "CSR_DIRECT"
    best_effective = direct_strength
    best_reason = "direct_explicit_safe_winner"

    candidates = [
        ("ROW_SPLIT_CUDA", _row_split_suitability(f, N), 0.70, 0.12, "row_split_strong_skew_or_dense_regular"),
        ("TC_REORDERED", _tc_suitability(f, N), 0.34, 0.03, "tc_reordered_narrow_reorder_locality"),
        ("HYBRID_TC_CUDA", _hybrid_suitability(f, N), 0.42, 0.08, "hybrid_real_mixed_tc_cuda_structure"),
        ("CUSPARSE", _cusparse_suitability(f, N), 0.20, 0.01, "cusparse_vendor_library_best_general_default"),
    ]

    # Guard: reject mixedness-only hybrid admission when locality selectivity
    # is absent and the estimated TC partition is large but tile utilization
    # remains weak. This targets heterogeneous_windows / mixed_block_skew cases.
    hybrid_mixed_empty_tile_case = (
        float(f["mixedness_proxy"]) >= 0.70 and
        float(f["locality_selectivity_proxy"]) <= 0.05 and
        float(f["estimated_tc_partition_ratio"]) >= 0.45 and
        float(f["avg_nnz_per_row"]) >= 20.0 and
        float(f["tile_fill_mean"]) <= 0.02
    )
    dense_regular_row_split_case = (
        float(f["avg_nnz_per_row"]) >= 96.0 and
        float(f["row_split_affinity_proxy"]) >= 0.30 and
        float(f["row_window_colspan_compactness"]) >= 0.03 and
        float(f["reordered_locality_proxy"]) <= 0.45 and
        float(f["locality_selectivity_proxy"]) <= 0.03 and
        float(f["long_row_nnz_fraction"]) <= 0.02 and
        float(f["mixedness_proxy"]) <= 0.05
    )
    roadnet_tc_case = (
        float(f["reordered_locality_proxy"]) >= 0.70 and
        float(f["locality_selectivity_proxy"]) <= 0.05 and
        float(f["row_window_colspan_compactness"]) <= 0.01 and
        float(f["avg_nnz_per_row"]) >= 2.4 and
        float(f["avg_nnz_per_row"]) <= 5.0 and
        float(f["long_row_nnz_fraction"]) <= 0.05 and
        float(f["mixedness_proxy"]) <= 0.10 and
        float(f["road_likeness_proxy"]) <= 0.18
    )
    selective_sparse_tc_case = (
        float(f["reordered_locality_proxy"]) >= 0.42 and
        float(f["locality_selectivity_proxy"]) >= 0.35 and
        float(f["row_window_colspan_compactness"]) <= 0.001 and
        float(f["avg_nnz_per_row"]) >= 4.0 and
        float(f["avg_nnz_per_row"]) <= 10.0 and
        float(f["mixedness_proxy"]) <= 0.10 and
        float(f["row_split_affinity_proxy"]) <= 0.60
    )
    dense_tc_hybrid_case = (
        N >= 128 and
        float(f["estimated_tc_partition_ratio"]) >= 0.95 and
        float(f["estimated_cuda_partition_ratio"]) <= 0.05 and
        float(f["tile_fill_mean"]) >= 0.30 and
        float(f["row_window_colspan_compactness"]) >= 0.30 and
        float(f["avg_nnz_per_row"]) >= 20.0
    )
    wide_sparse_tc_case = (
        float(f["reordered_locality_proxy"]) >= 0.75 and
        float(f["locality_selectivity_proxy"]) >= 0.12 and
        float(f["row_window_colspan_compactness"]) <= 0.01 and
        float(f["mixedness_proxy"]) <= 0.05 and
        float(f["road_likeness_proxy"]) <= 0.05
    )
    amazon_sparse_tc_case = (
        float(f["avg_nnz_per_row"]) >= 7.0 and
        float(f["avg_nnz_per_row"]) <= 10.0 and
        float(f["reordered_locality_proxy"]) >= 0.24 and
        float(f["row_window_colspan_compactness"]) <= 0.001 and
        float(f["tile_fill_mean"]) <= 0.02 and
        float(f["mixedness_proxy"]) <= 0.01 and
        float(f["road_likeness_proxy"]) <= 0.01
    )
    sparse_community_tc_case = (
        float(f["avg_nnz_per_row"]) >= 5.0 and
        float(f["avg_nnz_per_row"]) <= 9.5 and
        float(f["mixedness_proxy"]) <= 0.02 and
        float(f["road_likeness_proxy"]) <= 0.05 and
        float(f["row_split_affinity_proxy"]) >= 0.04 and
        float(f["row_split_affinity_proxy"]) <= 0.60 and
        float(f["tile_fill_mean"]) <= 0.02 and
        (
            (
                float(f["local_row_similarity_proxy"]) >= 0.60 and
                float(f["reordered_locality_proxy"]) >= 0.18
            ) or
            (
                float(f["locality_selectivity_proxy"]) >= 0.09 and
                float(f["reordered_locality_proxy"]) >= 0.17
            )
        )
    )
    dense_cluster_cusparse_case = (
        float(f["avg_nnz_per_row"]) >= 100.0 and
        float(f["degree_cv"]) <= 0.30 and
        float(f["locality_selectivity_proxy"]) <= 0.03 and
        float(f["row_window_colspan_compactness"]) >= 0.03 and
        float(f["row_split_affinity_proxy"]) <= 0.60 and
        float(f["reordered_locality_proxy"]) <= 0.45
    )

    feasibility: Dict[str, bool] = {
        "CSR_DIRECT": True,
        "ROW_SPLIT_CUDA": (
            (
                (
                    float(f["row_split_affinity_proxy"]) >= 0.65 and
                    float(f["avg_nnz_per_row"]) >= 4.0 and
                    (
                        float(f["long_row_nnz_fraction"]) >= 0.05 or
                        float(f["top_5_row_nnz_fraction"]) >= 0.02 or
                        float(f["degree_cv"]) >= 1.0
                    )
                ) or dense_regular_row_split_case
            ) and
            not (
                float(f["mixedness_proxy"]) >= 0.55 and
                float(f["estimated_tc_partition_ratio"]) >= 0.25 and
                float(f["estimated_cuda_partition_ratio"]) >= 0.25
            ) and
            not (
                float(f["avg_nnz_per_row"]) < 32.0 and
                float(f["row_split_affinity_proxy"]) >= 0.85 and
                float(f["locality_selectivity_proxy"]) < 0.10 and
                float(f["mixedness_proxy"]) < 0.10
            ) and
            not (
                not dense_regular_row_split_case and
                float(f["long_row_nnz_fraction"]) < 0.03 and
                float(f["top_5_row_nnz_fraction"]) < 0.01 and
                float(f["degree_cv"]) < 0.70 and
                float(f["reordered_locality_proxy"]) > 0.30
            )
        ),
        "TC_REORDERED": (
            N >= 64 and
            float(f["road_likeness_proxy"]) <= 0.18 and
            float(f["mixedness_proxy"]) <= 0.10 and
            (
                float(f["long_row_nnz_fraction"]) <= 0.16 or
                selective_sparse_tc_case or
                wide_sparse_tc_case or
                sparse_community_tc_case
            ) and
            (
                (
                    float(f["reordered_locality_proxy"]) >= 0.30 and
                    float(f["row_window_colspan_compactness"]) >= 0.03
                ) or
                (
                    float(f["locality_selectivity_proxy"]) >= 0.12 and
                    float(f["locality_gain_proxy"]) >= 0.12 and
                    float(f["reordered_locality_proxy"]) >= 0.24 and
                    float(f["row_window_colspan_compactness"]) >= 0.002 and
                    float(f["row_split_affinity_proxy"]) <= 0.70
                ) or
                (
                    float(f["reordered_locality_proxy"]) >= 0.60 and
                    float(f["locality_selectivity_proxy"]) >= 0.10
                ) or
                roadnet_tc_case or
                selective_sparse_tc_case or
                wide_sparse_tc_case or
                amazon_sparse_tc_case or
                sparse_community_tc_case
            ) and
            not dense_cluster_cusparse_case
        ),
        "HYBRID_TC_CUDA": (
            (
                dense_tc_hybrid_case or
                (
                    N >= 64 and
                    (
                float(f["mixedness_proxy"]) >= 0.55 and
                float(f["estimated_tc_partition_ratio"]) >= 0.18 and
                float(f["estimated_cuda_partition_ratio"]) >= 0.18 and
                (
                    float(f["irregular_window_fraction"]) >= 0.12 or
                    float(f["locality_selectivity_proxy"]) >= 0.08
                )
                    )
                )
            ) and
            not hybrid_mixed_empty_tile_case
        ),
        "CUSPARSE": not (
            (
                float(f["locality_selectivity_proxy"]) >= 0.20 and
                float(f["reordered_locality_proxy"]) >= 0.20 and
                float(f["row_window_colspan_compactness"]) >= 0.03 and
                float(f["road_likeness_proxy"]) < 0.50
            )
        ),
    }

    cusparse_candidate = next(c for c in candidates if c[0] == "CUSPARSE")
    cusparse_effective = float(cusparse_candidate[1]) - float(cusparse_candidate[3])
    road_like_cusparse_case = (
        float(f["road_likeness_proxy"]) >= 0.55 and
        float(f["avg_nnz_per_row"]) >= 3.5 and
        float(f["avg_nnz_per_row"]) <= 6.0 and
        float(f["row_split_affinity_proxy"]) <= 0.10 and
        float(f["mixedness_proxy"]) <= 0.10
    )
    if "CUSPARSE" in allowed and feasibility["CUSPARSE"] and float(cusparse_candidate[1]) >= float(cusparse_candidate[2]):
        best_path = "CUSPARSE"
        best_effective = cusparse_effective
        best_reason = "cusparse_vendor_library_best_general_default"
        if road_like_cusparse_case:
            best_reason = "cusparse_road_like_low_degree_general_case"
    elif "CSR_DIRECT" in allowed:
        best_path = "CSR_DIRECT"
        best_effective = direct_strength
        best_reason = "direct_explicit_safe_winner"

    if road_like_cusparse_case and "CUSPARSE" in allowed and feasibility["CUSPARSE"]:
        best_path = "CUSPARSE"
        best_effective = max(best_effective, cusparse_effective)
        best_reason = "cusparse_road_like_low_degree_general_case"

    tc_candidate = next(c for c in candidates if c[0] == "TC_REORDERED")
    row_split_candidate = next(c for c in candidates if c[0] == "ROW_SPLIT_CUDA")
    hybrid_candidate = next(c for c in candidates if c[0] == "HYBRID_TC_CUDA")
    tc_effective = float(tc_candidate[1]) - float(tc_candidate[3])
    row_split_effective = float(row_split_candidate[1]) - float(row_split_candidate[3])
    hybrid_effective = float(hybrid_candidate[1]) - float(hybrid_candidate[3])

    # Override: the updated stable build makes reordered locality clearly
    # worthwhile on sparse web-like recovery cases and real roadNet-style
    # ordered graphs even when cuSPARSE remains a strong general default.
    if (
        "TC_REORDERED" in allowed and
        feasibility["TC_REORDERED"] and
        (
            selective_sparse_tc_case or
            roadnet_tc_case or
            wide_sparse_tc_case or
            amazon_sparse_tc_case or
            sparse_community_tc_case
        )
    ):
        best_path = "TC_REORDERED"
        best_effective = max(best_effective, tc_effective) + 0.03
        best_reason = "tc_reordered_sparse_locality_override"

    if (
        "HYBRID_TC_CUDA" in allowed and
        feasibility["HYBRID_TC_CUDA"] and
        dense_tc_hybrid_case
    ):
        best_path = "HYBRID_TC_CUDA"
        best_effective = max(best_effective, hybrid_effective, tc_effective) + 0.03
        best_reason = "hybrid_dense_tc_window_override"

    # Override: dense regular community-style graphs now favor ROW_SPLIT on
    # the optimized kernels even though reordered locality remains non-trivial.
    if (
        "ROW_SPLIT_CUDA" in allowed and
        feasibility["ROW_SPLIT_CUDA"] and
        dense_regular_row_split_case and
        row_split_effective + 0.02 >= max(best_effective, tc_effective)
    ):
        best_path = "ROW_SPLIT_CUDA"
        best_effective = max(best_effective, row_split_effective)
        best_reason = "row_split_dense_regular_override"

    for path, suitability, min_strength, premium, reason in candidates:
        if path not in allowed or not feasibility[path]:
            continue
        if suitability < min_strength:
            continue
        effective = suitability - premium
        if path == "CSR_DIRECT":
            continue
        if effective > best_effective + 0.025:
            best_path = path
            best_effective = effective
            best_reason = reason

    if best_path not in allowed:
        if "CSR_DIRECT" in allowed:
            return "CSR_DIRECT", "direct_explicit_safe_winner"
        return sorted(allowed)[0], "restricted_portfolio_fallback"
    return best_path, best_reason


def restricted_choice_from_path_results(
    path_results: Mapping[str, Mapping[str, float]],
    features: Mapping[str, float],
    N: int,
    policy_mode: str,
    feature_ablation: Optional[str] = None,
) -> Tuple[str, str]:
    if policy_mode == "always_direct":
        return "CSR_DIRECT", "always_direct_ablation"
    if policy_mode == "current_router":
        return choose_policy_path(features, N, feature_ablation=feature_ablation)
    if policy_mode == "no_row_split":
        return choose_policy_path(features, N, allowed_paths=["CSR_DIRECT", "TC_REORDERED", "HYBRID_TC_CUDA", "CUSPARSE"],
                                  feature_ablation=feature_ablation)
    if policy_mode == "no_tc_paths":
        return choose_policy_path(features, N, allowed_paths=["CSR_DIRECT", "ROW_SPLIT_CUDA", "CUSPARSE"],
                                  feature_ablation=feature_ablation)
    if policy_mode == "direct_vs_row_split_only":
        return choose_policy_path(features, N, allowed_paths=["CSR_DIRECT", "ROW_SPLIT_CUDA"],
                                  feature_ablation=feature_ablation)
    raise ValueError(f"Unknown policy mode {policy_mode}")


def measure_cuda_ms(run_fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(max(1, iters))


def measure_row_split_ablation(
    runner: ExperimentRunner,
    case: EvalMatrixCase,
    N: int,
) -> Dict[str, object]:
    mat = runner.load_matrix(case)
    base_row = {
        "graph": case.name,
        "source": case.source,
        "graph_group": case.group,
        "category": case.category,
        "size_tag": case.size_tag,
        "M": case.M,
        "K": case.K,
        "N": N,
        "nnz": int(mat["nnz"]),
        "memory_policy": runner.memory_policy,
    }

    try:
        rowptr, colind, vals = prepare_gpu_matrix(mat)
        aligned_B = make_aligned_B(case.K, N, runner.seed + 101 + N)
        misaligned_B = make_misaligned_B(case.K, N, runner.seed + 303 + N)

        full_plan = runner.row_split_plan(case)
        no_long_plan = runner.row_split_plan_no_long_rows(case)

        row_split_aligned_ms = measure_cuda_ms(
            lambda: ra_spmm.run_row_split_plan(full_plan, colind, vals, aligned_B),
            runner.warmup, runner.iters)
        row_split_no_long_ms = measure_cuda_ms(
            lambda: ra_spmm.run_row_split_plan(no_long_plan, colind, vals, aligned_B),
            runner.warmup, runner.iters)
        row_split_scalar_proxy_ms = measure_cuda_ms(
            lambda: ra_spmm.run_row_split_plan(full_plan, colind, vals, misaligned_B),
            runner.warmup, runner.iters)
        direct_aligned_ms = measure_cuda_ms(
            lambda: ra_spmm.spmm_csr_direct(rowptr, colind, vals, aligned_B),
            runner.warmup, runner.iters)
        direct_scalar_proxy_ms = measure_cuda_ms(
            lambda: ra_spmm.spmm_csr_direct(rowptr, colind, vals, misaligned_B),
            runner.warmup, runner.iters)
    except Exception as exc:
        status = STATUS_OOM if is_oom_exception(exc) else STATUS_ERROR
        cleanup_cuda_state()
        return {
            **base_row,
            "status": status,
            "status_reason": str(exc),
            "attempted": True,
            "timed": False,
            "oom_retry_attempted": False,
            "oom_retry_succeeded": False,
        }

    return {
        **base_row,
        "status": STATUS_OK,
        "status_reason": "",
        "attempted": True,
        "timed": True,
        "oom_retry_attempted": False,
        "oom_retry_succeeded": False,
        "row_split_aligned_ms": row_split_aligned_ms,
        "row_split_no_long_rows_ms": row_split_no_long_ms,
        "row_split_scalar_proxy_ms": row_split_scalar_proxy_ms,
        "csr_direct_aligned_ms": direct_aligned_ms,
        "csr_direct_scalar_proxy_ms": direct_scalar_proxy_ms,
        "row_split_long_row_speedup": safe_div(row_split_no_long_ms, row_split_aligned_ms),
        "row_split_vectorization_speedup": safe_div(row_split_scalar_proxy_ms, row_split_aligned_ms),
        "csr_direct_vectorization_speedup": safe_div(direct_scalar_proxy_ms, direct_aligned_ms),
        "num_regular_rows": getattr(full_plan, "num_regular_rows", 0),
        "num_short_rows": getattr(full_plan, "num_short_rows", 0),
        "num_long_rows": getattr(full_plan, "num_long_rows", 0),
        "num_split_long_rows": getattr(full_plan, "num_split_long_rows", 0),
        "regular_nnz_fraction": getattr(full_plan, "regular_nnz_fraction", 0.0),
        "residual_nnz_fraction": getattr(full_plan, "residual_nnz_fraction", 0.0),
        "avg_segments_per_long_row": getattr(full_plan, "avg_segments_per_long_row", 0.0),
        "vectorization_proxy_note": "Misaligned contiguous B disables vec4 fast path without changing kernel code.",
    }


def dataset_inventory_rows(
    cases: Sequence[EvalMatrixCase],
    loaded_mats: Optional[Mapping[Tuple[str, str], Mapping[str, object]]] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        metadata = dict(case.metadata or {})
        if "nnz" not in metadata:
            key = case.case_key
            if loaded_mats and key in loaded_mats:
                mat = loaded_mats[key]
            else:
                mat = case.loader()
            metadata["nnz"] = int(mat["nnz"])
            metadata.setdefault("avg_nnz_per_row", float(mat.get("avg_nnz_per_row", 0.0)))
            metadata.setdefault("std_nnz_per_row", float(mat.get("std_nnz_per_row", 0.0)))
            metadata.setdefault("density", float(mat.get("density", 0.0)))
        rows.append({
            "name": case.name,
            "source": case.source,
            "category": case.category,
            "group": case.group,
            "M": case.M,
            "K": case.K,
            "nnz": int(metadata.get("nnz", 0)),
            "size_tag": case.size_tag,
            "Ns": ",".join(str(n) for n in case.Ns),
            "tags": ",".join(case.tags),
            "notes": case.notes,
            "disable_paths": ",".join(case.disable_paths),
            "memory_sensitive": case.memory_sensitive,
            "max_N": case.max_N if case.max_N is not None else "",
            "data_path": case.data_path,
        })
    return rows
