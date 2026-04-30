import importlib
import os
import sys
from glob import glob
from typing import Dict, List, Tuple

import torch


BLK_H = 16
BLK_W = 8

_DTC_MODULE = None


def dtc_module_dir() -> str:
    env_dir = os.environ.get("DTC_SPMM_DIR", "").strip()
    if env_dir:
        return env_dir
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "DTC-SpMM_ASPLOS24", "DTC-SpMM")


def load_dtc_module():
    global _DTC_MODULE
    if _DTC_MODULE is not None:
        return _DTC_MODULE
    module_dir = dtc_module_dir()
    if not os.path.isdir(module_dir):
        raise FileNotFoundError(f"DTC module directory not found: {module_dir}")
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    _DTC_MODULE = importlib.import_module("DTCSpMM")
    return _DTC_MODULE


def is_dtc_available() -> bool:
    module_dir = dtc_module_dir()
    if not os.path.isdir(module_dir):
        return False
    if glob(os.path.join(module_dir, "DTCSpMM*.so")):
        return True
    return False


def candidate_variants(N: int) -> List[Tuple[bool, str]]:
    variants: List[Tuple[bool, str]] = [
        (False, "float_nonsplit"),
        (False, "float2_nonsplit"),
        (False, "float4_nonsplit"),
        (True, "float_nonsplit"),
    ]
    if N >= 64:
        variants.append((False, "float2_split"))
    if N >= 128:
        variants.extend(
            [
                (False, "float4_split"),
                (True, "float4_split"),
            ]
        )
    return variants


def preprocess(module, rowptr: torch.Tensor, colind: torch.Tensor, num_rows: int, num_edges: int) -> Dict[str, object]:
    num_row_windows = (num_rows + BLK_H - 1) // BLK_H
    block_partition = torch.zeros(num_row_windows, dtype=torch.int32, device="cuda")
    edge_to_column = torch.zeros(num_edges, dtype=torch.int32, device="cuda")
    edge_to_row = torch.zeros(num_edges, dtype=torch.int32, device="cuda")
    row_window_offset, tcblock_rowid, tcblocktile_id, tcblock_offset, sparse_atox_index, block_count = module.preprocess_gpu(
        colind,
        rowptr,
        num_rows,
        BLK_H,
        BLK_W,
        block_partition,
        edge_to_column,
        edge_to_row,
    )
    return {
        "row_window_offset": row_window_offset,
        "tcblock_rowid": tcblock_rowid,
        "tcblocktile_id": tcblocktile_id,
        "tcblock_offset": tcblock_offset,
        "sparse_atox_index": sparse_atox_index,
        "block_count": int(block_count),
    }


def run_variant(
    module,
    state: Dict[str, object],
    B: torch.Tensor,
    num_rows: int,
    num_edges: int,
    use_balance: bool,
    exeplan: str,
) -> torch.Tensor:
    if use_balance:
        return module.run_DTCSpMM_balance(
            B,
            state["tcblock_rowid"],
            state["tcblocktile_id"],
            state["tcblock_offset"],
            state["sparse_atox_index"],
            num_rows,
            exeplan,
        )[0]
    return module.run_DTCSpMM(
        B,
        state["row_window_offset"],
        state["tcblocktile_id"],
        state["tcblock_offset"],
        state["sparse_atox_index"],
        num_rows,
        num_edges,
        exeplan,
    )[0]


def run_variant_timed(
    module,
    state: Dict[str, object],
    B: torch.Tensor,
    num_rows: int,
    num_edges: int,
    use_balance: bool,
    exeplan: str,
) -> Tuple[torch.Tensor, float]:
    if use_balance:
        outputs = module.run_DTCSpMM_balance(
            B,
            state["tcblock_rowid"],
            state["tcblocktile_id"],
            state["tcblock_offset"],
            state["sparse_atox_index"],
            num_rows,
            exeplan,
        )
    else:
        outputs = module.run_DTCSpMM(
            B,
            state["row_window_offset"],
            state["tcblocktile_id"],
            state["tcblock_offset"],
            state["sparse_atox_index"],
            num_rows,
            num_edges,
            exeplan,
        )
    if len(outputs) < 2:
        raise RuntimeError("dtc_timing_missing")
    return outputs[0], float(outputs[1].item())
