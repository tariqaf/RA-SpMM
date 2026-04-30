"""
download_new_datasets.py - Download and export new benchmark graphs

Downloads graphs that fill critical gaps in RA-SpMM evaluation:
- PPI, Flickr: Mixed/Irregular (fills R6/mixed gap)
- Yelp: Large GNN graph (bridges sparse→dense)
- com-Youtube: Large community graph
- Cora, CiteSeer: Classic GNN benchmarks (re-enable)

All graphs exported as NPZ (scipy CSR format) to datasets/gnn/exports/

Usage:
    python download_new_datasets.py          # Download all
    python download_new_datasets.py --list   # Show what will be downloaded
"""
import argparse
import os
import sys

import numpy as np

# Ensure datasets directory exists
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "gnn", "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def export_pyg_dataset(name, dataset_class, root="datasets/gnn", **kwargs):
    """Download a PyG dataset and export to NPZ."""
    try:
        import torch_geometric
        from torch_geometric.utils import to_scipy_sparse_matrix
    except ImportError:
        print(f"  SKIP {name}: torch_geometric not installed")
        return False

    print(f"  Downloading {name} via PyG...")
    try:
        if dataset_class == "Planetoid":
            from torch_geometric.datasets import Planetoid
            dataset = Planetoid(root=root, name=name, **kwargs)
        elif dataset_class == "PPI":
            from torch_geometric.datasets import PPI
            dataset = PPI(root=os.path.join(root, "PPI"), split="train")
        elif dataset_class == "Flickr":
            from torch_geometric.datasets import Flickr
            dataset = Flickr(root=os.path.join(root, "Flickr"))
        elif dataset_class == "Yelp":
            from torch_geometric.datasets import Yelp
            dataset = Yelp(root=os.path.join(root, "Yelp"))
        elif dataset_class == "Amazon":
            from torch_geometric.datasets import Amazon
            dataset = Amazon(root=root, name=name, **kwargs)
        else:
            print(f"  SKIP {name}: Unknown dataset class {dataset_class}")
            return False

        data = dataset[0]
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # Convert to scipy CSR
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

        # Export
        out_path = os.path.join(EXPORT_DIR, f"{name}.npz")
        np.savez(out_path,
                 indptr=adj.indptr.astype(np.int32),
                 indices=adj.indices.astype(np.int32),
                 data=adj.data.astype(np.float32),
                 shape=np.array(adj.shape, dtype=np.int32),
                 format=np.array("csr"))

        print(f"  Exported {name}: M={num_nodes}, nnz={adj.nnz}, "
              f"avg_deg={adj.nnz/num_nodes:.1f} -> {out_path}")
        return True

    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return False


def download_snap_graph(name, url, directed=False):
    """Download a SNAP graph (edge list)."""
    import urllib.request
    import gzip

    out_path = os.path.join(SNAP_DIR, f"{name}.txt")
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    print(f"  Downloading {name} from SNAP...")
    try:
        gz_path = out_path + ".gz"
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(gz_path)
        print(f"  Downloaded: {out_path}")
        return True
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return False


# Datasets to download
DATASETS = [
    # PyG datasets
    {"name": "PPI", "class": "PPI", "category": "Mixed/Irregular",
     "desc": "Protein-Protein Interaction (standard GNN benchmark)"},
    {"name": "Flickr", "class": "Flickr", "category": "Mixed/Irregular",
     "desc": "Flickr image graph (standard GNN benchmark)"},
    {"name": "Yelp", "class": "Yelp", "category": "Sparse Uniform (large)",
     "desc": "Yelp review graph (large GNN benchmark)"},
    {"name": "Cora", "class": "Planetoid", "category": "Mixed/Irregular",
     "desc": "Classic citation network (re-enable)"},
    {"name": "CiteSeer", "class": "Planetoid", "category": "Mixed/Irregular",
     "desc": "Classic citation network (re-enable)"},
    # SNAP datasets
    {"name": "com-youtube", "snap_url": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
     "category": "Community", "desc": "YouTube community graph (1.1M nodes)"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List datasets without downloading")
    args = parser.parse_args()

    print("=" * 60)
    print("RA-SpMM Dataset Downloader")
    print("=" * 60)

    for d in DATASETS:
        print(f"\n[{d['category']}] {d['name']}: {d['desc']}")
        if args.list:
            continue

        if "class" in d:
            export_pyg_dataset(d["name"], d["class"])
        elif "snap_url" in d:
            download_snap_graph(d["name"], d["snap_url"])

    if not args.list:
        print("\n" + "=" * 60)
        print("Download complete. Add entries to paper_datasets.json manually.")
        print("=" * 60)


if __name__ == "__main__":
    main()
