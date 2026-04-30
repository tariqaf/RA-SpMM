"""fetch_datasets.py — cross-platform Zenodo bundle downloader.

Downloads the RA-SpMM dataset archive from Zenodo and unpacks it into the
repository root so the paper_datasets.json / paper_combined_datasets.json
manifests resolve against on-disk files.

Usage (from the repository root):

    python scripts/fetch_datasets.py

Or override the Zenodo record id:

    ZENODO_RECORD=12345678 python scripts/fetch_datasets.py
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ZENODO_RECORD = os.environ.get("ZENODO_RECORD", "19903313")
ARCHIVE_NAME = "ra_spmm_data_v1.tar.gz"
EXPECTED_SHA256 = "eedcdc6285ce33a3af4e18ea8bd14d73cb43c2582c9ae362ee6bdc980f0a604f"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ARCHIVE_NAME}"


def main() -> int:
    import hashlib
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    archive_path = repo_root / ARCHIVE_NAME

    print(f"[fetch_datasets] Downloading {ARCHIVE_NAME} from Zenodo (~1.9 GB compressed)...")
    print(f"  URL: {ZENODO_URL}")

    def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        if totalsize <= 0:
            return
        downloaded = blocknum * blocksize
        pct = min(100.0, downloaded * 100.0 / totalsize)
        mb = downloaded / (1024 * 1024)
        total_mb = totalsize / (1024 * 1024)
        sys.stdout.write(f"\r  {pct:6.2f}%  ({mb:.1f} / {total_mb:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(ZENODO_URL, archive_path, reporthook)
    print()  # newline after progress bar

    print("[fetch_datasets] Verifying SHA-256 integrity...")
    h = hashlib.sha256()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != EXPECTED_SHA256:
        sys.stderr.write(f"  ERROR: SHA-256 mismatch\n    expected: {EXPECTED_SHA256}\n    actual:   {actual}\n")
        return 1
    print("  OK")

    print(f"[fetch_datasets] Extracting (mirrored layout, strip top-level wrapper)...")
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        # Strip the top-level "ra_spmm_data_v1/" prefix
        prefix = members[0].name.split("/", 1)[0] + "/"
        for m in members:
            if m.name.startswith(prefix):
                m.name = m.name[len(prefix):]
            if m.name:  # skip the (now-empty) root entry
                tf.extract(m, path=repo_root)

    print(f"[fetch_datasets] Cleaning up archive...")
    archive_path.unlink()

    print()
    print("[fetch_datasets] Sanity check:")
    try:
        with open(repo_root / "paper_datasets.json") as fh:
            m = json.load(fh)["datasets"]
        miss = [d["name"] for d in m if not (repo_root / d["path"]).exists()]
        if miss:
            print(f"  MISSING real graphs: {miss}")
            return 1
        print(f"  OK: all {len(m)} real graphs found.")
    except Exception as e:
        print(f"  Sanity check failed: {e}")
        return 1

    print()
    print("[fetch_datasets] Done. Next steps:")
    print("  1) cd bindings && python setup.py install   # build CUDA kernels")
    print("  2) python ra_router_parity_test.py          # expect 'PARITY OK 192/192'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
