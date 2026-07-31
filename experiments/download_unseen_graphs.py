#!/usr/bin/env python3
"""Download the nine previously-unused validation graphs into datasets/unseen/.

SNAP / PyG source data is not ours to redistribute, so the raw files are NOT
committed. This script fetches them from the source URLs recorded in
fgcs_results/revision/tf32/unseen/unseen_manifest.json and verifies each file's
sha256 against the manifest. experiments/unseen_validation.py then loads them
with the per-graph conventions the manifest documents (directed/undirected,
symmetrized, deduplicated, unit values).
"""
from __future__ import annotations
import hashlib, json, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "fgcs_results/revision/tf32/unseen/unseen_manifest.json"
DEST = REPO / "datasets/unseen"
# manifest graph-name -> local filename used by unseen_validation.py
FILES = {
    "ca-AstroPh": "ca-AstroPh.txt.gz", "Coauthor-CS": "Coauthor-CS.npz",
    "email-Enron": "email-Enron.txt.gz", "p2p-Gnutella31": "p2p-Gnutella31.txt.gz",
    "soc-Slashdot0902": "soc-Slashdot0922.txt.gz",  # loader key kept; SNAP file is 0902
    "web-NotreDame": "web-NotreDame.txt.gz", "wiki-Talk": "wiki-Talk.txt.gz",
    "as-Skitter": "as-Skitter.txt.gz", "cit-Patents": "cit-Patents.txt.gz",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    m = json.loads(MANIFEST.read_text())["graphs"]
    DEST.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, fn in FILES.items():
        rec = m[name]
        url, want = rec["source_url"], rec["sha256"]
        out = DEST / fn
        if out.exists() and sha256(out) == want:
            print(f"[skip] {name}: present, sha256 ok"); continue
        print(f"[get ] {name} <- {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print(f"[FAIL] {name}: {e}"); ok = False; continue
        got = sha256(out)
        if got != want:
            print(f"[FAIL] {name}: sha256 mismatch\n   want {want}\n   got  {got}"); ok = False
        else:
            print(f"[ok  ] {name}: sha256 verified")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
