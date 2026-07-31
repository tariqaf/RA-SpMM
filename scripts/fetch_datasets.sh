#!/usr/bin/env bash
# fetch_datasets.sh — download the RA-SpMM Zenodo bundle and unpack it
# into the repository root so the paper_datasets.json / paper_combined_datasets.json
# manifests resolve against on-disk files.
#
# Usage (from repo root):
#   bash scripts/fetch_datasets.sh
#
set -euo pipefail

ZENODO_RECORD="${ZENODO_RECORD:-19903313}"   # baked default; override via env if needed
ARCHIVE_NAME="ra_spmm_data_v1.tar.gz"
EXPECTED_SHA256="eedcdc6285ce33a3af4e18ea8bd14d73cb43c2582c9ae362ee6bdc980f0a604f"
ZENODO_URL="https://zenodo.org/records/${ZENODO_RECORD}/files/${ARCHIVE_NAME}"

# Run from repo root regardless of where the script is invoked from
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
cd "${REPO_ROOT}"

echo "[fetch_datasets] Downloading ${ARCHIVE_NAME} from Zenodo (~1.9 GB compressed)..."
if command -v curl >/dev/null 2>&1; then
    curl -L -o "${ARCHIVE_NAME}" "${ZENODO_URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -O "${ARCHIVE_NAME}" "${ZENODO_URL}"
else
    echo "ERROR: neither curl nor wget is installed." >&2
    exit 1
fi

echo "[fetch_datasets] Verifying SHA-256 integrity..."
if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_SHA256=$(sha256sum "${ARCHIVE_NAME}" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    ACTUAL_SHA256=$(shasum -a 256 "${ARCHIVE_NAME}" | awk '{print $1}')
else
    echo "  WARNING: no sha256 tool found; skipping integrity check."
    ACTUAL_SHA256="${EXPECTED_SHA256}"
fi
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "  ERROR: SHA-256 mismatch."
    echo "    expected: ${EXPECTED_SHA256}"
    echo "    actual:   ${ACTUAL_SHA256}"
    exit 1
fi
echo "  OK"

echo "[fetch_datasets] Extracting (mirrored layout, --strip-components=1)..."
tar -xzf "${ARCHIVE_NAME}" --strip-components=1

echo "[fetch_datasets] Cleaning up archive..."
rm -f "${ARCHIVE_NAME}"

echo ""
echo "[fetch_datasets] Sanity check:"
python3 - <<'PY'
import json, os, sys

COMBINED = 'fgcs_results/paper_combined_datasets.json'

def check(manifest, label):
    entries = json.load(open(manifest))['datasets']
    miss = [d['name'] for d in entries if not os.path.exists(d['path'])]
    if miss:
        print(f'  MISSING {label}:', miss)
        return False
    print(f'  OK: all {len(entries)} {label} found.')
    return True

try:
    ok = check('paper_datasets.json', 'real graphs')
    if not os.path.exists(COMBINED):
        print('  MISSING combined manifest: the synthetic half of the bundle was not')
        print('  extracted. The 192-configuration reproduction needs all 51 graphs.')
        ok = False
    else:
        ok = check(COMBINED, 'graphs in the full suite') and ok
    if not ok:
        sys.exit(1)
except Exception as e:
    print('  Sanity check failed:', e)
    sys.exit(1)
PY

echo ""
echo "[fetch_datasets] Done. Next steps:"
echo "  1) python setup.py install                  # build CUDA kernels"
echo "  2) python ra_router_parity_test.py          # expect 'PARITY OK 192/192'"
