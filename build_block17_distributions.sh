#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
cd "$ROOT"
"$PYTHON" verification/build_block17_distributions.py --output release/BLOCK17_DISTRIBUTION_BUILD.json
"$PYTHON" verification/block17_distribution_audit.py --output release/BLOCK17_DISTRIBUTION_VALIDATION.json
