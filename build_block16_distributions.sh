#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
cd "$ROOT"
"$PYTHON" verification/build_block16_distributions.py --output release/BLOCK16_DISTRIBUTION_BUILD.json
"$PYTHON" verification/block16_distribution_audit.py --output release/BLOCK16_DISTRIBUTION_VALIDATION.json
