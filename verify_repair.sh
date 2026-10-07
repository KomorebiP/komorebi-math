#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python verification/release_artifact_gate.py --runtime-tests --root . --output release/REPAIR_R2_ARTIFACT_GATE.json
python verification/block21_publication_gate.py --full-tests --skip-mkdocs --output release/REPAIR_R2_PUBLICATION_GATE.json --markdown release/REPAIR_R2_PUBLICATION_GATE.md
