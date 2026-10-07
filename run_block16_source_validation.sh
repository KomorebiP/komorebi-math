#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
MODE=${1:-batched}
export PYTHONPATH="$ROOT/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$ROOT"
"$PYTHON" -c "import platform,sys,pytest,numpy,komorebi_math; print('python',platform.python_version(),sys.executable); print('pytest',pytest.__version__,'numpy',numpy.__version__); print('komorebi',komorebi_math.__version__)"
"$PYTHON" "$ROOT/verification/optional_ad_backend_status.py"
if [ "$MODE" = "--focused" ]; then
    "$PYTHON" -m pytest -q --tb=short \
        tests/test_block15_polyphase_refinement.py \
        tests/test_block15_persistence.py \
        tests/test_block15_stable_api.py \
        tests/test_block16_analysis_synthesis.py \
        tests/test_block16_contracts.py \
        tests/test_block16_release_qualification.py \
        tests/test_block16_cli.py \
        tests/test_block16_release_contract.py \
        tests/test_block16_gpu_qualification.py \
        tests/test_block16_companion_crosswalk.py \
        tests/test_block16_release_layout.py \
        tests/test_block16_static_audit.py \
        tests/test_block16_package_audit.py \
        tests/test_optional_backend_probe.py
elif [ "$MODE" = "--monolithic" ]; then
    "$PYTHON" -m pytest -q --tb=short tests
else
    "$PYTHON" verification/run_isolated_test_batches.py --output release/BLOCK16_ISOLATED_SOURCE_VALIDATION.json
fi
"$PYTHON" -m komorebi_math api check --stable
"$PYTHON" -m komorebi_math release contract
"$PYTHON" -m komorebi_math verify stable
"$PYTHON" -m komorebi_math verify release
"$PYTHON" -m komorebi_math release qualify --output "$ROOT/release/BLOCK16_SOURCE_QUALIFICATION.json" --force
