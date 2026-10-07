"""Run the complete Block 14 test suite in bounded independent processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

BATCHES: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    (
        "algebra-algorithms-analytic-arithmetic-backends",
        141,
        (
            "tests/test_algebra_quotients.py",
            "tests/test_algorithms_compiler.py",
            "tests/test_algorithms_complexity_benchmarks.py",
            "tests/test_algorithms_discovery_gpu.py",
            "tests/test_algorithms_krylov.py",
            "tests/test_algorithms_layout_ir.py",
            "tests/test_algorithms_low_rank.py",
            "tests/test_algorithms_precision.py",
            "tests/test_algorithms_scheduling.py",
            "tests/test_algorithms_structured.py",
            "tests/test_algorithms_support.py",
            "tests/test_analytic_appell_zeros.py",
            "tests/test_analytic_cyclic_reciprocal.py",
            "tests/test_arithmetic_domains_valuation.py",
            "tests/test_arithmetic_euler.py",
            "tests/test_arithmetic_hurwitz_windows.py",
            "tests/test_arithmetic_mahler_polyphase.py",
            "tests/test_arithmetic_recurrence.py",
            "tests/test_backends.py",
        ),
    ),
    (
        "block10-block11",
        68,
        (
            "tests/test_block10_cli.py",
            "tests/test_block10_compatibility.py",
            "tests/test_block10_execution_persistence.py",
            "tests/test_block10_sparse_autodiff.py",
            "tests/test_block11_cli.py",
            "tests/test_block11_golden_profiling_beta.py",
            "tests/test_block11_streaming_migrations.py",
        ),
    ),
    (
        "block12-block13-block14-coordinates",
        163,
        (
            "tests/test_block12_cli_api.py",
            "tests/test_block12_denoise_streaming.py",
            "tests/test_block12_recursive_fft.py",
            "tests/test_block12_resampling_adaptive.py",
            "tests/test_block12_spectral_filtering.py",
            "tests/test_block12_windows_time_frequency.py",
            "tests/test_block13_cli_api.py",
            "tests/test_block14_cli_api.py",
            "tests/test_coordinates_exact.py",
            "tests/test_coordinates_general_affine_formal.py",
            "tests/test_coordinates_general_basis.py",
            "tests/test_coordinates_general_calculus.py",
            "tests/test_coordinates_general_classical.py",
            "tests/test_coordinates_general_combinatorial.py",
            "tests/test_coordinates_general_families_state.py",
            "tests/test_coordinates_general_numeric_packet.py",
            "tests/test_coordinates_general_persistence.py",
            "tests/test_coordinates_general_tensor_modular.py",
            "tests/test_coordinates_normalized_persistence.py",
            "tests/test_coordinates_operators.py",
            "tests/test_coordinates_polynomials.py",
            "tests/test_coordinates_realization.py",
            "tests/test_coordinates_recurrence.py",
        ),
    ),
    (
        "generating-foundation-domains",
        83,
        (
            "tests/test_generating_foundation.py",
            "tests/test_generating_domains.py",
        ),
    ),
    (
        "generating-cartier-production",
        105,
        (
            "tests/test_generating_cartier_padic.py",
            "tests/test_generating_block13.py",
        ),
    ),
    (
        "block4-certificates-deformation-dynamics-export",
        76,
        (
            "tests/test_block4_hardening.py",
            "tests/test_certificates_registry.py",
            "tests/test_deformation_complex_support_companions.py",
            "tests/test_deformation_forests_visibility_counting.py",
            "tests/test_dynamics_bayes_martingales.py",
            "tests/test_dynamics_feshbach.py",
            "tests/test_dynamics_fractional_green_feynman_survival.py",
            "tests/test_dynamics_semigroups_markov.py",
            "tests/test_export_cli.py",
            "tests/test_export_files.py",
            "tests/test_export_renderers.py",
            "tests/test_export_serialization.py",
        ),
    ),
    (
        "grammar-graphs-langlands-lifting",
        74,
        (
            "tests/test_grammar_realization.py",
            "tests/test_graphs.py",
            "tests/test_holonomy.py",
            "tests/test_langlands_finite_image.py",
            "tests/test_langlands_local_fields_tame_gauss.py",
            "tests/test_langlands_local_heisenberg_symplectic.py",
            "tests/test_langlands_ramification_global_completion.py",
            "tests/test_lattice_smith.py",
            "tests/test_lifting.py",
            "tests/test_linear_defects.py",
            "tests/test_morphisms.py",
        ),
    ),
    (
        "nonlinear",
        68,
        (
            "tests/test_nonlinear_coordinates_defects.py",
            "tests/test_nonlinear_correction_elimination.py",
            "tests/test_nonlinear_execution.py",
            "tests/test_nonlinear_filtered.py",
            "tests/test_nonlinear_graded.py",
            "tests/test_nonlinear_jets_composition.py",
            "tests/test_nonlinear_obstruction.py",
            "tests/test_nonlinear_snapshots_persistence.py",
            "tests/test_nonlinear_sparse_autodiff.py",
            "tests/test_nonlinear_transport.py",
            "tests/test_nonlinear_trees.py",
        ),
    ),
    (
        "operators-packets-quantum-signals-sequences",
        118,
        (
            "tests/test_operators_transfer.py",
            "tests/test_packets.py",
            "tests/test_quantum_channels_qec.py",
            "tests/test_quantum_symmetry_circuits.py",
            "tests/test_quantum_transport_effective_spectral.py",
            "tests/test_refinement_towers.py",
            "tests/test_residue_crt.py",
            "tests/test_seeds.py",
            "tests/test_sequences_execution.py",
            "tests/test_sequences_models.py",
            "tests/test_signals_extended.py",
            "tests/test_signals_polyphase_filterbanks.py",
            "tests/test_signals_recovery.py",
            "tests/test_signals_wavelets_singular.py",
            "tests/test_unified_extensions_cli.py",
            "tests/test_weighted_transfer.py",
        ),
    ),
)



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected", type=int, default=896)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    total = 0
    all_passed = True
    environment = dict(os.environ)

    for label, expected_count, files in BATCHES:
        command = [sys.executable, "-m", "pytest", "-q", *files]
        started = time.monotonic()
        print(f"\n=== {label} ===", flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=PACKAGE_ROOT,
                env=environment,
                timeout=args.timeout,
            )
            returncode = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = 124
            timed_out = True

        elapsed = time.monotonic() - started
        passed_count = expected_count if returncode == 0 else 0
        total += passed_count
        passed = returncode == 0 and not timed_out
        all_passed = all_passed and passed
        results.append(
            {
                "label": label,
                "files": len(files),
                "passed": passed_count,
                "returncode": returncode,
                "seconds": round(elapsed, 6),
                "timed_out": timed_out,
            }
        )
        if not passed:
            break

    payload = {
        "schema": "komorebi-math.block14-test-batch-run.v1",
        "package_version": "0.14.0a2",
        "expected": args.expected,
        "total_passed": total,
        "passed": all_passed and total == args.expected,
        "batches": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        destination = args.output
        if not destination.is_absolute():
            destination = PACKAGE_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
