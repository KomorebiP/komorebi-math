"""Command line interface for package inspection and verification."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.util import find_spec
from pathlib import Path

from .. import __version__
from ..backends import available_backends
from ..exceptions import ValidationError
from .contracts import VERIFICATION_SUITE_NAMES
from ..extensions import discover_extensions
from ..provenance import describe
from ..registry import features
from ..verify import (
    run_algorithms_verification,
    run_arithmetic_verification,
    run_analytic_verification,
    run_block2_verification,
    run_block3_verification,
    run_block4_verification,
    run_block5_verification,
    run_block6_verification,
    run_block7_verification,
    run_block8_verification,
    run_block9_verification,
    run_block10_verification,
    run_block11_verification,
    run_block12_verification,
    run_block13_verification,
    run_block14_verification,
    run_block15_verification,
    run_block16_verification,
    run_block17_verification,
    run_block18_verification,
    run_block19_verification,
    run_block20_verification,
    run_upgrade_verification,
    run_evidence_verification,
    run_documentation_verification,
    run_beta_readiness_verification,
    run_gpu_verification,
    run_release_qualification_verification,
    run_stable_verification,
    run_dsp_verification,
    run_scalable_nonlinear_verification,
    run_streaming_beta_verification,
    run_deformation_verification,
    run_langlands_completion_verification,
    run_langlands_finite_image_verification,
    run_langlands_local_verification,
    run_langlands_verification,
    run_core_verification,
    run_dynamics_verification,
    run_export_verification,
    run_frame_algebra_verification,
    run_integration_verification,
    run_expression_verification,
    run_generating_verification,
    run_coordinates_verification,
    run_nonlinear_verification,
    run_refinement_verification,
    run_quantum_verification,
    run_signals_verification,
    run_sequences_verification,
)


def _print_feature_table() -> None:
    specs = features()
    width = max(len(spec.key) for spec in specs)
    for spec in specs:
        print(f"{spec.key:<{width}}  {spec.status.value:<25}  {spec.summary}")


def command_info(_: argparse.Namespace) -> int:
    print(f"komorebi-math {__version__}")
    print("language: seed -> address -> operate -> project -> transfer -> refine -> constrain -> complete -> reconstruct")
    print("implemented blocks: finite dimensional core through Block 17, production finite DSP, plus research-tier Blocks 18-20 for frame algebra, finite integration, and typed exact expressions")
    return 0


def command_doctor(_: argparse.Namespace) -> int:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print("Backends:")
    for name, available in available_backends().items():
        print(f"  {name:<8} {'available' if available else 'not installed'}")
    print("Optional packages:")
    for name in ("scipy", "sympy", "mpmath", "matplotlib", "yaml", "jax", "torch", "cupy"):
        print(f"  {name:<10} {'available' if find_spec(name) else 'not installed'}")
    from ..export import available_latex_engines

    print("LaTeX engines:")
    for name, available in available_latex_engines().items():
        print(f"  {name:<10} {'available' if available else 'not installed'}")
    extensions = discover_extensions()
    print(f"Extensions discovered: {len(extensions)}")
    for result in extensions:
        state = "loaded" if result.loaded else f"failed: {result.error}"
        print(f"  {result.entry_point}: {state}")
    return 0 if all(result.loaded for result in extensions) else 1


def command_features(_: argparse.Namespace) -> int:
    _print_feature_table()
    return 0


def command_verify(args: argparse.Namespace) -> int:
    suites = {
        "core": run_core_verification,
        "refinement": run_refinement_verification,
        "nonlinear": run_nonlinear_verification,
        "block2": run_block2_verification,
        "algorithms": run_algorithms_verification,
        "block3": run_block3_verification,
        "signals": run_signals_verification,
        "dynamics": run_dynamics_verification,
        "block4": run_block4_verification,
        "analytic": run_analytic_verification,
        "quantum": run_quantum_verification,
        "block5": run_block5_verification,
        "arithmetic": run_arithmetic_verification,
        "sequences": run_sequences_verification,
        "block6": run_block6_verification,
        "deformation": run_deformation_verification,
        "langlands-local": run_langlands_local_verification,
        "langlands-finite-image": run_langlands_finite_image_verification,
        "langlands-completion": run_langlands_completion_verification,
        "langlands": run_langlands_verification,
        "block7": run_block7_verification,
        "export": run_export_verification,
        "block8": run_block8_verification,
        "block9": run_block9_verification,
        "hardening": run_scalable_nonlinear_verification,
        "block10": run_block10_verification,
        "streaming-beta": run_streaming_beta_verification,
        "block11": run_block11_verification,
        "dsp": run_dsp_verification,
        "block12": run_block12_verification,
        "generating": run_generating_verification,
        "block13": run_block13_verification,
        "coordinates": run_coordinates_verification,
        "block14": run_block14_verification,
        "block15": run_block15_verification,
        "stable": run_stable_verification,
        "release": run_release_qualification_verification,
        "gpu": run_gpu_verification,
        "block16": run_block16_verification,
        "upgrade": run_upgrade_verification,
        "evidence": run_evidence_verification,
        "documentation": run_documentation_verification,
        "beta": run_beta_readiness_verification,
        "block17": run_block17_verification,
        "frame-algebra": run_frame_algebra_verification,
        "block18": run_block18_verification,
        "integration": run_integration_verification,
        "block19": run_block19_verification,
        "expression": run_expression_verification,
        "block20": run_block20_verification,
        "all": run_block17_verification,
    }
    try:
        report = suites[args.suite]()
    except KeyError:
        print(f"unknown verification suite: {args.suite}", file=sys.stderr)
        return 2
    for check in report.checks:
        marker = "PASS" if check.passed else "FAIL"
        suffix = "" if not check.detail else f"  {check.detail}"
        print(f"[{marker}] {check.name}{suffix}")
    print(report.certificate().message)
    return 0 if report.passed else 1


def command_api_check(args: argparse.Namespace) -> int:
    """Check the current public API against a committed manifest."""

    from ..compatibility import check_beta_api, check_public_api, check_stable_api

    try:
        if args.stable:
            report = check_stable_api(args.manifest)
        elif args.beta:
            report = check_beta_api(args.manifest)
        else:
            report = check_public_api(
                args.manifest,
                strict_signatures=args.strict_signatures,
            )
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.compatible else 1


def command_api_snapshot(args: argparse.Namespace) -> int:
    """Write a deterministic public API manifest."""

    from ..compatibility import (
        create_beta_api_manifest,
        create_public_api_manifest,
        create_stable_api_manifest,
        save_public_api_manifest,
    )

    if args.stable:
        manifest = create_stable_api_manifest()
    elif args.beta:
        manifest = create_beta_api_manifest()
    else:
        manifest = create_public_api_manifest()
    try:
        destination = save_public_api_manifest(manifest, args.output, overwrite=args.force)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"path": str(destination), "digest": manifest.digest}, indent=2))
    return 0


def command_golden(_: argparse.Namespace) -> int:
    """Execute the committed closed behavioral fixture."""

    from ..golden import run_behavioral_golden_fixture

    try:
        report = run_behavioral_golden_fixture()
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def command_release_contract(_: argparse.Namespace) -> int:
    """Print and validate the committed current-beta release contract."""

    from ..release import check_release_contract

    report = check_release_contract()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def command_release_environment(_: argparse.Namespace) -> int:
    """Print the current finite runtime qualification record."""

    from ..release import current_runtime_record

    print(json.dumps(current_runtime_record(), indent=2, sort_keys=True))
    return 0


def command_release_qualify(args: argparse.Namespace) -> int:
    """Run the installed freeze-candidate qualification gate."""

    from ..qualification import run_release_qualification

    report = run_release_qualification(
        include_gpu=bool(args.gpu or args.require_gpu),
        require_gpu=bool(args.require_gpu),
    )
    if args.output:
        report.write(args.output, overwrite=args.force)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def _write_json_payload(payload: dict[str, object], path: str | None, *, force: bool) -> None:
    """Write one deterministic JSON report when a destination is requested."""

    if path is None:
        return
    destination = Path(path)
    if destination.exists() and not force:
        raise ValidationError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def command_release_upgrade(_: argparse.Namespace) -> int:
    """Validate predecessor API, semantic, codec, and portable-record upgrades."""

    from ..release import check_upgrade_contract

    report = check_upgrade_contract()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def command_release_evidence(args: argparse.Namespace) -> int:
    """Validate digest-checked platform evidence for beta or RC readiness."""

    from ..release import check_validation_evidence

    report = check_validation_evidence(
        require_current_all_platforms=bool(args.current),
    )
    payload = report.to_dict()
    try:
        _write_json_payload(payload, args.output, force=bool(args.force))
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.passed else 1


def command_release_documentation(_: argparse.Namespace) -> int:
    """Validate exact documentation coverage for the stable namespace."""

    from ..release import check_stable_documentation

    report = check_stable_documentation()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def command_release_readiness(args: argparse.Namespace) -> int:
    """Run the public-beta or stricter release-candidate readiness gate."""

    from ..release import check_beta_readiness

    report = check_beta_readiness(
        include_gpu=bool(args.gpu or args.require_gpu),
        require_gpu=bool(args.require_gpu),
        require_current_platforms=bool(args.final),
    )
    payload = report.to_dict()
    try:
        _write_json_payload(payload, args.output, force=bool(args.force))
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.passed else 1


def command_stream_inspect(args: argparse.Namespace) -> int:
    """Inspect one packet-jet stream manifest."""

    from ..nonlinear import open_packet_jet_snapshot_stream

    try:
        reader = open_packet_jet_snapshot_stream(
            args.path,
            allow_incomplete=args.allow_incomplete,
            verify_chunks=args.verify_chunks,
        )
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(reader.manifest.to_payload(), indent=2, sort_keys=True))
    return 0


def command_stream_verify(args: argparse.Namespace) -> int:
    """Verify every committed packet-jet stream chunk."""

    from ..nonlinear import open_packet_jet_snapshot_stream

    try:
        reader = open_packet_jet_snapshot_stream(
            args.path,
            allow_incomplete=args.allow_incomplete,
        )
        certificate = reader.verify()
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(certificate.to_json())
    return 0 if certificate.valid else 1


def command_describe(args: argparse.Namespace) -> int:
    record = describe(args.object)
    if record is None:
        print(f"no provenance record found for {args.object}", file=sys.stderr)
        return 1
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 0



def command_sequence(args: argparse.Namespace) -> int:
    """Evaluate one built in exact sequence."""

    from ..sequences import compile_sequence, get_sequence

    sequence = get_sequence(args.name)
    compiled = compile_sequence(sequence) if args.compile else None
    if args.index is not None:
        if args.modulus is not None:
            value = (compiled or compile_sequence(sequence)).modulo(args.index, args.modulus)
        else:
            value = (compiled.term(args.index) if compiled is not None else sequence.term(args.index))
        print(value)
        return 0
    values = compiled.first(args.terms) if compiled is not None else sequence.first(args.terms)
    if args.modulus is not None:
        model = compiled or compile_sequence(sequence)
        values = tuple(model.modulo(index, args.modulus) for index in range(args.terms))
    payload = {"name": sequence.name, "terms": [str(value) for value in values]}
    if compiled is not None:
        payload["strategy"] = compiled.strategy
        payload["exactness"] = compiled.certificate.exactness.value
    print(json.dumps(payload, indent=2))
    return 0


def command_standard_window(args: argparse.Namespace) -> int:
    """Compute one residual standard window."""

    from ..arithmetic import StandardWindow

    window = StandardWindow(args.prime, args.compression)
    certificate = window.certificate(args.address, periods=max(2, args.count // window.period + 1))
    certificate.certificate.require()
    values = window.residues(args.address, args.count - 1) if args.count else ()
    print(json.dumps({
        "prime": window.prime,
        "compression": window.compression,
        "address": args.address,
        "period": window.period,
        "minimum_period": window.minimum_period(),
        "coefficients": values,
    }, indent=2))
    return 0


def command_euler_coordinates(args: argparse.Namespace) -> int:
    """Compute one local Euler reconstruction."""

    from ..arithmetic import EulerLocalCoordinates

    coordinates = EulerLocalCoordinates(args.prime)
    if args.depth is None:
        value = coordinates.reconstruct(args.index)
        payload = {
            "prime": args.prime,
            "index": args.index,
            "value": str(value),
            "exact": True,
        }
    else:
        result = coordinates.truncate(args.index, args.depth)
        result.certificate.require()
        payload = {
            "prime": args.prime,
            "index": args.index,
            "depth": args.depth,
            "value": str(result.value),
            "exact_value": result.exact_value,
            "modulus": result.modulus,
            "error": str(result.error),
        }
    print(json.dumps(payload, indent=2))
    return 0


def command_deformation_count(args: argparse.Namespace) -> int:
    """Evaluate one exact finite deformation chart count."""

    from ..deformation import (
        nilpotent_chart_count,
        normal_crossing_count,
        smooth_chart_count,
        weighted_chart_count,
    )

    if args.chart == "smooth":
        value = smooth_chart_count(args.q, args.dimension, args.depth)
    elif args.chart == "weighted":
        weights = tuple(int(value) for value in args.weights.split(",") if value)
        value = weighted_chart_count(args.q, weights, args.depth)
    elif args.chart == "nilpotent":
        value = nilpotent_chart_count(args.q, args.exponent, args.depth)
    else:
        value = normal_crossing_count(args.q, args.depth)
    print(json.dumps({"chart": args.chart, "q": args.q, "depth": args.depth, "count": value}, indent=2))
    return 0


def command_tame_orbit(args: argparse.Namespace) -> int:
    """Inspect one finite tame Frobenius orbit."""

    from ..langlands.local import FrobeniusOrbit, regular_orbit_count

    orbit = FrobeniusOrbit(args.q, args.degree, args.exponent)
    orbit.certificate().require()
    print(json.dumps({
        "q": orbit.q,
        "degree": orbit.degree,
        "exponent": orbit.exponent,
        "orbit": orbit.elements,
        "regular": orbit.regular,
        "self_dual": orbit.self_dual,
        "regular_orbit_count": regular_orbit_count(orbit.q, orbit.degree),
    }, indent=2))
    return 0


def command_heisenberg(args: argparse.Namespace) -> int:
    """Validate one finite Schrödinger Heisenberg model."""

    from ..langlands.local import HeisenbergModel

    model = HeisenbergModel(args.prime, args.rank)
    model.trace_support_certificate().require()
    model.selector_certificate().require()
    print(json.dumps({
        "prime": model.prime,
        "rank": model.rank,
        "dimension": model.dimension,
        "selector_rank": int(__import__("numpy").linalg.matrix_rank(model.translation_selector())),
    }, indent=2))
    return 0


def command_completion(args: argparse.Namespace) -> int:
    """Build one explicit five-gate completion report."""

    from ..langlands.completion import AutomorphicCompletionReport, CompletionGate, GateStatus

    supplied = {value.strip() for value in args.satisfied.split(",") if value.strip()}
    unknown = supplied - {gate.value for gate in CompletionGate}
    if unknown:
        print(f"unknown completion gates: {sorted(unknown)}", file=sys.stderr)
        return 2
    report = AutomorphicCompletionReport({
        gate: GateStatus(gate.value in supplied, "supplied by command line" if gate.value in supplied else "not supplied")
        for gate in CompletionGate
    })
    result = report.result()
    print(result.certificate.to_json())
    return 0 if result.certificate.valid else 1



def _export_options(args: argparse.Namespace):
    from ..export import ExportOptions, HtmlRenderOptions, LatexCompileOptions, LatexRenderOptions

    return ExportOptions(
        title=args.title,
        author=args.author,
        date=args.date,
        abstract=args.abstract,
        overwrite=args.force,
        preserve_types=not args.no_type_metadata,
        keep_tex=not args.no_keep_tex,
        html=HtmlRenderOptions(include_mathjax=args.mathjax),
        latex=LatexRenderOptions(
            document_class=args.document_class,
            include_table_of_contents=args.toc,
        ),
        compiler=LatexCompileOptions(
            engine=args.engine,
            passes=args.passes,
            timeout=args.timeout,
            keep_auxiliary=args.keep_auxiliary,
        ),
    )


def _print_export_result(result) -> None:
    from ..export import to_serializable

    payload = {
        "path": str(result.path),
        "format": result.format.value,
        "bytes": result.bytes_written,
        "companion_files": tuple(str(path) for path in result.companion_files),
        "certificate": result.certificate,
    }
    print(json.dumps(to_serializable(payload), indent=2, sort_keys=True))


def _fraction_csv(text: str) -> tuple[object, ...]:
    """Parse comma-separated exact rational values for generating commands."""

    from fractions import Fraction

    values = tuple(Fraction(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValidationError("at least one coefficient is required")
    return values


def _fraction_matrix(text: str) -> tuple[tuple[object, ...], ...]:
    """Parse semicolon-separated rows of comma-separated rational values."""

    rows = tuple(_fraction_csv(row) for row in text.split(";") if row.strip())
    if not rows:
        raise ValidationError("at least one matrix row is required")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValidationError("matrix rows must have equal lengths")
    return rows


def _numeric_json(value: object) -> object:
    """Convert NumPy and complex values to deterministic JSON data."""

    import numpy as np

    if isinstance(value, np.ndarray):
        return _numeric_json(value.tolist())
    if isinstance(value, np.generic):
        return _numeric_json(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float):
        return float(value)
    if isinstance(value, tuple | list):
        return [_numeric_json(item) for item in value]
    return value


def _integer_csv(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValidationError("at least one integer coefficient is required")
    return values


def _fraction_strings(values: object) -> object:
    """Convert nested exact values to JSON-safe strings."""

    from fractions import Fraction

    if isinstance(values, Fraction):
        return str(values)
    if isinstance(values, tuple | list):
        return [_fraction_strings(value) for value in values]
    if isinstance(values, dict):
        return {str(key): _fraction_strings(value) for key, value in values.items()}
    return values


def command_coordinates_euler_fibonacci(args: argparse.Namespace) -> int:
    """Compute exact signed Euler coordinates in the Fibonacci companion system."""

    from ..coordinates import euler_fibonacci_coordinates, variation_of_constants_prefix

    try:
        record = euler_fibonacci_coordinates(args.count)
        record.certificate().require()
        reconstructed = variation_of_constants_prefix(record)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {
        "system": record.name,
        "count": record.length,
        "initial": _fraction_strings(record.initial),
        "defects": _fraction_strings(record.defects),
        "reconstructed": _fraction_strings(reconstructed),
        "homogeneous": record.homogeneous,
        "defect_egf": "sech(t)*tanh(t)-2*sech(t)^3",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_coordinates_recurrence(args: argparse.Namespace) -> int:
    """Change a finite prefix between two recurrence defect coordinate systems."""

    from ..coordinates import (
        analyze_recurrence_coordinates,
        coordinate_change_between_recurrences,
    )

    try:
        values = _fraction_csv(args.values)
        source_coefficients = _fraction_csv(args.source)
        target_coefficients = _fraction_csv(args.target)
        source_record = analyze_recurrence_coordinates(
            values, source_coefficients, name=args.source_name
        )
        target_record = analyze_recurrence_coordinates(
            values, target_coefficients, name=args.target_name
        )
        transform = coordinate_change_between_recurrences(
            source_coefficients,
            target_coefficients,
            len(values),
            source_name=args.source_name,
            target_name=args.target_name,
        )
        mapped = transform.apply(source_record.coordinates())
        if mapped != target_record.coordinates():
            raise ValidationError("recurrence coordinate transformation failed exact reconstruction")
    except (ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {
        "source_coordinates": _fraction_strings(source_record.coordinates()),
        "target_coordinates": _fraction_strings(target_record.coordinates()),
        "matrix": _fraction_strings(transform.matrix),
        "determinant": str(transform.determinant),
        "verified": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_coordinates_polynomial(args: argparse.Namespace) -> int:
    """Connect any two built-in graded polynomial families."""

    from ..coordinates import connection_transform, get_polynomial_family

    try:
        source_family = get_polynomial_family(args.source)
        target_family = get_polynomial_family(args.target)
        transform = connection_transform(source_family, target_family, args.degree)
        transform.certificate().require()
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {
        "degree": args.degree,
        "source": transform.source,
        "target": transform.target,
        "matrix": _fraction_strings(transform.matrix),
        "determinant": str(transform.determinant),
        "verified": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_coordinates_catalog(_: argparse.Namespace) -> int:
    """List the built-in coordinate systems and transform families."""

    from ..coordinates import classical_polynomial_catalog, recurrence_model_catalog

    payload = {
        "exact_linear": [
            "arbitrary finite bases",
            "coordinate atlases",
            "permutations and scalings",
            "direct sums and tensor products",
            "operator, covector, bilinear, and tensor transport",
        ],
        "polynomial_families": list(classical_polynomial_catalog().names()),
        "recurrence_models": [model.name for model in recurrence_model_catalog()],
        "combinatorial": [
            "Stirling first kind",
            "Stirling second kind",
            "Lah",
            "binomial",
            "forward difference",
            "subset zeta and Mobius",
            "divisor zeta and Mobius",
        ],
        "affine_and_formal": [
            "affine charts",
            "barycentric coordinates",
            "formal jet substitution",
            "formal inverse coordinates",
        ],
        "state_space": [
            "companion recurrence coordinates",
            "controllability coordinates",
            "observability coordinates",
            "modal coordinates",
            "intertwiner and similarity coordinates",
        ],
        "spectral": ["DFT", "DCT-II", "Walsh-Hadamard", "Haar", "eigenmode"],
        "packet": ["residue lanes", "digit reversal", "CRT order", "tensor-axis order"],
        "arithmetic": ["prime-field transforms", "exact rational reduction modulo a prime"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_coordinates_basis(args: argparse.Namespace) -> int:
    """Connect two arbitrary exact finite bases supplied as column matrices."""

    from ..coordinates import basis_from_matrix

    try:
        source_basis = basis_from_matrix(args.source_name, _fraction_matrix(args.source_matrix))
        target_basis = basis_from_matrix(args.target_name, _fraction_matrix(args.target_matrix))
        transform = source_basis.connection_to(target_basis)
        transform.certificate().require()
        payload: dict[str, object] = {
            "source": source_basis.name,
            "target": target_basis.name,
            "ambient_dimension": source_basis.ambient_dimension,
            "dimension": source_basis.dimension,
            "matrix": _fraction_strings(transform.matrix),
            "determinant": str(transform.determinant),
            "verified": True,
        }
        if args.vector is not None:
            ambient = _fraction_csv(args.vector)
            source_coordinates = source_basis.analyze(ambient)
            target_coordinates = target_basis.analyze(ambient)
            mapped = transform.apply(source_coordinates)
            if mapped != target_coordinates:
                raise ValidationError("arbitrary basis coordinates did not commute")
            payload["ambient_vector"] = _fraction_strings(ambient)
            payload["source_coordinates"] = _fraction_strings(source_coordinates)
            payload["target_coordinates"] = _fraction_strings(target_coordinates)
    except (ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_coordinates_model(args: argparse.Namespace) -> int:
    """Change a finite sequence between named recurrence-model coordinates."""

    from ..coordinates import get_recurrence_model, sequence_coordinates_between_models

    try:
        source = get_recurrence_model(args.source)
        target = get_recurrence_model(args.target)
        if args.values is None:
            values = source.sequence(args.count)
        else:
            values = _fraction_csv(args.values)
        source_record, target_record, transform = sequence_coordinates_between_models(
            values, source, target
        )
    except (ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {
        "values": _fraction_strings(values),
        "source": source.name,
        "target": target.name,
        "source_coordinates": _fraction_strings(source_record.coordinates()),
        "target_coordinates": _fraction_strings(target_record.coordinates()),
        "matrix": _fraction_strings(transform.matrix),
        "verified": transform.apply(source_record.coordinates()) == target_record.coordinates(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_coordinates_combinatorial(args: argparse.Namespace) -> int:
    """Construct one exact combinatorial coordinate transform."""

    from ..coordinates import (
        binomial_transform,
        falling_factorial_to_monomial_transform,
        forward_difference_transform,
        monomial_to_falling_factorial_transform,
        monomial_to_rising_factorial_transform,
        rising_factorial_to_monomial_transform,
        rising_to_falling_factorial_transform,
    )

    constructors = {
        "binomial": binomial_transform,
        "inverse-binomial": lambda size: binomial_transform(size, inverse=True),
        "forward-difference": forward_difference_transform,
        "monomial-to-falling": monomial_to_falling_factorial_transform,
        "falling-to-monomial": falling_factorial_to_monomial_transform,
        "monomial-to-rising": monomial_to_rising_factorial_transform,
        "rising-to-monomial": rising_factorial_to_monomial_transform,
        "rising-to-falling": rising_to_falling_factorial_transform,
    }
    try:
        transform = constructors[args.kind](args.size)
        transform.certificate().require()
        values = None if args.values is None else _fraction_csv(args.values)
        if values is not None and len(values) != transform.source_dimension:
            raise ValidationError("value count must equal the transform size")
    except (KeyError, ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {
        "kind": args.kind,
        "source": transform.source,
        "target": transform.target,
        "matrix": _fraction_strings(transform.matrix),
        "output": None if values is None else _fraction_strings(transform.apply(values)),
        "verified": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_coordinates_spectral(args: argparse.Namespace) -> int:
    """Construct one finite spectral or wavelet coordinate transform."""

    from ..coordinates import (
        discrete_cosine_transform,
        discrete_fourier_transform,
        haar_wavelet_transform,
        walsh_hadamard_transform,
    )

    constructors = {
        "dft": discrete_fourier_transform,
        "dct": discrete_cosine_transform,
        "hadamard": walsh_hadamard_transform,
        "haar": haar_wavelet_transform,
    }
    try:
        transform = constructors[args.kind](args.size)
        transform.certificate().require()
        values = None if args.values is None else [float(value) for value in args.values.split(",") if value.strip()]
        if values is not None and len(values) != transform.source_dimension:
            raise ValidationError("value count must equal the transform size")
    except (KeyError, ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {
        "kind": args.kind,
        "source": transform.source,
        "target": transform.target,
        "condition_number": transform.condition_number(),
        "matrix": _numeric_json(transform.matrix),
        "output": None if values is None else _numeric_json(transform.apply(values)),
        "verified": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_coordinates_packet(args: argparse.Namespace) -> int:
    """Construct an exact packet or index-order coordinate permutation."""

    from ..coordinates import (
        chinese_remainder_transform,
        digit_reversal_transform,
        residue_class_transform,
    )

    try:
        if args.kind == "residue":
            transform = residue_class_transform(args.length, args.modulus)
        elif args.kind == "digit-reversal":
            transform = digit_reversal_transform(args.length, args.radix)
        else:
            transform = chinese_remainder_transform(_integer_csv(args.moduli))
        transform.certificate().require()
    except (ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {
        "kind": args.kind,
        "source": transform.source,
        "target": transform.target,
        "matrix": _fraction_strings(transform.matrix),
        "verified": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_coordinates_formal(args: argparse.Namespace) -> int:
    """Apply an exact truncated formal change of variable."""

    from ..coordinates import formal_coordinate_from_iterable

    try:
        change = formal_coordinate_from_iterable(
            _fraction_csv(args.coordinate),
            name=args.name,
            source=args.source,
            target=args.target,
        )
        change.certificate().require()
        inverse = change.inverse()
        payload: dict[str, object] = {
            "name": change.name,
            "source": change.source,
            "target": change.target,
            "coefficients": _fraction_strings(change.coefficients),
            "inverse": _fraction_strings(inverse.coefficients),
            "substitution_matrix": _fraction_strings(change.substitution_transform().matrix),
            "verified": True,
        }
        if args.series is not None:
            coefficients = _fraction_csv(args.series)
            payload["pullback"] = _fraction_strings(change.pullback_coefficients(coefficients))
            payload["roundtrip"] = _fraction_strings(
                change.pushforward_coefficients(change.pullback_coefficients(coefficients))
            )
    except (ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_generating_mask(args: argparse.Namespace) -> int:
    """Compile and evaluate one finite Mahler mask."""

    from ..generating import compile_finite_mask_carry_plan

    try:
        plan = compile_finite_mask_carry_plan(_fraction_csv(args.mask), args.radix)
        certificate = plan.certificate(args.count)
        certificate.require()
    except (ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = {
        "radix": plan.radix,
        "mask": [str(value) for value in plan.mask],
        "carry_bound": plan.carry_bound,
        "dimension": plan.dimension,
        "coefficients": [str(value) for value in plan.sequence(args.count)],
        "verified": certificate.valid,
    }
    print(json.dumps(payload, indent=2))
    return 0


def command_generating_cartier(args: argparse.Namespace) -> int:
    """Compile one rational Mahler mask modulo a prime."""

    from ..generating import compile_cartier_linear_representation

    try:
        representation = compile_cartier_linear_representation(
            _integer_csv(args.numerator),
            _integer_csv(args.denominator),
            args.prime,
        )
        values = representation.sequence(args.count)
    except (ValidationError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "prime": representation.prime,
        "degree_bound": representation.degree_bound,
        "dimension": representation.dimension,
        "coefficients": values,
    }, indent=2))
    return 0


def command_generating_kernel(args: argparse.Namespace) -> int:
    """Evaluate one factorial-power kernel modulo a prime power."""

    from ..generating import PrimePowerKernelCompiler

    try:
        compiler = PrimePowerKernelCompiler(
            args.prime,
            args.exponent,
            args.power,
        )
        result = compiler.kernel(args.index, args.subindex)
    except (ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "index": result.index,
        "subindex": result.subindex,
        "prime": result.prime,
        "exponent": result.exponent,
        "factorial_power": result.factorial_power,
        "valuation": result.valuation,
        "unit": result.unit,
        "residue": result.residue,
        "survives": result.survives,
    }, indent=2))
    return 0


def command_generating_bell(args: argparse.Namespace) -> int:
    """Evaluate one generalized Bell digit-sum reduction."""

    from ..generating import generalized_bell_digit_reduction

    try:
        result = generalized_bell_digit_reduction(
            args.index,
            args.prime,
            args.power,
        )
        result.require()
    except (ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({
        "index": result.index,
        "prime": result.prime,
        "factorial_power": result.factorial_power,
        "digit_sum": result.digit_sum,
        "high_polynomial": result.high_polynomial,
        "reduced_polynomial": result.reduced_polynomial,
        "valid": result.valid,
    }, indent=2))
    return 0


def command_export_formats(_: argparse.Namespace) -> int:
    """List every supported export format."""

    from ..export import available_export_formats

    print(json.dumps({"formats": available_export_formats()}, indent=2))
    return 0


def command_export_sequence(args: argparse.Namespace) -> int:
    """Export one built in sequence as data or a mathematical report."""

    from ..export import export_document, sequence_document
    from ..sequences import compile_sequence, get_sequence

    sequence = get_sequence(args.name)
    compiled = compile_sequence(sequence) if args.compile else None
    values = compiled.first(args.terms) if compiled is not None else sequence.first(args.terms)
    if args.modulus is not None:
        model = compiled or compile_sequence(sequence)
        values = tuple(model.modulo(index, args.modulus) for index in range(args.terms))
    document = sequence_document(
        sequence.name,
        values,
        strategy=None if compiled is None else compiled.strategy,
        modulus=args.modulus,
        author=args.author,
    )
    options = _export_options(args)
    result = export_document(document, args.output, format=args.format, options=options)
    _print_export_result(result)
    return 0


def command_export_standard_window(args: argparse.Namespace) -> int:
    """Export one residual standard window report."""

    from ..arithmetic import StandardWindow
    from ..export import document_from_data, export_document

    window = StandardWindow(args.prime, args.compression)
    certificate = window.certificate(args.address, periods=max(2, args.count // window.period + 1))
    certificate.certificate.require()
    payload = {
        "prime": window.prime,
        "compression": window.compression,
        "address": args.address,
        "period": window.period,
        "minimum_period": window.minimum_period(),
        "coefficients": window.residues(args.address, args.count - 1) if args.count else (),
        "certificate": certificate.certificate,
    }
    document = document_from_data(
        payload,
        title=args.title,
        author=args.author,
        date=args.date,
        abstract=args.abstract,
    )
    result = export_document(document, args.output, format=args.format, options=_export_options(args))
    _print_export_result(result)
    return 0


def command_export_euler(args: argparse.Namespace) -> int:
    """Export one odd or dyadic Euler coordinate report."""

    from ..arithmetic import EulerLocalCoordinates
    from ..export import document_from_data, export_document

    coordinates = EulerLocalCoordinates(args.prime)
    if args.depth is None:
        payload = {
            "prime": args.prime,
            "index": args.index,
            "value": coordinates.reconstruct(args.index),
            "exact": True,
        }
    else:
        result = coordinates.truncate(args.index, args.depth)
        result.certificate.require()
        payload = {
            "prime": args.prime,
            "index": args.index,
            "depth": args.depth,
            "value": result.value,
            "exact_value": result.exact_value,
            "modulus": result.modulus,
            "error": result.error,
            "certificate": result.certificate,
        }
    document = document_from_data(
        payload,
        title=args.title,
        author=args.author,
        date=args.date,
        abstract=args.abstract,
    )
    exported = export_document(document, args.output, format=args.format, options=_export_options(args))
    _print_export_result(exported)
    return 0


def command_export_json(args: argparse.Namespace) -> int:
    """Convert a JSON data file to another supported output format."""

    from ..export import export_data

    input_path = __import__("pathlib").Path(args.input)
    value = json.loads(input_path.read_text(encoding="utf-8-sig"))
    result = export_data(value, args.output, format=args.format, options=_export_options(args))
    _print_export_result(result)
    return 0


def command_export_bundle(args: argparse.Namespace) -> int:
    """Create a multi-format ZIP bundle from a JSON data file."""

    from pathlib import Path

    from ..export import export_bundle, to_serializable

    value = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    result = export_bundle(
        value,
        args.output,
        base_name=args.base_name,
        formats=formats,
        options=_export_options(args),
    )
    print(json.dumps(to_serializable(result), indent=2, sort_keys=True))
    return 0


def command_latex_engines(_: argparse.Namespace) -> int:
    """Report available local LaTeX compilers."""

    from ..export import available_latex_engines

    print(json.dumps(available_latex_engines(), indent=2, sort_keys=True))
    return 0


def command_latex_compile(args: argparse.Namespace) -> int:
    """Compile one existing LaTeX source file."""

    from pathlib import Path

    from ..export import LatexCompileOptions, compile_latex, to_serializable

    source = Path(args.source)
    output = Path(args.output) if args.output else source.with_suffix(".pdf")
    if output.suffix.lower() != ".pdf":
        print("LaTeX compilation output must use a .pdf suffix", file=sys.stderr)
        return 2
    options = LatexCompileOptions(
        engine=args.engine,
        passes=args.passes,
        timeout=args.timeout,
        keep_auxiliary=args.keep_auxiliary,
        jobname=output.stem,
    )
    result = compile_latex(source, output.parent, options=options)
    print(json.dumps(to_serializable(result), indent=2, sort_keys=True))
    return 0 if result.valid else 1

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="komorebi", description="Komorebi Math command line tools")
    subcommands = parser.add_subparsers(dest="command", required=True)

    info = subcommands.add_parser("info", help="show package identity")
    info.set_defaults(function=command_info)

    doctor = subcommands.add_parser("doctor", help="inspect optional dependencies and extensions")
    doctor.set_defaults(function=command_doctor)

    feature_parser = subcommands.add_parser("features", help="show implemented and planned domains")
    feature_parser.set_defaults(function=command_features)

    verify = subcommands.add_parser("verify", help="run an installed verification suite")
    verify.add_argument("suite", choices=VERIFICATION_SUITE_NAMES)
    verify.set_defaults(function=command_verify)

    api_parser = subcommands.add_parser("api", help="inspect or snapshot the public API contract")
    api_commands = api_parser.add_subparsers(dest="api_command", required=True)
    api_check = api_commands.add_parser("check", help="compare the package with an API manifest")
    api_check.add_argument("--manifest")
    api_check.add_argument("--strict-signatures", action="store_true")
    api_check_tier = api_check.add_mutually_exclusive_group()
    api_check_tier.add_argument("--beta", action="store_true", help="check the curated strict beta namespace")
    api_check_tier.add_argument("--stable", action="store_true", help="check the frozen 1.0 candidate namespace")
    api_check.set_defaults(function=command_api_check)
    api_snapshot = api_commands.add_parser("snapshot", help="write the current API manifest")
    api_snapshot.add_argument("--output", required=True)
    api_snapshot.add_argument("--force", action="store_true")
    api_snapshot_tier = api_snapshot.add_mutually_exclusive_group()
    api_snapshot_tier.add_argument("--beta", action="store_true", help="snapshot the curated beta namespace")
    api_snapshot_tier.add_argument("--stable", action="store_true", help="snapshot the frozen 1.0 candidate namespace")
    api_snapshot.set_defaults(function=command_api_snapshot)

    golden_parser = subcommands.add_parser("golden", help="run committed behavioral semantics")
    golden_parser.set_defaults(function=command_golden)

    release_parser = subcommands.add_parser("release", help="run release qualification gates")
    release_commands = release_parser.add_subparsers(dest="release_command", required=True)
    release_contract = release_commands.add_parser("contract", help="validate and print the frozen release contract")
    release_contract.set_defaults(function=command_release_contract)
    release_environment = release_commands.add_parser("environment", help="print the current runtime qualification record")
    release_environment.set_defaults(function=command_release_environment)
    release_qualify = release_commands.add_parser("qualify", help="qualify the installed 1.0 freeze candidate")
    release_qualify.add_argument("--gpu", action="store_true", help="probe CUDA hardware when available")
    release_qualify.add_argument("--require-gpu", action="store_true", help="fail unless the CUDA hardware gate passes")
    release_qualify.add_argument("--output", help="write the machine-readable qualification record")
    release_qualify.add_argument("--force", action="store_true")
    release_qualify.set_defaults(function=command_release_qualify)
    release_upgrade = release_commands.add_parser(
        "upgrade", help="validate predecessor API, persistence, and portable records"
    )
    release_upgrade.set_defaults(function=command_release_upgrade)
    release_evidence = release_commands.add_parser(
        "evidence", help="validate digest-checked platform evidence"
    )
    release_evidence.add_argument(
        "--current",
        action="store_true",
        help="require current-version evidence on every declared platform family",
    )
    release_evidence.add_argument("--output", help="write the evidence report")
    release_evidence.add_argument("--force", action="store_true")
    release_evidence.set_defaults(function=command_release_evidence)
    release_documentation = release_commands.add_parser(
        "documentation", help="validate exact stable documentation coverage"
    )
    release_documentation.set_defaults(function=command_release_documentation)
    release_readiness = release_commands.add_parser(
        "readiness", help="run public-beta or release-candidate readiness"
    )
    release_readiness.add_argument(
        "--final",
        action="store_true",
        help="require current-version execution on every declared platform family",
    )
    release_readiness.add_argument("--gpu", action="store_true")
    release_readiness.add_argument("--require-gpu", action="store_true")
    release_readiness.add_argument("--output", help="write the readiness report")
    release_readiness.add_argument("--force", action="store_true")
    release_readiness.set_defaults(function=command_release_readiness)

    stream_parser = subcommands.add_parser("stream", help="inspect packet-jet stream directories")
    stream_commands = stream_parser.add_subparsers(dest="stream_command", required=True)
    stream_inspect = stream_commands.add_parser("inspect", help="print a packet-jet stream manifest")
    stream_inspect.add_argument("path")
    stream_inspect.add_argument("--allow-incomplete", action="store_true")
    stream_inspect.add_argument("--verify-chunks", action="store_true")
    stream_inspect.set_defaults(function=command_stream_inspect)
    stream_verify = stream_commands.add_parser("verify", help="verify every committed stream chunk")
    stream_verify.add_argument("path")
    stream_verify.add_argument("--allow-incomplete", action="store_true")
    stream_verify.set_defaults(function=command_stream_verify)

    sequence_parser = subcommands.add_parser("sequence", help="evaluate a built in exact sequence")
    sequence_parser.add_argument("name")
    sequence_parser.add_argument("--terms", type=int, default=20)
    sequence_parser.add_argument("--index", type=int)
    sequence_parser.add_argument("--modulus", type=int)
    sequence_parser.add_argument("--compile", action="store_true", help="use the strongest declared execution model")
    sequence_parser.set_defaults(function=command_sequence)

    coordinate_parser = subcommands.add_parser(
        "coordinates", help="compute exact changes between mathematical coordinate systems"
    )
    coordinate_commands = coordinate_parser.add_subparsers(
        dest="coordinate_command", required=True
    )
    coordinate_euler = coordinate_commands.add_parser(
        "euler-fibonacci", help="express signed Euler numbers in Fibonacci companion coordinates"
    )
    coordinate_euler.add_argument("--count", type=int, default=20)
    coordinate_euler.set_defaults(function=command_coordinates_euler_fibonacci)

    coordinate_recurrence = coordinate_commands.add_parser(
        "recurrence", help="change a finite prefix between recurrence defect coordinates"
    )
    coordinate_recurrence.add_argument("--values", required=True)
    coordinate_recurrence.add_argument("--source", required=True)
    coordinate_recurrence.add_argument("--target", required=True)
    coordinate_recurrence.add_argument("--source-name", default="source recurrence")
    coordinate_recurrence.add_argument("--target-name", default="target recurrence")
    coordinate_recurrence.set_defaults(function=command_coordinates_recurrence)

    coordinate_catalog = coordinate_commands.add_parser(
        "catalog", help="list available coordinate families and transform categories"
    )
    coordinate_catalog.set_defaults(function=command_coordinates_catalog)

    coordinate_basis = coordinate_commands.add_parser(
        "basis", help="connect arbitrary exact bases supplied as column matrices"
    )
    coordinate_basis.add_argument("--source-matrix", required=True)
    coordinate_basis.add_argument("--target-matrix", required=True)
    coordinate_basis.add_argument("--source-name", default="source basis")
    coordinate_basis.add_argument("--target-name", default="target basis")
    coordinate_basis.add_argument("--vector")
    coordinate_basis.set_defaults(function=command_coordinates_basis)

    coordinate_model = coordinate_commands.add_parser(
        "model", help="change between named recurrence-model coordinates"
    )
    coordinate_model.add_argument("--source", required=True)
    coordinate_model.add_argument("--target", required=True)
    coordinate_model.add_argument("--count", type=int, default=12)
    coordinate_model.add_argument("--values")
    coordinate_model.set_defaults(function=command_coordinates_model)

    coordinate_polynomial = coordinate_commands.add_parser(
        "polynomial", help="connect any two built-in graded polynomial families"
    )
    coordinate_polynomial.add_argument("--source", default="euler")
    coordinate_polynomial.add_argument("--target", default="fibonacci")
    coordinate_polynomial.add_argument("--degree", type=int, default=8)
    coordinate_polynomial.set_defaults(function=command_coordinates_polynomial)

    coordinate_combinatorial = coordinate_commands.add_parser(
        "combinatorial", help="construct Stirling, binomial, and finite-difference transforms"
    )
    coordinate_combinatorial.add_argument(
        "--kind",
        choices=[
            "binomial",
            "inverse-binomial",
            "forward-difference",
            "monomial-to-falling",
            "falling-to-monomial",
            "monomial-to-rising",
            "rising-to-monomial",
            "rising-to-falling",
        ],
        required=True,
    )
    coordinate_combinatorial.add_argument("--size", type=int, required=True)
    coordinate_combinatorial.add_argument("--values")
    coordinate_combinatorial.set_defaults(function=command_coordinates_combinatorial)

    coordinate_spectral = coordinate_commands.add_parser(
        "spectral", help="construct Fourier, cosine, Hadamard, or Haar coordinates"
    )
    coordinate_spectral.add_argument(
        "--kind", choices=["dft", "dct", "hadamard", "haar"], required=True
    )
    coordinate_spectral.add_argument("--size", type=int, required=True)
    coordinate_spectral.add_argument("--values")
    coordinate_spectral.set_defaults(function=command_coordinates_spectral)

    coordinate_packet = coordinate_commands.add_parser(
        "packet", help="construct residue, digit-reversal, or CRT index coordinates"
    )
    coordinate_packet.add_argument(
        "--kind", choices=["residue", "digit-reversal", "crt"], required=True
    )
    coordinate_packet.add_argument("--length", type=int, default=8)
    coordinate_packet.add_argument("--modulus", type=int, default=2)
    coordinate_packet.add_argument("--radix", type=int, default=2)
    coordinate_packet.add_argument("--moduli", default="2,3")
    coordinate_packet.set_defaults(function=command_coordinates_packet)

    coordinate_formal = coordinate_commands.add_parser(
        "formal", help="apply an invertible truncated formal coordinate change"
    )
    coordinate_formal.add_argument("--coordinate", required=True)
    coordinate_formal.add_argument("--series")
    coordinate_formal.add_argument("--name", default="formal coordinate change")
    coordinate_formal.add_argument("--source", default="source formal coordinate")
    coordinate_formal.add_argument("--target", default="target formal coordinate")
    coordinate_formal.set_defaults(function=command_coordinates_formal)

    generating_parser = subcommands.add_parser(
        "generating", help="compile normalized generating functions"
    )
    generating_commands = generating_parser.add_subparsers(
        dest="generating_command", required=True
    )
    mask_parser = generating_commands.add_parser(
        "mask", help="compile a finite Mahler mask"
    )
    mask_parser.add_argument("--mask", required=True, help="comma-separated rational coefficients")
    mask_parser.add_argument("--radix", type=int, required=True)
    mask_parser.add_argument("--count", type=int, default=20)
    mask_parser.set_defaults(function=command_generating_mask)

    cartier_parser = generating_commands.add_parser(
        "cartier", help="compile a rational mask modulo a prime"
    )
    cartier_parser.add_argument("--numerator", required=True)
    cartier_parser.add_argument("--denominator", required=True)
    cartier_parser.add_argument("--prime", type=int, required=True)
    cartier_parser.add_argument("--count", type=int, default=20)
    cartier_parser.set_defaults(function=command_generating_cartier)

    kernel_parser = generating_commands.add_parser(
        "kernel", help="evaluate a factorial-power prime-power kernel"
    )
    kernel_parser.add_argument("--index", type=int, required=True)
    kernel_parser.add_argument("--subindex", type=int, required=True)
    kernel_parser.add_argument("--prime", type=int, required=True)
    kernel_parser.add_argument("--exponent", type=int, required=True)
    kernel_parser.add_argument("--power", type=int, required=True)
    kernel_parser.set_defaults(function=command_generating_kernel)

    bell_parser = generating_commands.add_parser(
        "bell", help="evaluate a generalized Bell digit reduction"
    )
    bell_parser.add_argument("--index", type=int, required=True)
    bell_parser.add_argument("--prime", type=int, required=True)
    bell_parser.add_argument("--power", type=int, required=True)
    bell_parser.set_defaults(function=command_generating_bell)

    arithmetic_parser = subcommands.add_parser("arithmetic", help="run local arithmetic computations")
    arithmetic_commands = arithmetic_parser.add_subparsers(dest="arithmetic_command", required=True)

    window_parser = arithmetic_commands.add_parser("standard-window", help="compute a residual standard window")
    window_parser.add_argument("--prime", type=int, required=True)
    window_parser.add_argument("--compression", type=int, required=True)
    window_parser.add_argument("--address", type=int, default=0)
    window_parser.add_argument("--count", type=int, default=20)
    window_parser.set_defaults(function=command_standard_window)

    euler_parser = arithmetic_commands.add_parser("euler", help="compute an odd or dyadic Euler coordinate")
    euler_parser.add_argument("--prime", type=int, required=True)
    euler_parser.add_argument("--index", type=int, required=True)
    euler_parser.add_argument("--depth", type=int)
    euler_parser.set_defaults(function=command_euler_coordinates)

    deformation_parser = subcommands.add_parser("deformation", help="run finite deformation computations")
    deformation_commands = deformation_parser.add_subparsers(dest="deformation_command", required=True)
    count_parser = deformation_commands.add_parser("count", help="evaluate one finite chart count")
    count_parser.add_argument("--chart", choices=["smooth", "weighted", "nilpotent", "normal-crossing"], required=True)
    count_parser.add_argument("--q", type=int, required=True)
    count_parser.add_argument("--depth", type=int, required=True)
    count_parser.add_argument("--dimension", type=int, default=1)
    count_parser.add_argument("--weights", default="1")
    count_parser.add_argument("--exponent", type=int, default=2)
    count_parser.set_defaults(function=command_deformation_count)

    langlands_parser = subcommands.add_parser("langlands", help="run finite Langlands packet diagnostics")
    langlands_commands = langlands_parser.add_subparsers(dest="langlands_command", required=True)
    tame_parser = langlands_commands.add_parser("tame-orbit", help="inspect one tame Frobenius orbit")
    tame_parser.add_argument("--q", type=int, required=True)
    tame_parser.add_argument("--degree", type=int, required=True)
    tame_parser.add_argument("--exponent", type=int, required=True)
    tame_parser.set_defaults(function=command_tame_orbit)
    heisenberg_parser = langlands_commands.add_parser("heisenberg", help="validate one finite Heisenberg model")
    heisenberg_parser.add_argument("--prime", type=int, required=True)
    heisenberg_parser.add_argument("--rank", type=int, required=True)
    heisenberg_parser.set_defaults(function=command_heisenberg)
    completion_parser = langlands_commands.add_parser("completion", help="evaluate the five formal completion gates")
    completion_parser.add_argument("--satisfied", default="", help="comma separated gate names")
    completion_parser.set_defaults(function=command_completion)

    export_parser = subcommands.add_parser("export", help="export mathematical data and reports")
    export_commands = export_parser.add_subparsers(dest="export_command", required=True)
    formats_parser = export_commands.add_parser("formats", help="list supported export formats")
    formats_parser.set_defaults(function=command_export_formats)

    def add_export_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--output", required=True)
        command.add_argument("--format")
        command.add_argument("--title", default="Komorebi Math Export")
        command.add_argument("--author")
        command.add_argument("--date")
        command.add_argument("--abstract")
        command.add_argument("--document-class", default="article")
        command.add_argument("--toc", action="store_true")
        command.add_argument("--mathjax", action="store_true")
        command.add_argument("--engine", default="auto", choices=["auto", "pdflatex", "xelatex", "lualatex", "latexmk"])
        command.add_argument("--passes", type=int, default=2)
        command.add_argument("--timeout", type=float, default=120.0)
        command.add_argument("--keep-auxiliary", action="store_true")
        command.add_argument("--no-keep-tex", action="store_true")
        command.add_argument("--no-type-metadata", action="store_true")
        command.add_argument("--force", action="store_true")

    export_sequence = export_commands.add_parser("sequence", help="export one built in exact sequence")
    export_sequence.add_argument("name")
    export_sequence.add_argument("--terms", type=int, default=20)
    export_sequence.add_argument("--modulus", type=int)
    export_sequence.add_argument("--compile", action="store_true")
    add_export_options(export_sequence)
    export_sequence.set_defaults(function=command_export_sequence)

    export_window = export_commands.add_parser("standard-window", help="export one residual standard window")
    export_window.add_argument("--prime", type=int, required=True)
    export_window.add_argument("--compression", type=int, required=True)
    export_window.add_argument("--address", type=int, default=0)
    export_window.add_argument("--count", type=int, default=20)
    add_export_options(export_window)
    export_window.set_defaults(function=command_export_standard_window)

    export_euler = export_commands.add_parser("euler", help="export one local Euler reconstruction")
    export_euler.add_argument("--prime", type=int, required=True)
    export_euler.add_argument("--index", type=int, required=True)
    export_euler.add_argument("--depth", type=int)
    add_export_options(export_euler)
    export_euler.set_defaults(function=command_export_euler)

    export_json = export_commands.add_parser("json", help="convert one JSON data file")
    export_json.add_argument("--input", required=True)
    add_export_options(export_json)
    export_json.set_defaults(function=command_export_json)

    export_bundle_parser = export_commands.add_parser("bundle", help="create a multi-format ZIP from JSON")
    export_bundle_parser.add_argument("--input", required=True)
    export_bundle_parser.add_argument("--base-name", default="komorebi_export")
    export_bundle_parser.add_argument("--formats", default="json,markdown,latex")
    add_export_options(export_bundle_parser)
    export_bundle_parser.set_defaults(function=command_export_bundle)

    latex_parser = subcommands.add_parser("latex", help="inspect or run a local LaTeX compiler")
    latex_commands = latex_parser.add_subparsers(dest="latex_command", required=True)
    latex_engines = latex_commands.add_parser("engines", help="show available LaTeX engines")
    latex_engines.set_defaults(function=command_latex_engines)
    latex_compile = latex_commands.add_parser("compile", help="compile an existing LaTeX source")
    latex_compile.add_argument("source")
    latex_compile.add_argument("--output")
    latex_compile.add_argument("--engine", default="auto", choices=["auto", "pdflatex", "xelatex", "lualatex", "latexmk"])
    latex_compile.add_argument("--passes", type=int, default=2)
    latex_compile.add_argument("--timeout", type=float, default=120.0)
    latex_compile.add_argument("--keep-auxiliary", action="store_true")
    latex_compile.set_defaults(function=command_latex_compile)

    describe_parser = subcommands.add_parser("describe", help="show mathematical provenance for an object")
    describe_parser.add_argument("object", help="fully qualified Python object name")
    describe_parser.set_defaults(function=command_describe)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
