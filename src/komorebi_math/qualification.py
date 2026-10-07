"""Release qualification and stable semantic regression for Komorebi Math.

The source test suite remains the strongest development check.  This module is
an installed-package gate: it validates the frozen API aliases, the common
analysis--synthesis law, polyphase refinement, exact arithmetic preservation,
portable persistence, and optional hardware diagnostics without importing
native optional backends into the parent interpreter.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .certificates import Certificate, Exactness
from .compatibility import check_stable_api
from .nonlinear.autodiff import ad_framework_diagnostics
from .release import check_release_contract, check_upgrade_contract, check_validation_evidence, qualify_gpu
from .stable_contract import check_stable_contract
from .support import load_support_matrix

RELEASE_QUALIFICATION_SCHEMA = "komorebi-math.release-qualification"
RELEASE_QUALIFICATION_SCHEMA_VERSION = 1
SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX = (3, 13)
RELEASE_CANDIDATE_VERSION = "1.0.0b4"


@dataclass(frozen=True, slots=True)
class QualificationCheck:
    """One release qualification check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class QualificationVerificationReport:
    """Finite collection of qualification checks."""

    checks: tuple[QualificationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(check.passed for check in self.checks)

    @property
    def total_count(self) -> int:
        return len(self.checks)


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_inventory() -> dict[str, Any]:
    """Return deterministic runtime metadata without importing native extras."""

    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "distributions": {
            name: _distribution_version(name)
            for name in (
                "scipy",
                "sympy",
                "mpmath",
                "jax",
                "jaxlib",
                "torch",
                "cupy",
                "cupy-cuda12x",
                "numba",
                "llvmlite",
            )
        },
    }


def _capture(name: str, function: Any) -> QualificationCheck:
    try:
        function()
        return QualificationCheck(name, True)
    except Exception as exc:  # release gate must report all finite failures
        return QualificationCheck(name, False, f"{type(exc).__name__}: {exc}")


def run_stable_semantic_verification() -> QualificationVerificationReport:
    """Execute the closed semantic fixture for the 1.0 freeze candidate."""

    checks: list[QualificationCheck] = []

    def aliases() -> None:
        from . import stable, v1
        from .api import v1 as api_v1

        assert stable.__all__ == v1.__all__ == api_v1.__all__
        for name in stable.__all__:
            assert getattr(stable, name) is getattr(v1, name)
            assert getattr(stable, name) is getattr(api_v1, name)

    checks.append(_capture("versioned v1 aliases", aliases))

    def stable_manifest() -> None:
        report = check_stable_api()
        report.certificate.require()
        assert report.removed == ()
        assert report.kind_changed == ()
        assert report.signature_changed == ()

    checks.append(_capture("strict stable API manifest", stable_manifest))

    def square_analysis_synthesis() -> None:
        from .coordinates import AnalysisSynthesisSystem

        size = 4
        analysis = np.fft.fft(np.eye(size), axis=0) / np.sqrt(size)
        system = AnalysisSynthesisSystem.from_invertible(
            analysis,
            source="samples",
            coordinates="Fourier coordinates",
            tolerance=1e-11,
        )
        system.certificate().require()
        value = np.array([1.0, -2.0, 0.5, 4.0])
        system.reconstruction_certificate(value).require()
        assert np.allclose(system.reconstruct(value), value, atol=1e-11, rtol=1e-11)

    checks.append(_capture("square analysis--synthesis reconstruction", square_analysis_synthesis))

    def redundant_analysis_synthesis() -> None:
        from .coordinates import AnalysisSynthesisSystem

        analysis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        system = AnalysisSynthesisSystem.canonical_dual(
            analysis,
            source="two dimensional source",
            coordinates="three coefficient frame",
        )
        assert system.redundant
        system.certificate().require()
        value = np.array([2.0, -3.0])
        assert np.allclose(system.reconstruct(value), value)
        components = system.components()
        assert np.allclose(sum(components), np.eye(2))

    checks.append(_capture("redundant canonical-dual reconstruction", redundant_analysis_synthesis))

    def operator_blocks() -> None:
        from .coordinates import AnalysisSynthesisSystem

        analysis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        system = AnalysisSynthesisSystem.canonical_dual(analysis)
        operator = np.array([[2.0, -1.0], [3.0, 4.0]])
        blocks = system.operator_blocks(operator, groups=((0, 2), (1,)))
        blocks.certificate().require()
        assert np.allclose(blocks.reconstruct(), operator)

    checks.append(_capture("analysis--synthesis transfer blocks", operator_blocks))

    def signal_refinement() -> None:
        from .refinement import coarsen_polyphase, refine_polyphase
        from .signals import polyphase_analyze

        signal = np.arange(70, dtype=np.int64).reshape(5, 14)
        coarse = polyphase_analyze(signal, 2, axis=1)
        fine = refine_polyphase(coarse, 8)
        direct = polyphase_analyze(signal, 8, axis=1)
        assert all(
            np.array_equal(left, right)
            for left, right in zip(fine.channels, direct.channels, strict=True)
        )
        assert np.array_equal(coarsen_polyphase(fine, 2).reconstruct(), signal)

    checks.append(_capture("polyphase refinement coherence", signal_refinement))

    def exact_arithmetic() -> None:
        from fractions import Fraction

        from .arithmetic import polyphase_analyze

        values = tuple(Fraction(index + 1, index + 3) for index in range(19))
        coarse = polyphase_analyze(values, 3)
        fine = coarse.refine(12)
        assert fine == polyphase_analyze(values, 12)
        assert fine.coarsen(3) == coarse
        assert fine.reconstruct() == values

    checks.append(_capture("exact arithmetic refinement", exact_arithmetic))

    def persistence() -> None:
        from .coordinates import AnalysisSynthesisSystem
        from .export.persistence import dumps_portable, loads_portable
        from .refinement import PolyphaseRefinement

        system = AnalysisSynthesisSystem.canonical_dual(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        )
        restored = loads_portable(dumps_portable(system))
        assert isinstance(restored, AnalysisSynthesisSystem)
        restored.certificate().require()
        assert np.allclose(restored.analysis, system.analysis)
        assert np.allclose(restored.synthesis, system.synthesis)
        assert loads_portable(dumps_portable(PolyphaseRefinement(4, 12))) == PolyphaseRefinement(4, 12)

    checks.append(_capture("stable portable persistence", persistence))

    def core_construction() -> None:
        from .core import OperatorGrammar, OperatorRealization, PacketOperator, PacketSystem, SeedSystem

        grammar = OperatorGrammar(["X"], {"T": ("X", "X")})
        word = grammar.word("T")
        grammar.relation("idempotent", [(1, grammar.word("T", "T")), (-1, word)])
        operator = np.diag([1.0, 0.0, 1.0])
        realization = OperatorRealization(grammar, {"X": 3}, {"T": operator})
        realization.relation_certificate().require()
        packets = PacketSystem((np.diag([1, 1, 0]), np.diag([0, 0, 1])))
        blocked = PacketOperator.from_operator(operator, packets)
        blocked.certificate().require()
        seed = SeedSystem((operator,), (np.array([1.0, 1.0, 0.0]),))
        assert seed.hull_dimension() >= 1

    checks.append(_capture("stable construction pipeline", core_construction))

    def settings_scope() -> None:
        from .config import get_settings, settings_context

        original = get_settings()
        with settings_context(tolerance=1e-8, backend="numpy") as temporary:
            assert temporary.tolerance == 1e-8
            assert get_settings() == temporary
        assert get_settings() == original

    checks.append(_capture("stable settings context", settings_scope))

    return QualificationVerificationReport(tuple(checks))


def gpu_diagnostics(*, timeout: float = 120.0) -> dict[str, Any]:
    """Run PyTorch and CuPy CUDA checks in independent child interpreters."""

    report = qualify_gpu(timeout=timeout)
    by_name = {backend.backend: backend for backend in report.backends}
    torch_result = by_name.get("torch")
    cupy_result = by_name.get("cupy")
    torch_diagnostics = {} if torch_result is None else torch_result.diagnostics
    cupy_diagnostics = {} if cupy_result is None else cupy_result.diagnostics
    return {
        "available": report.passed,
        "passed": report.passed,
        "backends": [backend.to_dict() for backend in report.backends],
        "certificate": report.certificate.to_dict(),
        "torch_version": None if torch_result is None else torch_result.version,
        "torch_cuda": torch_diagnostics.get("compiled_cuda"),
        "gpu": None if torch_result is None else torch_result.device,
        "cupy_version": None if cupy_result is None else cupy_result.version,
        "device_count": cupy_diagnostics.get("device_count", 0),
        "torch_jvp": torch_diagnostics.get("jvp"),
        "packet_executor_residual": cupy_diagnostics.get("residual"),
        "detail": "; ".join(
            f"{backend.backend}: {backend.detail}" for backend in report.backends if not backend.passed
        ),
    }


@dataclass(frozen=True, slots=True)
class ReleaseQualificationReport:
    """Machine-readable installed-package release qualification record."""

    version: str
    runtime: dict[str, Any]
    optional_backends: dict[str, dict[str, Any]]
    verification: QualificationVerificationReport
    gpu: dict[str, Any] | None = None
    gpu_required: bool = False
    schema: str = RELEASE_QUALIFICATION_SCHEMA
    schema_version: int = RELEASE_QUALIFICATION_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        gpu_ok = not self.gpu_required or bool(self.gpu and self.gpu.get("available"))
        return self.verification.passed and gpu_ok

    def certificate(self) -> Certificate:
        return Certificate(
            valid=self.passed,
            exactness=Exactness.NUMERICAL,
            assumptions_checked=tuple(check.name for check in self.verification.checks if check.passed),
            assumptions_unchecked=(
                ()
                if self.gpu is not None
                else ("hardware-backed CUDA qualification not requested",)
            ),
            diagnostics={
                "version": self.version,
                "passed": self.verification.passed_count,
                "total": self.verification.total_count,
                "gpu_required": self.gpu_required,
                "gpu_available": None if self.gpu is None else bool(self.gpu.get("available")),
            },
            theorem_id="KM-REL-16",
            message="release qualification passed" if self.passed else "release qualification failed",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "version": self.version,
            "runtime": self.runtime,
            "optional_backends": self.optional_backends,
            "gpu": self.gpu,
            "gpu_required": self.gpu_required,
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in self.verification.checks
            ],
            "certificate": self.certificate().to_dict(),
        }

    def write(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination


def run_release_qualification(
    *,
    include_gpu: bool = False,
    require_gpu: bool = False,
) -> ReleaseQualificationReport:
    """Run the installed 1.0 public-beta qualification gate."""

    if require_gpu:
        include_gpu = True
    checks = list(run_stable_semantic_verification().checks)

    def aggregate_contract() -> None:
        check_stable_contract().certificate.require()

    checks.append(_capture("aggregate stable compatibility contract", aggregate_contract))

    def release_contract() -> None:
        check_release_contract().certificate.require()

    checks.append(_capture("machine-readable current beta release contract", release_contract))

    checks.append(_capture("predecessor upgrade contract", lambda: check_upgrade_contract().certificate.require()))
    checks.append(_capture("executed platform evidence", lambda: check_validation_evidence().certificate.require()))

    def support_policy() -> None:
        matrix = load_support_matrix()
        assert matrix.release_line
        assert matrix.python["minimum"] == "3.11"
        assert tuple(matrix.python["declared"]) == ("3.11", "3.12", "3.13")
        assert matrix.validated_environments

    checks.append(_capture("machine-readable support policy", support_policy))

    version_tuple = sys.version_info[:2]
    checks.append(
        QualificationCheck(
            "supported Python version",
            SUPPORTED_PYTHON_MIN <= version_tuple <= SUPPORTED_PYTHON_MAX,
            f"observed {platform.python_version()}",
        )
    )
    checks.append(
        QualificationCheck(
            "release version identity",
            __version__ == RELEASE_CANDIDATE_VERSION,
            f"observed {__version__}; expected {RELEASE_CANDIDATE_VERSION}",
        )
    )
    gpu = gpu_diagnostics() if include_gpu else None
    if include_gpu:
        checks.append(
            QualificationCheck(
                "CUDA hardware qualification",
                bool(gpu and gpu.get("available")) if require_gpu else True,
                json.dumps(gpu, sort_keys=True),
            )
        )
    return ReleaseQualificationReport(
        version=__version__,
        runtime=runtime_inventory(),
        optional_backends=ad_framework_diagnostics(),
        verification=QualificationVerificationReport(tuple(checks)),
        gpu=gpu,
        gpu_required=bool(require_gpu),
    )


__all__ = [
    "RELEASE_QUALIFICATION_SCHEMA",
    "RELEASE_QUALIFICATION_SCHEMA_VERSION",
    "RELEASE_CANDIDATE_VERSION",
    "QualificationCheck",
    "QualificationVerificationReport",
    "ReleaseQualificationReport",
    "gpu_diagnostics",
    "run_release_qualification",
    "run_stable_semantic_verification",
    "runtime_inventory",
]
