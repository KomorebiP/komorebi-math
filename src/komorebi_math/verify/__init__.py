"""Installed package verification entry points."""

from .algorithms import run_algorithms_verification
from .analytic import run_analytic_verification
from .arithmetic import run_arithmetic_verification
from .coordinates import run_coordinates_verification
from .core import VerificationCheck, VerificationReport, run_core_verification
from .deformation import run_deformation_verification
from .dsp import run_dsp_verification
from .dynamics import run_dynamics_verification
from .export import run_export_verification
from .frame_algebra import run_frame_algebra_verification
from .integration import run_integration_verification
from .expression import run_expression_verification
from .generating import run_generating_verification
from .gpu import run_gpu_verification
from .langlands import (
    run_langlands_completion_verification,
    run_langlands_finite_image_verification,
    run_langlands_local_verification,
    run_langlands_verification,
)
from .nonlinear import run_nonlinear_foundation_verification, run_nonlinear_verification
from .nonlinear_advanced import run_advanced_nonlinear_verification
from .quantum import run_quantum_verification
from .refinement import run_refinement_verification
from .release import run_release_qualification_verification
from .scalable_nonlinear import run_scalable_nonlinear_verification
from .sequences import run_sequences_verification
from .signals import run_signals_verification
from .stabilization import (
    run_polyphase_stabilization_verification,
    run_stable_contract_verification,
)
from .streaming_beta import run_streaming_beta_verification
from .beta import (
    run_beta_readiness_verification,
    run_documentation_verification,
    run_evidence_verification,
    run_upgrade_verification,
)


def _merge(*reports: VerificationReport) -> VerificationReport:
    return VerificationReport(tuple(check for report in reports for check in report.checks))


def run_block2_verification() -> VerificationReport:
    """Run the core, refinement, and nonlinear foundation suites."""

    return _merge(
        run_core_verification(),
        run_refinement_verification(),
        run_nonlinear_foundation_verification(),
    )


def run_block3_verification() -> VerificationReport:
    """Run every implemented suite through the structured algorithms block."""

    return _merge(run_block2_verification(), run_algorithms_verification())


def run_block4_verification() -> VerificationReport:
    """Run every implemented suite through signals, probability, and memory."""

    return _merge(run_block3_verification(), run_signals_verification(), run_dynamics_verification())


def run_block5_verification() -> VerificationReport:
    """Run every implemented suite through analytic and finite quantum domains."""

    return _merge(run_block4_verification(), run_analytic_verification(), run_quantum_verification())


def run_block6_verification() -> VerificationReport:
    """Run every implemented suite through local arithmetic and sequences."""

    return _merge(run_block5_verification(), run_arithmetic_verification(), run_sequences_verification())


def run_block7_verification() -> VerificationReport:
    """Run every implemented suite through deformation and finite Langlands models."""

    return _merge(run_block6_verification(), run_deformation_verification(), run_langlands_verification())


def run_block8_verification() -> VerificationReport:
    """Run every implemented suite through portable export and LaTeX output."""

    return _merge(run_block7_verification(), run_export_verification())


def run_block9_verification() -> VerificationReport:
    """Run every implemented suite through advanced nonlinear integration."""

    return _merge(run_block8_verification(), run_advanced_nonlinear_verification())


def run_block10_verification() -> VerificationReport:
    """Run every implemented suite through scalable nonlinear hardening."""

    return _merge(run_block9_verification(), run_scalable_nonlinear_verification())


def run_block11_verification() -> VerificationReport:
    """Run every implemented suite through streaming persistence and beta contracts."""

    return _merge(run_block10_verification(), run_streaming_beta_verification())


def run_block12_verification() -> VerificationReport:
    """Run every implemented suite through the production DSP expansion."""

    return _merge(run_block11_verification(), run_dsp_verification())


def run_block13_verification() -> VerificationReport:
    """Run every implemented suite through normalized generating functions."""

    return _merge(run_block12_verification(), run_generating_verification())


def run_block14_verification() -> VerificationReport:
    """Run every implemented suite through exact coordinate changes."""

    return _merge(run_block13_verification(), run_coordinates_verification())


def run_block15_verification() -> VerificationReport:
    """Run every implemented suite through canonical polyphase stabilization."""

    return _merge(run_block14_verification(), run_polyphase_stabilization_verification())


def run_stable_verification() -> VerificationReport:
    """Run the frozen API, semantic fixture, persistence, and alias checks."""

    return run_stable_contract_verification()


def run_block16_verification() -> VerificationReport:
    """Run every installed suite through the 1.0 release-qualification block."""

    return _merge(
        run_block15_verification(),
        run_stable_contract_verification(),
        run_release_qualification_verification(),
    )


def run_block17_verification() -> VerificationReport:
    """Run every installed suite through the 1.0 public-beta closure block."""

    return _merge(
        run_block16_verification(),
        run_upgrade_verification(),
        run_evidence_verification(),
        run_documentation_verification(),
        run_beta_readiness_verification(),
    )


def run_block18_verification() -> VerificationReport:
    """Run the additive frame algebra and the frozen stable contract."""

    return _merge(
        run_frame_algebra_verification(),
        run_stable_contract_verification(),
    )


def run_block19_verification() -> VerificationReport:
    """Run finite integration plus the Block 18 and frozen stable contracts."""

    return _merge(
        run_integration_verification(),
        run_frame_algebra_verification(),
        run_stable_contract_verification(),
    )


def run_block20_verification() -> VerificationReport:
    """Run typed expressions plus the Block 19 and frozen stable contracts."""

    return _merge(
        run_expression_verification(),
        run_block19_verification(),
    )


__all__ = [
    "VerificationCheck",
    "VerificationReport",
    "run_advanced_nonlinear_verification",
    "run_algorithms_verification",
    "run_analytic_verification",
    "run_arithmetic_verification",
    "run_block10_verification",
    "run_block11_verification",
    "run_block12_verification",
    "run_block13_verification",
    "run_block14_verification",
    "run_block15_verification",
    "run_block16_verification",
    "run_block2_verification",
    "run_block3_verification",
    "run_block4_verification",
    "run_block5_verification",
    "run_block6_verification",
    "run_block7_verification",
    "run_block8_verification",
    "run_block9_verification",
    "run_coordinates_verification",
    "run_core_verification",
    "run_deformation_verification",
    "run_dsp_verification",
    "run_dynamics_verification",
    "run_export_verification",
    "run_generating_verification",
    "run_gpu_verification",
    "run_langlands_completion_verification",
    "run_langlands_finite_image_verification",
    "run_langlands_local_verification",
    "run_langlands_verification",
    "run_nonlinear_foundation_verification",
    "run_nonlinear_verification",
    "run_polyphase_stabilization_verification",
    "run_quantum_verification",
    "run_refinement_verification",
    "run_release_qualification_verification",
    "run_scalable_nonlinear_verification",
    "run_sequences_verification",
    "run_signals_verification",
    "run_stable_contract_verification",
    "run_stable_verification",
    "run_streaming_beta_verification",
    "run_beta_readiness_verification",
    "run_block17_verification",
    "run_block18_verification",
    "run_block19_verification",
    "run_block20_verification",
    "run_documentation_verification",
    "run_evidence_verification",
    "run_frame_algebra_verification",
    "run_integration_verification",
    "run_expression_verification",
    "run_upgrade_verification",
]
