"""Finite quantum packet calculus, channels, transport, and diagnostics."""

from .._planned import development_status
from ..registry import FeatureSpec
from .channels import (
    ChannelPacketBlock,
    QuantumChannel,
    amplitude_damping,
    dephase,
    operator_packets,
    reconstruct_operator_packets,
)
from .circuits import CircuitPacketExpansion, CircuitPathAmplitude, InterferenceReport
from .constraints import ConstraintClosureReport, constraint_closure, transported_constraint_defect
from .effective import QuantumFeshbachReduction
from .error_correction import KnillLaflammeReport, QuantumCode
from .euclidean import EuclideanTransfer
from .kato import DirectRotationResult, KatoTransport, MovingPacketSystem, direct_packet_rotation
from .leakage import LeakageResult, fixed_packet_leakage, moving_geometric_coupling, moving_packet_leakage_bound
from .lindblad import LindbladGenerator
from .seeds import (
    QuantumSeedSystem,
    computational_basis_generation,
    local_orbit_dimension,
    observable_seed_rank,
    schmidt_rank,
)
from .spectral import (
    heat_trace,
    low_mode_dispersion_bound,
    periodic_laplacian,
    periodic_lattice_eigenvalues,
    spectral_dimension,
)
from .symmetry import (
    charge_certificate,
    charge_selection_certificate,
    cyclic_symmetry_projectors,
    finite_group_projector,
    symmetry_projector_certificate,
    total_charge_packets,
)
from .towers import BoundedHamiltonianTower, HamiltonianTowerReport

FEATURE_KEY = "quantum"


def status() -> FeatureSpec:
    """Return the registered implementation status for this domain."""

    return development_status(FEATURE_KEY)


def require() -> None:
    """Confirm that this package block is implemented."""

    return None


__all__ = [
    "FEATURE_KEY",
    "BoundedHamiltonianTower",
    "ChannelPacketBlock",
    "CircuitPacketExpansion",
    "CircuitPathAmplitude",
    "ConstraintClosureReport",
    "DirectRotationResult",
    "EuclideanTransfer",
    "HamiltonianTowerReport",
    "InterferenceReport",
    "KatoTransport",
    "KnillLaflammeReport",
    "LeakageResult",
    "LindbladGenerator",
    "MovingPacketSystem",
    "QuantumChannel",
    "QuantumCode",
    "QuantumFeshbachReduction",
    "QuantumSeedSystem",
    "amplitude_damping",
    "charge_certificate",
    "charge_selection_certificate",
    "computational_basis_generation",
    "constraint_closure",
    "cyclic_symmetry_projectors",
    "dephase",
    "direct_packet_rotation",
    "finite_group_projector",
    "fixed_packet_leakage",
    "heat_trace",
    "local_orbit_dimension",
    "low_mode_dispersion_bound",
    "moving_geometric_coupling",
    "moving_packet_leakage_bound",
    "observable_seed_rank",
    "operator_packets",
    "periodic_laplacian",
    "periodic_lattice_eigenvalues",
    "reconstruct_operator_packets",
    "schmidt_rank",
    "spectral_dimension",
    "require",
    "status",
    "symmetry_projector_certificate",
    "total_charge_packets",
    "transported_constraint_defect",
]
