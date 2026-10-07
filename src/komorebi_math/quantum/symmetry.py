"""Finite group, cyclic charge, and tensor symmetry packets."""

from __future__ import annotations

import cmath
from collections.abc import Hashable, Mapping, Sequence
from math import tau
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..core import PacketSystem
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import as_square_matrix, require_unitary


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-T08",
        title="Finite group isotypic projector",
        status=ClaimStatus.CLASSICAL,
        source="Paper VII",
        assumptions=("finite unitary representation", "irreducible character values"),
        verification_suite="quantum.symmetry",
    )
)
def finite_group_projector(
    representation: Mapping[Hashable, Any],
    character: Mapping[Hashable, complex],
    irrep_dimension: int,
) -> np.ndarray:
    """Construct one finite group central isotypic projector."""

    if not representation or set(representation) != set(character):
        raise ValidationError("representation and character must have the same nonempty group keys")
    dimension = int(irrep_dimension)
    if isinstance(irrep_dimension, bool) or dimension != irrep_dimension or dimension < 1:
        raise ValidationError("irrep_dimension must be a positive integer")
    matrices = {key: require_unitary(value, name="group representation matrix") for key, value in representation.items()}
    shape = next(iter(matrices.values())).shape
    if any(matrix.shape != shape for matrix in matrices.values()):
        raise DimensionError("all representation matrices must have one common dimension")
    return dimension / len(matrices) * sum(np.conj(character[key]) * matrices[key] for key in matrices)


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-T09",
        title="Cyclic character projectors",
        status=ClaimStatus.PROVED,
        source="Paper VII",
        assumptions=("unitary symmetry of finite order",),
        verification_suite="quantum.symmetry",
    )
)
def cyclic_symmetry_projectors(
    symmetry: Any,
    order: int,
    *,
    tolerance: float = 1e-10,
) -> PacketSystem:
    """Return the complete packet system of a finite cyclic unitary symmetry."""

    unitary = require_unitary(symmetry, tolerance=tolerance, name="cyclic symmetry")
    n = int(order)
    if isinstance(order, bool) or n != order or n < 1:
        raise ValidationError("order must be a positive integer")
    identity = np.eye(unitary.shape[0], dtype=complex)
    if not np.allclose(np.linalg.matrix_power(unitary, n), identity, atol=tolerance, rtol=0.0):
        raise ValidationError("symmetry does not satisfy U^order = I for the declared cyclic action")
    omega = cmath.exp(1j * tau / n)
    projectors = []
    power = identity.copy()
    powers = []
    for _ in range(n):
        powers.append(power)
        power = power @ unitary
    for charge in range(n):
        projectors.append(sum(omega ** (-charge * exponent) * powers[exponent] for exponent in range(n)) / n)
    return PacketSystem(projectors, names=tuple(str(charge) for charge in range(n)), tolerance=tolerance)


def charge_certificate(
    symmetry: Any,
    operator: Any,
    charge: int,
    order: int,
    *,
    tolerance: float = 1e-10,
) -> Certificate:
    """Check whether an operator has one declared cyclic charge."""

    unitary = require_unitary(symmetry, tolerance=tolerance, name="cyclic symmetry")
    value = as_square_matrix(operator, name="charged operator")
    if value.shape != unitary.shape:
        raise DimensionError("operator and symmetry dimensions do not agree")
    n = int(order)
    if isinstance(order, bool) or n != order or n < 1:
        raise ValidationError("order must be a positive integer")
    identity = np.eye(unitary.shape[0], dtype=complex)
    closure_residual = float(np.linalg.norm(np.linalg.matrix_power(unitary, n) - identity, ord=2))
    closure_tolerance = tolerance * max(1.0, float(n))
    q = int(charge) % n
    phase = cmath.exp(1j * tau * q / n)
    residual = float(np.linalg.norm(unitary @ value @ unitary.conj().T - phase * value, ord=2))
    scale = max(1.0, float(np.linalg.norm(value, ord=2)))
    charge_tolerance = tolerance * scale
    return Certificate(
        valid=(
            closure_residual <= closure_tolerance
            and residual <= charge_tolerance
        ),
        exactness=Exactness.NUMERICAL,
        residuals={
            "cyclic_closure": closure_residual,
            "charge_relation": residual,
        },
        tolerance=max(closure_tolerance, charge_tolerance),
        assumptions_checked=("unitary symmetry",),
        assumptions_unchecked=() if closure_residual <= closure_tolerance else ("U^order = I cyclic closure",),
        diagnostics={
            "order": n,
            "cyclic_closure_tolerance": closure_tolerance,
            "charge_relation_tolerance": charge_tolerance,
        },
        theorem_id="PSC-VII-T11",
        message=(
            "operator has the declared cyclic charge"
            if closure_residual <= closure_tolerance
            else "declared symmetry does not close at the requested cyclic order"
        ),
    )


def charge_selection_certificate(
    packets: PacketSystem,
    operator: Any,
    charge: int,
    *,
    tolerance: float = 1e-10,
) -> Certificate:
    """Check that a charged operator only connects allowed packet addresses."""

    value = as_square_matrix(operator, name="charged operator")
    if value.shape != (packets.dimension, packets.dimension):
        raise DimensionError("operator and packet dimensions do not agree")
    order = len(packets)
    q = int(charge) % order
    residual = 0.0
    for source in range(order):
        for target in range(order):
            if target != (source + q) % order:
                residual = max(residual, float(np.linalg.norm(packets.projectors[target] @ value @ packets.projectors[source], ord=2)))
    scale = max(1.0, float(np.linalg.norm(value, ord=2)))
    return Certificate(
        valid=residual <= tolerance * scale,
        exactness=Exactness.NUMERICAL,
        residuals={"forbidden_block": residual},
        tolerance=tolerance * scale,
        assumptions_checked=("complete cyclic packet family",),
        theorem_id="PSC-VII-T11",
        message="charged operator obeys the packet selection rule",
    )


def total_charge_packets(
    left: PacketSystem,
    right: PacketSystem,
    *,
    tolerance: float = 1e-10,
) -> PacketSystem:
    """Return the tensor packet system grouped by total cyclic charge."""

    if len(left) != len(right):
        raise ValidationError("left and right cyclic packet systems must have the same order")
    order = len(left)
    projectors = []
    for total in range(order):
        projectors.append(
            sum(
                np.kron(left.projectors[charge], right.projectors[(total - charge) % order])
                for charge in range(order)
            )
        )
    return PacketSystem(projectors, names=tuple(str(index) for index in range(order)), tolerance=tolerance)


def symmetry_projector_certificate(
    projectors: Sequence[Any],
    *,
    tolerance: float = 1e-10,
) -> ComputationResult[PacketSystem]:
    """Validate a supplied finite symmetry projector family."""

    packets = PacketSystem(projectors, tolerance=tolerance)
    return ComputationResult(packets, packets.certificate())


__all__ = [
    "charge_certificate",
    "charge_selection_certificate",
    "cyclic_symmetry_projectors",
    "finite_group_projector",
    "symmetry_projector_certificate",
    "total_charge_packets",
]
