"""Finite cochain complexes and packetwise cohomology over prime fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .._number_theory import require_integer, require_prime
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError


def _matrix_mod(value: Any, prime: int) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.ndim != 2:
        raise DimensionError("cochain maps must be matrices")
    # Store Python integers so products/sums are exact before modular reduction.
    output = np.empty(array.shape, dtype=object)
    for index, entry in np.ndenumerate(array):
        try:
            integer = int(entry)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("cochain maps over a prime field require integral entries") from exc
        if integer != entry:
            raise ValidationError("cochain maps over a prime field require integral entries")
        output[index] = integer % prime
    return output


def row_rank_mod(matrix: Any, prime: int) -> int:
    """Return the exact row rank over a prime field."""

    p = require_prime(prime)
    value = _matrix_mod(matrix, p).copy()
    rows, columns = value.shape
    rank = 0
    for column in range(columns):
        pivot = next((r for r in range(rank, rows) if value[r, column] % p), None)
        if pivot is None:
            continue
        if pivot != rank:
            value[[rank, pivot]] = value[[pivot, rank]]
        inverse = pow(int(value[rank, column]), -1, p)
        value[rank] = (value[rank] * inverse) % p
        for row in range(rows):
            if row == rank:
                continue
            factor = int(value[row, column]) % p
            if factor:
                value[row] = (value[row] - factor * value[rank]) % p
        rank += 1
        if rank == rows:
            break
    return rank


@dataclass(frozen=True, slots=True)
class CohomologyData:
    """Dimension data for one cohomological degree."""

    degree: int
    ambient_dimension: int
    cycle_dimension: int
    boundary_dimension: int
    cohomology_dimension: int


class FiniteCochainComplex:
    """A bounded finite cochain complex over a prime field."""

    def __init__(
        self,
        dimensions: Mapping[int, int],
        differentials: Mapping[int, Any],
        *,
        prime: int,
    ) -> None:
        self.prime = require_prime(prime)
        self.dimensions = {
            require_integer(degree, name="degree"): require_integer(dimension, name="dimension")
            for degree, dimension in dimensions.items()
        }
        if not self.dimensions:
            raise ValidationError("a cochain complex requires at least one degree")
        if any(dimension < 0 for dimension in self.dimensions.values()):
            raise ValidationError("cochain dimensions must be nonnegative")
        self.differentials: dict[int, np.ndarray] = {}
        for degree, matrix in differentials.items():
            d = require_integer(degree, name="degree")
            if d not in self.dimensions or d + 1 not in self.dimensions:
                raise ValidationError("differential degree is outside the complex")
            value = _matrix_mod(matrix, self.prime)
            expected = (self.dimensions[d + 1], self.dimensions[d])
            if value.shape != expected:
                raise DimensionError(
                    f"differential at degree {d} has shape {value.shape}, expected {expected}"
                )
            self.differentials[d] = value
        for degree in self.dimensions:
            if degree in self.differentials and degree + 1 in self.differentials:
                product_matrix = (
                    self.differentials[degree + 1] @ self.differentials[degree]
                ) % self.prime
                if np.any(product_matrix):
                    raise ValidationError("successive cochain differentials do not compose to zero")

    def differential(self, degree: int) -> np.ndarray:
        """Return one differential, inserting a zero map when absent."""

        d = require_integer(degree, name="degree")
        source = self.dimensions.get(d, 0)
        target = self.dimensions.get(d + 1, 0)
        value = _matrix_mod(self.differentials.get(d, np.zeros((target, source), dtype=object)), self.prime)
        if value.shape != (target, source):
            raise DimensionError("differential has the wrong current degree dimension")
        return value

    def cohomology(self, degree: int) -> CohomologyData:
        """Return exact dimension data for one degree."""

        d = require_integer(degree, name="degree")
        if d not in self.dimensions:
            raise ValidationError("degree is outside the complex")
        outgoing = self.differential(d)
        incoming = self.differential(d - 1)
        outgoing_rank = row_rank_mod(outgoing, self.prime)
        incoming_rank = row_rank_mod(incoming, self.prime)
        cycles = self.dimensions[d] - outgoing_rank
        if incoming_rank > cycles:
            raise ArithmeticError("boundary dimension exceeds the cycle dimension")
        return CohomologyData(d, self.dimensions[d], cycles, incoming_rank, cycles - incoming_rank)

    def certificate(self) -> Certificate:
        """Return a certificate for the complex identities."""

        residuals: dict[str, float] = {}
        valid = True
        for degree in sorted(self.dimensions):
            if degree in self.differentials and degree + 1 in self.differentials:
                residual = (self.differential(degree + 1) @ self.differential(degree)) % self.prime
                norm = float(np.max(np.abs(residual), initial=0))
                residuals[f"d{degree + 1}_d{degree}"] = norm
                valid = valid and not bool(np.any(residual))
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals=residuals,
            assumptions_checked=("prime field", "finite bounded complex", "successive differentials"),
            diagnostics={"arithmetic": "Python integers reduced modulo the prime; object arrays",
                         "residual_floats_are_diagnostics_only": True},
            theorem_id="PSC-X-COCHAIN-COMPLEX",
            message="valid finite cochain complex" if valid else "cochain identity failed",
        )

    def packet_cohomology(
        self,
        projectors: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, dict[int, CohomologyData]]:
        """Split cohomology by a compatible family of packet projectors.

        Every degree must use the same packet names.  Projectors must be
        pairwise orthogonal, complete, and commute with the differentials.
        """

        names: tuple[str, ...] | None = None
        normalized: dict[int, dict[str, np.ndarray]] = {}
        for degree, packet_map in projectors.items():
            d = require_integer(degree, name="degree")
            if d not in self.dimensions:
                raise ValidationError("packet degree is outside the complex")
            current = tuple(packet_map)
            if names is None:
                names = current
            elif current != names:
                raise ValidationError("every degree must use the same packet order")
            dimension = self.dimensions[d]
            entries: dict[str, np.ndarray] = {}
            total = np.zeros((dimension, dimension), dtype=object)
            for name, raw in packet_map.items():
                matrix = _matrix_mod(raw, self.prime)
                if matrix.shape != (dimension, dimension):
                    raise DimensionError("packet projector has the wrong degree dimension")
                if np.any((matrix @ matrix - matrix) % self.prime):
                    raise ValidationError("packet matrix is not idempotent")
                for earlier in entries.values():
                    if np.any((matrix @ earlier) % self.prime) or np.any((earlier @ matrix) % self.prime):
                        raise ValidationError("packet projectors are not orthogonal")
                entries[name] = matrix
                total = (total + matrix) % self.prime
            if np.any((total - np.eye(dimension, dtype=object)) % self.prime):
                raise ValidationError("packet projectors are not complete")
            normalized[d] = entries
        if names is None:
            raise ValidationError("packet cohomology requires projectors")
        if set(normalized) != set(self.dimensions):
            raise ValidationError("packet projectors are required in every degree")
        for degree in self.differentials:
            differential = self.differential(degree)
            for name in names:
                if np.any(
                    (normalized[degree + 1][name] @ differential - differential @ normalized[degree][name])
                    % self.prime
                ):
                    raise ValidationError("packet projectors do not commute with the differential")

        result: dict[str, dict[int, CohomologyData]] = {name: {} for name in names}
        for name in names:
            packet_dimensions = {
                degree: row_rank_mod(normalized[degree][name], self.prime)
                for degree in self.dimensions
            }
            restricted_ranks: dict[int, int] = {}
            for degree in self.differentials:
                differential = self.differential(degree)
                p_source = normalized[degree][name]
                p_target = normalized[degree + 1][name]
                restricted_ranks[degree] = row_rank_mod(
                    (p_target @ differential @ p_source) % self.prime,
                    self.prime,
                )
            for degree, ambient in packet_dimensions.items():
                outgoing_rank = restricted_ranks.get(degree, 0)
                incoming_rank = restricted_ranks.get(degree - 1, 0)
                cycles = ambient - outgoing_rank
                if incoming_rank > cycles:
                    raise ArithmeticError(
                        "packet boundary dimension exceeds the packet cycle dimension"
                    )
                result[name][degree] = CohomologyData(
                    degree,
                    ambient,
                    cycles,
                    incoming_rank,
                    cycles - incoming_rank,
                )
        return result


__all__ = [
    "CohomologyData",
    "FiniteCochainComplex",
    "row_rank_mod",
]
