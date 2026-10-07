"""Finite Heisenberg representations and isotropic selectors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

from ..._number_theory import require_integer, require_prime
from ...certificates import Certificate, Exactness
from ...exceptions import DimensionError, ValidationError


def _vector(value: Iterable[int], *, prime: int, rank: int, name: str) -> tuple[int, ...]:
    entries = tuple(require_integer(entry, name=f"{name} entry") % prime for entry in value)
    if len(entries) != rank:
        raise DimensionError(f"{name} must have length {rank}")
    return entries


def _dot(left: tuple[int, ...], right: tuple[int, ...], prime: int) -> int:
    return sum(a * b for a, b in zip(left, right, strict=True)) % prime


@dataclass(frozen=True, slots=True)
class HeisenbergElement:
    """One coordinate element of a finite Heisenberg group."""

    translation: tuple[int, ...]
    modulation: tuple[int, ...]
    center: int = 0


class HeisenbergModel:
    """The finite Schrödinger representation over ``F_p^rank``."""

    def __init__(self, prime: int, rank: int, *, max_dimension: int = 4096) -> None:
        self.prime = require_prime(prime)
        self.rank = require_integer(rank, name="rank")
        if self.rank < 1:
            raise ValidationError("Heisenberg rank must be positive")
        self.dimension = self.prime**self.rank
        if self.dimension > require_integer(max_dimension, name="max_dimension"):
            raise ValidationError("Heisenberg matrix dimension exceeds the declared safety limit")
        self.points = tuple(product(range(self.prime), repeat=self.rank))
        self.index = {point: index for index, point in enumerate(self.points)}
        self.root = np.exp(2j * np.pi / self.prime)

    def element(
        self,
        translation: Iterable[int],
        modulation: Iterable[int],
        center: int = 0,
    ) -> HeisenbergElement:
        """Construct one normalized group element."""

        return HeisenbergElement(
            _vector(translation, prime=self.prime, rank=self.rank, name="translation"),
            _vector(modulation, prime=self.prime, rank=self.rank, name="modulation"),
            require_integer(center, name="center") % self.prime,
        )

    def multiply(self, left: HeisenbergElement, right: HeisenbergElement) -> HeisenbergElement:
        """Multiply two Heisenberg coordinates in the chosen convention."""

        translation = tuple((a + b) % self.prime for a, b in zip(left.translation, right.translation, strict=True))
        modulation = tuple((a + b) % self.prime for a, b in zip(left.modulation, right.modulation, strict=True))
        center = (left.center + right.center - _dot(right.modulation, left.translation, self.prime)) % self.prime
        return HeisenbergElement(translation, modulation, center)

    def inverse(self, element: HeisenbergElement) -> HeisenbergElement:
        """Return the group inverse."""

        translation = tuple((-value) % self.prime for value in element.translation)
        modulation = tuple((-value) % self.prime for value in element.modulation)
        correction = _dot(element.modulation, element.translation, self.prime)
        center = (-element.center - correction) % self.prime
        return HeisenbergElement(translation, modulation, center)

    def translation_matrix(self, vector: Iterable[int]) -> np.ndarray:
        """Return the translation matrix of one vector."""

        shift = _vector(vector, prime=self.prime, rank=self.rank, name="translation")
        matrix = np.zeros((self.dimension, self.dimension), dtype=complex)
        for source, point in enumerate(self.points):
            target_point = tuple((point[index] + shift[index]) % self.prime for index in range(self.rank))
            matrix[self.index[target_point], source] = 1.0
        return matrix

    def modulation_matrix(self, vector: Iterable[int]) -> np.ndarray:
        """Return the modulation matrix of one dual vector."""

        frequency = _vector(vector, prime=self.prime, rank=self.rank, name="modulation")
        diagonal = [self.root ** _dot(frequency, point, self.prime) for point in self.points]
        return np.diag(diagonal)

    def matrix(self, element: HeisenbergElement) -> np.ndarray:
        """Return the Schrödinger matrix of one group element."""

        phase = self.root**element.center
        return phase * self.modulation_matrix(element.modulation) @ self.translation_matrix(element.translation)

    def group_law_certificate(
        self,
        left: HeisenbergElement,
        right: HeisenbergElement,
        *,
        tolerance: float = 1e-10,
    ) -> Certificate:
        """Check the exact coordinate law in the numerical representation."""

        residual = float(np.linalg.norm(self.matrix(left) @ self.matrix(right) - self.matrix(self.multiply(left, right))))
        return Certificate(
            valid=residual <= tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"group_law": residual},
            tolerance=tolerance,
            assumptions_checked=("finite Schrödinger representation",),
            theorem_id="PSC-XI-HEISENBERG-LAW",
            message="Heisenberg group law verified" if residual <= tolerance else "Heisenberg group law failed",
        )

    def trace(self, element: HeisenbergElement) -> complex:
        """Return the representation trace."""

        return complex(np.trace(self.matrix(element)))

    def trace_support_certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        """Check that trace is supported on the center."""

        maximum = 0.0
        for translation in self.points:
            for modulation in self.points:
                if all(value == 0 for value in translation + modulation):
                    continue
                maximum = max(maximum, abs(self.trace(HeisenbergElement(translation, modulation, 0))))
        return Certificate(
            valid=maximum <= tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"off_center_trace": maximum},
            tolerance=tolerance,
            assumptions_checked=("finite exhaustive phase space",),
            theorem_id="PSC-XI-HEISENBERG-TRACE",
            message="Heisenberg trace is supported on the center" if maximum <= tolerance else "off-center trace is nonzero",
        )

    def translation_selector(self) -> np.ndarray:
        """Average the standard maximal isotropic translation subgroup."""

        selector = sum((self.translation_matrix(vector) for vector in self.points), np.zeros((self.dimension, self.dimension), dtype=complex))
        return selector / self.dimension

    def selector_certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        """Check idempotence and rank one of the standard isotropic selector."""

        selector = self.translation_selector()
        idempotence = float(np.linalg.norm(selector @ selector - selector))
        self_adjoint = float(np.linalg.norm(selector.conj().T - selector))
        rank = int(np.linalg.matrix_rank(selector, tol=tolerance))
        valid = idempotence <= tolerance and self_adjoint <= tolerance and rank == 1
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"idempotence": idempotence, "self_adjoint": self_adjoint},
            tolerance=tolerance,
            assumptions_checked=("standard translation Lagrangian",),
            diagnostics={"rank": rank},
            theorem_id="PSC-XI-ISOTROPIC-SELECTOR",
            message="rank one isotropic selector verified" if valid else "isotropic selector validation failed",
        )


__all__ = ["HeisenbergElement", "HeisenbergModel"]
