"""Linear refinement defects and their propagated identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..backends import LinearAlgebraBackend, get_backend
from .._residuals import ResidualAudit
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import BackendUnavailableError, DimensionError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _matrix_exponential(matrix: Any, backend: LinearAlgebraBackend) -> Any:
    if backend.name == "sympy":
        return matrix.exp()
    try:
        from scipy.linalg import expm
    except ImportError as exc:
        raise BackendUnavailableError("matrix exponential defects require the 'scientific' SciPy extra") from exc
    return expm(matrix)


@documented(
    ProvenanceRecord(
        identifier="PSC-II-D19",
        title="Linear refinement defect",
        status=ClaimStatus.DEFINITION,
        source="Paper II",
        assumptions=("linear fine and coarse operators", "linear coarsening map"),
        verification_suite="refinement.defects",
    )
)
@dataclass(slots=True)
class LinearRefinementDefect:
    """The intertwining defect ``C T_f - T_c C``."""

    coarsen: Any
    fine_operator: Any
    coarse_operator: Any
    backend: str | LinearAlgebraBackend | None = None

    def __post_init__(self) -> None:
        self.backend = get_backend(self.backend)
        self.coarsen = self.backend.as_matrix(self.coarsen)
        self.fine_operator = self.backend.as_matrix(self.fine_operator)
        self.coarse_operator = self.backend.as_matrix(self.coarse_operator)
        coarse_dimension, fine_dimension = self.backend.shape(self.coarsen)
        if self.backend.shape(self.fine_operator) != (fine_dimension, fine_dimension):
            raise DimensionError("fine operator does not match the coarsening domain")
        if self.backend.shape(self.coarse_operator) != (coarse_dimension, coarse_dimension):
            raise DimensionError("coarse operator does not match the coarsening target")

    @property
    def matrix(self) -> Any:
        return self.backend.subtract(
            self.backend.matmul(self.coarsen, self.fine_operator),
            self.backend.matmul(self.coarse_operator, self.coarsen),
        )

    def is_exact(self, *, tolerance: float | None = None) -> bool:
        return self.backend.is_zero(self.matrix, tolerance=tolerance)

    @documented(
        ProvenanceRecord(
            identifier="PSC-II-T20",
            title="Power defect identity",
            status=ClaimStatus.PROVED,
            source="Paper II",
            assumptions=("positive integer power",),
            verification_suite="refinement.defects",
        )
    )
    def power(self, exponent: int) -> ComputationResult[Any]:
        if exponent < 1:
            raise ValueError("power defect exponent must be positive")
        coarse_dimension, fine_dimension = self.backend.shape(self.coarsen)
        total = self.backend.zeros(coarse_dimension, fine_dimension)
        delta = self.matrix
        for index in range(exponent):
            left = self._power(self.coarse_operator, exponent - 1 - index)
            right = self._power(self.fine_operator, index)
            total = self.backend.add(total, self.backend.matmul(left, self.backend.matmul(delta, right)))
        direct = self.backend.subtract(
            self.backend.matmul(self.coarsen, self._power(self.fine_operator, exponent)),
            self.backend.matmul(self._power(self.coarse_operator, exponent), self.coarsen),
        )
        tolerance = 1e-10
        audit = ResidualAudit(self.backend, tolerance)
        residual = audit.check("power_identity", self.backend.subtract(total, direct))
        valid = audit.valid
        return ComputationResult(
            total,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
                residuals={"power_identity": residual},
                tolerance=None if self.backend.exact else tolerance,
                assumptions_checked=("linear defect definition", "positive exponent"),
                assumptions_unchecked=audit.unresolved,
                diagnostics={"residual_decisions": audit.decisions},
                theorem_id="PSC-II-T20",
                message="power defect identity verified" if valid else "power defect identity failed",
            ),
        )

    def polynomial(self, coefficients: Iterable[Any]) -> ComputationResult[Any]:
        values = tuple(coefficients)
        coarse_dimension, fine_dimension = self.backend.shape(self.coarsen)
        total = self.backend.zeros(coarse_dimension, fine_dimension)
        for exponent, coefficient in enumerate(values):
            if exponent == 0 or coefficient == 0:
                continue
            total = self.backend.add(total, self.backend.scale(coefficient, self.power(exponent).require()))
        fine_value = self._polynomial(self.fine_operator, values)
        coarse_value = self._polynomial(self.coarse_operator, values)
        direct = self.backend.subtract(
            self.backend.matmul(self.coarsen, fine_value),
            self.backend.matmul(coarse_value, self.coarsen),
        )
        audit = ResidualAudit(self.backend, 1e-10)
        residual = audit.check("polynomial_identity", self.backend.subtract(total, direct))
        valid = audit.valid
        return ComputationResult(
            total,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
                residuals={"polynomial_identity": residual},
                tolerance=None if self.backend.exact else 1e-10,
                assumptions_checked=("finite polynomial",),
                assumptions_unchecked=audit.unresolved,
                diagnostics={"residual_decisions": audit.decisions},
                theorem_id="PSC-II-C21",
                message="polynomial defect identity verified" if valid else "polynomial defect identity failed",
            ),
        )

    @documented(
        ProvenanceRecord(
            identifier="PSC-II-T22",
            title="Resolvent defect identity",
            status=ClaimStatus.PROVED,
            source="Paper II",
            assumptions=("both resolvents exist",),
            verification_suite="refinement.defects",
        )
    )
    def resolvent(self, z: Any) -> ComputationResult[Any]:
        fine_identity = self.backend.eye(self.backend.shape(self.fine_operator)[0])
        coarse_identity = self.backend.eye(self.backend.shape(self.coarse_operator)[0])
        fine_resolvent = self.backend.inverse(self.backend.subtract(self.backend.scale(z, fine_identity), self.fine_operator))
        coarse_resolvent = self.backend.inverse(self.backend.subtract(self.backend.scale(z, coarse_identity), self.coarse_operator))
        formula = self.backend.matmul(coarse_resolvent, self.backend.matmul(self.matrix, fine_resolvent))
        direct = self.backend.subtract(
            self.backend.matmul(self.coarsen, fine_resolvent),
            self.backend.matmul(coarse_resolvent, self.coarsen),
        )
        audit = ResidualAudit(self.backend, 1e-9)
        residual = audit.check("resolvent_identity", self.backend.subtract(formula, direct))
        valid = audit.valid
        return ComputationResult(
            formula,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
                residuals={"resolvent_identity": residual},
                tolerance=None if self.backend.exact else 1e-9,
                assumptions_checked=("fine resolvent exists", "coarse resolvent exists"),
                assumptions_unchecked=audit.unresolved,
                diagnostics={"residual_decisions": audit.decisions},
                theorem_id="PSC-II-T22",
                message="resolvent defect identity verified" if valid else "resolvent defect identity failed",
            ),
        )

    @documented(
        ProvenanceRecord(
            identifier="PSC-II-T24",
            title="Exponential defect identity",
            status=ClaimStatus.PROVED,
            source="Paper II",
            assumptions=("finite matrices",),
            verification_suite="refinement.defects",
        )
    )
    def exponential(self, t: Any) -> ComputationResult[Any]:
        fine_exp = _matrix_exponential(self.backend.scale(t, self.fine_operator), self.backend)
        coarse_exp = _matrix_exponential(self.backend.scale(t, self.coarse_operator), self.backend)
        direct = self.backend.subtract(
            self.backend.matmul(self.coarsen, fine_exp),
            self.backend.matmul(coarse_exp, self.coarsen),
        )
        # The direct difference is the value of the Duhamel integral. We certify it by a block exponential identity.
        coarse_dimension = self.backend.shape(self.coarse_operator)[0]
        fine_dimension = self.backend.shape(self.fine_operator)[0]
        block_rows = coarse_dimension + fine_dimension
        if self.backend.name == "sympy":
            import sympy as sp

            block = sp.zeros(block_rows)
            block[:coarse_dimension, :coarse_dimension] = self.coarse_operator
            block[:coarse_dimension, coarse_dimension:] = self.matrix
            block[coarse_dimension:, coarse_dimension:] = self.fine_operator
            block_exp = _matrix_exponential(t * block, self.backend)
        else:
            import numpy as np

            block = np.zeros((block_rows, block_rows), dtype=complex)
            block[:coarse_dimension, :coarse_dimension] = self.coarse_operator
            block[:coarse_dimension, coarse_dimension:] = self.matrix
            block[coarse_dimension:, coarse_dimension:] = self.fine_operator
            block_exp = _matrix_exponential(t * block, self.backend)
        integral_value = block_exp[:coarse_dimension, coarse_dimension:]
        audit = ResidualAudit(self.backend, 1e-8)
        residual = audit.check("exponential_identity", self.backend.subtract(integral_value, direct))
        valid = audit.valid
        return ComputationResult(
            integral_value,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT if self.backend.exact else Exactness.NUMERICAL,
                residuals={"exponential_identity": residual},
                tolerance=None if self.backend.exact else 1e-8,
                assumptions_checked=("matrix exponentials exist", "block Duhamel identity"),
                assumptions_unchecked=audit.unresolved,
                diagnostics={"residual_decisions": audit.decisions},
                theorem_id="PSC-II-T24",
                message="exponential defect identity verified" if valid else "exponential defect identity failed",
            ),
        )

    def _power(self, matrix: Any, exponent: int) -> Any:
        size = self.backend.shape(matrix)[0]
        result = self.backend.eye(size)
        base = matrix
        value = exponent
        while value:
            if value & 1:
                result = self.backend.matmul(result, base)
            base = self.backend.matmul(base, base)
            value >>= 1
        return result

    def _polynomial(self, matrix: Any, coefficients: tuple[Any, ...]) -> Any:
        size = self.backend.shape(matrix)[0]
        result = self.backend.zeros(size, size)
        power = self.backend.eye(size)
        for coefficient in coefficients:
            result = self.backend.add(result, self.backend.scale(coefficient, power))
            power = self.backend.matmul(power, matrix)
        return result


@documented(
    ProvenanceRecord(
        identifier="PSC-II-T25",
        title="Multilevel defect decomposition",
        status=ClaimStatus.PROVED,
        source="Paper II",
        assumptions=("finite chain of compatible dimensions",),
        verification_suite="refinement.defects",
    )
)
def multilevel_defect(
    coarsenings: Iterable[Any],
    operators: Iterable[Any],
    *,
    backend: str | LinearAlgebraBackend | None = None,
) -> ComputationResult[Any]:
    """Return the telescoping total defect across several adjacent levels."""

    algebra = get_backend(backend)
    maps = tuple(algebra.as_matrix(value) for value in coarsenings)
    values = tuple(algebra.as_matrix(value) for value in operators)
    if len(values) != len(maps) + 1:
        raise DimensionError("a chain with L coarsening maps requires L + 1 operators")

    def compose(start: int, stop: int) -> Any:
        if start == stop:
            return algebra.eye(algebra.shape(values[start])[0])
        result = maps[stop - 1]
        for index in range(stop - 2, start - 1, -1):
            result = algebra.matmul(maps[index], result)
        return result

    total_map = compose(0, len(maps))
    direct = algebra.subtract(
        algebra.matmul(total_map, values[-1]),
        algebra.matmul(values[0], total_map),
    )
    total = algebra.zeros(*algebra.shape(direct))
    for level, coarsen in enumerate(maps):
        adjacent = LinearRefinementDefect(coarsen, values[level + 1], values[level], algebra).matrix
        left = compose(0, level)
        right = compose(level + 1, len(maps))
        total = algebra.add(total, algebra.matmul(left, algebra.matmul(adjacent, right)))
    audit = ResidualAudit(algebra, 1e-9)
    residual = audit.check("multilevel_identity", algebra.subtract(total, direct))
    valid = audit.valid
    return ComputationResult(
        total,
        Certificate(
            valid=valid,
            exactness=Exactness.EXACT if algebra.exact else Exactness.NUMERICAL,
            residuals={"multilevel_identity": residual},
            tolerance=None if algebra.exact else 1e-9,
            assumptions_checked=("adjacent dimensions", "finite refinement chain"),
            assumptions_unchecked=audit.unresolved,
            diagnostics={"residual_decisions": audit.decisions},
            theorem_id="PSC-II-T25",
            message="multilevel defect identity verified" if valid else "multilevel defect identity failed",
        ),
    )


__all__ = ["LinearRefinementDefect", "multilevel_defect"]
