"""Smith coordinates and congruence solving over prime power rings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import inf
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .._number_theory import require_integer, require_prime


def _sympy() -> Any:
    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError("Smith coordinate operations require the 'exact' SymPy extra") from exc
    return sp


def padic_valuation(value: int, prime: int) -> int | float:
    """Return the prime valuation of an integer, with infinity at zero."""

    prime = require_prime(prime)
    integer = int(value)
    if integer == 0:
        return inf
    integer = abs(integer)
    valuation = 0
    while integer % prime == 0:
        integer //= prime
        valuation += 1
    return valuation


def rational_padic_valuation(value: Any, prime: int) -> int | float:
    """Return the prime valuation of a rational number."""

    prime = require_prime(prime)
    sp = _sympy()
    rational = sp.Rational(value)
    if rational == 0:
        return inf
    return int(padic_valuation(int(rational.p), prime) - padic_valuation(int(rational.q), prime))


@dataclass(frozen=True, slots=True)
class SmithDecomposition:
    """An exact integer Smith decomposition ``left * matrix * right = diagonal``."""

    matrix: Any
    diagonal: Any
    left: Any
    right: Any
    invariant_factors: tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.invariant_factors)

    def certificate(self) -> Certificate:
        """Independently check integrality, unimodularity, and Smith form."""

        sp = _sympy()
        checks = {
            "matrix_shapes": False,
            "integer_entries": False,
            "smith_identity": False,
            "unimodular_transforms": False,
            "diagonal_form": False,
            "diagonal_divisibility": False,
            "invariant_factors": False,
        }
        diagnostics: dict[str, Any] = {"checks": checks}
        try:
            matrix, diagonal, left, right = (
                sp.Matrix(value) for value in (self.matrix, self.diagonal, self.left, self.right)
            )
        except (TypeError, ValueError) as exc:
            diagnostics["input_error"] = f"{type(exc).__name__}: {exc}"
        else:
            rows, columns = matrix.shape
            checks["matrix_shapes"] = (
                diagonal.shape == matrix.shape
                and left.shape == (rows, rows)
                and right.shape == (columns, columns)
            )
            checks["integer_entries"] = all(
                getattr(entry, "is_Integer", None) is True
                for value in (matrix, diagonal, left, right)
                for entry in value
            )
            if checks["matrix_shapes"] and checks["integer_entries"]:
                checks["smith_identity"] = left * matrix * right == diagonal
                left_determinant, right_determinant = left.det(), right.det()
                diagnostics["left_determinant"] = str(left_determinant)
                diagnostics["right_determinant"] = str(right_determinant)
                checks["unimodular_transforms"] = (
                    left_determinant in (-1, 1) and right_determinant in (-1, 1)
                )
                checks["diagonal_form"] = all(
                    diagonal[row, column] == 0
                    for row in range(rows)
                    for column in range(columns)
                    if row != column
                )
                if checks["diagonal_form"]:
                    entries = tuple(abs(int(diagonal[index, index])) for index in range(min(rows, columns)))
                    factors = tuple(entry for entry in entries if entry != 0)
                    # Nonzero entries precede zeros, and each divides its successor.
                    checks["diagonal_divisibility"] = (
                        entries == factors + (0,) * (len(entries) - len(factors))
                        and all(b % a == 0 for a, b in zip(factors, factors[1:], strict=False))
                    )
                    try:
                        recorded_factors = tuple(
                            require_integer(value, name="invariant factor") for value in self.invariant_factors
                        )
                    except (TypeError, ValidationError):
                        recorded_factors = None
                    checks["invariant_factors"] = recorded_factors == factors
                    diagnostics["diagonal_invariant_factors"] = factors
        valid = all(checks.values())
        failed = tuple(name for name, passed in checks.items() if not passed)
        diagnostics["failed_obligations"] = failed
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            # These are exact predicate failure counts, not rounded norms.
            residuals={
                "smith_identity": float(not checks["smith_identity"]),
                "failed_obligations": float(len(failed)),
            },
            assumptions_checked=tuple(name for name, passed in checks.items() if passed),
            diagnostics=diagnostics,
            theorem_id="PSC-II-SNF",
            message="valid Smith decomposition" if valid else "Smith decomposition obligations failed: " + ", ".join(failed),
        )


@documented(
    ProvenanceRecord(
        identifier="PSC-II-SNF",
        title="Smith normal form over the integers",
        status=ClaimStatus.CLASSICAL,
        source="Paper II",
        assumptions=("integer matrix",),
        verification_suite="refinement.smith",
    )
)
def smith_decomposition(matrix: Any) -> SmithDecomposition:
    """Return the Smith diagonal and unimodular transformation matrices."""

    sp = _sympy()
    from sympy.polys.domains import ZZ
    from sympy.polys.matrices import DomainMatrix
    try:
        from sympy.polys.matrices.normalforms import smith_normal_decomp
    except ImportError as exc:
        raise BackendUnavailableError(
            "Smith decomposition requires SymPy >=1.14 with smith_normal_decomp; "
            "install or upgrade the 'exact' optional dependency"
        ) from exc

    value = sp.ImmutableMatrix(matrix)
    if any(not entry.is_Integer for entry in value):
        raise ValidationError("Smith decomposition requires an integer matrix")
    rows, columns = value.shape
    domain_matrix = DomainMatrix.from_Matrix(value).convert_to(ZZ)
    diagonal_dm, left_dm, right_dm = smith_normal_decomp(domain_matrix)
    diagonal = sp.ImmutableMatrix(diagonal_dm.to_Matrix())
    left = sp.ImmutableMatrix(left_dm.to_Matrix())
    right = sp.ImmutableMatrix(right_dm.to_Matrix())
    factors: list[int] = []
    for index in range(min(rows, columns)):
        entry = int(diagonal[index, index])
        if entry != 0:
            factors.append(abs(entry))
    result = SmithDecomposition(value, diagonal, left, right, tuple(factors))
    result.certificate().require()
    return result


@dataclass(frozen=True, slots=True)
class SmithCongruenceSolution:
    """Solvability and branch data for one prime power congruence."""

    solvable: bool
    prime: int
    depth: int
    modulus: int
    rank: int
    smith_factors: tuple[int, ...]
    smith_exponents: tuple[int, ...]
    transformed_rhs: tuple[int, ...]
    obstruction_rows: tuple[int, ...]
    solution_count: int
    particular_solution: tuple[int, ...] | None
    kernel_generators: tuple[tuple[int, ...], ...]
    certificate: Certificate

    def require(self) -> "SmithCongruenceSolution":
        self.certificate.require()
        if not self.solvable:
            raise ValidationError("the congruence has a nonzero Smith obstruction")
        return self


@documented(
    ProvenanceRecord(
        identifier="PSC-II-T18",
        title="Smith weighted congruence solve",
        status=ClaimStatus.PROVED,
        source="Paper II",
        assumptions=("integer matrix", "prime power modulus"),
        verification_suite="refinement.smith",
    )
)
class SmithCongruenceProblem:
    """Solve ``matrix * x = rhs`` modulo a prime power."""

    def __init__(self, matrix: Any, rhs: Any, *, prime: int, depth: int) -> None:
        prime = require_prime(prime)
        depth = require_integer(depth, name="depth")
        if depth <= 0:
            raise ValueError("depth must be positive")
        sp = _sympy()
        self.matrix = sp.ImmutableMatrix(matrix)
        self.rhs = sp.ImmutableMatrix(rhs)
        if self.rhs.cols != 1:
            if self.rhs.rows == 1:
                self.rhs = self.rhs.T
            else:
                raise DimensionError("right hand side must be one dimensional")
        if self.matrix.rows != self.rhs.rows:
            raise DimensionError("matrix row count and right hand side dimension do not agree")
        if any(not entry.is_Integer for entry in self.matrix) or any(not entry.is_Integer for entry in self.rhs):
            raise ValidationError("Smith congruence data must be integral")
        self.prime = prime
        self.depth = int(depth)
        self.modulus = self.prime**self.depth
        self.decomposition = smith_decomposition(self.matrix)

    def solve(self) -> SmithCongruenceSolution:
        sp = _sympy()
        decomposition = self.decomposition
        # A cached decomposition is a witness for one specific matrix, not a
        # permanent permission to trust public fields after their replacement.
        decomposition.certificate().require()
        if decomposition.matrix != self.matrix:
            raise ValidationError("cached Smith decomposition does not match the current matrix")
        prime = require_prime(self.prime)
        depth = require_integer(self.depth, name="depth")
        if depth <= 0 or self.modulus != prime ** depth:
            raise ValidationError("current Smith modulus and depth do not agree")
        if self.rhs.shape != (self.matrix.rows, 1):
            raise DimensionError("current right hand side does not match the matrix")
        if any(getattr(entry, "is_Integer", None) is not True for entry in self.rhs):
            raise ValidationError("current Smith right hand side must be integral")
        transformed_rhs_matrix = decomposition.left * self.rhs
        transformed_rhs = tuple(int(value) % self.modulus for value in transformed_rhs_matrix)
        rank = decomposition.rank
        rows, columns = self.matrix.shape
        exponents = tuple(
            min(int(padic_valuation(factor, self.prime)), self.depth)
            for factor in decomposition.invariant_factors
        )
        obstruction_rows: list[int] = []
        for index in range(rank):
            required = self.prime ** exponents[index]
            if transformed_rhs[index] % required != 0:
                obstruction_rows.append(index)
        for index in range(rank, rows):
            if transformed_rhs[index] % self.modulus != 0:
                obstruction_rows.append(index)
        solvable = not obstruction_rows

        if solvable:
            count = self.prime ** ((columns - rank) * self.depth + sum(exponents))
            y = [0] * columns
            generators_y: list[list[int]] = []
            for index in range(rank):
                # The Smith diagonal may carry a negative sign.  The reported
                # invariant factors are normalized to positive integers, but
                # solving the transformed diagonal equation must use the
                # signed diagonal entry appearing in ``left * A * right``.
                factor = int(decomposition.diagonal[index, index])
                exponent = exponents[index]
                divisor = self.prime**exponent
                reduced_modulus = self.prime ** (self.depth - exponent)
                if reduced_modulus == 1:
                    y[index] = 0
                else:
                    unit = (factor // divisor) % reduced_modulus
                    inverse = pow(unit, -1, reduced_modulus)
                    y[index] = ((transformed_rhs[index] // divisor) * inverse) % reduced_modulus
                if exponent > 0:
                    generator = [0] * columns
                    generator[index] = self.prime ** (self.depth - exponent)
                    generators_y.append(generator)
            for index in range(rank, columns):
                generator = [0] * columns
                generator[index] = 1
                generators_y.append(generator)
            y_matrix = sp.Matrix(columns, 1, y)
            particular_matrix = decomposition.right * y_matrix
            particular = tuple(int(value) % self.modulus for value in particular_matrix)
            kernel_generators = tuple(
                tuple(int(value) % self.modulus for value in decomposition.right * sp.Matrix(columns, 1, generator))
                for generator in generators_y
            )
            residual = self.matrix * sp.Matrix(columns, 1, particular) - self.rhs
            valid = (all(int(value) % self.modulus == 0 for value in residual)
                     and all(all(int(value) % self.modulus == 0
                                 for value in self.matrix * sp.Matrix(columns, 1, generator))
                             for generator in kernel_generators))
        else:
            count = 0
            particular = None
            kernel_generators = ()
            valid = True

        certificate = Certificate(
            valid=bool(valid),
            exactness=Exactness.EXACT,
            residuals={"modular_residual": 0.0 if valid else 1.0},
            assumptions_checked=("fresh Smith decomposition", "matrix witness identity", "prime power modulus", "coordinate divisibility tests"),
            diagnostics={
                "solvable": solvable,
                "rank": rank,
                "solution_count": count,
                "obstruction_rows": tuple(obstruction_rows),
            },
            theorem_id="PSC-II-T18",
            message=(
                "Smith congruence solved"
                if solvable and valid
                else "Smith obstruction detected"
                if not solvable
                else "computed particular solution failed verification"
            ),
        )
        return SmithCongruenceSolution(
            solvable=solvable,
            prime=self.prime,
            depth=self.depth,
            modulus=self.modulus,
            rank=rank,
            smith_factors=decomposition.invariant_factors,
            smith_exponents=exponents,
            transformed_rhs=transformed_rhs,
            obstruction_rows=tuple(obstruction_rows),
            solution_count=count,
            particular_solution=particular,
            kernel_generators=kernel_generators,
            certificate=certificate,
        )

    def enumerate_solutions(self, *, limit: int = 100000) -> tuple[tuple[int, ...], ...]:
        """Enumerate every solution when the exact branch count is below ``limit``."""

        result = self.solve()
        if not result.solvable or result.particular_solution is None:
            return ()
        if result.solution_count > limit:
            raise ValueError(f"solution count {result.solution_count} exceeds enumeration limit {limit}")
        modulus = result.modulus
        if not result.kernel_generators:
            return (result.particular_solution,)
        orders: list[int] = []
        for generator in result.kernel_generators:
            nonzero = [value % modulus for value in generator if value % modulus]
            if not nonzero:
                orders.append(1)
                continue
            order = 1
            while any((order * value) % modulus for value in nonzero):
                order += 1
                if order > modulus:
                    raise ArithmeticError("failed to determine a kernel generator order")
            orders.append(order)
        solutions: set[tuple[int, ...]] = set()
        for coefficients in product(*(range(order) for order in orders)):
            vector = list(result.particular_solution)
            for coefficient, generator in zip(coefficients, result.kernel_generators, strict=True):
                vector = [
                    (value + coefficient * direction) % modulus
                    for value, direction in zip(vector, generator, strict=True)
                ]
            solutions.add(tuple(vector))
        if len(solutions) != result.solution_count:
            # The minimal generator list can contain dependent directions after the right transform.
            # Fall back to direct enumeration for small ambient spaces.
            solutions.clear()
            for vector in product(range(modulus), repeat=self.matrix.cols):
                residual = self.matrix * _sympy().Matrix(vector) - self.rhs
                if all(int(value) % modulus == 0 for value in residual):
                    solutions.add(tuple(int(value) for value in vector))
        if len(solutions) != result.solution_count:
            raise ArithmeticError("enumerated solution count does not match the Smith formula")
        return tuple(sorted(solutions))


__all__ = [
    "SmithCongruenceProblem",
    "SmithCongruenceSolution",
    "SmithDecomposition",
    "padic_valuation",
    "rational_padic_valuation",
    "smith_decomposition",
]
