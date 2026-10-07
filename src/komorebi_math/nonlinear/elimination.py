"""Finite nonlinear Schur elimination through second order."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .._residuals import finite_tolerance, value_audit
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .derivatives import DerivativeProvider


def _is_sympy(value: Any) -> bool:
    return value.__class__.__module__.startswith("sympy")


def _vector_length(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        normalized = tuple(int(entry) for entry in shape)
        if len(normalized) == 1:
            return normalized[0]
        if len(normalized) == 2 and 1 in normalized:
            return max(normalized)
    array = np.asarray(value, dtype=object)
    if array.ndim == 1:
        return int(array.shape[0])
    if array.ndim == 2 and 1 in array.shape:
        return int(max(array.shape))
    raise DimensionError("value must be vector-like")


def _as_vector(value: Any, dimension: int, *, name: str, exact: bool) -> Any:
    if exact:
        try:
            import sympy as sp
        except ImportError as exc:  # pragma: no cover
            from ..exceptions import BackendUnavailableError

            raise BackendUnavailableError("exact nonlinear elimination requires SymPy") from exc
        vector = value if isinstance(value, sp.MatrixBase) else sp.Matrix(value)
        if vector.rows == 1:
            vector = vector.T
        if vector.cols != 1 or vector.rows != dimension:
            raise DimensionError(f"{name} must have dimension {dimension}")
        return vector
    vector = np.asarray(value, dtype=complex)
    if vector.ndim == 2 and 1 in vector.shape:
        vector = vector.reshape(-1)
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise DimensionError(f"{name} must have dimension {dimension}")
    if not np.all(np.isfinite(vector)):
        raise ValidationError(f"{name} must be finite")
    return vector


def _join(left: Any, right: Any, *, exact: bool) -> Any:
    if exact:
        import sympy as sp

        return sp.Matrix.vstack(sp.Matrix(left), sp.Matrix(right))
    return np.concatenate((np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)))


def _basis(dimension: int, index: int, *, exact: bool) -> Any:
    if exact:
        import sympy as sp

        value = sp.zeros(dimension, 1)
        value[index] = 1
        return value
    value = np.zeros(dimension, dtype=complex)
    value[index] = 1.0
    return value


def _column_stack(columns: Sequence[Any], rows: int, *, exact: bool) -> Any:
    if exact:
        import sympy as sp

        return sp.Matrix.hstack(*(sp.Matrix(column) for column in columns)) if columns else sp.zeros(rows, 0)
    return np.column_stack(columns) if columns else np.zeros((rows, 0), dtype=complex)


def _matmul(left: Any, right: Any, *, exact: bool) -> Any:
    return left * right if exact else np.asarray(left) @ np.asarray(right)


def _add(left: Any, right: Any, *, exact: bool) -> Any:
    return left + right if exact else np.asarray(left) + np.asarray(right)


def _subtract(left: Any, right: Any, *, exact: bool) -> Any:
    return left - right if exact else np.asarray(left) - np.asarray(right)


def _scale(scalar: Any, value: Any, *, exact: bool) -> Any:
    return scalar * value if exact else scalar * np.asarray(value)


def _inverse(matrix: Any, *, exact: bool) -> Any:
    if exact:
        return matrix.inv()
    try:
        return np.linalg.inv(np.asarray(matrix, dtype=complex))
    except np.linalg.LinAlgError as exc:
        raise ValidationError("hidden derivative is singular") from exc


def _norm(value: Any) -> float:
    if _is_sympy(value):
        if not value:
            return 0.0
        result = float(value.norm())
        if result == 0.0 and any(entry != 0 for entry in value):
            # A nonzero exact value can lie below binary64 range.  The number is
            # only a display diagnostic; exact decisions use value_audit below.
            return float(np.nextafter(0.0, np.inf))
        return result
    array = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(array)):
        raise ValidationError("numerical residual must be finite")
    magnitudes = np.hypot(array.real, array.imag)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = float(np.hypot.reduce(magnitudes.ravel(), initial=0.0))
    if not isfinite(result):
        raise ValidationError("numerical residual norm is not representable")
    return result


def _zero_matrix(rows: int, columns: int, *, exact: bool) -> Any:
    if exact:
        import sympy as sp

        return sp.zeros(rows, columns)
    return np.zeros((rows, columns), dtype=complex)


def _identity(size: int, *, exact: bool) -> Any:
    if exact:
        import sympy as sp

        return sp.eye(size)
    return np.eye(size, dtype=complex)


@dataclass(frozen=True, slots=True)
class SchurDerivativeResult:
    """First-order implicit and reduced derivatives for nonlinear Schur elimination."""


    hidden_jacobian: Any
    hidden_inverse: Any
    implicit_derivative: Any
    reduced_derivative: Any
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class SchurSecondDerivativeResult:
    """Second-order implicit and reduced derivatives for nonlinear Schur elimination."""


    lifted_left: Any
    lifted_right: Any
    implicit_second_derivative: Any
    reduced_second_derivative: Any
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class HiddenSolveResult:
    """Finite hidden-variable solve with residual and iteration certificate."""


    hidden: np.ndarray
    point: np.ndarray
    residual: float
    iterations: int
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-III-T41",
        title="Finite nonlinear Schur elimination",
        status=ClaimStatus.PROVED,
        source="Paper III",
        assumptions=("C2 split equation", "invertible hidden derivative", "local implicit-function branch"),
        verification_suite="nonlinear.elimination",
    )
)
class NonlinearSchurElimination:
    """First and second derivatives after eliminating hidden variables.

    ``retained_provider`` represents ``F_P`` and ``hidden_provider`` represents
    ``F_Q`` on the same concatenated variable ``(u,v)``.  The hidden equation
    dimension must equal the hidden-variable dimension.
    """

    def __init__(
        self,
        retained_provider: DerivativeProvider,
        hidden_provider: DerivativeProvider,
        retained_dimension: int,
        hidden_dimension: int,
        base_retained: Any,
        base_hidden: Any,
        *,
        tolerance: float = 1e-10,
    ) -> None:
        retained = int(retained_dimension)
        hidden = int(hidden_dimension)
        if isinstance(retained_dimension, bool) or retained != retained_dimension or retained < 1:
            raise DimensionError("retained dimension must be a positive integer")
        if isinstance(hidden_dimension, bool) or hidden != hidden_dimension or hidden < 1:
            raise DimensionError("hidden dimension must be a positive integer")
        ambient = retained + hidden
        if retained_provider.source_dimension != ambient or hidden_provider.source_dimension != ambient:
            raise DimensionError("both equation providers must use the concatenated retained-hidden domain")
        if hidden_provider.target_dimension != hidden:
            raise DimensionError("hidden equation dimension must equal the hidden-variable dimension")
        if not retained_provider.supports_order(1) or not hidden_provider.supports_order(1):
            raise ValidationError("Schur elimination requires first derivatives")
        self.retained_provider = retained_provider
        self.hidden_provider = hidden_provider
        self.retained_dimension = retained
        self.hidden_dimension = hidden
        self.ambient_dimension = ambient
        self.retained_equation_dimension = retained_provider.target_dimension
        self.exact = (
            retained_provider.exactness is Exactness.EXACT
            and hidden_provider.exactness is Exactness.EXACT
        )
        self.exactness = Exactness.EXACT if self.exact else Exactness.NUMERICAL
        self.base_retained = _as_vector(base_retained, retained, name="base retained value", exact=self.exact)
        self.base_hidden = _as_vector(base_hidden, hidden, name="base hidden value", exact=self.exact)
        self.base_point = _join(self.base_retained, self.base_hidden, exact=self.exact)
        self.tolerance = float(tolerance)
        if not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValidationError("tolerance must be finite and positive")
        self._check_tolerance = finite_tolerance(self.tolerance * 10.0)
        hidden_value = _as_vector(
            hidden_provider.value(self.base_point),
            hidden,
            name="hidden equation at the base point",
            exact=self.exact,
        )
        base_audit = value_audit(
            exact=self.exact, tolerance=0.0 if self.exact else self._check_tolerance)
        self.base_hidden_residual = base_audit.check_value("base_hidden_equation", hidden_value)
        self.base_hidden_decision = base_audit.decisions["base_hidden_equation"]
        if not base_audit.valid:
            raise ValidationError("base point does not satisfy the hidden equation at the declared decision level")

    def _partial_jacobian(self, provider: DerivativeProvider, *, hidden_inputs: bool) -> Any:
        count = self.hidden_dimension if hidden_inputs else self.retained_dimension
        offset = self.retained_dimension if hidden_inputs else 0
        columns: list[Any] = []
        for local_index in range(count):
            direction = _basis(self.ambient_dimension, offset + local_index, exact=self.exact)
            column = _as_vector(
                provider.derivative(1, self.base_point, (direction,)),
                provider.target_dimension,
                name="partial derivative",
                exact=self.exact,
            )
            columns.append(column)
        return _column_stack(columns, provider.target_dimension, exact=self.exact)

    def first_derivatives(self) -> SchurDerivativeResult:
        du_p = self._partial_jacobian(self.retained_provider, hidden_inputs=False)
        dv_p = self._partial_jacobian(self.retained_provider, hidden_inputs=True)
        du_q = self._partial_jacobian(self.hidden_provider, hidden_inputs=False)
        dv_q = self._partial_jacobian(self.hidden_provider, hidden_inputs=True)
        inverse = _inverse(dv_q, exact=self.exact)
        implicit = _scale(-1, _matmul(inverse, du_q, exact=self.exact), exact=self.exact)
        feedback = _matmul(dv_p, _matmul(inverse, du_q, exact=self.exact), exact=self.exact)
        reduced = _subtract(du_p, feedback, exact=self.exact)
        inverse_difference = _subtract(
            _matmul(dv_q, inverse, exact=self.exact),
            _identity(self.hidden_dimension, exact=self.exact),
            exact=self.exact,
        )
        inverse_audit = value_audit(
            exact=self.exact, tolerance=0.0 if self.exact else self._check_tolerance)
        inverse_residual = inverse_audit.check_value("hidden_inverse", inverse_difference)
        valid = inverse_audit.valid
        return SchurDerivativeResult(
            dv_q,
            inverse,
            implicit,
            reduced,
            Certificate(
                valid=valid,
                exactness=self.exactness,
                residuals={"hidden_inverse": inverse_residual},
                tolerance=None if self.exact else self._check_tolerance,
                assumptions_checked=("hidden derivative inversion at the supplied base data", "first-order Schur algebra"),
                assumptions_unchecked=(
                    (() if self.exact else ("the numerical base point is an exact hidden-equation zero",))
                    + ("existence of a local implicit-function neighborhood",)
                ),
                diagnostics={
                    "retained_dimension": self.retained_dimension,
                    "hidden_dimension": self.hidden_dimension,
                    "retained_equation_dimension": self.retained_equation_dimension,
                    "base_hidden_residual": self.base_hidden_residual,
                    "base_hidden_decision": self.base_hidden_decision,
                    "inverse_residual_decisions": inverse_audit.decisions,
                },
                theorem_id="PSC-III-T41",
                message="Schur derivative data assembled; local branch existence remains external" if valid else "hidden derivative inversion failed",
            ),
        )

    def second_derivative(self, left: Any, right: Any) -> SchurSecondDerivativeResult:
        if not self.retained_provider.supports_order(2) or not self.hidden_provider.supports_order(2):
            raise ValidationError("second Schur derivative requires second derivatives of both equation blocks")
        h = _as_vector(left, self.retained_dimension, name="left retained direction", exact=self.exact)
        k = _as_vector(right, self.retained_dimension, name="right retained direction", exact=self.exact)
        first = self.first_derivatives()
        first.certificate.require()
        dphi_h = _matmul(first.implicit_derivative, h, exact=self.exact)
        dphi_k = _matmul(first.implicit_derivative, k, exact=self.exact)
        lifted_h = _join(h, dphi_h, exact=self.exact)
        lifted_k = _join(k, dphi_k, exact=self.exact)
        q_second = _as_vector(
            self.hidden_provider.derivative(2, self.base_point, (lifted_h, lifted_k)),
            self.hidden_dimension,
            name="hidden second derivative",
            exact=self.exact,
        )
        p_second = _as_vector(
            self.retained_provider.derivative(2, self.base_point, (lifted_h, lifted_k)),
            self.retained_equation_dimension,
            name="retained second derivative",
            exact=self.exact,
        )
        implicit_second = _scale(
            -1,
            _matmul(first.hidden_inverse, q_second, exact=self.exact),
            exact=self.exact,
        )
        dv_p = self._partial_jacobian(self.retained_provider, hidden_inputs=True)
        reduced_second = _add(
            p_second,
            _matmul(dv_p, implicit_second, exact=self.exact),
            exact=self.exact,
        )
        # Equivalent displayed form: D2 F_P - D_v F_P A^{-1} D2 F_Q.
        displayed = _subtract(
            p_second,
            _matmul(
                dv_p,
                _matmul(first.hidden_inverse, q_second, exact=self.exact),
                exact=self.exact,
            ),
            exact=self.exact,
        )
        second_difference = _subtract(reduced_second, displayed, exact=self.exact)
        second_audit = value_audit(
            exact=self.exact, tolerance=0.0 if self.exact else self._check_tolerance)
        residual = second_audit.check_value("second_schur_formula", second_difference)
        valid = second_audit.valid
        return SchurSecondDerivativeResult(
            lifted_h,
            lifted_k,
            implicit_second,
            reduced_second,
            Certificate(
                valid=valid,
                exactness=self.exactness,
                residuals={"second_schur_formula": residual},
                tolerance=None if self.exact else self._check_tolerance,
                assumptions_checked=("lifted retained directions", "second-order hidden feedback"),
                assumptions_unchecked=(
                    (() if self.exact else ("the numerical base point is an exact hidden-equation zero",))
                    + ("existence and C2 regularity of the local implicit branch",)
                ),
                theorem_id="PSC-III-T42",
                message="second nonlinear Schur derivative assembled" if valid else "second Schur formulas disagree",
            ),
        )

    def solve_hidden(
        self,
        retained_value: Any,
        *,
        initial_hidden: Any | None = None,
        max_iterations: int = 50,
    ) -> HiddenSolveResult:
        """Numerically solve the hidden equation on the local branch."""

        if self.exact:
            raise ValidationError("automatic hidden-variable solving is numerical; use the exact derivative formulas or supply a symbolic branch externally")
        u = _as_vector(retained_value, self.retained_dimension, name="retained value", exact=False)
        v = (
            np.asarray(self.base_hidden, dtype=complex).reshape(-1).copy()
            if initial_hidden is None
            else _as_vector(initial_hidden, self.hidden_dimension, name="initial hidden value", exact=False)
        )
        limit = int(max_iterations)
        if isinstance(max_iterations, bool) or limit != max_iterations or limit < 1:
            raise ValidationError("max_iterations must be a positive integer")
        residual = float("inf")
        iterations = 0
        for iterations in range(1, limit + 1):
            point = _join(u, v, exact=False)
            value = _as_vector(
                self.hidden_provider.value(point),
                self.hidden_dimension,
                name="hidden equation",
                exact=False,
            )
            residual = _norm(value)
            if residual <= self.tolerance:
                break
            columns = []
            for local_index in range(self.hidden_dimension):
                direction = _basis(self.ambient_dimension, self.retained_dimension + local_index, exact=False)
                columns.append(
                    _as_vector(
                        self.hidden_provider.derivative(1, point, (direction,)),
                        self.hidden_dimension,
                        name="hidden Jacobian column",
                        exact=False,
                    )
                )
            jacobian = np.column_stack(columns)
            try:
                step = np.linalg.solve(jacobian, -value)
            except np.linalg.LinAlgError as exc:
                raise ValidationError("hidden Jacobian became singular during Newton iteration") from exc
            accepted = False
            damping = 1.0
            for _ in range(24):
                candidate = v + damping * step
                candidate_point = _join(u, candidate, exact=False)
                candidate_value = _as_vector(
                    self.hidden_provider.value(candidate_point),
                    self.hidden_dimension,
                    name="hidden equation",
                    exact=False,
                )
                candidate_residual = _norm(candidate_value)
                if candidate_residual < residual or candidate_residual <= self.tolerance:
                    v = candidate
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                raise ValidationError("hidden Newton step failed to decrease the residual")
        point = _join(u, v, exact=False)
        residual = _norm(
            _as_vector(
                self.hidden_provider.value(point),
                self.hidden_dimension,
                name="hidden equation",
                exact=False,
            )
        )
        valid = residual <= self._check_tolerance
        return HiddenSolveResult(
            np.real_if_close(v, tol=1000),
            np.real_if_close(point, tol=1000),
            residual,
            iterations,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"hidden_equation": residual},
                tolerance=self._check_tolerance,
                assumptions_checked=("damped Newton residual is within the declared numerical tolerance",),
                assumptions_unchecked=(
                    "the returned point is an exact hidden-equation zero",
                    "the computed branch remains in the local implicit-function neighborhood",
                    "no other remote hidden solution is represented",
                ),
                diagnostics={"iterations": iterations, "max_iterations": limit},
                theorem_id="PSC-III-T40",
                message="approximate hidden-equation residual accepted" if valid else "hidden equation residual exceeds tolerance",
            ),
        )

    def reduced_value(
        self,
        retained_value: Any,
        *,
        initial_hidden: Any | None = None,
        max_iterations: int = 50,
    ) -> ComputationResult[np.ndarray]:
        solve = self.solve_hidden(
            retained_value,
            initial_hidden=initial_hidden,
            max_iterations=max_iterations,
        )
        solve.certificate.require()
        value = np.asarray(self.retained_provider.value(solve.point), dtype=complex).reshape(-1)
        return ComputationResult(
            np.real_if_close(value, tol=1000),
            Certificate(
                valid=True,
                exactness=Exactness.NUMERICAL,
                residuals={"hidden_equation": solve.residual},
                tolerance=self._check_tolerance,
                assumptions_checked=("hidden residual accepted numerically", "retained equation evaluated at the returned point"),
                assumptions_unchecked=solve.certificate.assumptions_unchecked,
                diagnostics={"hidden_iterations": solve.iterations},
                theorem_id="PSC-III-T40",
                message="retained equation evaluated at an approximate hidden solve; branch assumptions remain",
            ),
        )


__all__ = [
    "HiddenSolveResult",
    "NonlinearSchurElimination",
    "SchurDerivativeResult",
    "SchurSecondDerivativeResult",
]
