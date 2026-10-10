"""Exact finite Euclidean reconstruction and discrepancy certificates.

The source contracts are Sufficient Resolution (8 October 2026), SR-37--40,
SR-45--50, SR-52 and SR-57--60. Inputs are rational finite matrices with their
ordinary Euclidean metrics. A discrepancy root is enclosed by rational
parameters; an irrational real root is never returned as an exact rational.
Data-noise truth, infinite Hilbert limits and whole-operator measurements are
not inferred from a fit. SymPy is an optional exact-arithmetic dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ResourceLimitError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _sympy() -> Any:
    try:
        import sympy
    except ImportError as exc:
        raise BackendUnavailableError("exact Tikhonov machinery requires the 'exact' dependency") from exc
    return sympy


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValidationError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _rational(value: Any, name: str) -> Fraction:
    sp = _sympy()
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction, sp.Rational)):
        raise ValidationError(f"{name} must be an exact rational; floats, strings and symbols are rejected")
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        # Fraction(np.int64(...)) can retain a fixed-width NumPy numerator.
        # Own Python integers before any exact multiplication or squaring.
        return Fraction(int(value.numerator), int(value.denominator))
    return Fraction(int(value.p), int(value.q))


def _nonnegative(value: Any, name: str) -> Fraction:
    result = _rational(value, name)
    if result < 0:
        raise ValidationError(f"{name} must be nonnegative")
    return result


def _bits(value: Fraction) -> int:
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())


@dataclass(frozen=True, slots=True)
class TikhonovLimits:
    """Explicit allocation and conservative rational-arithmetic budgets."""

    max_dimension: int = 32
    max_entry_bits: int = 4096
    max_work_bits: int = 262_144

    def __post_init__(self) -> None:
        for name in ("max_dimension", "max_entry_bits", "max_work_bits"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


def _limits(value: TikhonovLimits | None) -> TikhonovLimits:
    if value is None:
        return TikhonovLimits()
    if not isinstance(value, TikhonovLimits):
        raise ValidationError("limits must be a TikhonovLimits record")
    return value


def _scalar(value: Any, name: str, limits: TikhonovLimits, *, positive: bool = False) -> Fraction:
    result = _rational(value, name)
    if positive and result <= 0:
        raise ValidationError(f"{name} must be strictly positive")
    if _bits(result) > limits.max_entry_bits:
        raise ResourceLimitError(f"{name} exceeds the rational entry-bit budget")
    return result


def _matrix(value: Any, name: str, limits: TikhonovLimits, *, column: bool = False) -> Any:
    sp = _sympy()
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) not in (1, 2) or any(not isinstance(s, Integral) or s <= 0 for s in shape):
            raise DimensionError(f"{name} must have positive dimensions")
        rows, cols = (int(shape[0]), 1) if len(shape) == 1 else tuple(map(int, shape))
        if max(rows, cols) > limits.max_dimension:
            raise ResourceLimitError(f"{name} exceeds the dimension budget before conversion")
        if column and cols != 1:
            raise DimensionError(f"{name} must be a column vector")
    raw = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(raw, (list, tuple)) or not raw:
        raise DimensionError(f"{name} must be a nonempty rational matrix")
    rows = len(raw)
    if rows > limits.max_dimension:
        raise ResourceLimitError(f"{name} exceeds the dimension budget")
    nested = isinstance(raw[0], (list, tuple))
    cols = len(raw[0]) if nested else 1
    if cols <= 0 or any(isinstance(row, (list, tuple)) != nested for row in raw):
        raise DimensionError(f"{name} must be a nonempty rectangular matrix")
    if nested and any(len(row) != cols for row in raw):
        raise DimensionError(f"{name} must be rectangular")
    if cols > limits.max_dimension:
        raise ResourceLimitError(f"{name} exceeds the dimension budget")
    if column and cols != 1:
        raise DimensionError(f"{name} must be a column vector")
    converted = [
        [_scalar(entry, name, limits) for entry in row] if nested else [_scalar(row, name, limits)]
        for row in raw
    ]
    return sp.ImmutableMatrix(converted)


def _work_guard(matrices: tuple[Any, ...], limits: TikhonovLimits, *, inverse_size: int = 1) -> None:
    # Clear-denominator determinant bounds also cover rational inverse entries.
    # This intentionally overestimates coefficient growth before elimination.
    dimension = max((max(matrix.shape) for matrix in matrices), default=1)
    input_bits = max((_bits(_rational(entry, "entry")) for matrix in matrices for entry in matrix), default=1)
    forecast = 8 * inverse_size * dimension * (input_bits + dimension.bit_length() + 2)
    if forecast > limits.max_work_bits:
        raise ResourceLimitError("conservative exact elimination forecast exceeds max_work_bits")


def _owned(matrix: Any, limits: TikhonovLimits) -> Any:
    if any(_bits(_rational(entry, "working entry")) > limits.max_work_bits for entry in matrix):
        raise ResourceLimitError("computed rational entry exceeds max_work_bits")
    return _sympy().ImmutableMatrix(matrix)


def _sq(vector: Any) -> Fraction:
    return _rational((vector.T * vector)[0, 0], "squared norm")


def _zero(matrix: Any) -> bool:
    return all(entry == 0 for entry in matrix)


def _max_abs(matrix: Any) -> Fraction:
    return max((_rational(abs(entry), "residual") for entry in matrix), default=Fraction(0))


def _certificate(
    theorem: str,
    checks: dict[str, bool],
    *,
    residuals: dict[str, Fraction] | None = None,
    diagnostics: dict[str, Any] | None = None,
    unchecked: tuple[str, ...] = (),
    message: str,
) -> Certificate:
    return Certificate(
        valid=all(checks.values()),
        exactness=Exactness.EXACT,
        residuals=residuals or {},
        assumptions_checked=("finite rational matrices with Euclidean metrics", *checks),
        assumptions_unchecked=unchecked,
        diagnostics={"checks": dict(checks), **(diagnostics or {})},
        theorem_id=theorem,
        message=message,
    )


def _projector(vectors: list[Any], dimension: int, limits: TikhonovLimits) -> Any:
    sp = _sympy()
    if not vectors:
        return sp.ImmutableMatrix(sp.zeros(dimension))
    basis = sp.Matrix.hstack(*vectors)
    gram = basis.T * basis
    _work_guard((basis, gram), limits, inverse_size=gram.rows)
    return _owned(basis * gram.inv() * basis.T, limits)


def _positive_semidefinite(matrix: Any, limits: TikhonovLimits) -> bool:
    """Exact rational Schur-complement test, including zero diagonal pivots."""
    if matrix.rows != matrix.cols or matrix != matrix.T:
        return False
    _work_guard((matrix,), limits, inverse_size=max(matrix.rows, 1))
    current = _sympy().Matrix(matrix)
    while current.rows:
        if any(current[i, i] < 0 for i in range(current.rows)):
            return False
        # For a PSD matrix a zero diagonal implies the entire row is zero.
        # Rejecting a nonzero zero-diagonal row prevents unsafe pivot dropping.
        for i in range(current.rows):
            if current[i, i] == 0 and any(current[i, j] != 0 for j in range(current.rows)):
                return False
        pivot = next((i for i in range(current.rows) if current[i, i] > 0), None)
        if pivot is None:
            return _zero(current)
        if pivot:
            current.row_swap(0, pivot)
            current.col_swap(0, pivot)
        current = current[1:, 1:] - current[1:, 0:1] * current[0:1, 1:] / current[0, 0]
        current = _sympy().Matrix(_owned(current, limits))
    return True


@dataclass(frozen=True, slots=True)
class TikhonovSolution:
    source: Any
    fitted: Any
    fitting_residual: Any
    normal_residual: Any
    source_squared: Fraction
    fitting_squared: Fraction
    objective: Fraction
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class ObjectiveIncrement:
    solution: TikhonovSolution
    candidate_objective: Fraction
    image_increment_squared: Fraction
    source_increment_squared: Fraction
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class BiasNoiseDecomposition:
    identifiable_source: Any
    kernel_source: Any
    bias: Any
    propagated_noise: Any
    error: Any
    identifiable_error_squared: Fraction
    kernel_squared: Fraction
    total_error_squared: Fraction
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class ResolventDifference:
    difference: Any
    factorization: Any
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class TikhonovSystem:
    """Constructed positive normal inverse in finite rational Euclidean spaces.

    Use :func:`tikhonov_system`; retained matrices are immutable. Directly
    instantiating this dataclass also checks its defining matrix identities;
    an unrelated true certificate cannot validate a forged normal operator.
    """

    operator: Any
    parameter: Fraction
    normal: Any
    inverse: Any
    regularized: Any
    kernel_projector: Any
    range_projector: Any
    limits: TikhonovLimits
    certificate: Certificate

    def __post_init__(self) -> None:
        sp = _sympy()
        if not isinstance(self.limits, TikhonovLimits):
            raise ValidationError("limits must be a TikhonovLimits record")
        object.__setattr__(self, "parameter", _scalar(self.parameter, "parameter", self.limits, positive=True))
        names = ("operator", "normal", "inverse", "regularized", "kernel_projector", "range_projector")
        for name in names:
            matrix = getattr(self, name)
            if not isinstance(matrix, sp.ImmutableMatrix):
                raise ValidationError(f"{name} must be an owned immutable rational matrix")
            if min(matrix.shape) <= 0 or max(matrix.shape) > self.limits.max_dimension:
                raise DimensionError(f"{name} dimensions violate the model contract")
            for entry in matrix:
                if _bits(_rational(entry, name)) > self.limits.max_work_bits:
                    raise ResourceLimitError(f"{name} violates max_work_bits")
        rows, cols = self.operator.shape
        expected_shapes = (("normal", (cols, cols)), ("inverse", (cols, cols)),
                           ("regularized", (cols, rows)), ("kernel_projector", (cols, cols)),
                           ("range_projector", (rows, rows)))
        if any(getattr(self, name).shape != shape for name, shape in expected_shapes):
            raise DimensionError("Tikhonov system matrix dimensions do not match its operator")
        _work_guard(tuple(getattr(self, name) for name in names), self.limits)
        identity = sp.eye(cols)
        rank = self.operator.rank()
        kernel, image = self.kernel_projector, self.range_projector
        valid = (
            self.normal == self.operator.T*self.operator + self.parameter*identity
            and self.normal*self.inverse == identity and self.inverse*self.normal == identity
            and self.regularized == self.inverse*self.operator.T
            and kernel == kernel.T and kernel*kernel == kernel and _zero(self.operator*kernel)
            and sp.trace(kernel) == cols-rank
            and image == image.T and image*image == image and image*self.operator == self.operator
            and sp.trace(image) == rank
        )
        if not valid:
            raise ValidationError("Tikhonov system failed its defining actual matrix identities")
        self.certificate.require()

    def whole_map_bounds(self) -> Certificate:
        """Check the four SR-38 whole-map bounds by exact PSD inequalities."""
        self.certificate.require()
        sp = _sympy()
        _work_guard((self.inverse, self.regularized, self.operator), self.limits,
                    inverse_size=self.operator.cols)
        response_map = self.operator * self.regularized
        residual_map = response_map - sp.eye(self.operator.rows)
        margins = {
            "normal inverse norm bound": sp.eye(self.operator.cols) / self.parameter**2 - self.inverse.T * self.inverse,
            "regularized map norm bound": sp.eye(self.operator.cols) / (4*self.parameter) - self.regularized * self.regularized.T,
            "whole fitted map contraction": sp.eye(self.operator.rows) - response_map.T * response_map,
            "whole fitting residual contraction": sp.eye(self.operator.rows) - residual_map.T * residual_map,
        }
        checks = {name: _positive_semidefinite(matrix, self.limits) for name, matrix in margins.items()}
        return _certificate("SR-38-finite", checks,
                            diagnostics={"resolvent_norm_squared_upper": 1/self.parameter**2,
                                         "regularized_norm_squared_upper": 1/(4*self.parameter)},
                            message="Actual finite whole-map squared norm bounds verified by rational PSD tests.")

    def solve(self, data: Any) -> TikhonovSolution:
        self.certificate.require()
        v = _matrix(data, "data", self.limits, column=True)
        if v.rows != self.operator.rows:
            raise DimensionError("data dimension does not match operator response")
        _work_guard((self.regularized, v), self.limits)
        source = _owned(self.regularized * v, self.limits)
        fitted = _owned(self.operator * source, self.limits)
        residual = _owned(fitted - v, self.limits)
        normal_residual = _owned(self.normal * source - self.operator.T * v, self.limits)
        source_sq, fitting_sq = _sq(source), _sq(residual)
        energy_error = _sq(fitted) + self.parameter * source_sq - _rational((v.T * fitted)[0], "energy")
        checks = {
            "actual normal equation": _zero(normal_residual),
            "energy identity": energy_error == 0,
            "source orthogonal to kernel": _zero(self.kernel_projector * source),
            "source squared bound": 4 * self.parameter * source_sq <= _sq(v),
            "fitted contraction": _sq(fitted) <= _sq(v),
            "fitting contraction": fitting_sq <= _sq(v),
        }
        certificate = _certificate(
            "SR-37/SR-38-finite", checks,
            residuals={"normal_equation": _max_abs(normal_residual), "energy": abs(energy_error)},
            message="Exact positive normal solve and finite energy identities recomputed.",
        )
        return TikhonovSolution(source, fitted, residual, normal_residual, source_sq, fitting_sq,
                                fitting_sq + self.parameter * source_sq, certificate)

    def objective_increment(self, data: Any, candidate: Any) -> ObjectiveIncrement:
        solution = self.solve(data)
        v = _matrix(data, "data", self.limits, column=True)
        x = _matrix(candidate, "candidate", self.limits, column=True)
        if x.rows != self.operator.cols:
            raise DimensionError("candidate dimension does not match source")
        _work_guard((self.operator, x, v), self.limits)
        difference = x - solution.source
        image_sq, source_sq = _sq(self.operator * difference), _sq(difference)
        candidate_objective = _sq(self.operator * x - v) + self.parameter * _sq(x)
        residual = candidate_objective - solution.objective - image_sq - self.parameter * source_sq
        certificate = _certificate(
            "SR-38-finite", {"objective increment identity": residual == 0,
                             "nonnegative objective increment": candidate_objective >= solution.objective},
            residuals={"objective_increment": abs(residual)},
            message="Candidate objective compared with its actual unique regularized minimizer.",
        )
        return ObjectiveIncrement(solution, candidate_objective, image_sq, source_sq, certificate)

    def bias_noise(self, source: Any, noise: Any) -> BiasNoiseDecomposition:
        self.certificate.require()
        g = _matrix(source, "source", self.limits, column=True)
        e = _matrix(noise, "noise", self.limits, column=True)
        if g.rows != self.operator.cols or e.rows != self.operator.rows:
            raise DimensionError("source/noise dimensions do not match operator")
        _work_guard((self.operator, self.inverse, g, e), self.limits)
        k = _owned(self.kernel_projector * g, self.limits)
        p = _owned(g - k, self.limits)
        bias = _owned(-self.parameter * self.inverse * g, self.limits)
        propagated = _owned(self.regularized * e, self.limits)
        error = _owned(self.regularized * (self.operator * g + e) - g, self.limits)
        identifiable_error = -self.parameter * self.inverse * p + propagated
        error_sq, identifiable_sq, kernel_sq = _sq(error), _sq(identifiable_error), _sq(k)
        filter_residual = self.regularized * self.operator * g + self.parameter * self.inverse * g - g
        checks = {
            "filter decomposition": _zero(filter_residual),
            "noise decomposition": error == bias + propagated,
            "kernel orthogonality": (identifiable_error.T * k)[0] == 0,
            "squared error decomposition": error_sq == identifiable_sq + kernel_sq,
        }
        return BiasNoiseDecomposition(
            p, k, bias, propagated, error, identifiable_sq, kernel_sq, error_sq,
            _certificate("SR-39-finite", checks, residuals={"filter": _max_abs(filter_residual)},
                         message="Bias, propagated noise and irreducible kernel content recomputed exactly."),
        )

    def resolvent_difference(self, other: TikhonovSystem) -> ResolventDifference:
        self.certificate.require()
        other.certificate.require()
        if self.operator.shape != other.operator.shape:
            raise DimensionError("resolvent systems must have the same operator shape")
        _work_guard((self.inverse, other.inverse, self.normal, other.normal), self.limits)
        difference = _owned(self.inverse - other.inverse, self.limits)
        factorization = _owned(self.inverse * (other.normal - self.normal) * other.inverse, self.limits)
        residual = difference - factorization
        return ResolventDifference(
            difference, factorization,
            _certificate("SR-40-finite", {"resolvent difference identity": _zero(residual)},
                         residuals={"identity": _max_abs(residual)},
                         message="Exact two-operator/two-parameter resolvent identity; no uniform vanishing-parameter bound."),
        )

    def operator_sensitivity(self, other: TikhonovSystem, *, operator_bound: Any,
                             other_operator_bound: Any, difference_bound: Any) -> Certificate:
        """Verify supplied whole-map bounds before using the SR-40 allowance."""
        self.certificate.require()
        other.certificate.require()
        if self.operator.shape != other.operator.shape or self.parameter != other.parameter:
            raise ValidationError("fixed-parameter sensitivity needs matching shapes and parameters")
        a, c, epsilon = (_nonnegative(value, name) for name, value in (
            ("operator_bound", operator_bound), ("other_operator_bound", other_operator_bound),
            ("difference_bound", difference_bound),
        ))
        for name, value in (("operator_bound", a), ("other_operator_bound", c), ("difference_bound", epsilon)):
            _scalar(value, name, self.limits)
        sp = _sympy()
        _work_guard((self.operator, other.operator, self.regularized, other.regularized,
                     sp.ImmutableMatrix([[a, c, epsilon]])), self.limits,
                    inverse_size=self.operator.cols)
        identity = sp.eye(self.operator.cols)
        difference = self.operator - other.operator
        allowance = epsilon/self.parameter + (a+c)*epsilon*c/self.parameter**2
        regularized_difference = self.regularized - other.regularized
        checks = {
            "verified operator norm enclosure": _positive_semidefinite(a*a*identity - self.operator.T*self.operator, self.limits),
            "verified other operator norm enclosure": _positive_semidefinite(c*c*identity - other.operator.T*other.operator, self.limits),
            "verified whole-operator difference enclosure": _positive_semidefinite(epsilon**2*identity - difference.T*difference, self.limits),
            "actual regularized difference bound": _positive_semidefinite(
                allowance**2*sp.eye(self.operator.rows) - regularized_difference.T*regularized_difference, self.limits),
        }
        return _certificate("SR-40-finite", checks,
                            diagnostics={"regularized_difference_norm_upper": allowance},
                            message="Supplied finite whole-operator enclosures and the resulting fixed-parameter sensitivity verified.")

    def source_condition_bias(self, source: Any, witness: Any) -> Certificate:
        """Check g=A*Aw and the resulting SR-39 normal-range bias bound."""
        self.certificate.require()
        g = _matrix(source, "source", self.limits, column=True)
        w = _matrix(witness, "source-condition witness", self.limits, column=True)
        if g.rows != self.operator.cols or w.rows != self.operator.cols:
            raise DimensionError("source-condition vectors must match source dimension")
        _work_guard((self.operator, self.inverse, g, w), self.limits)
        bias = self.regularized*self.operator*g - g
        checks = {"actual normal-range source condition": g == self.operator.T*self.operator*w,
                  "source-condition squared bias bound": _sq(bias) <= self.parameter**2*_sq(w)}
        return _certificate("SR-39-finite", checks,
                            diagnostics={"actual_bias_squared": _sq(bias),
                                         "bias_squared_upper": self.parameter**2*_sq(w)},
                            message="A supplied normal-range source witness and its exact squared bias bound checked.")

    def conditioned_bias_noise(self, source: Any, noise: Any, alpha_squared: Any) -> Certificate:
        """Check restricted source coercivity and the SR-39 conditioned error."""
        result = self.bias_noise(source, noise)
        alpha_sq = _scalar(alpha_squared, "alpha_squared", self.limits, positive=True)
        noise_vector = _matrix(noise, "noise", self.limits, column=True)
        sp = _sympy()
        _work_guard((self.operator, self.kernel_projector, sp.ImmutableMatrix([[alpha_sq]])),
                    self.limits, inverse_size=self.operator.cols)
        identifiable = sp.eye(self.operator.cols) - self.kernel_projector
        coercivity = _positive_semidefinite(self.operator.T*self.operator - alpha_sq*identifiable, self.limits)
        p_sq, e_sq = _sq(result.identifiable_source), _sq(noise_vector)
        bias_term = self.parameter**2*p_sq/(alpha_sq+self.parameter)**2
        noise_term = e_sq/(4*self.parameter)
        remainder = result.identifiable_error_squared - bias_term - noise_term
        # Compare the cross radical only after checking its sign, without
        # replacing the sum of norm allowances by a sum of squares.
        bound = remainder <= 0 or remainder**2 <= self.parameter*p_sq*e_sq/(alpha_sq+self.parameter)**2
        return _certificate("SR-39-finite",
                            {"verified identifiable-source coercivity": coercivity,
                             "conditioned bias-noise squared bound": bound,
                             "exact kernel split": result.certificate.valid},
                            diagnostics={"alpha_squared": alpha_sq, "actual_error_squared": result.total_error_squared,
                                         "bias_term_squared": bias_term, "noise_term_squared": noise_term,
                                         "irreducible_kernel_squared": result.kernel_squared},
                            message="Finite source-coercivity premise and conditioned bias/noise/kernel bound checked exactly.")


@documented(ProvenanceRecord(
    identifier="SR-37-40-finite", title="Finite rational positive normal inverse",
    status=ClaimStatus.FINITE_CERTIFICATE,
    source="Sufficient Resolution, 8 October 2026, Section 11, SR-37--40",
    assumptions=("finite rational matrices", "ordinary Euclidean metrics", "strictly positive parameter"),
    limitations=("no infinite Hilbert-space construction or filter-limit theorem",),
    verification_suite="signals.tikhonov",
))
def tikhonov_system(
    operator: Any, parameter: Any, *, limits: TikhonovLimits | None = None,
) -> TikhonovSystem:
    limits = _limits(limits)
    sp = _sympy()
    a = _matrix(operator, "operator", limits)
    lam = _scalar(parameter, "parameter", limits, positive=True)
    # Forecast Gram formation and elimination before constructing the normal.
    _work_guard((a, sp.ImmutableMatrix([[lam]])), limits, inverse_size=a.cols)
    normal = _owned(a.T * a + lam * sp.eye(a.cols), limits)
    _work_guard((normal,), limits, inverse_size=a.cols)
    inverse = _owned(normal.inv(), limits)
    regularized = _owned(inverse * a.T, limits)
    kernel = _projector(a.nullspace(), a.cols, limits)
    image = _projector(a.columnspace(), a.rows, limits)
    identity = sp.eye(a.cols)
    checks = {
        "positive parameter": lam > 0,
        "both normal inverse laws": normal * inverse == identity and inverse * normal == identity,
        "self-adjoint inverse": inverse == inverse.T,
        "orthogonal kernel projector": kernel == kernel.T and kernel * kernel == kernel and _zero(a * kernel),
        "orthogonal range projector": image == image.T and image * image == image and image * a == a,
        "regularized source is identifiable": _zero(kernel * regularized),
    }
    certificate = _certificate(
        "SR-37-finite", checks,
        residuals={"normal_inverse": _max_abs(normal * inverse - identity)},
        message="Positive normal inverse constructed and checked in exact rational Euclidean arithmetic.",
    )
    certificate.require()
    return TikhonovSystem(a, lam, normal, inverse, regularized, kernel, image, limits, certificate)


@dataclass(frozen=True, slots=True)
class NormalResidualCorrection:
    normal_residual: Any
    correction: Any
    image_correction: Any
    normal_squared: Fraction
    source_correction_squared: Fraction
    image_correction_squared: Fraction
    source_bound_squared: Fraction
    image_bound_squared: Fraction
    certificate: Certificate


def normal_residual_correction(system: TikhonovSystem, data: Any, candidate: Any) -> NormalResidualCorrection:
    """Recompute the actual normal residual and the two distinct correction scales."""
    solution = system.solve(data)
    v = _matrix(data, "data", system.limits, column=True)
    x = _matrix(candidate, "candidate", system.limits, column=True)
    if x.rows != system.operator.cols:
        raise DimensionError("candidate dimension does not match source")
    _work_guard((system.normal, x, v, system.inverse), system.limits)
    residual = _owned(system.normal * x - system.operator.T * v, system.limits)
    correction = _owned(x - solution.source, system.limits)
    image = _owned(system.operator * correction, system.limits)
    residual_sq, correction_sq, image_sq = _sq(residual), _sq(correction), _sq(image)
    source_bound = residual_sq / system.parameter ** 2
    image_bound = residual_sq / (4 * system.parameter)
    checks = {
        "actual correction identity": correction == system.inverse * residual,
        "source correction squared bound": correction_sq <= source_bound,
        "image correction squared bound": image_sq <= image_bound,
    }
    return NormalResidualCorrection(
        residual, correction, image, residual_sq, correction_sq, image_sq, source_bound, image_bound,
        _certificate("SR-49-finite", checks, message="Actual normal residual controls source and image on distinct parameter scales."),
    )


@dataclass(frozen=True, slots=True)
class DiscrepancyBracket:
    lower: Fraction
    upper: Fraction
    lower_residual_squared: Fraction
    upper_residual_squared: Fraction
    target_squared: Fraction
    attainable_floor_squared: Fraction
    data_squared: Fraction
    evaluations: int
    converged: bool
    exact_parameter: Fraction | None
    certificate: Certificate


def _residual_curve(operator: Any, data: Any, limits: TikhonovLimits) -> tuple[Any, Any, Any, Fraction]:
    sp = _sympy()
    a = _matrix(operator, "operator", limits)
    v = _matrix(data, "data", limits, column=True)
    if v.rows != a.rows:
        raise DimensionError("data dimension does not match operator response")
    _work_guard((a, v), limits, inverse_size=a.cols)
    image = _projector(a.columnspace(), a.rows, limits)
    floor = _sq((sp.eye(a.rows) - image) * v)
    return a, v, _owned(a.T * a, limits), floor


def _bracket(
    operator: Any, data: Any, target_squared: Any, *, width: Fraction,
    max_evaluations: int, limits: TikhonovLimits,
    accept_band: tuple[Fraction, Fraction] | None = None,
) -> tuple[DiscrepancyBracket, Fraction | None]:
    sp = _sympy()
    a, v, gram, floor = _residual_curve(operator, data, limits)
    target = _scalar(target_squared, "target_squared", limits, positive=True)
    data_sq = _sq(v)
    if not floor < target < data_sq:
        raise ValidationError("target_squared requires strict attainable_floor_squared < target_squared < data_squared")
    evaluations = 0
    selected = None

    def evaluate(parameter: Fraction) -> Fraction:
        nonlocal evaluations, selected
        if evaluations >= max_evaluations:
            raise ResourceLimitError("discrepancy bracketing exceeded max_evaluations")
        _scalar(parameter, "bracket parameter", limits, positive=True)
        normal = _owned(gram + parameter * sp.eye(a.cols), limits)
        _work_guard((normal, v), limits, inverse_size=a.cols)
        source = _owned(normal.inv() * a.T * v, limits)
        result = _sq(a * source - v)
        evaluations += 1
        if accept_band is not None and accept_band[0] <= result <= accept_band[1]:
            selected = parameter
        return result

    lower = upper = Fraction(1)
    lower_sq = upper_sq = evaluate(Fraction(1))
    # Geometric searches are bounded and never substitute a nonexistent root.
    while lower_sq > target:
        lower /= 2
        lower_sq = evaluate(lower)
    while upper_sq < target:
        upper *= 2
        upper_sq = evaluate(upper)
    exact = lower if lower_sq == target else upper if upper_sq == target else None
    if exact is not None:
        lower = upper = exact
        lower_sq = upper_sq = target
    while exact is None and upper - lower > width and selected is None:
        middle = (lower + upper) / 2
        middle_sq = evaluate(middle)
        if middle_sq == target:
            lower = upper = exact = middle
            lower_sq = upper_sq = target
        elif middle_sq < target:
            lower, lower_sq = middle, middle_sq
        else:
            upper, upper_sq = middle, middle_sq
    converged = upper - lower <= width
    checks = {"positive parameter bracket": 0 < lower <= upper,
              "strict root domain": floor < target < data_sq,
              "recomputed endpoint signs": lower_sq <= target <= upper_sq}
    certificate = _certificate(
        "SR-45-finite", checks,
        diagnostics={"width": upper - lower, "requested_width": width, "converged": converged,
                     "exact_rational_root_found": exact is not None, "evaluations": evaluations},
        message="Rational parameters enclose the unique real discrepancy root; endpoint residuals were recomputed exactly.",
    )
    return DiscrepancyBracket(lower, upper, lower_sq, upper_sq, target, floor, data_sq,
                             evaluations, converged, exact, certificate), selected


def bracket_discrepancy(
    operator: Any, data: Any, target_squared: Any, *, width: Any = Fraction(1, 1 << 20),
    max_evaluations: int = 256, limits: TikhonovLimits | None = None,
) -> DiscrepancyBracket:
    """Enclose the SR-45 real root using exact squared residual endpoint tests.

    ``target_squared`` is the square of the desired discrepancy. Strict finite
    floor and data-norm inequalities are checked, including rank-deficient
    operators. Exhausting a resource budget raises ``ResourceLimitError``.
    """
    caps = _limits(limits)
    tolerance = _scalar(width, "width", caps, positive=True)
    count = _integer(max_evaluations, "max_evaluations")
    return _bracket(operator, data, target_squared, width=tolerance,
                    max_evaluations=count, limits=caps)[0]


@dataclass(frozen=True, slots=True)
class DiscrepancySelection:
    branch: str
    parameter: Fraction | None
    source: Any
    fitting_squared: Fraction
    bracket: DiscrepancyBracket | None
    certificate: Certificate


def select_discrepancy(
    operator: Any, data: Any, eta: Any, tau: Any, *, max_evaluations: int = 256,
    limits: TikhonovLimits | None = None,
) -> DiscrepancySelection:
    """Select an exact rational source in the SR-47 band, or the zero branch.

    A positive-parameter result is an interior-band rational selection, rather
    than the paper's possibly irrational upper-edge root. Finite feasibility
    ``distance(data, range(operator)) <= eta`` is checked in the positive
    branch. The zero branch only checks its defining small-data inequality.
    Neither check can
    authenticate ``eta`` as a true-data allowance for an unknown source.
    """
    caps = _limits(limits)
    allowance = _scalar(eta, "eta", caps, positive=True)
    multiplier = _scalar(tau, "tau", caps)
    if multiplier <= 1:
        raise ValidationError("tau must be strictly greater than one for selection")
    count = _integer(max_evaluations, "max_evaluations")
    a, v, _, floor = _residual_curve(operator, data, caps)
    if _sq(v) <= (multiplier * allowance) ** 2:
        source = _sympy().ImmutableMatrix(_sympy().zeros(a.cols, 1))
        cert = _certificate("SR-47-finite", {"small-data bound": True, "zero source": True},
                            unchecked=("eta is a genuine true-data noise allowance",),
                            message="Zero source accepted by the total small-data branch.")
        return DiscrepancySelection("small_data", None, source, _sq(v), None, cert)
    if floor > allowance ** 2:
        raise ValidationError("finite feasibility failed: attainable floor exceeds eta")
    target = (allowance * (multiplier + 1) / 2) ** 2
    bracket, selected = _bracket(
        a, v, target, width=Fraction(0), max_evaluations=count, limits=caps,
        accept_band=(allowance ** 2, (multiplier * allowance) ** 2),
    )
    if selected is None:
        # An exact interior target is itself an accepted rational sample.
        selected = bracket.exact_parameter
    if selected is None:
        raise ResourceLimitError("no rational accepted parameter found within the discrepancy budget")
    solution = tikhonov_system(a, selected, limits=caps).solve(v)
    cert = _certificate(
        "SR-47-finite", {"positive parameter": selected > 0,
                         "actual normal solution": solution.certificate.valid,
                         "accepted squared residual band": allowance ** 2 <= solution.fitting_squared <= (multiplier * allowance) ** 2,
                         "finite feasibility": floor <= allowance ** 2},
        unchecked=("eta is a genuine true-data noise allowance",),
        message="An exact rational positive-parameter source was selected inside the discrepancy band.",
    )
    return DiscrepancySelection("positive_parameter", selected, solution.source,
                                solution.fitting_squared, bracket, cert)


@dataclass(frozen=True, slots=True)
class ProtectedDiscrepancyWindow:
    lower: Fraction
    upper: Fraction
    midpoint: Fraction
    exists: bool
    certificate: Certificate

    def require_nonempty(self) -> ProtectedDiscrepancyWindow:
        self.certificate.require()
        if not self.exists:
            raise ValidationError("protected discrepancy window is empty")
        return self


def protected_discrepancy_window(eta: Any, tau: Any, omega: Any, xi: Any, *,
                                 limits: TikhonovLimits | None = None) -> ProtectedDiscrepancyWindow:
    """Construct the SR-50 protected scalar window, retaining empty windows."""
    caps = _limits(limits)
    e, t, w, i = (_nonnegative(value, name) for name, value in (
        ("eta", eta), ("tau", tau), ("omega", omega), ("xi", xi),
    ))
    for name, value in (("eta", e), ("tau", t), ("omega", w), ("xi", i)):
        _scalar(value, name, caps)
    lower, upper, midpoint = e+w+i, t*e-w-i, (t+1)*e/2
    exists = lower <= upper
    margin_exists = 2*(w+i) <= (t-1)*e
    checks = {"protected-window existence equivalence": exists == margin_exists,
              "midpoint accepted when nonempty": not exists or lower <= midpoint <= upper}
    return ProtectedDiscrepancyWindow(
        lower, upper, midpoint, exists,
        _certificate("SR-50-finite", checks, diagnostics={"exists": exists},
                     unchecked=("a particular computed source attains this scalar window", "genuine data-noise allowance"),
                     message="Protected scalar endpoints and existence margin computed exactly."),
    )


@dataclass(frozen=True, slots=True)
class SquaredDiscrepancyCertificate:
    actual_normal_squared: Fraction
    actual_fitting_squared: Fraction
    actual_source_correction_squared: Fraction
    exact_source_fitting_squared: Fraction
    checks: tuple[tuple[str, bool], ...]
    certificate: Certificate


def certify_squared_discrepancy(
    system: TikhonovSystem, data: Any, candidate: Any, *, eta: Any, tau: Any,
    rho: Any, xi: Any, kappa: Any, measured: Any, omega: Any,
) -> SquaredDiscrepancyCertificate:
    """Validate every SR-52 premise against actual rational squared norms.

    ``rho`` is an asserted normal-residual norm bound, ``xi`` an image budget,
    ``kappa`` a source budget, and ``omega`` an evaluation budget. No supplied
    squared norm or solver-reported residual is trusted in place of the actual
    matrix expression. Failed nonnegative margins return an invalid record.
    """
    scalars = {name: _nonnegative(value, name) for name, value in {
        "eta": eta, "tau": tau, "rho": rho, "xi": xi, "kappa": kappa,
        "measured": measured, "omega": omega,
    }.items()}
    for name, value in scalars.items():
        _scalar(value, name, system.limits)
    correction = normal_residual_correction(system, data, candidate)
    v = _matrix(data, "data", system.limits, column=True)
    x = _matrix(candidate, "candidate", system.limits, column=True)
    fit_sq = _sq(system.operator * x - v)
    exact = system.solve(v)
    e, t, r, i, k, m, w = (scalars[name] for name in ("eta", "tau", "rho", "xi", "kappa", "measured", "omega"))
    checks = {
        "actual normal residual enclosure": correction.normal_squared <= r ** 2,
        "source residual budget": r ** 2 <= (system.parameter * k) ** 2,
        "image residual budget": r ** 2 <= 4 * system.parameter * i ** 2,
        "actual fitting lower enclosure": max(Fraction(0), m - w) ** 2 <= fit_sq,
        "actual fitting upper enclosure": fit_sq <= (m + w) ** 2,
        "protected lower margin": e + w + i <= m,
        "protected upper margin": m <= t * e - w - i,
    }
    premises = all(checks.values())
    # Recompute the finite conclusion as a separate producer sanity check.
    conclusion = correction.source_correction_squared <= k ** 2 and e ** 2 <= exact.fitting_squared <= (t * e) ** 2
    checks["verified near-accepted conclusion"] = (not premises) or conclusion
    certificate = _certificate(
        "SR-52-finite", checks,
        diagnostics={"source_budget": k, "image_budget": i, "evaluation_budget": w,
                     "near_accepted": premises and conclusion},
        unchecked=("eta is a genuine true-data noise allowance", "infinite-family consistency hypotheses"),
        message="Squared protected discrepancy premises recomputed; data-noise truth remains a separate premise.",
    )
    return SquaredDiscrepancyCertificate(correction.normal_squared, fit_sq,
                                       correction.source_correction_squared, exact.fitting_squared,
                                       tuple(checks.items()), certificate)


def _sqrt_le_affine_root(a: Fraction, b: Fraction, c: Fraction, d: Fraction) -> bool:
    """Decide sqrt(a) <= b+c*sqrt(d) for nonnegative rationals, exactly."""
    remainder = a - b ** 2 - c ** 2 * d
    return remainder <= 0 or remainder ** 2 <= 4 * b ** 2 * c ** 2 * d


def _affine_root_le_sqrt(b: Fraction, c: Fraction, d: Fraction, a: Fraction) -> bool:
    """Decide b+c*sqrt(d) <= sqrt(a) without unsafe signed squaring."""
    remainder = a - b ** 2 - c ** 2 * d
    return remainder >= 0 and remainder ** 2 >= 4 * b ** 2 * c ** 2 * d


@dataclass(frozen=True, slots=True)
class CandidateNoiseAcceptance:
    accepted: bool
    branch: str | None
    source_squared: Fraction
    fitting_squared: Fraction
    normal_squared: Fraction
    certificate: Certificate


def candidate_noise_acceptance(
    system: TikhonovSystem, data: Any, candidate: Any, *, eta: Any, delta: Any, tau: Any,
) -> CandidateNoiseAcceptance:
    """Check cap-free SR-57--60 acceptance by exact radical comparisons.

    The positive branch requires the candidate to be the *actual* regularized
    source, not just to have a fitting norm in the requested band. For the
    separate small-data branch the candidate must be zero. Truth allowances,
    relative feasibility and existence are not inferred from acceptance.
    """
    system.certificate.require()
    e, d, t = (_nonnegative(value, name) for name, value in (("eta", eta), ("delta", delta), ("tau", tau)))
    for name, value in (("eta", e), ("delta", d), ("tau", t)):
        _scalar(value, name, system.limits)
    v = _matrix(data, "data", system.limits, column=True)
    x = _matrix(candidate, "candidate", system.limits, column=True)
    if v.rows != system.operator.rows or x.rows != system.operator.cols:
        raise DimensionError("candidate/data dimensions do not match operator")
    _work_guard((system.normal, x, v), system.limits)
    source_sq, fit_sq = _sq(x), _sq(system.operator * x - v)
    normal_sq = _sq(system.normal * x - system.operator.T * v)
    small = _zero(x) and _sq(v) <= (t * e) ** 2
    actual = normal_sq == 0
    lower = _affine_root_le_sqrt(e, d, source_sq, fit_sq)
    upper = _sqrt_le_affine_root(fit_sq, t * e, t * d, source_sq)
    positive = actual and lower and upper
    accepted = small or positive
    certificate = _certificate(
        "SR-57--60-finite", {"accepted by a defined branch": accepted},
        diagnostics={"zero_small_data": small, "actual_positive_normal_solution": actual,
                     "candidate_lower_band": lower, "candidate_upper_band": upper},
        unchecked=("eta is a genuine true-data noise allowance", "delta encloses the whole true/available operator difference",
                   "relative-feasibility hypotheses for existence", "infinite-family consistency hypotheses"),
        message="Generalized candidate acceptance checked with exact signed radical comparisons.",
    )
    return CandidateNoiseAcceptance(accepted, "small_data" if small else "positive_parameter" if positive else None,
                                    source_sq, fit_sq, normal_sq, certificate)


@dataclass(frozen=True, slots=True)
class CandidateNormDomination:
    solution: TikhonovSolution
    comparison_squared: Fraction
    comparison_fitting_squared: Fraction
    certificate: Certificate


def candidate_norm_domination(
    system: TikhonovSystem, data: Any, comparison: Any, *, eta: Any, delta: Any,
) -> CandidateNormDomination:
    """Check SR-57's feasible-comparison premises and actual norm conclusion."""
    e, d = _nonnegative(eta, "eta"), _nonnegative(delta, "delta")
    _scalar(e, "eta", system.limits)
    _scalar(d, "delta", system.limits)
    solution = system.solve(data)
    v = _matrix(data, "data", system.limits, column=True)
    p = _matrix(comparison, "comparison", system.limits, column=True)
    if p.rows != system.operator.cols:
        raise DimensionError("comparison dimension does not match source")
    _work_guard((system.operator, p, v), system.limits)
    comparison_sq, comparison_fit = _sq(p), _sq(system.operator * p - v)
    checks = {
        "comparison candidate-noise feasibility": _sqrt_le_affine_root(comparison_fit, e, d, comparison_sq),
        "actual source lower band": _affine_root_le_sqrt(e, d, solution.source_squared, solution.fitting_squared),
        "source norm dominated by comparison": solution.source_squared <= comparison_sq,
    }
    return CandidateNormDomination(solution, comparison_sq, comparison_fit,
                                   _certificate("SR-57-finite", checks,
                                                message="Finite feasible-comparison self-bounding checked exactly."))


def candidate_relative_feasibility(comparison: Any, eta: Any, delta: Any, tau: Any, *,
                                   limits: TikhonovLimits | None = None) -> Certificate:
    """Check the strict SR-59 scalar margin, retaining separate truth premises."""
    caps = _limits(limits)
    p = _matrix(comparison, "comparison", caps, column=True)
    e, d, t = (_nonnegative(value, name) for name, value in (("eta", eta), ("delta", delta), ("tau", tau)))
    for name, value in (("eta", e), ("delta", d), ("tau", t)):
        _scalar(value, name, caps)
    margin = (t - 1) * e
    checks = {"positive data allowance": e > 0, "strict band multiplier": t > 1,
              "strict relative feasibility": margin > 0 and d ** 2 * _sq(p) < margin ** 2}
    return _certificate("SR-59-finite", checks,
                        diagnostics={"comparison_squared": _sq(p), "margin_squared": margin ** 2},
                        unchecked=("true data bound for comparison", "whole-operator model bound"),
                        message="Strict relative-feasibility scalar inequality checked; existence retains its truth premises.")
