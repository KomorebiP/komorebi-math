"""Exact finite Gaussian gain algebra and direct outward dyadic verification.

The model and residual score follow Certified Bayesian Inference (8 October
2026), Sections 2--7. Dyadic enclosures are computed from exact rational final
values. This is a distinct finite evaluator, not a port of Core 1.0.0's
operation-by-operation rounded graph, selective cache, fuel or Lean program.
There are no eager optional dependencies and no floating-point promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral, Rational
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ResourceLimitError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented

Matrix = tuple[tuple[Fraction, ...], ...]
Vector = tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class GaussianBudget:
    """Explicit caps on dimensions, exact scalar size and dyadic precision."""

    max_dimension: int = 32
    max_exact_bits: int = 4096
    max_precision: int = 1024
    max_packets: int = 32

    def __post_init__(self) -> None:
        for name in ("max_dimension", "max_exact_bits", "max_packets"):
            _integer(getattr(self, name), name, minimum=1)
        _integer(self.max_precision, "max_precision", minimum=0)


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValidationError(f"{name} must be an integer at least {minimum}")
    return int(value)


_DEFAULT_BUDGET = GaussianBudget()


def _budget(value: GaussianBudget | None) -> GaussianBudget:
    if value is None:
        return _DEFAULT_BUDGET
    if not isinstance(value, GaussianBudget):
        raise ValidationError("budget must be a GaussianBudget")
    return value


def _check(value: Fraction, budget: GaussianBudget) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > budget.max_exact_bits:
        raise ResourceLimitError("exact Gaussian rational exceeds max_exact_bits")
    return value


def _rational(value: Any, name: str, budget: GaussianBudget) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Rational):
        raise ValidationError(
            f"{name} must be an exact rational; floats, strings and symbols are not promoted"
        )
    # Fraction(value) can retain fixed-width Integral numerator/denominator
    # components. Own them before bit checks or exact matrix arithmetic.
    return _check(Fraction(int(value.numerator), int(value.denominator)), budget)


def _shape_guard(
    value: Any, name: str, budget: GaussianBudget, *, vector: bool = False, empty_columns: bool = False
) -> None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return
    if not isinstance(shape, tuple) or len(shape) not in ((1, 2) if vector else (2,)):
        raise DimensionError(f"{name} has invalid dimensions")
    for index, size in enumerate(shape):
        minimum = 0 if empty_columns and index == 1 else 1
        if isinstance(size, bool) or not isinstance(size, Integral) or size < minimum:
            raise DimensionError(f"{name} has invalid dimensions")
        if size > budget.max_dimension:
            raise ResourceLimitError(f"{name} exceeds max_dimension")
    if vector and len(shape) == 2 and shape[1] != 1:
        raise DimensionError(f"{name} must be a column vector")


def _matrix(value: Any, name: str, budget: GaussianBudget, *, empty_columns: bool = False) -> Matrix:
    _shape_guard(value, name, budget, empty_columns=empty_columns)
    raw = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(raw, (list, tuple)) or not raw or not isinstance(raw[0], (list, tuple)):
        raise DimensionError(f"{name} must be a nonempty rectangular matrix")
    rows, cols = len(raw), len(raw[0])
    if rows > budget.max_dimension or cols > budget.max_dimension:
        raise ResourceLimitError(f"{name} exceeds max_dimension")
    if cols == 0 and not empty_columns:
        raise DimensionError(f"{name} must have positive dimensions")
    if any(not isinstance(row, (list, tuple)) or len(row) != cols for row in raw):
        raise DimensionError(f"{name} must be rectangular")
    return tuple(tuple(_rational(entry, name, budget) for entry in row) for row in raw)


def _vector(value: Any, name: str, budget: GaussianBudget) -> Vector:
    _shape_guard(value, name, budget, vector=True)
    raw = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(raw, (list, tuple)) or not raw:
        raise DimensionError(f"{name} must be a nonempty vector")
    if len(raw) > budget.max_dimension:
        raise ResourceLimitError(f"{name} exceeds max_dimension")
    if isinstance(raw[0], (list, tuple)):
        if any(not isinstance(row, (list, tuple)) or len(row) != 1 for row in raw):
            raise DimensionError(f"{name} must be a column vector")
        raw = [row[0] for row in raw]
    return tuple(_rational(entry, name, budget) for entry in raw)


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(zip(*matrix, strict=True))


def _dot(first: Vector, second: Vector, budget: GaussianBudget) -> Fraction:
    result = Fraction(0)
    for left, right in zip(first, second, strict=True):
        result = _check(result + _check(left * right, budget), budget)
    return result


def _multiply(first: Matrix, second: Matrix, budget: GaussianBudget) -> Matrix:
    return tuple(tuple(_dot(row, column, budget) for column in _transpose(second)) for row in first)


def _add(first: Matrix, second: Matrix, budget: GaussianBudget, *, sign: int = 1) -> Matrix:
    return tuple(
        tuple(_check(a + sign * b, budget) for a, b in zip(left, right, strict=True))
        for left, right in zip(first, second, strict=True)
    )


def _identity(size: int) -> Matrix:
    return tuple(tuple(Fraction(i == j) for j in range(size)) for i in range(size))


def _gram(factor: Matrix, budget: GaussianBudget) -> Matrix:
    # Direct row products also handle a represented n-by-zero factor.
    return tuple(tuple(_dot(row, other, budget) for other in factor) for row in factor)


def _energy(matrix: Matrix, budget: GaussianBudget) -> Fraction:
    flattened = tuple(entry for row in matrix for entry in row)
    return _dot(flattened, flattened, budget)


def _zero(matrix: Matrix) -> bool:
    return all(entry == 0 for row in matrix for entry in row)


def _cert(
    identifier: str,
    *,
    valid: bool,
    checked: tuple[str, ...],
    message: str,
    residuals: dict[str, Fraction] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> Certificate:
    return Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        theorem_id=identifier,
        residuals=residuals or {},
        assumptions_checked=checked,
        assumptions_unchecked=("law interpretation is the declared independent Gaussian model",),
        diagnostics=diagnostics or {},
        message=message,
    )


@dataclass(frozen=True, slots=True, init=False)
class FactorGaussianModel:
    """Rational P=LL^T, R=NN^T+sigma*I for one Gaussian observation.

    Prior factors may be singular and H may be rectangular. Factors with zero
    columns are represented by one empty row per output dimension.
    """

    prior_factor: Matrix
    observation_operator: Matrix
    noise_factor: Matrix
    noise_floor: Fraction
    prior_mean: Vector
    observation: Vector
    prior_covariance: Matrix
    noise_covariance: Matrix
    innovation_covariance: Matrix
    innovation: Vector
    budget: GaussianBudget = field(repr=False)

    def __init__(
        self,
        *,
        prior_factor: Any,
        observation_operator: Any,
        noise_factor: Any,
        noise_floor: Any,
        prior_mean: Any,
        observation: Any,
        budget: GaussianBudget | None = None,
    ) -> None:
        cap = _budget(budget)
        floor = _rational(noise_floor, "noise_floor", cap)
        if floor <= 0:
            raise ValidationError("noise_floor must be positive")
        prior = _matrix(prior_factor, "prior_factor", cap, empty_columns=True)
        operator = _matrix(observation_operator, "observation_operator", cap)
        noise = _matrix(noise_factor, "noise_factor", cap, empty_columns=True)
        mean = _vector(prior_mean, "prior_mean", cap)
        observed = _vector(observation, "observation", cap)
        n, p = len(prior), len(operator)
        if len(operator[0]) != n or len(mean) != n or len(noise) != p or len(observed) != p:
            raise DimensionError("factor, observation operator and vector dimensions do not match")
        covariance = _gram(prior, cap)
        noise_covariance = _add(
            _gram(noise, cap), tuple(tuple(floor * entry for entry in row) for row in _identity(p)), cap
        )
        innovation_covariance = _add(
            _multiply(_multiply(operator, covariance, cap), _transpose(operator), cap), noise_covariance, cap
        )
        innovation = tuple(
            _check(y - _dot(row, mean, cap), cap) for y, row in zip(observed, operator, strict=True)
        )
        for name, value in (
            ("prior_factor", prior),
            ("observation_operator", operator),
            ("noise_factor", noise),
            ("noise_floor", floor),
            ("prior_mean", mean),
            ("observation", observed),
            ("prior_covariance", covariance),
            ("noise_covariance", noise_covariance),
            ("innovation_covariance", innovation_covariance),
            ("innovation", innovation),
            ("budget", cap),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class GaussianGainEvaluation:
    gain: Matrix
    mean: Vector
    covariance: Matrix
    residual: Matrix
    residual_energy: Fraction
    innovation_energy: Fraction
    score: Fraction
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="CB-1.0-JOSEPH-SCORE",
        title="Finite Joseph gain and residual score",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Certified Bayesian Inference, §§4-5, Proposition 4.1 and Theorem 5.2",
        assumptions=("rational factor-coded one-observation Gaussian model", "positive noise floor"),
        limitations=("exact rational finite evaluator, not the formal dyadic graph or a sampler",),
        verification_suite="dynamics.gaussian_verifier",
    )
)
def evaluate_gaussian_gain(model: FactorGaussianModel, gain: Any) -> GaussianGainEvaluation:
    """Compute the Joseph parameters and inverse-free squared-law upper score."""
    if not isinstance(model, FactorGaussianModel):
        raise ValidationError("model must be a FactorGaussianModel")
    cap = model.budget
    matrix = _matrix(gain, "gain", cap)
    n, p = len(model.prior_mean), len(model.observation)
    if len(matrix) != n or len(matrix[0]) != p:
        raise DimensionError("gain must map observation space to state space")
    mean = tuple(
        _check(m + _dot(row, model.innovation, cap), cap)
        for m, row in zip(model.prior_mean, matrix, strict=True)
    )
    difference = _add(_identity(n), _multiply(matrix, model.observation_operator, cap), cap, sign=-1)
    covariance = _add(
        _multiply(_multiply(difference, model.prior_covariance, cap), _transpose(difference), cap),
        _multiply(_multiply(matrix, model.noise_covariance, cap), _transpose(matrix), cap),
        cap,
    )
    residual = _add(
        _multiply(model.prior_covariance, _transpose(model.observation_operator), cap),
        _multiply(matrix, model.innovation_covariance, cap),
        cap,
        sign=-1,
    )
    energy = _energy(residual, cap)
    innovation_energy = _dot(model.innovation, model.innovation, cap)
    weight = _check(
        _check(model.noise_floor + innovation_energy, cap) / _check(model.noise_floor**2, cap), cap
    )
    score = _check(energy * weight, cap)
    cert = _cert(
        "CB-1.0-JOSEPH-SCORE",
        valid=True,
        checked=(
            "exact rational factor-coded covariance",
            "positive noise floor",
            "Joseph covariance congruences",
            "exact whole residual energy",
        ),
        diagnostics={"score": score, "score_semantics": "upper bound on squared W2 under the declared model"},
        message="finite Joseph parameters and inverse-free residual score computed exactly",
    )
    return GaussianGainEvaluation(matrix, mean, covariance, residual, energy, innovation_energy, score, cert)


def _solve(matrix: Matrix, rhs: Matrix, cap: GaussianBudget) -> Matrix:
    size, width = len(matrix), len(rhs[0])
    augmented = [list(left + right) for left, right in zip(matrix, rhs, strict=True)]
    for column in range(size):
        pivot = next((i for i in range(column, size) if augmented[i][column]), None)
        if pivot is None:
            raise ValidationError("innovation covariance is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [_check(entry / divisor, cap) for entry in augmented[column]]
        for index in range(size):
            if index != column:
                multiplier = augmented[index][column]
                augmented[index] = [
                    _check(a - _check(multiplier * b, cap), cap)
                    for a, b in zip(augmented[index], augmented[column], strict=True)
                ]
    return tuple(tuple(row[size : size + width]) for row in augmented)


@documented(
    ProvenanceRecord(
        identifier="CB-1.0-EXACT-GAIN",
        title="Exact finite Gaussian innovation solve",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Certified Bayesian Inference, §2 equation (3) and §3 Lemma 3.1",
        assumptions=("factor-coded rational model with positive noise floor",),
        limitations=("finite matrix solve; no new disintegration proof or formal program refinement",),
        verification_suite="dynamics.gaussian_verifier",
    )
)
def exact_gaussian_posterior(model: FactorGaussianModel) -> GaussianGainEvaluation:
    """Solve S*K^T=H*P exactly; no inverse of the singular prior is used."""
    if not isinstance(model, FactorGaussianModel):
        raise ValidationError("model must be a FactorGaussianModel")
    gain = _transpose(
        _solve(
            model.innovation_covariance,
            _multiply(model.observation_operator, model.prior_covariance, model.budget),
            model.budget,
        )
    )
    result = evaluate_gaussian_gain(model, gain)
    if not _zero(result.residual):
        raise ValidationError("exact posterior solve did not satisfy the gain equation")
    return result


@documented(
    ProvenanceRecord(
        identifier="CB-1.0-T5.2-FINITE",
        title="Exact finite Gaussian residual tolerance",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Certified Bayesian Inference, §5 Theorem 5.2, equation (22)",
        assumptions=(
            "declared one-observation independent Gaussian model",
            "rational factor-coded covariance and positive floor",
        ),
        limitations=("sufficient score test, not optimal Wasserstein distance",),
        verification_suite="dynamics.gaussian_verifier",
    )
)
def check_gaussian_tolerance(model: FactorGaussianModel, gain: Any, tolerance: Any) -> Certificate:
    """Recompute the candidate and compare the source's exact rational inequality."""
    result = evaluate_gaussian_gain(model, gain)
    epsilon = _rational(tolerance, "tolerance", model.budget)
    if epsilon < 0:
        raise ValidationError("tolerance must be nonnegative")
    left = _check(
        result.residual_energy * _check(model.noise_floor + result.innovation_energy, model.budget),
        model.budget,
    )
    right = _check(
        _check(epsilon**2, model.budget) * _check(model.noise_floor**2, model.budget), model.budget
    )
    return _cert(
        "CB-1.0-T5.2-FINITE",
        valid=left <= right,
        checked=("candidate recomputed from model and gain", "exact cross-multiplied residual tolerance"),
        diagnostics={"left": left, "right": right, "tolerance": epsilon},
        message="residual law tolerance established"
        if left <= right
        else "sufficient residual law tolerance not established",
    )


@dataclass(frozen=True, slots=True)
class DyadicInterval:
    lower: Fraction
    upper: Fraction
    center: Fraction
    radius: Fraction
    precision: int


def outward_dyadic(value: Any, precision: int, *, budget: GaussianBudget | None = None) -> DyadicInterval:
    """Enclose one exact rational with floor/ceiling endpoints on 2^-p grids."""
    cap = _budget(budget)
    exact = _rational(value, "value", cap)
    bits = _integer(precision, "precision", minimum=0)
    if bits > cap.max_precision:
        raise ResourceLimitError("dyadic precision exceeds max_precision")
    if max(abs(exact.numerator).bit_length() + bits, bits + 1) > cap.max_exact_bits:
        raise ResourceLimitError("dyadic scaling exceeds max_exact_bits")
    mesh_denominator = 1 << bits
    floor, remainder = divmod(exact.numerator << bits, exact.denominator)
    lower = _check(Fraction(floor, mesh_denominator), cap)
    upper = _check(Fraction(floor + bool(remainder), mesh_denominator), cap)
    center = _check((lower + upper) / 2, cap)
    radius = _check((upper - lower) / 2, cap)
    return DyadicInterval(lower, upper, center, radius, bits)


@dataclass(frozen=True, slots=True)
class GaussianVerification:
    evaluation: GaussianGainEvaluation
    score_interval: DyadicInterval
    mean_intervals: tuple[DyadicInterval, ...]
    covariance_intervals: tuple[tuple[DyadicInterval, ...], ...]
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="CB-DIRECT-DYADIC-FINITE",
        title="Direct rational Gaussian dyadic verifier",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Certified Bayesian Inference, §§5,11-12; distinct direct finite evaluator",
        assumptions=("exact rational final score and Joseph parameters", "supplied finite dyadic precision"),
        limitations=(
            "does not implement the formal operation-by-operation rounded graph, fallback, cache or fuel",
        ),
        verification_suite="dynamics.gaussian_verifier",
    )
)
def verify_gaussian_gain(
    model: FactorGaussianModel, gain: Any, *, tolerance: Any, precision: int, output_radius: Any | None = None
) -> GaussianVerification:
    """Independently recompute and enclose; accept only a passing upper score.

    A coarse enclosure overlapping the squared tolerance fails verification.
    Parameter intervals enclose this returned Joseph law, not the exact target.
    """
    result = evaluate_gaussian_gain(model, gain)
    cap = model.budget
    epsilon = _rational(tolerance, "tolerance", cap)
    if epsilon < 0:
        raise ValidationError("tolerance must be nonnegative")
    radius = None if output_radius is None else _rational(output_radius, "output_radius", cap)
    if radius is not None and radius <= 0:
        raise ValidationError("output_radius must be positive")
    interval = outward_dyadic(result.score, precision, budget=cap)
    mean = tuple(outward_dyadic(value, precision, budget=cap) for value in result.mean)
    covariance = tuple(
        tuple(outward_dyadic(value, precision, budget=cap) for value in row) for row in result.covariance
    )
    tolerance_ok = interval.upper <= _check(epsilon**2, cap)
    radius_ok = radius is None or all(
        item.radius <= radius for item in mean + tuple(item for row in covariance for item in row)
    )
    valid = tolerance_ok and radius_ok
    cert = _cert(
        "CB-DIRECT-DYADIC-FINITE",
        valid=valid,
        checked=(
            "gain and score independently recomputed",
            "exact outward floor/ceiling endpoints",
            "upper squared score meets tolerance"
            if tolerance_ok
            else "upper squared score does not meet tolerance",
        ),
        diagnostics={
            "score_upper": interval.upper,
            "tolerance": epsilon,
            "precision": interval.precision,
            "output_radius_met": radius_ok,
            "evaluator": "exact rational final values then direct dyadic endpoints",
        },
        message="direct dyadic finite Gaussian verification passed"
        if valid
        else "direct dyadic verification lacks the requested score or radius bound",
    )
    return GaussianVerification(result, interval, mean, covariance, cert)


@dataclass(frozen=True, slots=True)
class GaussianPacketEnergy:
    energies: tuple[tuple[Fraction, ...], ...]
    total: Fraction
    certificate: Certificate


def _packets(values: Any, dimension: int, name: str, cap: GaussianBudget) -> tuple[Matrix, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValidationError(f"{name} must be a nonempty finite packet sequence")
    if len(values) > cap.max_packets:
        raise ResourceLimitError(f"{name} exceeds max_packets")
    packets = tuple(_matrix(value, name, cap) for value in values)
    if any(len(matrix) != dimension or len(matrix[0]) != dimension for matrix in packets):
        raise DimensionError(f"{name} has a wrong packet dimension")
    total = tuple(tuple(Fraction(0) for _ in range(dimension)) for _ in range(dimension))
    for index, matrix in enumerate(packets):
        if matrix != _transpose(matrix) or _multiply(matrix, matrix, cap) != matrix:
            raise ValidationError(f"{name} must be exactly self-adjoint idempotents")
        total = _add(total, matrix, cap)
        for other_index, other in enumerate(packets):
            if index != other_index and not _zero(_multiply(matrix, other, cap)):
                raise ValidationError(f"{name} must be mutually orthogonal")
    if total != _identity(dimension):
        raise ValidationError(f"{name} must cover the whole space")
    return packets


@documented(
    ProvenanceRecord(
        identifier="CB-1.0-T6.1-FINITE",
        title="Complete finite Gaussian residual packet energy",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Certified Bayesian Inference, §6 Theorem 6.1 equation (27)",
        assumptions=("finite exact complete self-adjoint orthogonal input and output packet families",),
        limitations=("fixed declared packet families; no selective cache trajectory",),
        verification_suite="dynamics.gaussian_verifier",
    )
)
def gaussian_packet_energy(
    model: FactorGaussianModel, gain: Any, *, output_packets: Any, input_packets: Any
) -> GaussianPacketEnergy:
    """Validate complete orthogonal packets and compute every E(Q_a*F*D_b)."""
    result = evaluate_gaussian_gain(model, gain)
    cap = model.budget
    outputs = _packets(output_packets, len(model.prior_mean), "output_packets", cap)
    inputs = _packets(input_packets, len(model.observation), "input_packets", cap)
    energies = tuple(
        tuple(_energy(_multiply(_multiply(q, result.residual, cap), d, cap), cap) for d in inputs)
        for q in outputs
    )
    flattened = tuple(value for row in energies for value in row)
    total = _dot(flattened, tuple(Fraction(1) for _ in flattened), cap)
    valid = total == result.residual_energy
    cert = _cert(
        "CB-1.0-T6.1-FINITE",
        valid=valid,
        checked=("exact complete self-adjoint orthogonal packets", "all two-sided residual packet energies"),
        residuals={"energy_accounting": total - result.residual_energy},
        message="complete residual packet energy agrees exactly"
        if valid
        else "residual packet energy identity failed",
    )
    return GaussianPacketEnergy(energies, total, cert)


__all__ = [
    "DyadicInterval",
    "FactorGaussianModel",
    "GaussianBudget",
    "GaussianGainEvaluation",
    "GaussianPacketEnergy",
    "GaussianVerification",
    "check_gaussian_tolerance",
    "evaluate_gaussian_gain",
    "exact_gaussian_posterior",
    "gaussian_packet_energy",
    "outward_dyadic",
    "verify_gaussian_gain",
]
