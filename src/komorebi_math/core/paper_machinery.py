"""Finite executable constructions extracted from Papers II and III.

The mapping cone works over a specified prime field. Spectral packets work over
the rational field and require a supplied complete coprime characteristic
factorization. Neither construction claims arbitrary-field discovery, analytic
completion, or proof of an infinite-dimensional theorem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._number_theory import require_integer
from ..deformation.complexes import FiniteCochainComplex
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from .packets import PacketSystem
from .seeds import _polynomial_at_matrix


def _field_matrix(value: Any, shape: tuple[int, int], prime: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=object)
    if matrix.ndim != 2 or matrix.shape != shape:
        raise DimensionError(f"cochain map has shape {matrix.shape}, expected {shape}")
    result = np.empty(shape, dtype=object)
    for index, entry in np.ndenumerate(matrix):
        result[index] = require_integer(entry, name="cochain map entry") % prime
    return result


def finite_mapping_cone(
    source: FiniteCochainComplex,
    target: FiniteCochainComplex,
    cochain_maps: Mapping[int, Any],
) -> FiniteCochainComplex:
    """Construct ``Cone(f)^n = target^n + source^(n+1)`` exactly.

    The differential, in that order of summands, is
    ``[[d_target, f_(n+1)], [0, -d_source]]``. Missing component maps are zero.
    All cochain-map squares are checked before construction. Input complexes
    are revalidated and copied; output coefficients are Python integers reduced
    modulo their common prime. This is Paper III Definition 21.10 specialized
    to finite prime-field data, including zero-dimensional and negative degrees.
    """
    if not isinstance(source, FiniteCochainComplex) or not isinstance(target, FiniteCochainComplex):
        raise ValidationError("source and target must be finite cochain complexes")
    if source.prime != target.prime:
        raise ValidationError("mapping-cone complexes must use the same prime field")
    # Reconstruct instead of trusting a previously emitted certificate for
    # potentially changed public dictionaries.
    source_copy = FiniteCochainComplex(source.dimensions, source.differentials, prime=source.prime)
    target_copy = FiniteCochainComplex(target.dimensions, target.differentials, prime=target.prime)
    prime = source_copy.prime
    active_degrees = set(source_copy.dimensions) | set(target_copy.dimensions)
    component_maps: dict[int, np.ndarray] = {}
    for raw_degree, value in cochain_maps.items():
        degree = require_integer(raw_degree, name="cochain map degree")
        if degree not in active_degrees:
            raise ValidationError("cochain map degree is outside both complexes")
        shape = (target_copy.dimensions.get(degree, 0), source_copy.dimensions.get(degree, 0))
        component_maps[degree] = _field_matrix(value, shape, prime)

    def component(degree: int) -> np.ndarray:
        shape = (target_copy.dimensions.get(degree, 0), source_copy.dimensions.get(degree, 0))
        return component_maps.get(degree, np.zeros(shape, dtype=object))

    for degree in range(min(active_degrees) - 1, max(active_degrees) + 1):
        square = (
            target_copy.differential(degree) @ component(degree)
            - component(degree + 1) @ source_copy.differential(degree)
        ) % prime
        if np.any(square):
            raise ValidationError(f"cochain map does not commute with the differentials at degree {degree}")

    first_degree = min(min(target_copy.dimensions), min(source_copy.dimensions) - 1)
    last_degree = max(max(target_copy.dimensions), max(source_copy.dimensions) - 1)
    dimensions = {
        degree: target_copy.dimensions.get(degree, 0) + source_copy.dimensions.get(degree + 1, 0)
        for degree in range(first_degree, last_degree + 1)
    }
    differentials: dict[int, np.ndarray] = {}
    for degree in range(first_degree, last_degree):
        matrix = np.zeros((dimensions[degree + 1], dimensions[degree]), dtype=object)
        target_source = target_copy.dimensions.get(degree, 0)
        target_target = target_copy.dimensions.get(degree + 1, 0)
        matrix[:target_target, :target_source] = target_copy.differential(degree)
        matrix[:target_target, target_source:] = component(degree + 1)
        matrix[target_target:, target_source:] = -source_copy.differential(degree + 1)
        differentials[degree] = matrix % prime
    return FiniteCochainComplex(dimensions, differentials, prime=prime)


def rational_spectral_projectors(
    operator: Any,
    factors: Sequence[Any],
    *,
    variable: Any | None = None,
    names: Sequence[str] | None = None,
) -> PacketSystem:
    """Construct exact rational packets for supplied characteristic factors.

    Each factor must be monic, nonconstant, and rational. Their product must
    equal the characteristic polynomial, and distinct factors must be coprime.
    For ``chi = f_i * g_i`` and ``u_i*f_i + v_i*g_i = 1``, the projector is
    ``v_i(A)*g_i(A)`` onto ``ker(f_i(A))``. Repeated roots within one factor are
    allowed. The input factor order determines output packet order.

    This implements the finite rational specialization of Paper II Theorems
    7.1--7.2. It does not infer the input factors, a lattice intersection basis,
    resultant-optimal denominators, or an arbitrary DVR realization.
    """
    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError("rational spectral packets require the exact SymPy extra") from exc
    matrix = sp.ImmutableMatrix(operator)
    if matrix.rows != matrix.cols or matrix.rows == 0:
        raise DimensionError("spectral packet operator must be a nonempty square matrix")
    if any(not isinstance(entry, sp.Rational) for entry in matrix):
        raise ValidationError("spectral packet operator requires exact rational entries")
    expressions = tuple(
        value.as_expr() if isinstance(value, sp.Poly) else sp.sympify(value) for value in factors
    )
    if not expressions:
        raise ValidationError("spectral packets require at least one characteristic factor")
    if variable is None:
        symbols = set().union(*(expression.free_symbols for expression in expressions))
        if len(symbols) != 1:
            raise ValidationError("characteristic factors must identify one polynomial variable")
        variable = symbols.pop()
    if not isinstance(variable, sp.Symbol):
        raise ValidationError("polynomial variable must be a SymPy Symbol")
    polynomials: list[Any] = []
    for expression in expressions:
        if expression.atoms(sp.Float):
            raise ValidationError("characteristic factors require exact rational coefficients")
        try:
            polynomial = sp.Poly(expression, variable, domain=sp.QQ)
        except (sp.PolynomialError, sp.CoercionFailed) as exc:
            raise ValidationError(
                "characteristic factors must be polynomials over the rational field"
            ) from exc
        if polynomial.degree() < 1 or polynomial.LC() != 1:
            raise ValidationError("characteristic factors must be monic and nonconstant")
        polynomials.append(polynomial)
    characteristic = sp.Poly(matrix.charpoly(variable).as_expr(), variable, domain=sp.QQ)
    factor_product = sp.Poly(1, variable, domain=sp.QQ)
    for polynomial in polynomials:
        factor_product *= polynomial
    if factor_product != characteristic:
        raise ValidationError("supplied factors do not multiply to the characteristic polynomial")
    for index, left in enumerate(polynomials):
        if any(sp.gcd(left, right).degree() > 0 for right in polynomials[index + 1 :]):
            raise ValidationError("distinct characteristic factors must be coprime")
    projectors: list[Any] = []
    for factor in polynomials:
        complementary = characteristic.exquo(factor)
        _, multiplier, gcd = sp.gcdex(factor, complementary)
        if gcd != sp.Poly(1, variable, domain=sp.QQ):
            raise ArithmeticError("coprime characteristic factors did not produce a unit Bezout identity")
        projector = _polynomial_at_matrix((multiplier * complementary).as_expr(), matrix, variable)
        if _polynomial_at_matrix(factor.as_expr(), matrix, variable) * projector != sp.zeros(matrix.rows):
            raise ArithmeticError("spectral projector does not have the declared annihilator")
        if matrix * projector != projector * matrix:
            raise ArithmeticError("spectral projector does not commute with its operator")
        projectors.append(projector)
    packets = PacketSystem(projectors, names=names, backend="sympy")
    packets.certificate().require()
    return packets


def rational_primary_packets(operator: Any, *, names: Sequence[str] | None = None) -> PacketSystem:
    """Derive primary packets of one rational operator by exact factorization.

    Irreducible rational characteristic factors are raised to their full
    characteristic multiplicities, then passed to
    :func:`rational_spectral_projectors`. This is a canonical primary
    decomposition of the supplied finite rational single-operator module. It
    is not a simultaneous Hecke-algebra, arbitrary-field, or global packet
    discovery algorithm.
    """
    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError("rational primary packets require the exact SymPy extra") from exc
    matrix = sp.ImmutableMatrix(operator)
    if matrix.rows != matrix.cols or matrix.rows == 0:
        raise DimensionError("primary packet operator must be a nonempty square matrix")
    if any(not isinstance(entry, sp.Rational) for entry in matrix):
        raise ValidationError("primary packet operator requires exact rational entries")
    variable = sp.Symbol("x")
    characteristic = sp.Poly(matrix.charpoly(variable).as_expr(), variable, domain=sp.QQ)
    factors = tuple(factor**multiplicity for factor, multiplicity in sp.factor_list(characteristic)[1])
    return rational_spectral_projectors(matrix, factors, variable=variable, names=names)


__all__ = ["finite_mapping_cone", "rational_primary_packets", "rational_spectral_projectors"]
