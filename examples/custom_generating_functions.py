"""Exact custom generating functions in explicit normalized bases."""

from __future__ import annotations

from fractions import Fraction

from komorebi_math.export import dumps_portable, loads_portable
from komorebi_math.generating import (
    FiniteNormalizedSeries,
    FunctionRegistry,
    RationalDomain,
    algebraic_function,
    factorial_power_basis,
    freeze_custom_function,
    rational_function,
)

registry = FunctionRegistry()
fibonacci = rational_function("fibonacci", (0, 1), (1, -1, -1))
catalan = algebraic_function(
    "catalan",
    {(0, 0): 1, (0, 1): -1, (1, 2): 1},
    1,
)
registry.register(fibonacci)
registry.register(catalan)

assert fibonacci.series(12).coefficients[12] == 144
assert catalan.series(8).coefficients[8] == 1430

basis = factorial_power_basis(2)
left = FiniteNormalizedSeries(basis, RationalDomain(), (1, 2, -3, 5))
right = FiniteNormalizedSeries(basis, RationalDomain(), (2, -1, 4))
product = left.multiply(right)
product.certificate().require()

frozen = freeze_custom_function(catalan, 12)
restored = loads_portable(dumps_portable(frozen))
assert restored.series().coefficients == frozen.coefficients

print(
    {
        "registry": registry.names(),
        "catalan_8": str(catalan.series(8).coefficients[8]),
        "product": tuple(str(value) for value in product.coefficients),
        "portable_degree": restored.degree,
    }
)
