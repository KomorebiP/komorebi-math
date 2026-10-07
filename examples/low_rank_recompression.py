"""Apply, solve, multiply, and recompress block diagonal plus low rank operators."""

from __future__ import annotations

import numpy as np

from komorebi_math.algorithms import BlockDiagonalLowRankOperator, BlockDiagonalOperator

rng = np.random.default_rng(7)
diagonal = BlockDiagonalOperator((2.0 * np.eye(2), 3.0 * np.eye(2)))
left = 0.1 * rng.normal(size=(4, 2))
right = 0.1 * rng.normal(size=(4, 2))
operator = BlockDiagonalLowRankOperator.from_factors(diagonal, left, right)

x = rng.normal(size=4)
y = operator.apply(x)
recovered = operator.solve(y)

product = operator.exact_product(operator)
compressed, certificate = product.recompress(rank=2)
certificate.certificate.require()

print("solve residual", np.linalg.norm(recovered - x))
print("product rank bound", product.rank_bound)
print("compressed rank", compressed.rank_bound)
print("compression spectral error", certificate.spectral_error)
