"""Crash-isolated smoke checks for optional automatic-differentiation backends.

This worker is intentionally executed in a child interpreter.  Native backend
imports can abort the interpreter before Python can raise an exception.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from komorebi_math.nonlinear.autodiff import JaxDerivativeProvider, TorchDerivativeProvider


def _block10_expected(point: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    return float(
        24 * point[0] * u[0] * v[0] * w[0]
        + 2
        * (
            u[0] * v[1] * w[1]
            + v[0] * u[1] * w[1]
            + w[0] * u[1] * v[1]
        )
    )


def _nonlinear_expected(
    order: int,
    point: np.ndarray,
    directions: tuple[np.ndarray, ...],
) -> np.ndarray:
    x, y = point
    if order == 1:
        (u,) = directions
        return np.array([3 * x * x * u[0] + y * u[0] + x * u[1], 3 * y * y * u[1]])
    if order == 2:
        u, v = directions
        return np.array(
            [6 * x * u[0] * v[0] + u[0] * v[1] + u[1] * v[0], 6 * y * u[1] * v[1]]
        )
    if order == 3:
        u, v, w = directions
        return np.array([6 * u[0] * v[0] * w[0], 6 * u[1] * v[1] * w[1]])
    raise AssertionError(order)


def _run_jax(case: str) -> str:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    if case == "block10":
        provider = JaxDerivativeProvider(
            lambda x: jnp.array([x[0] ** 4 + x[0] * x[1] ** 2]),
            source_dimension=2,
            target_dimension=1,
            max_order=3,
        )
        point = np.array([0.8, -0.3])
        u = np.array([0.4, 0.7])
        v = np.array([-0.2, 0.5])
        w = np.array([0.6, -0.1])
        first = (4 * point[0] ** 3 + point[1] ** 2) * u[0] + 2 * point[0] * point[1] * u[1]
        if not np.allclose(provider.derivative(1, point, (u,)), [first], rtol=2e-5, atol=2e-6):
            raise AssertionError("JAX first directional derivative mismatch")
        expected = _block10_expected(point, u, v, w)
        if not np.allclose(provider.derivative(3, point, (u, v, w)), [expected], rtol=3e-5, atol=3e-6):
            raise AssertionError("JAX third directional derivative mismatch")
    else:
        provider = JaxDerivativeProvider(
            lambda x: jnp.stack((x[0] ** 3 + x[0] * x[1], x[1] ** 3)),
            source_dimension=2,
            target_dimension=2,
            max_order=3,
            dtype="float64",
        )
        point = np.array([0.7, -0.4])
        directions = (
            np.array([0.2, 0.9]),
            np.array([-0.5, 0.3]),
            np.array([0.8, -0.6]),
        )
        for order in range(1, 4):
            observed = provider.derivative(order, point, directions[:order])
            expected = _nonlinear_expected(order, point, directions[:order])
            if not np.allclose(observed, expected, atol=1e-11):
                raise AssertionError(f"JAX order-{order} derivative mismatch")
        provider.validation_certificate(point, directions, tolerance=1e-10).require()
    return str(getattr(jax, "__version__", "unknown"))


def _run_torch(case: str) -> str:
    import torch

    if case == "block10":
        provider = TorchDerivativeProvider(
            lambda x: torch.stack((x[0] ** 4 + x[0] * x[1] ** 2,)),
            source_dimension=2,
            target_dimension=1,
            max_order=3,
        )
        point = np.array([0.8, -0.3])
        u = np.array([0.4, 0.7])
        v = np.array([-0.2, 0.5])
        w = np.array([0.6, -0.1])
        expected = _block10_expected(point, u, v, w)
        if not np.allclose(provider.derivative(3, point, (u, v, w)), [expected], atol=1e-10):
            raise AssertionError("PyTorch third directional derivative mismatch")
    else:
        provider = TorchDerivativeProvider(
            lambda x: torch.stack((x[0] ** 3 + x[0] * x[1], x[1] ** 3)),
            source_dimension=2,
            target_dimension=2,
            max_order=3,
        )
        point = np.array([0.7, -0.4])
        directions = (
            np.array([0.2, 0.9]),
            np.array([-0.5, 0.3]),
            np.array([0.8, -0.6]),
        )
        for order in range(1, 4):
            observed = provider.derivative(order, point, directions[:order])
            expected = _nonlinear_expected(order, point, directions[:order])
            if not np.allclose(observed, expected, atol=1e-11):
                raise AssertionError(f"PyTorch order-{order} derivative mismatch")
    return str(getattr(torch, "__version__", "unknown"))


def _emit(payload: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    print(json.dumps(payload, sort_keys=True), file=stream, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("jax", "torch"), required=True)
    parser.add_argument("--case", choices=("block10", "nonlinear"), required=True)
    args = parser.parse_args(argv)
    try:
        version = _run_jax(args.case) if args.framework == "jax" else _run_torch(args.case)
    except Exception as exc:
        _emit(
            {
                "available": True,
                "case": args.case,
                "error": f"{type(exc).__name__}: {exc}",
                "framework": args.framework,
                "passed": False,
            },
            stream=sys.stderr,
        )
        return 1
    _emit(
        {
            "available": True,
            "case": args.case,
            "framework": args.framework,
            "passed": True,
            "version": version,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
