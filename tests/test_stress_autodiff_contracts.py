"""Backend-independent AD contract regressions and lazy import checks."""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.certificates import Exactness
from komorebi_math.nonlinear.autodiff import (
    AutomaticDifferentiationProvider,
    JaxDerivativeProvider,
    TorchDerivativeProvider,
)


def _neutral_provider(tangent: float = 1.0) -> AutomaticDifferentiationProvider:
    return AutomaticDifferentiationProvider(
        lambda x: np.asarray(x),
        lambda function, point, direction: (function(point), np.array([tangent])),
        1,
        1,
        max_order=4,
    )


@pytest.mark.parametrize("provider_class", [AutomaticDifferentiationProvider,
                                           JaxDerivativeProvider, TorchDerivativeProvider])
@pytest.mark.parametrize("order", [True, np.bool_(True), 1.5, "1", None,
                                   float("nan"), float("inf"), -float("inf")])
def test_ad_rejects_an_order_that_supports_order_rejects(provider_class, order) -> None:
    provider = (_neutral_provider() if provider_class is AutomaticDifferentiationProvider
                else provider_class(lambda x: x, 1, 1, max_order=4))
    assert provider.supports_order(order) is False
    with pytest.raises(ValidationError):
        provider.derivative(order, [1.0], ([1.0],))


@pytest.mark.parametrize("tangent", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_jvp_cannot_produce_a_live_symmetry_certificate(tangent) -> None:
    certificate = _neutral_provider(tangent).validation_certificate([1.0], ([1.0],))
    assert certificate.valid is False
    assert certificate.current_valid is False
    assert not np.isfinite(certificate.residuals["directional_symmetry"])
    with pytest.raises(ValidationError):
        certificate.require()


@pytest.mark.parametrize("which", ["point", "direction", "primal"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_ad_data_cannot_be_certified_even_with_finite_tangents(which, bad) -> None:
    provider = _neutral_provider()
    point = [bad] if which == "point" else [1.0]
    directions = ([bad] if which == "direction" else [1.0],)
    if which == "primal":
        provider.function = lambda x: np.array([bad])
    certificate = provider.validation_certificate(point, directions)
    assert certificate.current_valid is False
    with pytest.raises(ValidationError):
        certificate.require()


def test_finite_exact_primal_need_not_fit_in_machine_float() -> None:
    provider = AutomaticDifferentiationProvider(
        lambda x: np.array([Fraction(10**400)], dtype=object),
        lambda function, point, direction: (function(point), np.array([Fraction(0)], dtype=object)),
        1, 1, max_order=1, exactness=Exactness.EXACT,
    )
    provider.validation_certificate([Fraction(10**400)], ([Fraction(1)],)).require()


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_neutral_ad_rejects_nonfinite_or_negative_validation_tolerance(tolerance) -> None:
    with pytest.raises(ValidationError, match="finite and nonnegative"):
        _neutral_provider().validation_certificate([1.0], ([1.0],), tolerance=tolerance)


def _run_child(program: str, *arguments: str, timeout: float = 180.0) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source = root / "src"
    if source.is_dir():
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(source) + (os.pathsep + existing if existing else "")
    # These are CPU checks. No GPU runtime is requested or qualified.
    environment["JAX_PLATFORMS"] = "cpu"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run([sys.executable, "-c", program, *arguments],
                          cwd=root, env=environment, capture_output=True, text=True,
                          check=False, timeout=timeout)


def test_optional_ad_import_and_construction_remain_lazy() -> None:
    program = """
import sys
from komorebi_math.nonlinear.autodiff import JaxDerivativeProvider, TorchDerivativeProvider
JaxDerivativeProvider(lambda x: x, 1, 1)
TorchDerivativeProvider(lambda x: x, 1, 1)
assert 'jax' not in sys.modules
assert 'torch' not in sys.modules
print('lazy imports passed')
"""
    result = _run_child(program)
    assert result.returncode == 0, result.stdout + result.stderr


