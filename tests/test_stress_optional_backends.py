"""Crash-isolated CPU controls for optional JAX and PyTorch AD adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from komorebi_math.nonlinear.autodiff import ad_framework_diagnostics


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


# Analytic controls are closed formulas for exp(a.x) and sin(b.x), including
# mixed directions through order four. They do not use finite differences,
# the adapter implementation, or another AD backend as their oracle.
_ANALYTIC_CPU_PROGRAM = r"""
import json
import sys
import numpy as np
from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear.autodiff import JaxDerivativeProvider, TorchDerivativeProvider

framework = sys.argv[1]
if framework == 'jax':
    import jax
    import jax.numpy as xp
    jax.config.update('jax_enable_x64', True)
    Provider = JaxDerivativeProvider
    version = jax.__version__
    assert all(device.platform == 'cpu' for device in jax.devices())
    def stack(values): return xp.stack(values)
else:
    import torch
    xp = torch
    Provider = TorchDerivativeProvider
    version = torch.__version__
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    def stack(values): return torch.stack(values)

rng = np.random.default_rng(182613)
checks = 0
max_normalized_error = 0.0
for dtype in ('float32', 'float64', 'complex128'):
    tolerance = 4e-5 if dtype == 'float32' else 5e-12
    for dimension in (1, 2, 8):
        a = np.linspace(0.05, 0.2, dimension)
        b = np.linspace(-0.3, 0.15, dimension)
        # A scalar constant vector supported by the backend avoids converting
        # tracers or dual tensors to NumPy inside the mathematical function.
        if framework == 'jax':
            aa, bb = xp.asarray(a, dtype=dtype), xp.asarray(b, dtype=dtype)
        else:
            aa, bb = torch.tensor(a, dtype=getattr(torch, dtype)), torch.tensor(b, dtype=getattr(torch, dtype))
        function = lambda x: stack((xp.exp((x * aa).sum()), xp.sin((x * bb).sum())))
        provider = Provider(function, dimension, 2, max_order=4, dtype=dtype)
        for scale in (0.01, 1.0, 3.0):
            for sample in range(2):
                point = rng.normal(size=dimension) * scale
                directions = tuple(rng.normal(size=dimension) for _ in range(4))
                if dtype == 'complex128':
                    point = point + 0.1j * rng.normal(size=dimension)
                    directions = tuple(direction + 0.1j * rng.normal(size=dimension) for direction in directions)
                expected_value = np.array([np.exp(a @ point), np.sin(b @ point)])
                observed_value = provider.value(point)
                np.testing.assert_allclose(observed_value, expected_value, rtol=tolerance, atol=tolerance)
                checks += 1
                for order in range(1, 5):
                    expected = np.array([
                        np.exp(a @ point) * np.prod([a @ direction for direction in directions[:order]]),
                        np.sin(b @ point + order * np.pi / 2) * np.prod([b @ direction for direction in directions[:order]]),
                    ])
                    observed = provider.derivative(order, point, directions[:order])
                    error = float(np.max(np.abs(observed - expected) / (1.0 + np.abs(expected))))
                    max_normalized_error = max(max_normalized_error, error)
                    np.testing.assert_allclose(observed, expected, rtol=tolerance, atol=tolerance)
                    checks += 1
                zero_direction = np.zeros(dimension, dtype=complex if dtype == 'complex128' else float)
                np.testing.assert_allclose(provider.derivative(4, point, (*directions[:3], zero_direction)), 0.0, atol=tolerance)
                checks += 1
                if sample == 0 and scale == 1.0:
                    certificate = provider.validation_certificate(point, directions[:3], tolerance=10 * tolerance)
                    certificate.require()
                    assert certificate.diagnostics['checked_orders'] == (1, 2, 3)
                    checks += 1

# sqrt has nonfinite tangents outside its real domain, while log can return
# a nonfinite primal with finite tangents. Neither may be certified.
for operation in (xp.sqrt, xp.log):
    bad = Provider(lambda x: stack((operation(x[0]),)), 1, 1, max_order=2, dtype='float64')
    certificate = bad.validation_certificate(np.array([-1.0]), (np.array([1.0]), np.array([2.0])))
    assert not certificate.current_valid
    try:
        certificate.require()
    except ValidationError:
        pass
    else:
        raise AssertionError('nonfinite backend certificate was accepted')
    checks += 1

# Unsupported differentiation dtypes and absent dtype names must fail.
for dtype in ('int64', 'komorebi_nonexistent_dtype'):
    invalid = Provider(lambda x: stack((x[0] ** 2,)), 1, 1, dtype=dtype)
    try:
        invalid.derivative(1, [1.0], ([1.0],))
    except (TypeError, ValueError, ValidationError):
        pass
    else:
        raise AssertionError('unsupported dtype accepted: ' + dtype)
    checks += 1

print(json.dumps({'framework': framework, 'version': version, 'cpu_only': True,
                  'checks': checks, 'max_normalized_error': max_normalized_error,
                  'seed': 182613, 'passed': True}), flush=True)
"""


@pytest.mark.optional_backend
@pytest.mark.parametrize("framework", ["jax", "torch"])
def test_optional_cpu_ad_matches_independent_analytic_controls(framework) -> None:
    status = ad_framework_diagnostics()[framework]
    if not status["available"]:
        pytest.skip(f"{framework} unavailable: {status['detail']}")
    result = _run_child(_ANALYTIC_CPU_PROGRAM, framework)
    assert result.returncode == 0, (result.stdout + result.stderr)[-8000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["passed"] is True
    assert report["checks"] == 337
    print(json.dumps(report, sort_keys=True))
