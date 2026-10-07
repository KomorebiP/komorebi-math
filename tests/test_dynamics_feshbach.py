from __future__ import annotations

import numpy as np

from komorebi_math.dynamics import (
    FeshbachReduction,
    MemoryKernel,
    RationalMemoryRealization,
    memory_kernel_error_bound,
    volterra_trajectory_bound,
)


def test_feshbach_resolvent_and_memory_transform() -> None:
    matrix = np.array(
        [
            [-0.4, 0.2, 0.3, 0.0],
            [0.0, -0.6, 0.0, -0.2],
            [0.4, 0.0, -1.0, 0.1],
            [0.0, 0.5, 0.0, -1.4],
        ]
    )
    reduction = FeshbachReduction(matrix, [0, 1])
    reduction.resolvent_certificate(2.0 + 0.3j).require()
    kernel = reduction.memory_kernel()
    assert np.allclose(kernel.at(0.0), reduction.retained_to_hidden @ reduction.hidden_to_retained)
    kernel.certificate(2.0 + 0.2j, horizon=20.0, points=5000).require()


def test_local_closure_criterion() -> None:
    app = np.array([[-0.4, 0.2], [0.0, -0.6]])
    aqq = np.diag([-1.0, -1.4])
    apq = np.array([[1.0, 0.0], [0.0, 0.0]])
    aqp = np.array([[0.0, 0.0], [0.0, 1.0]])
    matrix = np.block([[app, apq], [aqp, aqq]])
    reduction = FeshbachReduction(matrix, [0, 1])
    reduction.local_closure_certificate().require()


def test_rational_memory_realization() -> None:
    realization = RationalMemoryRealization(
        np.array([[0.7, -0.2]]),
        np.array([[-1.3, 1.0], [0.0, -1.3]]),
        np.array([[0.4], [0.9]]),
    )
    kernel = realization.kernel()
    assert kernel.at(0.5).shape == (1, 1)
    augmented = realization.augment([[-0.4]])
    assert augmented.shape == (3, 3)
    initial = np.array([1.0, 0.0, 0.0])
    assert realization.retained_trajectory([[-0.4]], initial, 0.2).shape == (1,)


def test_memory_error_and_volterra_bound() -> None:
    first = MemoryKernel([[1.0]], [[-1.0]], [[1.0]])
    second = MemoryKernel([[0.9]], [[-1.0]], [[1.0]])
    error = memory_kernel_error_bound(first, second, horizon=5.0, samples=1000)
    error.certificate.require()
    bound = volterra_trajectory_bound(
        error.value,
        horizon=2.0,
        trajectory_bound=1.5,
        retained_generator_norm=0.4,
        reference_kernel_l1=1.0,
    )
    assert bound >= 0
