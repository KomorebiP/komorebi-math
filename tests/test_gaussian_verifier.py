"""Independent Schur-complement, packet-energy and enclosure controls."""

import random
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from itertools import combinations

import numpy as np
import pytest

from komorebi_math.dynamics.gaussian_verifier import (
    FactorGaussianModel,
    GaussianBudget,
    check_gaussian_tolerance,
    evaluate_gaussian_gain,
    exact_gaussian_posterior,
    gaussian_packet_energy,
    outward_dyadic,
    verify_gaussian_gain,
)
from komorebi_math.exceptions import DimensionError, ResourceLimitError, ValidationError

sp = pytest.importorskip("sympy")


def singular_model(**overrides):
    arguments = dict(
        prior_factor=[[1], [0]],
        observation_operator=[[1, 1]],
        noise_factor=[[]],
        noise_floor=1,
        prior_mean=[0, 0],
        observation=[1],
    )
    arguments.update(overrides)
    return FactorGaussianModel(**arguments)


def test_source_singular_example_and_distinct_tolerances():
    model = singular_model()
    gain = [[F(1, 3)], [0]]
    evaluated = evaluate_gaussian_gain(model, gain)
    assert evaluated.mean == (F(1, 3), F(0))
    assert evaluated.covariance == ((F(5, 9), F(0)), (F(0), F(0)))
    assert evaluated.residual_energy == F(1, 9) and evaluated.score == F(2, 9)
    target = exact_gaussian_posterior(model)
    assert target.gain == ((F(1, 2),), (F(0),)) and target.score == 0
    assert check_gaussian_tolerance(model, gain, F(1, 2)).valid
    assert not check_gaussian_tolerance(model, gain, F(1, 10)).valid
    coarse = verify_gaussian_gain(model, gain, tolerance=F(1, 2), precision=0)
    assert coarse.score_interval.lower == 0 and coarse.score_interval.upper == 1
    assert not coarse.certificate.valid
    fine = verify_gaussian_gain(model, gain, tolerance=F(1, 2), precision=8, output_radius=F(1, 100))
    assert fine.certificate.valid
    assert fine.covariance_intervals[0][0].lower <= F(5, 9) <= fine.covariance_intervals[0][0].upper
    # The interval describes the returned Joseph covariance 5/9, not target 1/2.
    assert fine.covariance_intervals[0][0].lower > F(1, 2)
    assert not verify_gaussian_gain(
        model, gain, tolerance=F(1, 2), precision=8, output_radius=F(1, 10_000)
    ).certificate.valid


@pytest.mark.parametrize("seed", range(32))
def test_exact_posterior_and_candidates_against_independent_full_inverse(seed):
    rng = random.Random(704 + seed)
    n, p = 2 + seed % 2, 1 + seed % 2
    prior_factor = [[F(rng.randrange(-2, 3), 3) for _ in range(2)] for _ in range(n)]
    noise_factor = [[F(rng.randrange(-2, 3), 2)] for _ in range(p)]
    h = [[rng.randrange(-2, 3) for _ in range(n)] for _ in range(p)]
    m = [F(rng.randrange(-2, 3), 2) for _ in range(n)]
    y = [F(rng.randrange(-2, 3), 3) for _ in range(p)]
    sigma = F(1 + seed % 3, 3)
    b = [[F(rng.randrange(-2, 3), 3) for _ in range(p)] for _ in range(n)]
    model = FactorGaussianModel(
        prior_factor=prior_factor,
        observation_operator=h,
        noise_factor=noise_factor,
        noise_floor=sigma,
        prior_mean=m,
        observation=y,
    )
    # Independent reference uses a full SymPy inverse and Schur complement.
    l_ref, n_ref, h_ref, b_ref = map(sp.Matrix, (prior_factor, noise_factor, h, b))
    p_ref, r_ref = l_ref * l_ref.T, n_ref * n_ref.T + sigma * sp.eye(p)
    s_ref = h_ref * p_ref * h_ref.T + r_ref
    k_ref = p_ref * h_ref.T * s_ref.inv()
    covariance_target = p_ref - k_ref * h_ref * p_ref
    posterior = exact_gaussian_posterior(model)
    assert sp.Matrix(posterior.gain) == k_ref
    assert sp.Matrix(posterior.covariance) == covariance_target
    evaluated = evaluate_gaussian_gain(model, b)
    delta = k_ref - b_ref
    innovation = sp.Matrix(y) - h_ref * sp.Matrix(m)
    covariance_candidate = covariance_target + delta * s_ref * delta.T
    assert sp.Matrix(evaluated.covariance) == covariance_candidate
    assert sp.Matrix(evaluated.mean) == sp.Matrix(m) + b_ref * innovation
    residual_ref = p_ref * h_ref.T - b_ref * s_ref
    direct_energy = sum(entry**2 for entry in residual_ref)
    direct_score = direct_energy * (sigma + (innovation.T * innovation)[0]) / sigma**2
    assert evaluated.residual_energy == direct_energy and evaluated.score == direct_score
    coupling_cost = (delta * innovation).dot(delta * innovation) + sp.trace(delta * s_ref * delta.T)
    assert 0 <= coupling_cost <= evaluated.score
    # All principal minors provide an independent finite PSD control.
    for size in range(1, n + 1):
        for indices in combinations(range(n), size):
            assert covariance_candidate.extract(indices, indices).det() >= 0


def test_zero_prior_is_supported_and_arbitrary_gain_stays_a_candidate():
    model = singular_model(prior_factor=[[], []])
    candidate = evaluate_gaussian_gain(model, [[1], [-1]])
    assert candidate.mean == (F(1), F(-1))
    assert candidate.covariance == ((F(1), F(-1)), (F(-1), F(1)))
    assert candidate.score == 4 and not check_gaussian_tolerance(model, candidate.gain, 1).valid
    target = exact_gaussian_posterior(model)
    assert target.mean == (F(0), F(0)) and target.score == 0
    assert check_gaussian_tolerance(model, target.gain, 0).valid


def test_rotated_packet_energy_against_direct_component_scalar_sums():
    model = FactorGaussianModel(
        prior_factor=[[1], [1]],
        observation_operator=[[1, 0], [0, 1]],
        noise_factor=[[], []],
        noise_floor=1,
        prior_mean=[0, 0],
        observation=[1, -1],
    )
    gain = [[F(1, 3), 0], [F(1, 5), F(-1, 7)]]
    q = [[[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]], [[F(1, 2), F(-1, 2)], [F(-1, 2), F(1, 2)]]]
    d = [[[1, 0], [0, 0]], [[0, 0], [0, 1]]]
    residual = evaluate_gaussian_gain(model, gain).residual
    result = gaussian_packet_energy(model, gain, output_packets=q, input_packets=d)
    expected = tuple(
        tuple(
            sum(
                sum(output[i][k] * residual[k][ell] * inner[ell][j] for k in range(2) for ell in range(2))
                ** 2
                for i in range(2)
                for j in range(2)
            )
            for inner in d
        )
        for output in q
    )
    assert result.energies == expected
    assert result.total == sum(value**2 for row in residual for value in row)
    result.certificate.require()


@pytest.mark.parametrize(
    "packets",
    [
        [[[1, 0], [0, 0]]],
        [[[1, 1], [0, 0]], [[0, -1], [0, 1]]],
        [[[1, 0], [0, 1]], [[1, 0], [0, 1]]],
        [[[F(1, 2), 0], [0, F(1, 2)]], [[F(1, 2), 0], [0, F(1, 2)]]],
    ],
)
def test_false_partial_oblique_overlapping_or_nonidempotent_packets_rejected(packets):
    with pytest.raises(ValidationError):
        gaussian_packet_energy(singular_model(), [[0], [0]], output_packets=packets, input_packets=[[[1]]])


@pytest.mark.parametrize("value", [F(-7, 3), F(-1, 1025), F(0), F(1, 3), F(2, 9), F(5, 9), F(11)])
@pytest.mark.parametrize("precision", [0, 1, 8, 17])
def test_dyadic_endpoints_against_independent_integer_floor_and_ceiling(value, precision):
    interval = outward_dyadic(value, precision)
    scale = 2**precision
    lower_integer = (value * scale).__floor__()
    upper_integer = -((-value * scale).__floor__())
    assert interval.lower == F(lower_integer, scale) and interval.upper == F(upper_integer, scale)
    assert interval.lower <= value <= interval.upper
    assert interval.radius <= F(1, 2 * scale)
    assert interval.center - interval.radius == interval.lower
    assert interval.center + interval.radius == interval.upper


def test_exact_threshold_and_independent_recomputation_ignore_modified_diagnostics():
    model = singular_model(observation=[0])
    gain = [[F(1, 4)], [0]]
    evaluated = evaluate_gaussian_gain(model, gain)
    assert evaluated.score == F(1, 4)
    assert check_gaussian_tolerance(model, gain, F(1, 2)).valid
    assert verify_gaussian_gain(model, gain, tolerance=F(1, 2), precision=2).certificate.valid
    modified = replace(evaluated, score=F(0), residual_energy=F(0))
    modified.certificate.diagnostics["score"] = 0
    assert not verify_gaussian_gain(model, modified.gain, tolerance=F(1, 10), precision=8).certificate.valid


def test_inputs_and_outputs_are_owned_immutable_rationals():
    factor, h, observed = [[1], [0]], [[1, 1]], [1]
    model = singular_model(prior_factor=factor, observation_operator=h, observation=observed)
    factor[0][0] = h[0][0] = observed[0] = 99
    assert model.prior_factor == ((F(1),), (F(0),))
    assert model.observation_operator == ((F(1), F(1)),) and model.observation == (F(1),)
    gain = [[F(1, 3)], [0]]
    evaluated = evaluate_gaussian_gain(model, gain)
    gain[0][0] = 99
    assert evaluated.gain[0][0] == F(1, 3)
    with pytest.raises(TypeError):
        evaluated.covariance[0][0] = 99
    with pytest.raises(FrozenInstanceError):
        model.noise_floor = 0
    assert singular_model(prior_factor=np.array([[1], [0]], dtype=int)).prior_factor == model.prior_factor


@pytest.mark.parametrize("bad", [0.0, float("nan"), True, "1/3", sp.sqrt(2), sp.Symbol("x"), 1j])
def test_inexact_and_symbolic_inputs_are_rejected(bad):
    with pytest.raises(ValidationError):
        singular_model(noise_floor=bad)
    with pytest.raises(ValidationError):
        evaluate_gaussian_gain(singular_model(), [[bad], [0]])
    with pytest.raises(ValidationError):
        outward_dyadic(bad, 8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"noise_floor": 0},
        {"noise_floor": -1},
        {"observation_operator": [[1]]},
        {"noise_factor": [[], []]},
        {"prior_mean": [0]},
        {"observation": [0, 1]},
        {"prior_factor": []},
        {"prior_factor": [[1], [0, 1]]},
    ],
)
def test_nonpositive_floor_and_malformed_model_rejected(overrides):
    with pytest.raises((ValidationError, DimensionError)):
        singular_model(**overrides)


@pytest.mark.parametrize(
    "call",
    [
        lambda: singular_model(budget=GaussianBudget(max_dimension=1)),
        lambda: singular_model(budget=GaussianBudget(max_exact_bits=1)),
        lambda: singular_model(prior_factor=[[2**20], [0]], budget=GaussianBudget(max_exact_bits=8)),
        lambda: outward_dyadic(F(1, 3), 9, budget=GaussianBudget(max_precision=8)),
        lambda: outward_dyadic(1, 8, budget=GaussianBudget(max_exact_bits=8)),
        lambda: outward_dyadic(F(2**7), 1, budget=GaussianBudget(max_exact_bits=8)),
        lambda: outward_dyadic(F(1, 3), True),
        lambda: verify_gaussian_gain(singular_model(), [[0], [0]], tolerance=-1, precision=8),
        lambda: verify_gaussian_gain(singular_model(), [[0], [0]], tolerance=1, precision=8, output_radius=0),
        lambda: evaluate_gaussian_gain(singular_model(), [[1, 0], [0, 1]]),
    ],
)
def test_resource_precision_parameter_and_gain_dimension_guards(call):
    with pytest.raises((ValidationError, ResourceLimitError, DimensionError)):
        call()


def test_oversized_array_rejected_before_list_copy():
    class NoListCopy(np.ndarray):
        def tolist(self):
            raise AssertionError("oversized Gaussian array reached tolist")

    factor = np.zeros((33, 1), dtype=int).view(NoListCopy)
    with pytest.raises(ResourceLimitError):
        singular_model(prior_factor=factor)


def test_tiny_nonzero_residual_never_underflows_to_zero():
    model = singular_model(observation=[0])
    gain = [[F(1, 2) + F(1, 10**100)], [0]]
    evaluated = evaluate_gaussian_gain(model, gain)
    assert evaluated.residual_energy > 0
    assert not check_gaussian_tolerance(model, gain, 0).valid
