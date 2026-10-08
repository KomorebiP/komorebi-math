"""Independent regressions for frame nullspaces, integer arithmetic and state transport."""

import numpy as np
import pytest

from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.signals.denoise import haar_decompose
from komorebi_math.signals.edge_recovery import recover_jumps, recover_jumps_subgaussian
from komorebi_math.signals.filterbanks import FiniteFilterBank, HaarFilterBank, PeriodicFilterBank
from komorebi_math.signals.filtering import StreamingFIR, fir_filter
from komorebi_math.signals.recovery import detect_piecewise_polynomial_breaks, recover_separated_jumps
from komorebi_math.signals.recursive import LinearFilter, StreamingLinearFilter
from komorebi_math.signals.wavelets import HaarPacketTree


@pytest.mark.parametrize("analysis", [np.array([[1.0, 0.0]]), np.array([[1j, 0j, 0j]])])
def test_lower_frame_bound_includes_invisible_input_directions(analysis):
    report = FiniteFilterBank(analysis, analysis.conj().T).report()
    null_input = np.zeros(analysis.shape[1])
    null_input[-1] = 1
    assert np.linalg.norm(analysis @ null_input) == 0
    assert report.lower_frame_bound == 0
    assert report.upper_frame_bound == pytest.approx(1)


def test_periodic_undersampled_bank_has_zero_lower_frame_bound():
    bank = PeriodicFilterBank(([1.0],), ([1.0],), 2)
    lower, upper = bank.frame_bounds(4)
    assert lower == 0
    assert upper == pytest.approx(1)
    np.testing.assert_array_equal(bank.analyze([0, 1, 0, 0])[0], [0, 0])


@pytest.mark.parametrize("dtype,values", [(np.uint8, [255, 255]), (np.int8, [100, 100])])
@pytest.mark.parametrize("normalized", [True, False])
def test_haar_pair_arithmetic_does_not_wrap(dtype, values, normalized):
    source = np.array(values, dtype=dtype)
    bank = HaarFilterBank(normalized)
    low, high = bank.analyze(source)
    divisor = np.sqrt(2) if normalized else 1
    np.testing.assert_allclose(low, [(float(values[0]) + float(values[1])) / divisor])
    np.testing.assert_allclose(high, [0])
    result = bank.certificate(source)
    np.testing.assert_allclose(result.value, values)
    assert result.certificate.valid


def test_integer_haar_synthesis_promotes_before_adding_channels():
    output = HaarFilterBank(False).synthesize((np.array([200], np.uint8), np.array([100], np.uint8)))
    np.testing.assert_allclose(output, [150, 50])


def test_integer_multilevel_and_packet_haar_roundtrip_and_energy():
    source = np.array([255, 0, 254, 1, 253, 2, 252, 3], dtype=np.uint8)
    reference_energy = sum(float(value) ** 2 for value in source)
    decomposition = haar_decompose(source, 3)
    np.testing.assert_allclose(decomposition.reconstruct(), source, atol=1e-12)
    assert decomposition.certificate(source).valid
    tree = HaarPacketTree.analyze(source, 3)
    np.testing.assert_allclose(tree.reconstruct(), source, atol=1e-12)
    assert tree.energy() == pytest.approx(reference_energy)
    assert tree.certificate().valid


@pytest.mark.parametrize("chunk_values", [([1j], [0.0]), ([1j], [], [0.0]), ([1.0], [1j], [0.0])])
def test_fir_continuation_retains_complex_history_across_real_chunks(chunk_values):
    processor = StreamingFIR([1.0, 1.0])
    output = np.concatenate([processor.process(chunk) for chunk in chunk_values])
    source = np.concatenate([np.asarray(chunk) for chunk in chunk_values])
    expected = np.convolve(source, [1.0, 1.0])[:source.size]
    np.testing.assert_allclose(output, expected)
    np.testing.assert_allclose(np.concatenate((output, processor.flush())), np.convolve(source, [1.0, 1.0]))


def test_integer_fir_multiplication_does_not_wrap():
    source = np.array([255, 255], dtype=np.uint8)
    taps = np.array([2, 2], dtype=np.uint8)
    expected = np.array([510.0, 1020.0, 510.0])
    np.testing.assert_array_equal(fir_filter(source, taps, mode="full"), expected)
    processor = StreamingFIR(taps)
    np.testing.assert_array_equal(np.concatenate((processor.process(source[:1]),
                                                  processor.process(source[1:]), processor.flush())), expected)


def test_iir_continuation_retains_complex_state_across_real_chunks():
    filter_ = LinearFilter([1], [1, -.5])
    processor = StreamingLinearFilter(filter_)
    output = np.concatenate((processor.process([1j]), processor.process([]), processor.process([0.0, 0.0])))
    np.testing.assert_allclose(output, [1j, .5j, .25j])


@pytest.mark.parametrize("state", [[1j], np.array([1j])])
def test_iir_complex_initial_state_with_real_input(state):
    output = LinearFilter([1], [1, -.5]).process([0.0, 0.0], initial_state=state)
    np.testing.assert_allclose(output.samples, [1j, .5j])
    np.testing.assert_allclose(output.final_state, [.25j])


def test_uint_recovery_detects_downward_jump_with_correct_amplitude():
    result = recover_jumps(np.array([255, 0], np.uint8), spacing=1, lipschitz_bound=0,
                           noise_bound=0, minimum_jump=250)
    assert result.certificate.valid
    assert result.cells == (0,)
    np.testing.assert_array_equal(result.amplitudes, [-255])


def test_complex_vector_recovery_preserves_jump_components():
    result = recover_jumps(np.array([[0j, 0j], [3j, 4.0]]), spacing=1, lipschitz_bound=0,
                           noise_bound=0, minimum_jump=4)
    assert result.certificate.valid
    assert result.cells == (0,)
    np.testing.assert_array_equal(result.amplitudes, [[3j, 4.0]])


@pytest.mark.parametrize("api", [recover_jumps, recover_separated_jumps])
@pytest.mark.parametrize("samples", [[0, float("nan"), 100], [0, float("inf")], [0, -float("inf")]])
def test_recovery_rejects_nonfinite_data_before_certification(api, samples):
    with pytest.raises(ValidationError, match="finite"):
        api(samples, spacing=1, lipschitz_bound=0, noise_bound=0, minimum_jump=10)


@pytest.mark.parametrize("origin", [float("nan"), float("inf"), -float("inf")])
def test_recovery_rejects_nonfinite_grid_origin(origin):
    with pytest.raises(ValidationError, match="origin"):
        recover_jumps([0, 100], spacing=1, lipschitz_bound=0, noise_bound=0, minimum_jump=10, origin=origin)


def test_recovery_rejects_empty_vector_components():
    with pytest.raises(DimensionError, match="component"):
        recover_jumps_subgaussian(np.empty((2, 0)), spacing=1, lipschitz_bound=0,
                                  sigma=1, minimum_jump=10, failure_probability=.1)


def test_recovery_rejects_nonfinite_separation_declaration():
    with pytest.raises(ValidationError, match="minimum_separation"):
        recover_separated_jumps([0, 100], spacing=1, lipschitz_bound=0, noise_bound=0,
                                 minimum_jump=10, minimum_separation=float("nan"))


def test_piecewise_polynomial_unsigned_jump_differences_are_not_modular():
    result = detect_piecewise_polynomial_breaks(np.array([255, 0], np.uint8), degree=0,
                                              noise_bound=0, threshold=100)
    assert result.indices == (0,)
    np.testing.assert_array_equal(result.response, [-255])


@pytest.mark.parametrize("values", [[0, float("nan")], [0, float("inf")]])
def test_haar_and_packet_haar_reject_nonfinite_inputs(values):
    with pytest.raises(ValidationError, match="finite"):
        HaarFilterBank().analyze(values)
    with pytest.raises(ValidationError, match="finite"):
        HaarPacketTree.analyze(values, 1)


def test_haar_certificate_respects_signal_scaling():
    source = np.array([1, -2, 3, 4.0]) * 1e12
    assert HaarFilterBank().certificate(source).certificate.valid
    assert HaarPacketTree.analyze(source, 2).certificate().valid

