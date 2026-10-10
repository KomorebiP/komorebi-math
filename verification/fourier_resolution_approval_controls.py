"""Independent controls for Fourier fit versus conditional recovery approval.

Uses direct integrals of constant segments, never packet.fourier(), to generate
the observed coefficients. The driver tests a bounded set of regressions and
does not modify the package. Exit zero means these assertions passed, not a
universal noise or superresolution theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def integrated_coefficients(locations, amplitudes, *, period, degree, bound):
    """Integrate the right-continuous step levels on each constant segment."""
    pairs = sorted(zip(locations, amplitudes, strict=True))
    mean = sum(a * (1 - x / period) for x, a in pairs)
    breaks = [0.0] + [x for x, _ in pairs] + [period]
    levels = [-mean]
    for _, jump in pairs:
        levels.append(levels[-1] + jump)
    answer = {}
    for index in range(degree + 1, degree + 1 + 2 * bound):
        omega = 2 * np.pi * index / period
        answer[index] = complex(sum(
            level * (np.exp(-1j * omega * right) - np.exp(-1j * omega * left))
            / (-1j * omega) / period
            for left, right, level in zip(breaks[:-1], breaks[1:], levels, strict=True)
        ))
        answer[-index] = answer[index].conjugate()
    return answer


def step_value(x, locations, amplitudes, period):
    return sum(a for tau, a in zip(locations, amplitudes, strict=True) if x >= tau) - sum(
        a * (1 - tau / period) for tau, a in zip(locations, amplitudes, strict=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'src')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import komorebi_math
    from komorebi_math.certificates import Exactness
    from komorebi_math.exceptions import AssumptionError, ValidationError
    from komorebi_math.signals.fourier_recovery import PeriodicJumpPacket, recover_fourier_jumps

    began = time.monotonic()
    module_path = args.source / 'komorebi_math/signals/fourier_recovery.py'
    initial_hash = hashlib.sha256(module_path.read_bytes()).hexdigest()
    rows = []
    defaults = dict(background_degree=3, maximum_jumps=5, period=5.0)

    def check(name, assertion):
        try:
            detail = assertion()
            rows.append(dict(name=name, passed=True, detail=detail))
        except Exception as error:
            rows.append(dict(name=name, passed=False, exception=type(error).__name__, message=str(error)))

    def blocked(result):
        try:
            result.require_recovery()
        except ValidationError:
            return True
        raise AssertionError('require_recovery() approved an unapproved result')

    def recovery_evidence(result):
        return dict(order=result.order, fit_valid=result.fit_certificate.current_valid,
                    recovery_valid=result.certificate.current_valid,
                    resolution_status=result.resolution_status,
                    fit_exactness=result.fit_certificate.exactness.value,
                    recovery_exactness=result.certificate.exactness.value,
                    residuals=dict(result.fit_certificate.residuals),
                    diagnostics=dict(result.fit_certificate.diagnostics))

    separated_locations = [.41, 1.92, 3.84]
    separated_amplitudes = [2, -.7, -1.3]
    separated = integrated_coefficients(separated_locations, separated_amplitudes, period=5, degree=3, bound=5)
    close_locations = [1.0, 1.00000005, 3.84]
    close_amplitudes = [2, -.7, -1.3]
    close = integrated_coefficients(close_locations, close_amplitudes, period=5, degree=3, bound=5)

    def close_inferred():
        result = recover_fourier_jumps(close, **defaults)
        assert result.order == 2
        assert result.fit_certificate.current_valid
        assert result.fit_certificate.exactness is Exactness.NUMERICAL
        assert not result.certificate.current_valid
        assert result.resolution_status == 'candidate_only'
        assert result.declared_order is None
        assert blocked(result)
        probe = (close_locations[0] + close_locations[1]) / 2
        pointwise_error = abs(float(result.packet.evaluate(probe)) - step_value(probe, close_locations, close_amplitudes, 5))
        assert pointwise_error > .69
        return recovery_evidence(result) | dict(pointwise_error=pointwise_error, probe=probe)

    check('close_pair_fit_does_not_approve_recovery', close_inferred)

    def reject_hint(modes, hint, extra=None):
        try:
            recover_fourier_jumps(modes, known_order=hint, **defaults, **(extra or {}))
        except AssumptionError as error:
            assert 'rank' in str(error).lower() or 'order' in str(error).lower()
            return dict(exception=type(error).__name__, message=str(error))
        raise AssertionError('mismatching declared and numerical orders were accepted')

    check('close_pair_known_three_rejected', lambda: reject_hint(close, 3))
    check('underdeclared_zero_cannot_use_loose_fit_tolerance', lambda: reject_hint(separated, 0, {'tolerance': 2.0}))
    check('underdeclared_two_cannot_use_loose_fit_tolerance', lambda: reject_hint(separated, 2, {'tolerance': 2.0}))
    check('overdeclared_four_rejected', lambda: reject_hint(separated, 4))

    def separated_inferred():
        result = recover_fourier_jumps(separated, **defaults)
        assert result.order == 3 and result.fit_certificate.current_valid
        assert result.resolution_status == 'candidate_only' and not result.certificate.current_valid
        assert blocked(result)
        return recovery_evidence(result)

    check('separated_inferred_remains_candidate', separated_inferred)

    def declared_success(modes, count, locations=None, amplitudes=None):
        result = recover_fourier_jumps(modes, known_order=count, **defaults)
        assert result.order == count and result.fit_certificate.current_valid
        assert result.certificate.current_valid
        assert result.certificate.exactness is Exactness.CONDITIONAL
        assert result.fit_certificate.exactness is Exactness.NUMERICAL
        assert result.resolution_status == 'conditional'
        assert result.declared_order == count
        assert result.require_recovery() is result.packet
        # A true flag must retain the target-order, bandlimit, and noise assumptions.
        assumption_text = ' '.join(result.certificate.assumptions_unchecked).lower()
        assert 'order' in assumption_text or 'count' in assumption_text
        assert 'background' in assumption_text or 'bandlimit' in assumption_text
        assert 'noise' in assumption_text
        if locations is not None:
            np.testing.assert_allclose(result.packet.locations, locations, rtol=0, atol=1e-10)
            np.testing.assert_allclose(result.packet.amplitudes, amplitudes, rtol=0, atol=1e-10)
        return recovery_evidence(result)

    check('separated_declared_three_conditionally_approved', lambda: declared_success(separated, 3, separated_locations, separated_amplitudes))
    two_locations, two_amplitudes = [.29, 4.13], [1.3, -1.3]
    two = integrated_coefficients(two_locations, two_amplitudes, period=5, degree=3, bound=5)
    check('exact_two_jump_conditionally_approved', lambda: declared_success(two, 2, two_locations, two_amplitudes))
    zero = {n: 0j for n in range(4, 14)}
    check('declared_zero_conditionally_approved', lambda: declared_success(zero, 0, [], []))

    def inferred_zero():
        result = recover_fourier_jumps(zero, **defaults)
        assert result.order == 0 and result.fit_certificate.current_valid
        assert not result.certificate.current_valid and result.resolution_status == 'candidate_only'
        assert blocked(result)
        np.testing.assert_array_equal(result.packet.evaluate([0, .2, 1]), [0, 0, 0])
        return recovery_evidence(result)

    check('inferred_zero_does_not_authenticate_absence_of_edges', inferred_zero)

    def moderate_noise():
        noisy = dict(separated)
        for offset, n in enumerate(range(4, 14)):
            noisy[n] += 1e-6 * complex(np.cos(offset), np.sin(2 * offset))
            noisy[-n] = noisy[n].conjugate()
        try:
            result = recover_fourier_jumps(noisy, known_order=3, **defaults)
        except AssumptionError as error:
            return dict(exception=type(error).__name__, message=str(error))
        assert not result.certificate.current_valid
        assert result.resolution_status == 'invalid'
        assert blocked(result)
        return recovery_evidence(result)

    check('moderate_noise_never_silently_approved', moderate_noise)

    def tiny_noise():
        noisy = dict(separated)
        for offset, n in enumerate(range(4, 14)):
            noisy[n] += 1e-14 * complex(np.cos(offset), np.sin(2 * offset))
            noisy[-n] = noisy[n].conjugate()
        return declared_success(noisy, 3)

    check('subtolerance_noise_is_explicitly_conditional', tiny_noise)

    def ill_conditioned():
        modes = integrated_coefficients([1, 1.0001, 3.84], [2, -.7, -1.3], period=5, degree=3, bound=5)
        try:
            result = recover_fourier_jumps(modes, known_order=3, rank_tolerance=1e-12, **defaults)
        except AssumptionError as error:
            return dict(exception=type(error).__name__, message=str(error))
        # A good moment fit cannot erase evidence from roots before unit projection.
        assert not result.fit_certificate.current_valid
        assert not result.certificate.current_valid
        assert result.resolution_status == 'invalid'
        assert blocked(result)
        assert result.fit_certificate.diagnostics['hankel_condition'] > 1e7
        return recovery_evidence(result)

    check('ill_conditioned_distinct_nodes_fail_closed', ill_conditioned)

    def source_mutation():
        inputs = dict(two)
        result = recover_fourier_jumps(inputs, known_order=2, **defaults)
        observed = result.measured_moments.copy()
        predicted = result.predicted_moments.copy()
        status = result.resolution_status
        evidence = recovery_evidence(result)
        for n in list(inputs):
            inputs[n] = 0j
        np.testing.assert_array_equal(result.measured_moments, observed)
        np.testing.assert_array_equal(result.predicted_moments, predicted)
        assert not result.measured_moments.flags.writeable
        assert not result.predicted_moments.flags.writeable
        assert result.resolution_status == status
        assert recovery_evidence(result) == evidence
        return dict(input_mutation_changed_evidence=False, resolution_status=status)

    check('measurement_mapping_mutation_does_not_change_evidence', source_mutation)

    def diagnostics_cannot_promote():
        result = recover_fourier_jumps(close, **defaults)
        # Diagnostics are descriptive metadata. Changing their mutable mapping
        # must not substitute for the immutable declared_order field or flag.
        result.certificate.diagnostics['declared_order'] = result.order
        result.certificate.diagnostics['resolution_status'] = 'conditional'
        result.fit_certificate.diagnostics['numerical_hankel_rank'] = 3
        assert result.declared_order is None
        assert result.resolution_status == 'candidate_only'
        assert not result.certificate.current_valid
        assert blocked(result)
        return dict(diagnostic_mutation_changed_approval=False)

    check('mutable_diagnostics_cannot_promote_candidate', diagnostics_cannot_promote)

    def packet_mutation():
        positions = np.array(two_locations)
        amplitudes = np.array(two_amplitudes)
        packet = PeriodicJumpPacket(positions, amplitudes, period=5)
        positions[:] = 0
        amplitudes[:] = 0
        np.testing.assert_array_equal(packet.locations, two_locations)
        np.testing.assert_array_equal(packet.amplitudes, two_amplitudes)
        assert not packet.locations.flags.writeable and not packet.amplitudes.flags.writeable
        return dict(input_mutation_changed_packet=False)

    check('packet_input_arrays_are_copied', packet_mutation)
    source_unchanged = initial_hash == hashlib.sha256(module_path.read_bytes()).hexdigest()
    payload = dict(
        contract='Finite numerical model fit versus explicit conditional recovery approval',
        non_claims=['No uniform noise theorem', 'No arbitrarily close edge resolution guarantee', 'No recovery approval authenticates its unchecked target assumptions'],
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                         komorebi_version=komorebi_math.__version__, module=str(module_path.resolve()),
                         source_sha256=initial_hash),
        controls=rows, summary=dict(count=len(rows), passed=sum(row['passed'] for row in rows),
                                   source_unchanged=source_unchanged, runtime_seconds=time.monotonic()-began),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload['summary'], sort_keys=True))
    for row in rows:
        if not row['passed']:
            print(json.dumps(row, sort_keys=True))
    return 0 if source_unchanged and all(row['passed'] for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
