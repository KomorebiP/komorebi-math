"""Exact worked example from LSM Foundations 0.8.0, Sections 13 and 18."""

from fractions import Fraction as F

from komorebi_math.signals.foundations import (
    compression_closure,
    delay_closure,
    identify_data_span,
    minimal_predictive_completion,
    projected_memory,
)


def main() -> None:
    operator = [[F(1, 2), 1, 0], [F(1, 4), F(1, 3), 0], [0, 0, F(5, 6)]]
    retained = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    obstruction = compression_closure(operator, retained)
    assert not obstruction.closed
    assert obstruction.composition_defect[0, 0] == F(1, 4)

    completion = minimal_predictive_completion([[1, 0, 0]], {"L": operator}).require_complete()
    assert completion.basis.rows == 2  # The independent third coordinate is irrelevant.
    delay = delay_closure(operator, [[1, 0, 0]], delay=2)
    known_transition = delay.require_transition()

    # This inference routine receives paired observed windows, not the generator.
    fit = identify_data_span([[1, F(1, 2)], [F(1, 2), F(1, 2)]], [[F(1, 2), F(1, 2)], [F(1, 2), F(11, 24)]])
    assert fit.require_closed_span() == known_transition
    assert fit.global_coverage == "not_established"
    predicted = fit.require_fit() * fit.successor_states[:, -1]
    assert predicted[-1] == F(61, 144)

    memory = projected_memory(operator, retained, horizon=2)
    next_retained = memory.next_state([[1, 0, 0], [F(1, 2), 0, 0], [F(1, 2), 0, 0]], [0, 0, 0])
    assert next_retained[0] == F(11, 24)
    print("Return-path defect:", obstruction.composition_defect[0, 0])
    print("Minimal predictive dimension:", completion.basis.rows)
    print("Exact delay transition:", known_transition)
    print("Next withheld observed output:", predicted[-1])
    print("Memory reconstruction:", next_retained[0])
    print("Unknown-process coverage:", fit.global_coverage)


if __name__ == "__main__":
    main()
