"""Replay finite paper machinery plus an exact signal closure worked example.

Install komorebi-math[exact] and run this script offline. A missing required
extra is recorded as a skipped check and yields nonzero exit status.
"""

from fractions import Fraction

from komorebi_math.audit import CheckOutcome, SkipCheck, StressCheck, paper_checks, run_stress
from komorebi_math.audit.special_function_checks import special_function_checks
from komorebi_math.certificates import Exactness


def predictive_return_path(context):
    try:
        import sympy  # noqa: F401
    except ImportError as exc:
        raise SkipCheck("the exact worked example requires the SymPy extra") from exc
    from komorebi_math.signals.foundations import compression_closure, delay_closure

    operator = [[Fraction(1, 2), 1, 0], [Fraction(1, 4), Fraction(1, 3), 0], [0, 0, Fraction(5, 6)]]
    compression = compression_closure(operator, [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    delay = delay_closure(operator, [[1, 0, 0]], delay=2)
    transition = delay.require_transition()
    passed = (
        not compression.closed and transition[1, 0] == Fraction(1, 12) and transition[1, 1] == Fraction(5, 6)
    )
    # Convert SymPy entries to exact Fraction evidence, retaining the actual matrix.
    rows = [[Fraction(int(transition[i, j].p), int(transition[i, j].q)) for j in range(2)] for i in range(2)]
    return CheckOutcome(
        bool(passed),
        {"one_coordinate_closed": compression.closed, "two_delay_transition": rows},
        "A return path obstructs one-coordinate prediction; two exact delays close this supplied finite model.",
    )


def main():
    worked = StressCheck(
        "example.paper.predictive-return-path",
        "worked",
        predictive_return_path,
        Exactness.EXACT,
        {"ambient_dimension": 3, "observed_dimension": 1, "delay": 2},
    )
    report = run_stress(
        (*paper_checks(8), *special_function_checks(8), worked),
        seed=20261010,
        timeout_seconds=30,
        total_timeout_seconds=180,
    )
    print(report.to_json())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
