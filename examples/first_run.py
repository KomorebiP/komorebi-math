"""Run two finite examples and print their expected results.

Run with the interpreter into which Komorebi Math was installed.
"""
from fractions import Fraction

import numpy as np

from komorebi_math.stable import ExactCoordinateTransform, PacketSystem


def main():
    packets = PacketSystem(
        [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])],
        names=("left", "right"),
    )
    packets.certificate().require()
    vector = np.array([2.0, 3.0])
    reconstructed = packets.analyze(vector).reconstruct()
    np.testing.assert_array_equal(reconstructed, vector)

    transform = ExactCoordinateTransform(
        "A", "B", ((1, Fraction(1, 2)), (0, 1)),
        "EXAMPLE-SHEAR", "exact rational shear",
    )
    exact_vector = (Fraction(2), Fraction(3))
    mapped = transform.apply(exact_vector)
    restored = transform.inverse().apply(mapped)
    assert mapped == (Fraction(7, 2), Fraction(3))
    assert restored == exact_vector
    transform.certificate().require()

    print(f"Numerical reconstruction: {reconstructed.real.tolist()}")
    print("Exact coordinate change: (" + ", ".join(map(str, mapped)) + ")")
    print("Exact inverse: (" + ", ".join(map(str, restored)) + ")")
    print("PASS: first examples")


if __name__ == "__main__":
    main()
