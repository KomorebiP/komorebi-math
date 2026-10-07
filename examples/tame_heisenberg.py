"""Construct a tame Frobenius packet and a finite Heisenberg selector."""

from __future__ import annotations

import numpy as np

from komorebi_math.langlands.local import FrobeniusOrbit, HeisenbergModel, TameWeilPacket


def main() -> None:
    """Run the local finite packet example."""

    orbit = FrobeniusOrbit(q=2, degree=3, exponent=1)
    orbit.certificate().require()
    tame = TameWeilPacket(orbit)
    tame.relation_certificate().require()

    heisenberg = HeisenbergModel(prime=3, rank=1)
    left = heisenberg.element((1,), (2,), 1)
    right = heisenberg.element((2,), (1,), 2)
    heisenberg.group_law_certificate(left, right).require()
    heisenberg.selector_certificate().require()

    print(
        {
            "frobenius_orbit": orbit.elements,
            "tame_dimension": tame.dimension,
            "determinant_orientation": complex(np.linalg.det(tame.frobenius_matrix())),
            "heisenberg_dimension": heisenberg.dimension,
            "selector_rank": int(np.linalg.matrix_rank(heisenberg.translation_selector())),
        }
    )


if __name__ == "__main__":
    main()
