"""Separate finite Frobenius constituents and select a compatible carrier."""

from __future__ import annotations

from komorebi_math.deformation import (
    CompatibleCarrier,
    FrobeniusSpine,
    finite_frobenius_separator,
)


def main() -> None:
    """Run the finite Frobenius separator example."""

    spine = FrobeniusSpine(
        {
            2: (1, -3, 2),
            3: (1, -4, 3),
        }
    )
    separator = finite_frobenius_separator(
        {
            2: (1, 1),
            3: (1, -1),
            5: (2, 0),
        }
    )
    separator.certificate.require()

    multiplicities = separator.recover((5, 1))
    carrier = CompatibleCarrier(("rho1", "rho2"), (4, 2))
    selection = carrier.select(multiplicities)

    print(
        {
            "spine_rank": spine.rank,
            "separating_places": separator.places,
            "multiplicities": multiplicities,
            "selected_carrier": selection,
        }
    )


if __name__ == "__main__":
    main()
