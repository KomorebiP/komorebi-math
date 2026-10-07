"""Resolve a finite induced representation and inspect a Clifford descent word."""

from __future__ import annotations

from komorebi_math.langlands.finite_image import (
    CliffordStep,
    CliffordTower,
    InducedRepresentation,
    LinearCharacter,
    MonomialAction,
    cyclic_group,
)


def main() -> None:
    """Run the finite image and Clifford tower example."""

    group = cyclic_group(4)
    subgroup = group.subgroup((0, 2), name="C2")
    character = LinearCharacter(subgroup, {0: 1, 2: -1})
    representation = InducedRepresentation(group, character)
    representation.certificate().require()

    action = MonomialAction(representation)
    packets = action.symmetric_power_packets(3)
    action.packet_certificate(packets, expected_dimension=4).require()

    tower = CliffordTower(
        (
            CliffordStep(24, 12, 3, 3, "E", attached_descent=False),
            CliffordStep(12, 4, 3, 1, "I"),
            CliffordStep(4, 2, 1, 1, "E"),
        )
    )
    report = tower.report()
    report.certificate.require()

    print(
        {
            "induced_dimension": representation.dimension,
            "symmetric_packet_dimensions": tuple(packet.dimension for packet in packets),
            "clifford_word": report.word,
            "clifford_depth": report.depth,
            "unresolved_extension_steps": report.unresolved_extension_steps,
        }
    )


if __name__ == "__main__":
    main()
