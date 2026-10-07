"""Installed-wheel smoke probe for the Block 13 provenance packet."""
from __future__ import annotations
import json
import numpy as np
from komorebi_math.nonlinear.convex_packets import polyhedral_support_packet
from komorebi_math.nonlinear.provenance_packets import (
    finite_provenance_observer,
    compile_policy_packet,
    compile_observer_convex_packet,
)

def main() -> None:
    # Parity observer: the four-state presentation deliberately contains
    # continuation-equivalent duplicate states and must minimize to two.
    observer = finite_provenance_observer(
        ((1, 2), (0, 3), (3, 0), (2, 1)),
        ("even", "odd", "odd", "even"),
    )
    minimized = observer.minimize().observer
    packet = polyhedral_support_packet(
        np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        provenance=((0,), (1,), (0, 0), (1, 1)),
    )
    flat = compile_policy_packet(packet, observer)
    conditioned = compile_observer_convex_packet(packet, observer)
    zero = np.zeros((2, 2))
    resurrected = conditioned.linear_image(zero).query(np.array([1.0, 0.0]))
    checks = {
        "observer_minimized_to_two": minimized.state_count == 2,
        "flat_and_conditioned_values_match": abs(flat.support(np.array([0.7, -1.2])) - conditioned.support(np.array([0.7, -1.2]))) < 1e-12,
        "conditioned_packet_has_two_states": conditioned.observer.state_count == 2,
        "singular_collapse_keeps_all_observer_outputs": set(resurrected.active_outputs) == {"even", "odd"},
    }
    print(json.dumps({"checks": checks, "passed": all(checks.values())}, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
