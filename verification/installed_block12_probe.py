"""Installed-wheel smoke probe for the Block 12 research-tier support packet."""

from __future__ import annotations

import json

import numpy as np

from komorebi_math.nonlinear.convex_packets import polyhedral_support_packet


def main() -> None:
    diamond = polyhedral_support_packet(
        np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    )
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 1.0], [0.0, -1.0]])
    envelope = diamond.bellman_envelope([first, second])
    checks = {
        "support_fixed_point": abs(envelope.support(np.array([2.0, -0.5])) - 2.0) < 1e-12,
        "x_branch": diamond.branch_query([first, second], np.array([2.0, 1.0])).active_branches == (0,),
        "y_branch": diamond.branch_query([first, second], np.array([1.0, 2.0])).active_branches == (1,),
        "tie_branches": diamond.branch_query([first, second], np.array([1.0, 1.0])).active_branches == (0, 1),
        "provenance": set(envelope.query(np.array([1.0, 1.0])).active_provenance) == {(0,), (1,)},
    }
    print(json.dumps({"checks": checks, "passed": all(checks.values())}, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
