from __future__ import annotations

import json

import numpy as np

from komorebi_math import PacketSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.export import dumps_portable, loads_portable, portable_digest
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    snapshot_packet_jet,
)


def main() -> None:
    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[0, 0, 0] = 3.0
    tensor[1, 1, 1] = -2.0
    provider = PolynomialDerivativeProvider(
        {2: tensor},
        source_dimension=2,
        target_dimension=2,
    )
    packets = PacketSystem(
        (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
        names=("x", "y"),
    )
    family = PacketJetFamily(provider, np.zeros(2), packets, packets, order=2)
    jet = family.jet("x", ("x", "x"))
    snapshot = snapshot_packet_jet(jet)
    snapshot.capture_certificate.require()

    encoded = dumps_portable(snapshot)
    restored = loads_portable(encoded)
    directions = (np.array([2.0, 5.0]), np.array([-1.0, 4.0]))
    restored.comparison_certificate(jet, directions).require()

    envelope = json.loads(encoded)
    envelope["payload_sha256"] = "0" * 64
    try:
        loads_portable(json.dumps(envelope))
    except ValidationError:
        tamper_status = "rejected"
    else:
        raise AssertionError("tampered portable payload was accepted")

    print("snapshot address:", restored.address)
    print("stored coefficients:", restored.nnz)
    print("portable digest:", portable_digest(restored))
    print("evaluation:", np.asarray(restored.evaluate(directions)).tolist())
    print("tampered payload:", tamper_status)


if __name__ == "__main__":
    main()
