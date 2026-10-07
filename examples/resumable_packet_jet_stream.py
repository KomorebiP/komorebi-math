from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from komorebi_math import PacketSystem
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    capture_packet_jet_stream,
    open_packet_jet_snapshot_stream,
)


tensor = np.array([[[1.0, 2.0], [3.0, 4.0]]])
provider = PolynomialDerivativeProvider(
    {2: tensor},
    source_dimension=2,
    target_dimension=1,
    name="resumable example",
)
source = PacketSystem((np.eye(2),), names=("all",))
target = PacketSystem((np.eye(1),), names=("all",))
family = PacketJetFamily(provider, np.zeros(2), source, target, 2)
jet = family.jet("all", ("all", "all"))
directions = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))

with tempfile.TemporaryDirectory() as directory:
    destination = Path(directory) / "quadratic.kmstream"
    partial = capture_packet_jet_stream(
        jet,
        destination,
        chunk_size=1,
        evaluation_budget=2,
    )
    complete = capture_packet_jet_stream(
        jet,
        destination,
        chunk_size=1,
        resume=True,
    )
    reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
    evaluation = reader.evaluate(directions)
    evaluation.certificate.require()
    print(
        json.dumps(
            {
                "paused_after": partial.manifest.evaluations_completed,
                "resumed": complete.resumed,
                "chunks": len(reader.manifest.chunks),
                "stored_coefficients": reader.manifest.coefficients_completed,
                "stream_sha256": reader.manifest.stream_sha256,
                "value": np.real_if_close(evaluation.value).tolist(),
            },
            indent=2,
        )
    )
