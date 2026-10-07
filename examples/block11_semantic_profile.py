from __future__ import annotations

import json

import numpy as np

from komorebi_math import PacketSystem
from komorebi_math.golden import run_behavioral_golden_fixture
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    packet_jet_support,
    profile_packet_jet_runtime,
)


tensor = np.zeros((2, 2, 2))
tensor[0, 0, 0] = 2.0
tensor[1, 1, 1] = -3.0
provider = PolynomialDerivativeProvider(
    {2: tensor},
    source_dimension=2,
    target_dimension=2,
)
packets = PacketSystem(
    (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
    names=("x", "y"),
)
family = PacketJetFamily(provider, np.zeros(2), packets, packets, 2)
directions = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))
support = packet_jet_support(family)
profile = profile_packet_jet_runtime(
    family,
    directions,
    support=support,
    jet=family.jet("x", ("x", "x")),
    repetitions=2,
    warmup=1,
    include_stream_write=True,
    stream_chunk_size=1,
)
golden = run_behavioral_golden_fixture()
profile.certificate.require()
golden.certificate.require()
print(
    json.dumps(
        {
            "exact_counts": profile.estimate.to_dict(),
            "observational_stages": [stage.to_dict() for stage in profile.stages],
            "environment": profile.environment,
            "golden_fixture_digest": golden.fixture_digest,
            "golden_cases": len(golden.results),
        },
        indent=2,
        sort_keys=True,
    )
)
