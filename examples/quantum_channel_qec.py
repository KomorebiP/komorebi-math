from __future__ import annotations

import numpy as np

from komorebi_math.quantum import QuantumCode, amplitude_damping

channel = amplitude_damping(0.2)
channel.certificate().require()
excited = np.diag([0, 1]).astype(complex)
assert np.allclose(np.diag(channel.apply_density(excited)), [0.2, 0.8])

identity = np.eye(2)
x = np.array([[0, 1], [1, 0]], dtype=complex)
zero = np.zeros(8)
zero[0] = 1
one = np.zeros(8)
one[-1] = 1
projector = np.outer(zero, zero) + np.outer(one, one)
errors = (
    np.eye(8),
    np.kron(np.kron(x, identity), identity),
    np.kron(np.kron(identity, x), identity),
    np.kron(np.kron(identity, identity), x),
)
code = QuantumCode(projector, errors)
code.knill_laflamme_report().certificate.require()
code.recovered_channel_certificate().require()
print("quantum channel and error correction example passed")
