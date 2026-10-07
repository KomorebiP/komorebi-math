from __future__ import annotations

import numpy as np

from komorebi_math import OperatorGrammar, OperatorRealization

grammar = OperatorGrammar(["X"], {"T": ("X", "X")})
grammar.relation(
    "cubic",
    [
        (1, grammar.word("T", "T", "T")),
        (-1, grammar.word("T")),
    ],
)

realization = OperatorRealization(
    grammar,
    {"X": 3},
    {"T": np.diag([-1.0, 0.0, 1.0])},
)

print(realization.relation_certificate().to_json())
