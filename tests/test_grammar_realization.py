from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from komorebi_math import OperatorGrammar, OperatorRealization
from komorebi_math.exceptions import RelationError, ValidationError


def test_typed_path_endpoints_and_identity() -> None:
    grammar = OperatorGrammar(["X", "Y", "Z"], {"a": ("X", "Y"), "b": ("Y", "Z")})
    word = grammar.word("a", "b")
    assert word.source == "X"
    assert word.target == "Z"
    assert word.length == 2
    identity = grammar.identity("Y")
    assert identity.is_identity


def test_noncomposable_path_rejected() -> None:
    grammar = OperatorGrammar(["X", "Y"], {"a": ("X", "Y"), "b": ("X", "Y")})
    with pytest.raises(ValidationError):
        grammar.word("a", "b")


def test_relation_terms_must_be_parallel() -> None:
    grammar = OperatorGrammar(["X", "Y"], {"a": ("X", "Y")})
    with pytest.raises(RelationError):
        grammar.relation("bad", [(1, grammar.word("a")), (-1, grammar.identity("X"))])


def test_numpy_realization_relation() -> None:
    grammar = OperatorGrammar(["X"], {"T": ("X", "X")})
    grammar.relation(
        "cubic",
        [(1, grammar.word("T", "T", "T")), (-1, grammar.word("T"))],
    )
    realization = OperatorRealization(grammar, {"X": 3}, {"T": np.diag([-1, 0, 1])})
    certificate = realization.relation_certificate()
    assert certificate.valid
    assert certificate.residuals["cubic"] == 0


def test_invalid_realization_relation_rejected() -> None:
    grammar = OperatorGrammar(["X"], {"T": ("X", "X")})
    grammar.relation("square_zero", [(1, grammar.word("T", "T"))])
    with pytest.raises(RelationError):
        OperatorRealization(grammar, {"X": 2}, {"T": np.eye(2)})


def test_exact_sympy_realization() -> None:
    grammar = OperatorGrammar(["X"], {"N": ("X", "X")})
    grammar.relation("nilpotent", [(1, grammar.word("N", "N"))])
    realization = OperatorRealization(
        grammar,
        {"X": 2},
        {"N": sp.Matrix([[0, 1], [0, 0]])},
        backend="sympy",
    )
    assert realization.relation_certificate().exactness.value == "exact"


def test_path_enumeration() -> None:
    grammar = OperatorGrammar(["X", "Y"], {"a": ("X", "Y"), "b": ("Y", "X")})
    loops = grammar.paths(source="X", target="X", max_length=2)
    strings = {str(word) for word in loops}
    assert "id_X" in strings
    assert "a · b" in strings
