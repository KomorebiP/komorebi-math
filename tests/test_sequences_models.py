from fractions import Fraction

import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.sequences import (
    AutomaticSequence,
    Sequence,
    apery_recurrence,
    apery_sequence,
    base_digits,
    compile_sequence,
    motzkin_recurrence,
    motzkin_sequence,
    thue_morse_automaton,
    thue_morse_sequence,
)


def test_base_digits_and_automatic_execution() -> None:
    assert base_digits(0, 2) == (0,)
    assert base_digits(13, 2) == (1, 1, 0, 1)
    model = thue_morse_automaton()
    model.certificate(1000).certificate.require()
    assert model.first(16) == (0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0)
    for exponent in range(5):
        modulus = 2**exponent
        for address in range(modulus):
            packet = model.residue_model(exponent, address)
            assert packet.first(64) == tuple(model.term(modulus * n + address) for n in range(64))


def test_automatic_validation_rejects_noncanonical_leading_zero_model() -> None:
    with pytest.raises(ValidationError, match="zero digit loop"):
        AutomaticSequence(2, ((1, 0), (1, 0)), (0, 1))


def test_holonomic_models() -> None:
    motzkin = motzkin_recurrence()
    apery = apery_recurrence()
    motzkin.certificate(30).certificate.require()
    apery.certificate(20).certificate.require()
    assert motzkin.first(11) == tuple(Fraction(v) for v in (1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188))
    assert apery.first(8) == tuple(Fraction(v) for v in (1, 5, 73, 1445, 33001, 819005, 21460825, 584307365))


def test_compiler_selects_declared_models() -> None:
    automatic = compile_sequence(thue_morse_sequence())
    assert automatic.strategy == "declared finite automaton"
    automatic.certificate.require()
    assert automatic.first(100) == thue_morse_sequence().first(100)

    motzkin = compile_sequence(motzkin_sequence())
    assert motzkin.strategy == "declared holonomic recurrence"
    motzkin.certificate.require()
    assert motzkin.first(30) == motzkin_sequence().first(30)

    apery = compile_sequence(apery_sequence())
    assert apery.strategy == "declared holonomic recurrence"
    apery.certificate.require()
    assert apery.first(16) == apery_sequence().first(16)


def test_untrusted_declared_model_is_conditional() -> None:
    model = thue_morse_automaton()
    sequence = Sequence(model.term, metadata={"automatic_model": model})
    compiled = compile_sequence(sequence)
    assert compiled.certificate.exactness.value == "conditional"
    assert compiled.certificate.assumptions_unchecked
