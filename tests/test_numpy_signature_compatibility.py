"""Dependency repr changes must not alter the committed callable contract."""

from __future__ import annotations

import inspect

import numpy as np

from komorebi_math import compatibility
from komorebi_math.signals import design_biquad


def test_biquad_default_is_portable_without_mutating_callable() -> None:
    before = inspect.signature(design_biquad)
    text = compatibility._public_signature(design_biquad)
    assert "q: 'float' = np.float64(0.7071067811865475)" in text
    assert inspect.signature(design_biquad) == before
    assert type(before.parameters["q"].default) is np.float64
    compatibility.check_public_api(strict_signatures=True).certificate.require()


def test_numpy_default_value_and_type_changes_remain_visible() -> None:
    def scalar(value=np.float64(0.5)):
        pass

    def different_value(value=np.float64(0.25)):
        pass

    def python_float(value=0.5):
        pass

    assert compatibility._public_signature(scalar) == "(value=np.float64(0.5))"
    assert compatibility._public_signature(different_value) == "(value=np.float64(0.25))"
    assert compatibility._public_signature(python_float) == "(value=0.5)"
