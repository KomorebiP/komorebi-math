from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from komorebi_math import Certificate, ComputationResult, Exactness
from komorebi_math.exceptions import ValidationError
from komorebi_math.export import register_export_adapter, table_rows, to_serializable


@dataclass
class Sample:
    value: Fraction
    path: Path


class Adapted:
    def __init__(self, value: int) -> None:
        self.value = value


def test_serialization_preserves_exact_and_array_types() -> None:
    normalized = to_serializable(
        {
            "fraction": Fraction(2, 5),
            "complex": 1 + 3j,
            "array": np.arange(6).reshape(2, 3),
        }
    )
    assert normalized["fraction"] == {"__type__": "Fraction", "numerator": 2, "denominator": 5}
    assert normalized["complex"] == {"__type__": "complex", "real": 1.0, "imag": 3.0}
    assert normalized["array"]["shape"] == [2, 3]
    assert normalized["array"]["data"] == [[0, 1, 2], [3, 4, 5]]


def test_serialization_dataclass_and_result() -> None:
    sample = Sample(Fraction(1, 7), Path("value.txt"))
    normalized = to_serializable(sample)
    assert normalized["value"]["denominator"] == 7
    result = ComputationResult(
        sample,
        Certificate(True, Exactness.EXACT, theorem_id="TEST"),
    )
    result_data = to_serializable(result)
    assert result_data["certificate"]["theorem_id"] == "TEST"


def test_registered_export_adapter() -> None:
    register_export_adapter(Adapted, lambda value: {"adapted": value.value}, replace=True)
    assert to_serializable(Adapted(9)) == {"adapted": 9}


def test_unknown_type_is_rejected() -> None:
    class Unknown:
        pass

    with pytest.raises(ValidationError):
        to_serializable(Unknown())


def test_table_rows_support_common_shapes() -> None:
    headers, rows = table_rows(np.array([[1, 2], [3, 4]]))
    assert headers == ("column_0", "column_1")
    assert rows == ((1, 2), (3, 4))

    headers, rows = table_rows([{"n": 0, "value": 1}, {"n": 1, "value": 2}])
    assert headers == ("n", "value")
    assert rows == ((0, 1), (1, 2))

    headers, rows = table_rows({"n": [0, 1], "value": [1, 2]})
    assert headers == ("n", "value")
    assert rows == ((0, 1), (1, 2))
