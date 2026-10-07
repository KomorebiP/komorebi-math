from fractions import Fraction

import pytest

from komorebi_math.deformation import finite_frobenius_separator
from komorebi_math.exceptions import ValidationError


def test_rank_deficient_finite_trace_table_has_invalid_exact_certificate() -> None:
    result = finite_frobenius_separator({2: (1, 1), 3: (2, 2), 5: (3, 3)})
    assert result.rank == 1
    assert result.constituent_count == 2
    assert result.separating is False
    assert result.certificate.valid is False
    assert result.certificate.exactness.value == "exact"
    assert result.certificate.diagnostics["rank_deficiency"] == 1
    with pytest.raises(ValidationError, match="do not separate"):
        result.certificate.require()


def test_full_column_rank_table_certifies_and_recovers_exactly() -> None:
    result = finite_frobenius_separator(
        {
            2: (Fraction(1, 2), 1),
            3: (1, Fraction(-1, 2)),
            5: (4, 7),
        }
    )
    assert result.separating is True
    result.certificate.require()
    multiplicities = (3, -2)
    observed = tuple(
        sum(Fraction(entry) * coefficient for entry, coefficient in zip(row, multiplicities, strict=True))
        for row in result.trace_matrix
    )
    assert result.recover(observed) == multiplicities


def test_finite_separator_does_not_claim_global_chebotarev_step() -> None:
    result = finite_frobenius_separator({2: (1, 1), 3: (1, -1)})
    result.certificate.require()
    assert "Chebotarev density beyond the supplied table" in result.certificate.assumptions_unchecked


def test_underdetermined_table_rejects_require() -> None:
    result = finite_frobenius_separator({2: (1, 2, 3)})
    assert result.separating is False
    assert result.certificate.diagnostics["rank_deficiency"] == 2
    with pytest.raises(ValidationError):
        result.certificate.require()
