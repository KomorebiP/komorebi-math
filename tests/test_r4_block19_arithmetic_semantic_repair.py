from __future__ import annotations

from fractions import Fraction

import pytest

from komorebi_math.arithmetic import MahlerSeries, TruncatedPadic, UnitModeMahler
from komorebi_math.generating import TruncatedPadicDomain


def test_truncated_padic_visible_valuation_is_capped_at_precision() -> None:
    local = TruncatedPadic(3, 4)
    assert local.valuation(27) == 3
    assert local.valuation(0) == 4
    assert local.valuation(81) == 4
    assert local.valuation(162) == 4

    # Once enough precision is available, the same chosen integer lift reveals
    # its exact finite valuation rather than a spurious transition from infinity.
    refined = local.refine(2)
    assert refined.valuation(81) == 4
    assert refined.valuation(162) == 4
    assert refined.valuation(0) == 6


def test_generating_truncated_padic_domain_uses_same_visible_semantics() -> None:
    local = TruncatedPadicDomain(5, 3)
    assert local.valuation(25) == 2
    assert local.valuation(125) == 3
    assert local.valuation(0) == 3


def test_unit_mode_certificate_rejects_undertruncated_requested_range() -> None:
    mode = UnitModeMahler(base=2, jump=1, residue=0)
    certificate = mode.certificate(5, degree=0)
    assert not certificate.valid
    assert certificate.diagnostics["first_failure"] == 1
    with pytest.raises(Exception):
        certificate.require()

    truncated = MahlerSeries(mode.coefficients(0))
    assert truncated.first(6) == (Fraction(1),) * 6
    assert tuple(2**x for x in range(6)) != truncated.first(6)


def test_unit_mode_certificate_checks_entire_requested_range() -> None:
    mode = UnitModeMahler(base=3, jump=2, residue=1)
    assert mode.certificate(8, degree=8).valid
    assert not mode.certificate(8, degree=7).valid
    assert mode.certificate(8, degree=7).diagnostics["first_failure"] == 8


def test_low_degree_can_still_be_exact_when_increment_vanishes() -> None:
    for base, jump, residue in ((1, 7, 4), (-1, 2, 3)):
        mode = UnitModeMahler(base=base, jump=jump, residue=residue)
        certificate = mode.certificate(12, degree=0)
        certificate.require()
        assert certificate.diagnostics["increment_zero"] is True
