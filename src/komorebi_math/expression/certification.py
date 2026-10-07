"""Sound certificate composition for the research expression calculus.

This module is deliberately independent of the expression AST.  It supplies
the proof-status meet used by checking, rewriting, planning, and execution
without changing the frozen :mod:`komorebi_math.certificates` records.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError

_EXACTNESS_STRENGTH: Mapping[Exactness, int] = {
    Exactness.EXACT: 0,
    Exactness.NUMERICAL: 1,
    Exactness.CONDITIONAL: 2,
    Exactness.HEURISTIC: 3,
    Exactness.PLANNED: 4,
}


def exactness_strength(value: Exactness) -> int:
    """Return the total-order rank of one proof status.

    A larger rank denotes a weaker guarantee.  Requiring an ``Exactness``
    instance rather than accepting a string prevents accidental status
    coercion at a certificate boundary.
    """

    if not isinstance(value, Exactness):
        raise TypeError("value must be an Exactness status")
    return _EXACTNESS_STRENGTH[value]


def weakest_exactness(*values: Exactness) -> Exactness:
    """Return the weakest supplied status, with ``EXACT`` as the empty meet."""

    if not values:
        return Exactness.EXACT
    for value in values:
        if not isinstance(value, Exactness):
            raise TypeError("every value must be an Exactness status")
    return max(values, key=exactness_strength)


def _ordered_assumptions(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{name} must contain nonempty strings")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _diagnostics(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValidationError(f"{name} keys must be nonempty strings")
        result[key] = item
    return result


def _finite_float(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a finite real number") from exc
    if not isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise ValidationError(f"{name} must be a {qualifier} real number")
    return result


def compose_certificates(
    certificates: Iterable[Certificate],
    *,
    operation: str | None = None,
    theorem_id: str | None = None,
    assumptions_checked: Iterable[str] = (),
    assumptions_unchecked: Iterable[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
) -> Certificate:
    """Compose child evidence without strengthening any proof claim.

    Residuals receive deterministic ``child{index}:`` prefixes.  Child and
    local diagnostics occupy separate nested namespaces, so evidence cannot
    disappear through a mapping-key collision.  If the same assumption is
    checked in one place and unchecked in another, the unchecked status wins.
    """

    children = tuple(certificates)
    if any(not isinstance(certificate, Certificate) for certificate in children):
        raise TypeError("every child must be a Certificate")

    if operation is not None and (
        not isinstance(operation, str) or not operation.strip()
    ):
        raise ValidationError("operation must be a nonempty string when supplied")
    if theorem_id is not None and (
        not isinstance(theorem_id, str) or not theorem_id.strip()
    ):
        raise ValidationError("theorem_id must be a nonempty string when supplied")

    residuals: dict[str, float] = {}
    tolerances: list[float] = []
    child_checked: list[str] = []
    child_unchecked: list[str] = []
    child_diagnostics: list[dict[str, Any]] = []

    for index, certificate in enumerate(children):
        for key, raw_value in certificate.residuals.items():
            if not isinstance(key, str) or not key:
                raise ValidationError("child residual keys must be nonempty strings")
            residuals[f"child{index}:{key}"] = _finite_float(
                raw_value,
                name=f"child {index} residual {key!r}",
            )
        if certificate.tolerance is not None:
            tolerances.append(
                _finite_float(
                    certificate.tolerance,
                    name=f"child {index} tolerance",
                    nonnegative=True,
                )
            )
        child_checked.extend(certificate.assumptions_checked)
        child_unchecked.extend(certificate.assumptions_unchecked)
        child_diagnostics.append(
            _diagnostics(
                certificate.diagnostics,
                name=f"child {index} diagnostics",
            )
        )

    checked = _ordered_assumptions(
        (*child_checked, *tuple(assumptions_checked)),
        name="checked assumptions",
    )
    unchecked = _ordered_assumptions(
        (*child_unchecked, *tuple(assumptions_unchecked)),
        name="unchecked assumptions",
    )
    unchecked_set = set(unchecked)
    checked = tuple(value for value in checked if value not in unchecked_set)

    exactness = weakest_exactness(
        *(certificate.exactness for certificate in children)
    )
    if unchecked and exactness_strength(exactness) < exactness_strength(
        Exactness.CONDITIONAL
    ):
        exactness = Exactness.CONDITIONAL

    valid = all(certificate.valid for certificate in children)
    label = "certificate composition" if operation is None else operation.strip()
    local_diagnostics = _diagnostics(diagnostics, name="local diagnostics")
    merged_diagnostics: dict[str, Any] = {
        "composition": {
            "operation": label,
            "child_count": len(children),
            "child_validity": tuple(certificate.valid for certificate in children),
            "child_exactness": tuple(
                certificate.exactness.value for certificate in children
            ),
            "child_theorem_ids": tuple(
                certificate.theorem_id for certificate in children
            ),
        },
        "children": tuple(child_diagnostics),
        "local": local_diagnostics,
    }

    return Certificate(
        valid=valid,
        exactness=exactness,
        residuals=residuals,
        tolerance=max(tolerances) if tolerances else None,
        assumptions_checked=checked,
        assumptions_unchecked=unchecked,
        diagnostics=merged_diagnostics,
        theorem_id=theorem_id,
        message=(
            f"{label} certified"
            if valid
            else f"{label} failed because at least one child certificate is invalid"
        ),
    )


def exact_internal_certificate(
    operation: str,
    *,
    theorem_id: str | None = None,
    assumptions_checked: Iterable[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
) -> Certificate:
    """Create evidence for a finite internal operation exact by construction."""

    if not isinstance(operation, str) or not operation.strip():
        raise ValidationError("operation must be a nonempty string")
    if theorem_id is not None and (
        not isinstance(theorem_id, str) or not theorem_id.strip()
    ):
        raise ValidationError("theorem_id must be a nonempty string when supplied")
    checked = _ordered_assumptions(
        (
            f"internal exact operation: {operation.strip()}",
            *tuple(assumptions_checked),
        ),
        name="checked assumptions",
    )
    return Certificate(
        valid=True,
        exactness=Exactness.EXACT,
        assumptions_checked=checked,
        diagnostics=_diagnostics(diagnostics, name="diagnostics"),
        theorem_id=theorem_id,
        message=f"{operation.strip()} is exact by construction",
    )


@dataclass(frozen=True, slots=True)
class RewriteStep:
    """One certificate-carrying expression rewrite record."""

    rule: str
    theorem_id: str | None
    before_digest: str
    after_digest: str
    side_conditions: tuple[str, ...]
    certificate: Certificate

    def __post_init__(self) -> None:
        if not isinstance(self.rule, str) or not self.rule.strip():
            raise ValidationError("rewrite rule must be a nonempty string")
        if self.theorem_id is not None and (
            not isinstance(self.theorem_id, str) or not self.theorem_id.strip()
        ):
            raise ValidationError("rewrite theorem_id must be nonempty when supplied")
        for name, digest in (
            ("before_digest", self.before_digest),
            ("after_digest", self.after_digest),
        ):
            if not isinstance(digest, str) or not digest.strip():
                raise ValidationError(f"{name} must be a nonempty string")
        if not isinstance(self.certificate, Certificate):
            raise TypeError("rewrite certificate must be a Certificate")
        object.__setattr__(
            self,
            "side_conditions",
            _ordered_assumptions(
                self.side_conditions,
                name="rewrite side conditions",
            ),
        )


@dataclass(frozen=True, slots=True)
class NormalForm:
    """A canonical expression payload paired with its derivation evidence."""

    expression_type: Any
    data: Any
    digest: str
    certificate: Certificate

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not self.digest.strip():
            raise ValidationError("normal-form digest must be a nonempty string")
        if not isinstance(self.certificate, Certificate):
            raise TypeError("normal-form certificate must be a Certificate")


@dataclass(frozen=True, slots=True)
class ExpressionBinding:
    """A runtime value with an explicit declared type and evidence record."""

    value: Any
    declared_type: Any
    certificate: Certificate

    def __post_init__(self) -> None:
        if self.declared_type is None:
            raise ValidationError("expression binding requires a declared type")
        if not isinstance(self.certificate, Certificate):
            raise TypeError("expression binding certificate must be a Certificate")


__all__ = [
    "ExpressionBinding",
    "NormalForm",
    "RewriteStep",
    "compose_certificates",
    "exact_internal_certificate",
    "exactness_strength",
    "weakest_exactness",
]
