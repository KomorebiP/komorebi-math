"""Internal residual decisions, deliberately separate from display norms.

Exact SymPy checks use simplification and three-valued zero predicates. Unknown
never grants a certificate. Numerical checks remain tolerance checks, not proofs
about real-valued inputs before floating-point arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, isfinite
from typing import Any

from .exceptions import ValidationError, DimensionError


def finite_tolerance(value: Any) -> float:
    """Normalize a finite nonnegative tolerance or reject it before use."""
    try:
        tol = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("tolerance must be finite and nonnegative") from exc
    if not isfinite(tol) or tol < 0.0:
        raise ValidationError("tolerance must be finite and nonnegative")
    return tol


def _exact_status(backend: Any, value: Any) -> tuple[str, dict[str, Any]]:
    if backend.name != "sympy":
        decision = backend.is_zero(value, tolerance=0.0)
        if decision is True:
            return "zero", {}
        if decision is False:
            return "nonzero", {"method": "backend exact zero predicate"}
        return "unknown", {"reason": "exact backend returned no Boolean decision"}
    import sympy as sp
    matrix = sp.Matrix(value)
    unknown = None
    nonzero = None
    for position, entry in enumerate(matrix):
        simplified = sp.simplify(entry)
        if simplified.is_finite is False or simplified.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            return "nonfinite", {"entry": position, "expression": sp.srepr(simplified)}
        zero = simplified.is_zero
        if zero is False and nonzero is None:
            nonzero = {"entry": position, "expression": sp.srepr(simplified)}
        if zero is not True and unknown is None:
            unknown = {"entry": position, "expression": sp.srepr(simplified)}
    # Do not stop at a nonzero entry: a later entry can be nonfinite.
    if nonzero is not None:
        return "nonzero", nonzero
    return ("unknown", unknown) if unknown is not None else ("zero", {})


@dataclass
class ResidualAudit:
    """A producer-local record. It is not a persistent, authenticated proof."""
    backend: Any
    tolerance: float
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tolerance = finite_tolerance(self.tolerance)

    def check(self, name: str, residual: Any) -> float:
        if name in self.decisions:
            raise ValidationError("duplicate residual decision label")
        if self.backend.exact:
            status, witness = _exact_status(self.backend, residual)
            accepted = status == "zero"
        else:
            accepted = bool(self.backend.is_zero(residual, tolerance=self.tolerance))
            status = "within_tolerance" if accepted else "outside_tolerance"
            witness = {}
        # These floats are diagnostics ONLY. Infinity denotes an unavailable
        # diagnostic, including exact nonzero quantities below float range.
        norm_status = "available"
        try:
            magnitude = 0.0 if self.backend.exact and accepted else float(self.backend.norm(residual))
            if not isfinite(magnitude) or magnitude < 0:
                magnitude, norm_status = inf, "unavailable_or_nonfinite"
            elif self.backend.exact and not accepted and magnitude == 0.0:
                magnitude, norm_status = inf, "nonzero_or_unknown_below_float_resolution"
        except (TypeError, ValueError, OverflowError):
            magnitude, norm_status = inf, "unavailable_symbolic_or_out_of_range"
        if not self.backend.exact and not isfinite(magnitude):
            accepted, status = False, "nonfinite_or_unrepresentable_norm"
        self.decisions[name] = {
            "status": status, "accepted": accepted,
            "arithmetic": "exact_predicate" if self.backend.exact else "numerical_tolerance",
            "shape": list(self.backend.shape(residual)),
            "witness": witness, "diagnostic_norm_status": norm_status,
        }
        return magnitude

    def check_value(self, name: str, residual: Any) -> float:
        """Check a scalar/vector/matrix without first converting exact entries to floats.

        Higher-rank tensors are outside this helper's contract. Numerical values
        are normalized to NumPy complex arrays, not certified real-input bounds.
        """
        import numpy as np
        array = np.asarray(residual, dtype=object if self.backend.exact else complex)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise DimensionError("residual must be a scalar, vector, or matrix")
        if self.backend.name == "sympy":
            import sympy as sp
            value = sp.Matrix(array.shape[0], array.shape[1], list(array.flat))
        else:
            value = self.backend.as_matrix(array)
        return self.check(name, value)

    @property
    def valid(self) -> bool:
        return all(item["accepted"] for item in self.decisions.values())

    @property
    def unresolved(self) -> tuple[str, ...]:
        return tuple("exact zero decision unresolved: " + key
                     for key, item in self.decisions.items() if item["status"] == "unknown")


def value_audit(*, exact: bool, tolerance: float) -> ResidualAudit:
    """Build a value comparison using the producer's declared arithmetic scope.

    An exact residual check does not authenticate a supplied derivative or other
    callback. Its mathematical hypotheses must remain in the outer certificate.
    """
    from .backends import get_backend
    return ResidualAudit(get_backend("sympy" if exact else "numpy"), tolerance)


def python_integer_entries(value: Any) -> Any:
    """Copy to object storage and normalize NumPy integers before arithmetic.

    Object storage alone is insufficient when its entries are np.int64 scalars.
    Noninteger values retain their declared semantics; no float-to-exact-real
    conversion or validated arbitrary object protocol is implied.
    """
    import numpy as np
    array = np.asarray(value, dtype=object).copy()
    for index in np.ndindex(array.shape):
        if isinstance(array[index], np.integer):
            array[index] = int(array[index])
    return array


def coefficient_storage_decision(value: Any, *, exact: bool,
                                 tolerance: float) -> tuple[bool, float, str]:
    """Decide coefficient omission before computing any display-only norm.

    Exact unresolved coefficients are retained. Known nonfinite values are
    rejected. Numerical decisions concern the stored values and a finite
    tolerance, not an a-priori rounding or uniform tensor-error bound.
    """
    import numpy as np
    audit = value_audit(exact=exact, tolerance=tolerance)
    magnitude = audit.check_value("coefficient", value)
    item = audit.decisions["coefficient"]
    if exact:
        nonfinite = item["status"] == "nonfinite"
    else:
        nonfinite = not bool(np.all(np.isfinite(np.asarray(value, dtype=complex))))
    if nonfinite:
        raise ValidationError("captured coefficients must not contain nonfinite values")
    return bool(item["accepted"]), magnitude, str(item["status"])
