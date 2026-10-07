"""Explicit resource contracts for generating-function compilation."""

from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import ResourceLimitError, ValidationError


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class GeneratingBudget:
    """Upper bounds checked before potentially expensive operations.

    The default values are deliberately conservative.  A caller may provide a
    larger budget when the resulting memory and arithmetic costs are acceptable.
    """

    maximum_degree: int = 100_000
    maximum_coefficient_bits: int = 4_000_000
    maximum_matrix_dimension: int = 4_096
    maximum_reachable_states: int = 100_000
    maximum_local_combinations: int = 1_000_000
    maximum_modulus: int = 2_000_000
    maximum_serialized_bytes: int = 64 * 1024 * 1024
    maximum_recursion_depth: int = 1_000

    def __post_init__(self) -> None:
        for name in (
            "maximum_degree",
            "maximum_coefficient_bits",
            "maximum_matrix_dimension",
            "maximum_reachable_states",
            "maximum_local_combinations",
            "maximum_modulus",
            "maximum_serialized_bytes",
            "maximum_recursion_depth",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    def require_degree(self, degree: int) -> int:
        """Validate and return a requested truncation degree."""

        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValidationError("degree must be a nonnegative integer")
        if degree > self.maximum_degree:
            raise ResourceLimitError(
                f"degree {degree} exceeds maximum_degree={self.maximum_degree}"
            )
        return degree


    def require_terms(self, count: int, *, label: str = "term count") -> int:
        """Validate a finite enumeration count before work begins."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationError(f"{label} must be a nonnegative integer")
        if count > self.maximum_local_combinations:
            raise ResourceLimitError(
                f"{label} {count} exceeds maximum_local_combinations="
                f"{self.maximum_local_combinations}"
            )
        return count

    @property
    def max_modulus_table(self) -> int:
        """Compatibility name used by prime-power table constructors."""

        return self.maximum_modulus

    def require_matrix_dimension(self, dimension: int) -> int:
        """Validate a finite-state matrix dimension before allocation."""

        value = _positive(dimension, "dimension")
        if value > self.maximum_matrix_dimension:
            raise ResourceLimitError(
                f"matrix dimension {value} exceeds "
                f"maximum_matrix_dimension={self.maximum_matrix_dimension}"
            )
        return value

    def require_states(self, states: int) -> int:
        """Validate a reachable-state request."""

        value = _positive(states, "states")
        if value > self.maximum_reachable_states:
            raise ResourceLimitError(
                f"state request {value} exceeds "
                f"maximum_reachable_states={self.maximum_reachable_states}"
            )
        return value

    def require_local_combinations(self, combinations: int) -> int:
        """Validate a local digit-enumeration count."""

        value = _positive(combinations, "combinations")
        if value > self.maximum_local_combinations:
            raise ResourceLimitError(
                f"local enumeration requires {value} combinations, exceeding "
                f"maximum_local_combinations={self.maximum_local_combinations}"
            )
        return value

    def require_modulus(self, modulus: int) -> int:
        """Validate a modular prefix-table size."""

        value = _positive(modulus, "modulus")
        if value > self.maximum_modulus:
            raise ResourceLimitError(
                f"modulus {value} exceeds maximum_modulus={self.maximum_modulus}"
            )
        return value

    def require_serialized_bytes(self, size: int) -> int:
        """Validate a serialized payload size."""

        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValidationError("serialized size must be a nonnegative integer")
        if size > self.maximum_serialized_bytes:
            raise ResourceLimitError(
                f"serialized payload has {size} bytes, exceeding "
                f"maximum_serialized_bytes={self.maximum_serialized_bytes}"
            )
        return size

    def require_coefficient_bits(self, bits: int) -> int:
        """Validate an exact coefficient bit length."""

        if isinstance(bits, bool) or not isinstance(bits, int) or bits < 0:
            raise ValidationError("coefficient bit length must be nonnegative")
        if bits > self.maximum_coefficient_bits:
            raise ResourceLimitError(
                f"coefficient bit length {bits} exceeds "
                f"maximum_coefficient_bits={self.maximum_coefficient_bits}"
            )
        return bits


DEFAULT_GENERATING_BUDGET = GeneratingBudget()


__all__ = ["DEFAULT_GENERATING_BUDGET", "GeneratingBudget"]
