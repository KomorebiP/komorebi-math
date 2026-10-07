"""Finite gate models for patched support and characteristic-zero fibers."""

from __future__ import annotations

from dataclasses import dataclass

from ...certificates import Certificate, Exactness


@dataclass(frozen=True, slots=True)
class PatchingSupportReport:
    """A conditional support report for one patched module or complex."""

    patched_object_exists: bool
    patched_full_support: bool
    finite_level_descent: bool
    comparison_preserves_support: bool
    characteristic_zero_fiber_nonzero: bool
    derived: bool = False

    @property
    def complete(self) -> bool:
        """Return whether every formal support gate is supplied."""

        return all(
            (
                self.patched_object_exists,
                self.patched_full_support,
                self.finite_level_descent,
                self.comparison_preserves_support,
                self.characteristic_zero_fiber_nonzero,
            )
        )

    @property
    def unmet(self) -> tuple[str, ...]:
        """Return the missing support gates."""

        labels = (
            (self.patched_object_exists, "patched object exists"),
            (self.patched_full_support, "patched object has full support"),
            (self.finite_level_descent, "support descends to finite level"),
            (self.comparison_preserves_support, "comparison preserves required support"),
            (self.characteristic_zero_fiber_nonzero, "characteristic zero fiber is nonzero"),
        )
        return tuple(label for value, label in labels if not value)

    def certificate(self) -> Certificate:
        """Return a conditional formal support certificate."""

        statuses = (
            (self.patched_object_exists, "patched object exists"),
            (self.patched_full_support, "patched object has full support"),
            (self.finite_level_descent, "support descends to finite level"),
            (self.comparison_preserves_support, "comparison preserves required support"),
            (self.characteristic_zero_fiber_nonzero, "characteristic zero fiber is nonzero"),
        )
        return Certificate(
            valid=self.complete,
            exactness=Exactness.CONDITIONAL,
            assumptions_checked=tuple(label for value, label in statuses if value),
            assumptions_unchecked=tuple(label for value, label in statuses if not value),
            diagnostics={"derived": self.derived},
            theorem_id="PSC-XII-PATCHING-SUPPORT",
            message="all formal patching support gates are supplied" if self.complete else "patching support gates remain",
        )


__all__ = ["PatchingSupportReport"]
