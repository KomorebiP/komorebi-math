"""Canonical finite polyphase refinement and coarsening.

This module consolidates the previously separate signal, residue, arithmetic,
and rectangular-lattice interfaces around the address law

    a = r + M s,

where ``r`` is a residue modulo the coarse order ``M`` and ``s`` is a
refinement digit modulo ``K = N / M``.  Consequently

    (x_r^(M))_s^(K) = x_(r + M s)^(N)

for every finite sequence or array and every divisible pair ``M | N``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any, Iterable, Mapping

import numpy as np

from ..arithmetic.polyphase import ArithmeticPolyphase
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..signals.polyphase import LatticePolyphaseCoordinates, PolyphaseCoordinates
from .residue import ResidueCoordinates

PolyphaseCoordinateRecord = PolyphaseCoordinates | ResidueCoordinates | ArithmeticPolyphase


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a positive integer") from exc
    if normalized != value or normalized < 1:
        raise ValidationError(f"{name} must be a positive integer")
    return normalized


def _coordinate_order(coordinates: PolyphaseCoordinateRecord) -> int:
    if isinstance(coordinates, PolyphaseCoordinates):
        return coordinates.order
    if isinstance(coordinates, ResidueCoordinates):
        return coordinates.modulus
    if isinstance(coordinates, ArithmeticPolyphase):
        return coordinates.modulus
    raise ValidationError(
        "coordinates must be PolyphaseCoordinates, ResidueCoordinates, or ArithmeticPolyphase"
    )


def _coordinate_channels(coordinates: PolyphaseCoordinateRecord) -> tuple[np.ndarray, ...]:
    if isinstance(coordinates, ArithmeticPolyphase):
        return tuple(np.asarray(channel, dtype=object) for channel in coordinates.channels)
    return tuple(np.asarray(channel) for channel in coordinates.channels)


def _arrays_residual(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if np.array_equal(left, right):
        return 0.0
    try:
        difference = np.asarray(left - right, dtype=np.complex128)
        return float(np.linalg.norm(difference.reshape(-1)))
    except (TypeError, ValueError, OverflowError):
        return float("inf")


def _coordinate_residual(
    left: PolyphaseCoordinateRecord,
    right: PolyphaseCoordinateRecord,
) -> float:
    if type(left) is not type(right):
        return float("inf")
    if _coordinate_order(left) != _coordinate_order(right):
        return float("inf")
    if isinstance(left, PolyphaseCoordinates):
        assert isinstance(right, PolyphaseCoordinates)
        if left.axis != right.axis or left.original_shape != right.original_shape:
            return float("inf")
    elif isinstance(left, ResidueCoordinates):
        assert isinstance(right, ResidueCoordinates)
        if left.length != right.length:
            return float("inf")
    else:
        assert isinstance(left, ArithmeticPolyphase)
        assert isinstance(right, ArithmeticPolyphase)
        if left.length != right.length:
            return float("inf")
    residuals = [
        _arrays_residual(a, b)
        for a, b in zip(_coordinate_channels(left), _coordinate_channels(right), strict=True)
    ]
    return max(residuals, default=0.0)


def _coordinate_scale(coordinates: PolyphaseCoordinateRecord) -> float:
    channels = _coordinate_channels(coordinates)
    values: list[np.ndarray] = []
    for channel in channels:
        try:
            values.append(np.asarray(channel, dtype=np.complex128).reshape(-1))
        except (TypeError, ValueError, OverflowError):
            return 1.0
    if not values:
        return 1.0
    joined = np.concatenate(values) if values else np.asarray([], dtype=float)
    return max(1.0, float(np.linalg.norm(joined)))


def _coordinates_are_exact(coordinates: PolyphaseCoordinateRecord) -> bool:
    if isinstance(coordinates, ArithmeticPolyphase):
        return True
    return all(channel.dtype.kind not in {"f", "c"} for channel in _coordinate_channels(coordinates))


def _copy_array(array: np.ndarray, copy: bool) -> np.ndarray:
    return np.array(array, copy=True) if copy else np.asarray(array)


@dataclass(frozen=True, slots=True)
class PolyphaseRefinement:
    """One canonical refinement between divisible one-dimensional packet orders.

    The object supports finite NumPy polyphase coordinates, one-dimensional
    residue coordinates, and exact arithmetic polyphase coordinates.
    """

    coarse_order: int
    fine_order: int

    def __post_init__(self) -> None:
        coarse = _positive_int(self.coarse_order, "coarse_order")
        fine = _positive_int(self.fine_order, "fine_order")
        if fine % coarse:
            raise ValidationError("fine_order must be divisible by coarse_order")
        object.__setattr__(self, "coarse_order", coarse)
        object.__setattr__(self, "fine_order", fine)

    @property
    def ratio(self) -> int:
        """Return the inner refinement radix ``fine_order / coarse_order``."""

        return self.fine_order // self.coarse_order

    def address(self, coarse_residue: int, child_digit: int) -> int:
        """Encode ``(coarse_residue, child_digit)`` as one fine residue."""

        residue = int(coarse_residue)
        digit = int(child_digit)
        if not 0 <= residue < self.coarse_order:
            raise ValidationError("coarse_residue lies outside the coarse residue range")
        if not 0 <= digit < self.ratio:
            raise ValidationError("child_digit lies outside the refinement radix")
        return residue + self.coarse_order * digit

    def split_address(self, fine_residue: int) -> tuple[int, int]:
        """Decode one fine residue into its parent residue and child digit."""

        address = int(fine_residue)
        if not 0 <= address < self.fine_order:
            raise ValidationError("fine_residue lies outside the fine residue range")
        return address % self.coarse_order, address // self.coarse_order

    def parent(self, fine_residue: int) -> int:
        """Return the coarse parent of one fine residue."""

        return self.split_address(fine_residue)[0]

    def child_digit(self, fine_residue: int) -> int:
        """Return the inner refinement digit of one fine residue."""

        return self.split_address(fine_residue)[1]

    def children(self, coarse_residue: int) -> tuple[int, ...]:
        """Return the ordered fine children of one coarse residue."""

        return tuple(self.address(coarse_residue, digit) for digit in range(self.ratio))

    @property
    def parent_map(self) -> tuple[int, ...]:
        """Return the parent of every fine residue in fine-address order."""

        return tuple(self.parent(address) for address in range(self.fine_order))

    def refine(
        self,
        coordinates: PolyphaseCoordinateRecord,
        *,
        copy: bool = True,
    ) -> PolyphaseCoordinateRecord:
        """Refine coordinates without reconstructing the original signal first."""

        if _coordinate_order(coordinates) != self.coarse_order:
            raise ValidationError("coordinate order does not match coarse_order")
        if isinstance(coordinates, PolyphaseCoordinates):
            return self._refine_signal(coordinates, copy=copy)
        if isinstance(coordinates, ResidueCoordinates):
            return self._refine_residue(coordinates, copy=copy)
        if isinstance(coordinates, ArithmeticPolyphase):
            return self._refine_arithmetic(coordinates)
        raise ValidationError("unsupported polyphase coordinate type")

    def coarsen(
        self,
        coordinates: PolyphaseCoordinateRecord,
        *,
        copy: bool = True,
    ) -> PolyphaseCoordinateRecord:
        """Coarsen fine coordinates by interleaving children inside each parent."""

        if _coordinate_order(coordinates) != self.fine_order:
            raise ValidationError("coordinate order does not match fine_order")
        if isinstance(coordinates, PolyphaseCoordinates):
            return self._coarsen_signal(coordinates, copy=copy)
        if isinstance(coordinates, ResidueCoordinates):
            return self._coarsen_residue(coordinates, copy=copy)
        if isinstance(coordinates, ArithmeticPolyphase):
            return self._coarsen_arithmetic(coordinates)
        raise ValidationError("unsupported polyphase coordinate type")

    def certified_refine(
        self,
        coordinates: PolyphaseCoordinateRecord,
        *,
        tolerance: float = 1e-12,
        copy: bool = True,
    ) -> ComputationResult[PolyphaseCoordinateRecord]:
        """Return refined coordinates paired with the full coherence certificate."""

        refined = self.refine(coordinates, copy=copy)
        return ComputationResult(refined, self.certificate(coordinates, tolerance=tolerance))

    def certificate(
        self,
        coordinates: PolyphaseCoordinateRecord,
        *,
        tolerance: float = 1e-12,
    ) -> Certificate:
        """Certify round-trip and direct-analysis equivalence.

        The two checked identities are

        ``coarsen(refine(c)) = c``

        and

        ``refine(analyze_M(x)) = analyze_N(x)``.
        """

        threshold = float(tolerance)
        if threshold < 0 or not np.isfinite(threshold):
            raise ValidationError("tolerance must be finite and nonnegative")
        refined = self.refine(coordinates)
        roundtrip = self.coarsen(refined)
        direct = self._direct_fine_analysis(coordinates)
        roundtrip_residual = _coordinate_residual(coordinates, roundtrip)
        direct_residual = _coordinate_residual(refined, direct)
        scale = _coordinate_scale(coordinates)
        exact = _coordinates_are_exact(coordinates)
        if exact:
            valid = roundtrip_residual == 0.0 and direct_residual == 0.0
            exactness = Exactness.EXACT
        else:
            bound = threshold * scale
            valid = roundtrip_residual <= bound and direct_residual <= bound
            exactness = Exactness.NUMERICAL
        return Certificate(
            valid=valid,
            exactness=exactness,
            residuals={
                "coarsen_after_refine": roundtrip_residual,
                "direct_fine_analysis": direct_residual,
            },
            tolerance=None if exact else threshold,
            assumptions_checked=(
                "positive divisible packet orders",
                "finite polyphase coordinate record",
                "mixed-radix address law a = r + M s",
                "coarsen after refine identity",
                "nested analysis equals direct fine analysis",
            ),
            diagnostics={
                "coarse_order": self.coarse_order,
                "fine_order": self.fine_order,
                "ratio": self.ratio,
                "coordinate_type": type(coordinates).__name__,
            },
            theorem_id="KM-B15-T01",
            message=(
                "polyphase refinement coherence verified"
                if valid
                else "polyphase refinement coherence failed"
            ),
        )

    def compose(self, next_refinement: "PolyphaseRefinement") -> "PolyphaseRefinement":
        """Compose adjacent refinement laws into their direct refinement."""

        if not isinstance(next_refinement, PolyphaseRefinement):
            raise ValidationError("next_refinement must be a PolyphaseRefinement")
        if self.fine_order != next_refinement.coarse_order:
            raise ValidationError("refinement endpoints do not compose")
        return PolyphaseRefinement(self.coarse_order, next_refinement.fine_order)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable versioned refinement record."""

        return {
            "schema": "komorebi-math.polyphase-refinement",
            "schema_version": 1,
            "coarse_order": self.coarse_order,
            "fine_order": self.fine_order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolyphaseRefinement":
        """Reconstruct a refinement from its validated stable record."""

        required = {"schema", "schema_version", "coarse_order", "fine_order"}
        if set(value) != required:
            raise ValidationError("polyphase refinement fields do not match the schema")
        if value["schema"] != "komorebi-math.polyphase-refinement":
            raise ValidationError("unknown polyphase refinement schema")
        if int(value["schema_version"]) != 1:
            raise ValidationError("unsupported polyphase refinement schema version")
        return cls(int(value["coarse_order"]), int(value["fine_order"]))

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize this refinement to deterministic JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "PolyphaseRefinement":
        """Deserialize one deterministic JSON refinement record."""

        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValidationError("polyphase refinement JSON is invalid") from exc
        if not isinstance(data, Mapping):
            raise ValidationError("polyphase refinement JSON root must be an object")
        return cls.from_dict(data)

    def _refine_signal(
        self,
        coordinates: PolyphaseCoordinates,
        *,
        copy: bool,
    ) -> PolyphaseCoordinates:
        channels: list[np.ndarray | None] = [None] * self.fine_order
        for residue, channel in enumerate(coordinates.channels):
            for digit in range(self.ratio):
                index = [slice(None)] * channel.ndim
                index[coordinates.axis] = slice(digit, None, self.ratio)
                channels[self.address(residue, digit)] = _copy_array(channel[tuple(index)], copy)
        if any(channel is None for channel in channels):
            raise RuntimeError("internal refinement failed to assign every fine channel")
        return PolyphaseCoordinates(
            tuple(channel for channel in channels if channel is not None),
            self.fine_order,
            coordinates.axis,
            coordinates.original_shape,
        )

    def _coarsen_signal(
        self,
        coordinates: PolyphaseCoordinates,
        *,
        copy: bool,
    ) -> PolyphaseCoordinates:
        coarse_channels: list[np.ndarray] = []
        for residue in range(self.coarse_order):
            children = tuple(coordinates.channels[address] for address in self.children(residue))
            shape = list(coordinates.original_shape)
            shape[coordinates.axis] = len(
                range(residue, coordinates.original_shape[coordinates.axis], self.coarse_order)
            )
            dtype = np.result_type(*(child.dtype for child in children))
            parent = np.empty(tuple(shape), dtype=dtype)
            for digit, child in enumerate(children):
                index = [slice(None)] * parent.ndim
                index[coordinates.axis] = slice(digit, None, self.ratio)
                expected = parent[tuple(index)].shape
                if child.shape != expected:
                    raise DimensionError(
                        f"fine child {self.address(residue, digit)} has shape {child.shape}, "
                        f"expected {expected}"
                    )
                parent[tuple(index)] = child
            coarse_channels.append(_copy_array(parent, copy))
        return PolyphaseCoordinates(
            tuple(coarse_channels),
            self.coarse_order,
            coordinates.axis,
            coordinates.original_shape,
        )

    def _refine_residue(
        self,
        coordinates: ResidueCoordinates,
        *,
        copy: bool,
    ) -> ResidueCoordinates:
        channels: list[np.ndarray | None] = [None] * self.fine_order
        for residue, channel in enumerate(coordinates.channels):
            for digit in range(self.ratio):
                channels[self.address(residue, digit)] = _copy_array(
                    np.asarray(channel)[digit :: self.ratio],
                    copy,
                )
        if any(channel is None for channel in channels):
            raise RuntimeError("internal refinement failed to assign every fine residue channel")
        return ResidueCoordinates(
            self.fine_order,
            tuple(channel for channel in channels if channel is not None),
            coordinates.length,
        )

    def _coarsen_residue(
        self,
        coordinates: ResidueCoordinates,
        *,
        copy: bool,
    ) -> ResidueCoordinates:
        channels: list[np.ndarray] = []
        for residue in range(self.coarse_order):
            children = tuple(np.asarray(coordinates.channels[address]) for address in self.children(residue))
            length = len(range(residue, coordinates.length, self.coarse_order))
            dtype = np.result_type(*(child.dtype for child in children))
            parent = np.empty(length, dtype=dtype)
            for digit, child in enumerate(children):
                expected = parent[digit :: self.ratio].shape
                if child.shape != expected:
                    raise DimensionError(
                        f"fine child {self.address(residue, digit)} has shape {child.shape}, "
                        f"expected {expected}"
                    )
                parent[digit :: self.ratio] = child
            channels.append(_copy_array(parent, copy))
        return ResidueCoordinates(self.coarse_order, tuple(channels), coordinates.length)

    def _refine_arithmetic(self, coordinates: ArithmeticPolyphase) -> ArithmeticPolyphase:
        channels: list[tuple[Any, ...] | None] = [None] * self.fine_order
        for residue, channel in enumerate(coordinates.channels):
            for digit in range(self.ratio):
                channels[self.address(residue, digit)] = tuple(channel[digit :: self.ratio])
        if any(channel is None for channel in channels):
            raise RuntimeError("internal refinement failed to assign every exact channel")
        return ArithmeticPolyphase(
            self.fine_order,
            tuple(channel for channel in channels if channel is not None),
            coordinates.length,
        )

    def _coarsen_arithmetic(self, coordinates: ArithmeticPolyphase) -> ArithmeticPolyphase:
        channels: list[tuple[Any, ...]] = []
        for residue in range(self.coarse_order):
            children = tuple(coordinates.channels[address] for address in self.children(residue))
            length = len(range(residue, coordinates.length, self.coarse_order))
            parent = ArithmeticPolyphase(self.ratio, children, length).reconstruct()
            channels.append(parent)
        return ArithmeticPolyphase(self.coarse_order, tuple(channels), coordinates.length)

    def _direct_fine_analysis(
        self,
        coordinates: PolyphaseCoordinateRecord,
    ) -> PolyphaseCoordinateRecord:
        if isinstance(coordinates, PolyphaseCoordinates):
            from ..signals.polyphase import polyphase_analyze

            return polyphase_analyze(
                coordinates.reconstruct(),
                self.fine_order,
                axis=coordinates.axis,
            )
        if isinstance(coordinates, ResidueCoordinates):
            from .residue import residue_analyze

            return residue_analyze(coordinates.reconstruct(), self.fine_order)
        if isinstance(coordinates, ArithmeticPolyphase):
            from ..arithmetic.polyphase import polyphase_analyze

            return polyphase_analyze(coordinates.reconstruct(), self.fine_order)
        raise ValidationError("unsupported polyphase coordinate type")


@dataclass(frozen=True, slots=True)
class PolyphaseRefinementTower:
    """A finite chain of compatible polyphase orders."""

    orders: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized = tuple(_positive_int(value, "order") for value in self.orders)
        if len(normalized) < 2:
            raise ValidationError("a refinement tower requires at least two orders")
        if len(set(normalized)) != len(normalized):
            raise ValidationError("refinement tower orders must be distinct")
        for coarse, fine in zip(normalized, normalized[1:]):
            if fine % coarse:
                raise ValidationError("every tower order must divide the next order")
        object.__setattr__(self, "orders", normalized)

    @property
    def refinements(self) -> tuple[PolyphaseRefinement, ...]:
        return tuple(
            PolyphaseRefinement(coarse, fine)
            for coarse, fine in zip(self.orders, self.orders[1:])
        )

    def level(self, order: int) -> int:
        value = int(order)
        try:
            return self.orders.index(value)
        except ValueError as exc:
            raise ValidationError("order is not present in this refinement tower") from exc

    def refinement(self, source_order: int, target_order: int) -> PolyphaseRefinement:
        source_level = self.level(source_order)
        target_level = self.level(target_order)
        if target_level <= source_level:
            raise ValidationError("target_order must occur after source_order")
        return PolyphaseRefinement(self.orders[source_level], self.orders[target_level])

    def refine(
        self,
        coordinates: PolyphaseCoordinateRecord,
        target_order: int | None = None,
        *,
        copy: bool = True,
    ) -> PolyphaseCoordinateRecord:
        source = _coordinate_order(coordinates)
        source_level = self.level(source)
        target = self.orders[-1] if target_order is None else int(target_order)
        target_level = self.level(target)
        if target_level < source_level:
            raise ValidationError("target_order precedes the coordinate order")
        result: PolyphaseCoordinateRecord = coordinates
        for level in range(source_level, target_level):
            result = PolyphaseRefinement(
                self.orders[level],
                self.orders[level + 1],
            ).refine(result, copy=copy)
        return result

    def coarsen(
        self,
        coordinates: PolyphaseCoordinateRecord,
        target_order: int | None = None,
        *,
        copy: bool = True,
    ) -> PolyphaseCoordinateRecord:
        source = _coordinate_order(coordinates)
        source_level = self.level(source)
        target = self.orders[0] if target_order is None else int(target_order)
        target_level = self.level(target)
        if target_level > source_level:
            raise ValidationError("target_order follows the coordinate order")
        result: PolyphaseCoordinateRecord = coordinates
        for level in range(source_level, target_level, -1):
            result = PolyphaseRefinement(
                self.orders[level - 1],
                self.orders[level],
            ).coarsen(result, copy=copy)
        return result

    def descendants(
        self,
        residue: int,
        source_order: int,
        target_order: int,
    ) -> tuple[int, ...]:
        refinement = self.refinement(source_order, target_order)
        return refinement.children(residue)

    def path_address(
        self,
        residue: int,
        child_digits: Iterable[int],
        *,
        source_order: int | None = None,
    ) -> int:
        source = self.orders[0] if source_order is None else int(source_order)
        level = self.level(source)
        address = int(residue)
        if not 0 <= address < source:
            raise ValidationError("residue lies outside source_order")
        digits = tuple(int(value) for value in child_digits)
        if level + len(digits) >= len(self.orders):
            if level + len(digits) != len(self.orders) - 1:
                raise ValidationError("child digit count exceeds the available tower levels")
        for offset, digit in enumerate(digits):
            coarse = self.orders[level + offset]
            fine = self.orders[level + offset + 1]
            ratio = fine // coarse
            if not 0 <= digit < ratio:
                raise ValidationError("a child digit lies outside its tower radix")
            address += coarse * digit
        return address

    def coherence_certificate(
        self,
        coordinates: PolyphaseCoordinateRecord,
        target_order: int | None = None,
        *,
        tolerance: float = 1e-12,
    ) -> Certificate:
        source = _coordinate_order(coordinates)
        target = self.orders[-1] if target_order is None else int(target_order)
        sequential = self.refine(coordinates, target)
        direct = PolyphaseRefinement(source, target).refine(coordinates)
        residual = _coordinate_residual(sequential, direct)
        exact = _coordinates_are_exact(coordinates)
        scale = _coordinate_scale(coordinates)
        valid = residual == 0.0 if exact else residual <= float(tolerance) * scale
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if exact else Exactness.NUMERICAL,
            residuals={"sequential_vs_direct_refinement": residual},
            tolerance=None if exact else float(tolerance),
            assumptions_checked=(
                "nested divisible refinement tower",
                "associativity of mixed-radix address composition",
            ),
            diagnostics={
                "orders": self.orders,
                "source_order": source,
                "target_order": target,
            },
            theorem_id="KM-B15-T02",
            message=(
                "multilevel polyphase refinement coherence verified"
                if valid
                else "multilevel polyphase refinement coherence failed"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "komorebi-math.polyphase-refinement-tower",
            "schema_version": 1,
            "orders": list(self.orders),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolyphaseRefinementTower":
        required = {"schema", "schema_version", "orders"}
        if set(value) != required:
            raise ValidationError("polyphase refinement tower fields do not match the schema")
        if value["schema"] != "komorebi-math.polyphase-refinement-tower":
            raise ValidationError("unknown polyphase refinement tower schema")
        if int(value["schema_version"]) != 1:
            raise ValidationError("unsupported polyphase refinement tower schema version")
        orders = value["orders"]
        if not isinstance(orders, (list, tuple)):
            raise ValidationError("polyphase refinement tower orders must be an array")
        return cls(tuple(int(order) for order in orders))


@dataclass(frozen=True, slots=True)
class LatticePolyphaseRefinement:
    """Componentwise refinement of rectangular lattice polyphase coordinates."""

    coarse_strides: tuple[int, ...]
    fine_strides: tuple[int, ...]

    def __post_init__(self) -> None:
        coarse = tuple(_positive_int(value, "coarse_stride") for value in self.coarse_strides)
        fine = tuple(_positive_int(value, "fine_stride") for value in self.fine_strides)
        if not coarse or len(coarse) != len(fine):
            raise DimensionError("coarse and fine strides must have the same positive dimension")
        if any(target % source for source, target in zip(coarse, fine, strict=True)):
            raise ValidationError("every fine stride must be divisible by its coarse stride")
        object.__setattr__(self, "coarse_strides", coarse)
        object.__setattr__(self, "fine_strides", fine)

    @property
    def ratios(self) -> tuple[int, ...]:
        return tuple(
            fine // coarse
            for coarse, fine in zip(self.coarse_strides, self.fine_strides, strict=True)
        )

    @property
    def coarse_channel_count(self) -> int:
        return prod(self.coarse_strides)

    @property
    def fine_channel_count(self) -> int:
        return prod(self.fine_strides)

    def address(
        self,
        coarse_address: Iterable[int],
        child_digits: Iterable[int],
    ) -> tuple[int, ...]:
        residue = tuple(int(value) for value in coarse_address)
        digits = tuple(int(value) for value in child_digits)
        if len(residue) != len(self.coarse_strides) or len(digits) != len(self.ratios):
            raise DimensionError("addresses must match the lattice dimension")
        output: list[int] = []
        for r, digit, coarse, ratio in zip(
            residue,
            digits,
            self.coarse_strides,
            self.ratios,
            strict=True,
        ):
            if not 0 <= r < coarse:
                raise ValidationError("a coarse address component lies outside its stride")
            if not 0 <= digit < ratio:
                raise ValidationError("a child digit lies outside its lattice refinement radix")
            output.append(r + coarse * digit)
        return tuple(output)

    def split_address(self, fine_address: Iterable[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        address = tuple(int(value) for value in fine_address)
        if len(address) != len(self.fine_strides):
            raise DimensionError("fine address must match the lattice dimension")
        parents: list[int] = []
        digits: list[int] = []
        for value, coarse, fine in zip(
            address,
            self.coarse_strides,
            self.fine_strides,
            strict=True,
        ):
            if not 0 <= value < fine:
                raise ValidationError("a fine address component lies outside its stride")
            parents.append(value % coarse)
            digits.append(value // coarse)
        return tuple(parents), tuple(digits)

    def parent(self, fine_address: Iterable[int]) -> tuple[int, ...]:
        return self.split_address(fine_address)[0]

    def children(self, coarse_address: Iterable[int]) -> tuple[tuple[int, ...], ...]:
        residue = tuple(int(value) for value in coarse_address)
        return tuple(
            self.address(residue, digits)
            for digits in product(*(range(ratio) for ratio in self.ratios))
        )

    def refine(
        self,
        coordinates: LatticePolyphaseCoordinates,
        *,
        copy: bool = True,
    ) -> LatticePolyphaseCoordinates:
        if not isinstance(coordinates, LatticePolyphaseCoordinates):
            raise ValidationError("coordinates must be LatticePolyphaseCoordinates")
        if coordinates.strides != self.coarse_strides:
            raise ValidationError("lattice coordinate strides do not match coarse_strides")
        channels: dict[tuple[int, ...], np.ndarray] = {}
        for coarse_address, channel in coordinates.channels.items():
            for digits in product(*(range(ratio) for ratio in self.ratios)):
                index = tuple(slice(digit, None, ratio) for digit, ratio in zip(digits, self.ratios, strict=True))
                channels[self.address(coarse_address, digits)] = _copy_array(channel[index], copy)
        return LatticePolyphaseCoordinates(
            channels,
            self.fine_strides,
            coordinates.original_shape,
        )

    def coarsen(
        self,
        coordinates: LatticePolyphaseCoordinates,
        *,
        copy: bool = True,
    ) -> LatticePolyphaseCoordinates:
        if not isinstance(coordinates, LatticePolyphaseCoordinates):
            raise ValidationError("coordinates must be LatticePolyphaseCoordinates")
        if coordinates.strides != self.fine_strides:
            raise ValidationError("lattice coordinate strides do not match fine_strides")
        channels: dict[tuple[int, ...], np.ndarray] = {}
        for coarse_address in product(*(range(stride) for stride in self.coarse_strides)):
            child_addresses = self.children(coarse_address)
            children = tuple(coordinates.channels[address] for address in child_addresses)
            shape = tuple(
                len(range(residue, length, stride))
                for residue, length, stride in zip(
                    coarse_address,
                    coordinates.original_shape,
                    self.coarse_strides,
                    strict=True,
                )
            )
            dtype = np.result_type(*(child.dtype for child in children))
            parent = np.empty(shape, dtype=dtype)
            for digits in product(*(range(ratio) for ratio in self.ratios)):
                child = coordinates.channels[self.address(coarse_address, digits)]
                index = tuple(
                    slice(digit, None, ratio)
                    for digit, ratio in zip(digits, self.ratios, strict=True)
                )
                expected = parent[index].shape
                if child.shape != expected:
                    raise DimensionError(
                        f"fine child {self.address(coarse_address, digits)} has shape {child.shape}, "
                        f"expected {expected}"
                    )
                parent[index] = child
            channels[coarse_address] = _copy_array(parent, copy)
        return LatticePolyphaseCoordinates(
            channels,
            self.coarse_strides,
            coordinates.original_shape,
        )

    def certificate(
        self,
        coordinates: LatticePolyphaseCoordinates,
        *,
        tolerance: float = 1e-12,
    ) -> Certificate:
        from ..signals.polyphase import lattice_polyphase_analyze

        if coordinates.strides != self.coarse_strides:
            raise ValidationError("lattice coordinate strides do not match coarse_strides")
        refined = self.refine(coordinates)
        roundtrip = self.coarsen(refined)
        direct = lattice_polyphase_analyze(coordinates.reconstruct(), self.fine_strides)
        roundtrip_residual = max(
            (
                _arrays_residual(coordinates.channels[address], roundtrip.channels[address])
                for address in coordinates.channels
            ),
            default=0.0,
        )
        direct_residual = max(
            (
                _arrays_residual(refined.channels[address], direct.channels[address])
                for address in refined.channels
            ),
            default=0.0,
        )
        exact = all(channel.dtype.kind not in {"f", "c"} for channel in coordinates.channels.values())
        scale = max(
            1.0,
            float(np.linalg.norm(np.asarray(coordinates.reconstruct(), dtype=np.complex128).reshape(-1))),
        )
        if exact:
            valid = roundtrip_residual == 0.0 and direct_residual == 0.0
        else:
            bound = float(tolerance) * scale
            valid = roundtrip_residual <= bound and direct_residual <= bound
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if exact else Exactness.NUMERICAL,
            residuals={
                "coarsen_after_refine": roundtrip_residual,
                "direct_fine_analysis": direct_residual,
            },
            tolerance=None if exact else float(tolerance),
            assumptions_checked=(
                "componentwise divisible rectangular strides",
                "finite rectangular lattice polyphase coordinates",
                "componentwise address law a = r + M*s",
            ),
            diagnostics={
                "coarse_strides": self.coarse_strides,
                "fine_strides": self.fine_strides,
                "ratios": self.ratios,
            },
            theorem_id="KM-B15-T03",
            message=(
                "lattice polyphase refinement coherence verified"
                if valid
                else "lattice polyphase refinement coherence failed"
            ),
        )

    def compose(
        self,
        next_refinement: "LatticePolyphaseRefinement",
    ) -> "LatticePolyphaseRefinement":
        if not isinstance(next_refinement, LatticePolyphaseRefinement):
            raise ValidationError("next_refinement must be a LatticePolyphaseRefinement")
        if self.fine_strides != next_refinement.coarse_strides:
            raise ValidationError("lattice refinement endpoints do not compose")
        return LatticePolyphaseRefinement(self.coarse_strides, next_refinement.fine_strides)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "komorebi-math.lattice-polyphase-refinement",
            "schema_version": 1,
            "coarse_strides": list(self.coarse_strides),
            "fine_strides": list(self.fine_strides),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LatticePolyphaseRefinement":
        required = {"schema", "schema_version", "coarse_strides", "fine_strides"}
        if set(value) != required:
            raise ValidationError("lattice refinement fields do not match the schema")
        if value["schema"] != "komorebi-math.lattice-polyphase-refinement":
            raise ValidationError("unknown lattice refinement schema")
        if int(value["schema_version"]) != 1:
            raise ValidationError("unsupported lattice refinement schema version")
        coarse = value["coarse_strides"]
        fine = value["fine_strides"]
        if not isinstance(coarse, (list, tuple)) or not isinstance(fine, (list, tuple)):
            raise ValidationError("lattice refinement strides must be arrays")
        return cls(tuple(int(value) for value in coarse), tuple(int(value) for value in fine))


def refine_polyphase(
    coordinates: PolyphaseCoordinateRecord,
    fine_order: int,
    *,
    copy: bool = True,
) -> PolyphaseCoordinateRecord:
    """Refine any supported one-dimensional coordinate record to ``fine_order``."""

    return PolyphaseRefinement(_coordinate_order(coordinates), fine_order).refine(
        coordinates,
        copy=copy,
    )


def coarsen_polyphase(
    coordinates: PolyphaseCoordinateRecord,
    coarse_order: int,
    *,
    copy: bool = True,
) -> PolyphaseCoordinateRecord:
    """Coarsen any supported one-dimensional coordinate record to ``coarse_order``."""

    return PolyphaseRefinement(coarse_order, _coordinate_order(coordinates)).coarsen(
        coordinates,
        copy=copy,
    )


def polyphase_refinement_certificate(
    coordinates: PolyphaseCoordinateRecord,
    fine_order: int,
    *,
    tolerance: float = 1e-12,
) -> Certificate:
    """Certify refinement and direct-analysis equivalence for one coordinate record."""

    return PolyphaseRefinement(_coordinate_order(coordinates), fine_order).certificate(
        coordinates,
        tolerance=tolerance,
    )


def refine_lattice_polyphase(
    coordinates: LatticePolyphaseCoordinates,
    fine_strides: Iterable[int],
    *,
    copy: bool = True,
) -> LatticePolyphaseCoordinates:
    """Refine rectangular lattice coordinates to componentwise finer strides."""

    return LatticePolyphaseRefinement(
        coordinates.strides,
        tuple(int(value) for value in fine_strides),
    ).refine(coordinates, copy=copy)


def coarsen_lattice_polyphase(
    coordinates: LatticePolyphaseCoordinates,
    coarse_strides: Iterable[int],
    *,
    copy: bool = True,
) -> LatticePolyphaseCoordinates:
    """Coarsen rectangular lattice coordinates to componentwise coarser strides."""

    return LatticePolyphaseRefinement(
        tuple(int(value) for value in coarse_strides),
        coordinates.strides,
    ).coarsen(coordinates, copy=copy)


__all__ = [
    "LatticePolyphaseRefinement",
    "PolyphaseCoordinateRecord",
    "PolyphaseRefinement",
    "PolyphaseRefinementTower",
    "coarsen_lattice_polyphase",
    "coarsen_polyphase",
    "polyphase_refinement_certificate",
    "refine_lattice_polyphase",
    "refine_polyphase",
]
