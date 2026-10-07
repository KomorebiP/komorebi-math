"""Optional device executors for packet intermediate representations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any


from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, DimensionError, ValidationError
from .ir import PacketIR


def cupy_available() -> bool:
    """Report whether a CuPy module can be discovered."""

    return find_spec("cupy") is not None


@dataclass(frozen=True, slots=True)
class DeviceExecutionResult:
    """Value and certificate returned by one execution backend."""

    value: Any
    backend: str
    returned_to_host: bool
    certificate: Certificate


class NumpyPacketExecutor:
    """CPU executor that preserves PacketIR block semantics."""

    name = "numpy"

    def __init__(self, packet_ir: PacketIR) -> None:
        self.packet_ir = packet_ir

    def apply(self, value: Any) -> DeviceExecutionResult:
        result = self.packet_ir.apply(value)
        certificate = Certificate(
            valid=True,
            exactness=Exactness.NUMERICAL,
            assumptions_checked=("NumPy packet execution",),
            diagnostics={"active_blocks": len(self.packet_ir.support)},
            theorem_id="PSC-VIII-C01",
            message="NumPy packet execution completed",
        )
        return DeviceExecutionResult(result, self.name, True, certificate)


class CupyPacketExecutor:
    """CuPy executor for finite packet blocks.

    Structured blocks are materialized once when the executor is constructed.
    The object therefore measures device execution after setup rather than
    structure discovery or transfer to the device.
    """

    name = "cupy"

    def __init__(self, packet_ir: PacketIR) -> None:
        if not cupy_available():
            raise BackendUnavailableError("CuPy is not installed")
        import cupy as cp

        self._cp = cp
        self.packet_ir = packet_ir
        self._blocks = {
            address: cp.asarray(block.materialize())
            for address, block in packet_ir.blocks.items()
        }

    def apply(
        self,
        value: Any,
        *,
        return_host: bool = True,
        input_on_device: bool = False,
    ) -> DeviceExecutionResult:
        cp = self._cp
        array = value if input_on_device else cp.asarray(value)
        vector = array.ndim == 1
        if vector:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] != self.packet_ir.source_layout.dimension:
            raise DimensionError("right hand side has an incompatible shape")
        outputs = [
            cp.zeros((size, array.shape[1]), dtype=cp.result_type(array.dtype, *[block.dtype for block in self._blocks.values()]))
            for size in self.packet_ir.target_layout.sizes
        ]
        for (target, source), block in self._blocks.items():
            source_slice = self.packet_ir.source_layout.slices[source]
            outputs[target] += block @ array[source_slice]
        result = cp.concatenate(outputs, axis=0)
        if vector:
            result = result[:, 0]
        returned = cp.asnumpy(result) if return_host else result
        certificate = Certificate(
            valid=True,
            exactness=Exactness.NUMERICAL,
            assumptions_checked=("CuPy installed", "packet blocks transferred to device"),
            assumptions_unchecked=("device execution has not been synchronized for timing",),
            diagnostics={
                "active_blocks": len(self._blocks),
                "input_on_device": bool(input_on_device),
            },
            theorem_id="PSC-VIII-C02",
            message="CuPy packet execution completed",
        )
        return DeviceExecutionResult(returned, self.name, bool(return_host), certificate)

    def synchronize(self) -> None:
        self._cp.cuda.Stream.null.synchronize()


def packet_executor(packet_ir: PacketIR, backend: str = "numpy") -> NumpyPacketExecutor | CupyPacketExecutor:
    """Construct a packet executor for a supported backend name."""

    name = str(backend).lower()
    if name == "numpy":
        return NumpyPacketExecutor(packet_ir)
    if name == "cupy":
        return CupyPacketExecutor(packet_ir)
    raise ValidationError(f"unknown packet execution backend: {backend}")


__all__ = [
    "CupyPacketExecutor",
    "DeviceExecutionResult",
    "NumpyPacketExecutor",
    "cupy_available",
    "packet_executor",
]
