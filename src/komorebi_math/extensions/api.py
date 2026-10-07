"""Public extension contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class ExtensionSpec:
    name: str
    version: str
    api_version: str
    modules: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    description: str = ""
    register: Callable[[], None] | None = None

    def activate(self) -> None:
        if self.register is not None:
            self.register()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["register"] = None if self.register is None else self.register.__qualname__
        return data
