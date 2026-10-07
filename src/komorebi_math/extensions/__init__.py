"""Extension API and runtime discovery."""

from .api import ExtensionSpec
from .discovery import ENTRY_POINT_GROUP, ExtensionLoadResult, discover_extensions

__all__ = ["ENTRY_POINT_GROUP", "ExtensionLoadResult", "ExtensionSpec", "discover_extensions"]
