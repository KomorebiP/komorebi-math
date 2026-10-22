"""Compatibility import for existing development regression tests."""
from __future__ import annotations

try:
    from . import canonicalize_sdist as _implementation
except ImportError:
    import canonicalize_sdist as _implementation

globals().update({name: getattr(_implementation, name)
                  for name in dir(_implementation) if not name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
