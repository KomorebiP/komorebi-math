"""Compatibility import for the current package audit."""
try:
    from .package_audit import *
except ImportError:
    from package_audit import *

if __name__ == "__main__":
    raise SystemExit(main())
