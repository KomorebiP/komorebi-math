"""Compatibility import for the current source audit."""
try:
    from .source_audit import *
    from .source_audit import _shell_mode_diagnostics
except ImportError:
    from source_audit import *
    from source_audit import _shell_mode_diagnostics

if __name__ == "__main__":
    raise SystemExit(main())
