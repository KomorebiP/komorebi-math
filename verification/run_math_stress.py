"""Thin source-checkout launcher for the installed mathematical stress CLI.

Prefer ``python -m komorebi_math.audit`` after installing the wheel. This file
does not add the source checkout to sys.path or import repository-only tests.
"""

from komorebi_math.audit.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
