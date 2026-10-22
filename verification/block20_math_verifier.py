"""Compatibility import for the independent subject verifier."""
from pathlib import Path
import sys
_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)
from verification import verify_expression_math as _implementation
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
