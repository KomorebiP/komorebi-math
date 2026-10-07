# R4 Block 3: tensor and numerical decision repairs

Checkpoint: `1.0.0b4-repair-r4-block3`. Wheel build: `4b3`.
Package/API version remains `1.0.0b4`. This is a scoped development checkpoint,
not final R4, production 1.0, a universal proof checker or publication approval.
The parent is the unchanged R4 Block 2 checkpoint. All inherited tests remain.

## Changed scope

- Exact sparse provider construction does not tolerance-prune nonzero exact coefficients.
- Exact sparse contractions, stored-jet evaluation and streaming evaluation promote
  NumPy integer operands to Python integers before multiplying. Object output arrays
  alone were not sufficient to avoid fixed-width intermediate overflow.
- Snapshot and streaming capture retain undecidable symbolic coefficients in memory
  and reject known nonfinite coefficients. Portable symbolic codecs are not added.
- The selected CPU execution, snapshot comparison, dense symmetry, sparse roundtrip,
  storage, tree and obstruction-square certificates decide residual objects before
  computing floating diagnostics. Unknown equality does not grant a positive equation.
- Global callback correctness and multilinearity remain supplied premises, not facts
  established by finite sampling. Finite storage comparisons do not prove global equality.
- The Hilbert obstruction projector derives rank, inverse, image and cokernel from one
  retained SVD mask. Numerical model assumptions stay explicit. Finite-entry checks and
  a robust norm prevent the demonstrated zero-tolerance underflow false-solvability.

## Compatibility

Public names, kinds and reflected signatures are intended to remain unchanged.
Acceptance decisions intentionally become stricter. Exact sparse constructor tolerance
no longer discards coefficients. Unknown symbolic coefficients may be retained, but the
existing stream codecs still reject unsupported symbolic objects. An exact arithmetic
label does not make input binary floats exact representations of pre-rounding real data.
Diagnostic infinity may report that an exact residual has no usable float diagnostic.

## Evidence and reproduction

The enclosing checkpoint's BLOCK3_VERIFICATION.json and REPAIR_REPORT.md select the
final evidence. Historical R1-R3 and Block 1-2 receipts do not certify changed bytes.
See KNOWN_ISSUES_BLOCK3.md for the residual audit boundary. The licence is unchanged.
