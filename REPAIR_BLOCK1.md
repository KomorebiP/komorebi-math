# R4 Block 1 — certificate soundness

Development checkpoint: 1.0.0b4-repair-r4-block1, wheel build 4b1.
Predecessor: byte-exact R3, SHA-256 24cbd13e88dd5ab9aa4cf13e4530ece2a6643a09ce79c160b99b6f2d4562436a.

## Contract

Seven exact certificate paths now decide equality from the residual object,
not its floating norm. The private residual audit uses SymPy simplification
and three-valued zero predicates. Unknown yields valid=False with an explicit
unresolved diagnostic; it is not a proof of inequality. Nonzero/unknown witnesses
retain their symbolic expression. Numeric residuals remain tolerance checks on
already-computed arrays, not error bounds for their real-arithmetic antecedents.

A diagnostic norm unavailable as a finite positive float uses infinity as an
unavailable sentinel. It must not be interpreted as the true residual size.
Exact proved zeros have zero diagnostics. SymPy Float inputs and arbitrary
symbolic domain/assumption validity are not upgraded into a proof of exact input
semantics by this change.

Packet-operator, packet-morphism and refinement certificates recompute parent
packet axioms. Existing returned certificates remain historical observations,
not self-invalidating proofs. Arbitrary persistent certificates can still be
constructed or decoded; full snapshot binding is a later block.

PacketOperator.adjoint re-resolves the ambient adjoint in the original reversed
packet systems. This preserves the return-system and naming API. It deliberately
does not merely transpose/adjoint embedded blocks into incompatible oblique
families. Reprojection can cost more; no speedup is claimed.

PacketSystem, Settings/configuration, OperatorMorphism and NumPy zero checks
reject nonfinite/negative tolerances. NumPy zero tolerance uses elementwise zero
and finite-input checks. Other numerical tolerances use a hypot-reduced Frobenius
norm to avoid the demonstrated squaring underflow. Alternate backends remain
outside this numeric repair's qualification.

Scalar self-map and strict contraction guards use Fraction arithmetic on the
exact rational values of the normalized binary-float inputs. Tolerance never
relaxes the guards. Displayed image/contraction values are rounded upward.
Picard tails freshly recheck scalar norm data, compute an exact rational tail,
and round upward. The actual operator-norm bounds and Banach completeness remain
external assumptions. This is not a proof of those assumptions.

The quadratic Catalan helper proposes a cancellation-resistant root and checks
up to 16 adjacent float radii exactly. Unsuccessful positive-discriminant search
is UNRESOLVED_BOUNDARY, not a no-solution theorem. Exact power/tail calculations
use a declared 1,000,000-bit-work proxy cap, checked before exponentiation. This
is a rejection policy, not a certified worst-case bit/time/memory complexity.
Large rational diagnostics use a lossless hexadecimal representation.

## Compatibility

The stable namespace retains its 131 names, kinds and reflected signatures.
The shared Certificate data-class fields and the package version are unchanged.
Invalid refinements and nonfinite/negative tolerances are now rejected rather
than silently accepted. Some certificates formerly marked true become false.
No test is allowed to accept an arbitrary exception in place of these deliberate
ValidationError contracts. New mathematical feature APIs are not introduced.

## Reproduction

With the recorded dependencies already present:

    python -m pip install --no-index --no-deps --force-reinstall release/dist/komorebi_math-1.0.0b4-4b1-py3-none-any.whl

Run from this source directory, ensuring the intended source or installed import
path is selected explicitly:

    python -B -m pytest tests/test_repair_r4_block1.py -q -p no:cacheprovider
    python -B -m pytest tests -q -p no:cacheprovider

The accompanying checkpoint bundle supplies a bounded command runner, raw outputs,
JUnit results, input hashes and source/installed parity checks. Original R3
regressions and historical failed exploratory runs remain in separate evidence.
Full-suite tests inspect release/dist, so rebuilding distributions is required
after any source/test change. Do not weaken the stale-distribution check.

Windows, PowerShell, CUDA, full dependency isolation, strict MkDocs, all 220
review candidates and full publication qualification are not established here.
No Lean compilation or independent mathematical referee was performed.
