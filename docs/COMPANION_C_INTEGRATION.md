# Companion C Integration Map

The Companion C maximal visual atlas is useful to Komorebi Math, but it should not be copied wholesale into the production API immediately before the 1.0 freeze.

## Material adopted in Block 16

Block 16 adopts its strongest architectural framing:

\[
P_\alpha=U^{-1}E_\alpha U,
\qquad
x=\sum_\alpha P_\alpha x,
\qquad
T=\sum_{\alpha,\beta}P_\beta T P_\alpha,
\]

and the general analysis-synthesis obligation

\[
SA=I
\]

on the declared object or model space.

This framing is reflected in the stable semantic contract for complete decomposition, polyphase refinement, reconstruction, and certificates.

## Cross-checks relevant to the current package

The Companion C numerical ledger includes finite checks for:

- packet synthesis
- operator-block synthesis
- polyphase synthesis
- two-to-four refinement
- mixed-radix packet synthesis
- dyadic and ternary residue refinement
- perfect-reconstruction filter banks
- Fourier and wavelet coordinate synthesis
- coordinate transport
- CRT packet synthesis
- overlap-add synthesis
- quantum and modal synthesis

These are useful independent targets. Block 16 uses the overlapping polyphase, mixed-radix, lattice, persistence, and reconstruction identities in package-native golden fixtures.

## Material retained as research or documentation

The following Companion C areas should remain outside the frozen API until separately reviewed:

- standalone plotting and page-layout helpers
- duplicate Fourier, wavelet, DCT, DST, Walsh, Gabor, and sampling implementations
- visual-only diagnostics without package certificates
- PDE, multigrid, Krylov, and low-rank examples that do not yet map to one stable interface
- heuristic comparison plots and model-specific RMSE claims

## Promotion rule

A Companion C construction may enter the stable package only after it has:

1. a package-native mathematical convention
2. a typed public interface
3. exactness or numerical certificate semantics
4. deterministic regression tests
5. a persistence decision
6. API tier classification
7. documentation that states its finite scope

This policy extracts value from the atlas without reopening uncontrolled feature expansion before 1.0.
