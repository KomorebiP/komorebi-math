# Komorebi Math 1.0.0b2 — Block 18 report

Block 18 closes the finite algebra of redundant analysis--synthesis
presentations at the research tier.

## Added mathematics

- Oblique coefficient projections and complementary gauge projections
- Quotient metrics and canonical coefficient representatives
- Canonical lifts of rectangular physical maps
- Composition across different middle presentations through explicit bridges
- Small coefficient blocks with ordinary block-matrix multiplication
- Separate physical and dual adjoints
- Exact rational systems, maps, compositions, transitions, sums, and tensors
- A consistent SVD threshold for numerical canonical duals

The physical reconstruction components remain available, but are explicitly
not treated as ordinary compositional blocks in a redundant presentation.

## Compatibility

The stable surface remains 131 symbols, beta remains 83 symbols, the codec
inventory remains 51 entries, and all ten stable semantic fixtures pass.  The
six new public objects are research-only and are additive in both the root and
coordinate research namespaces.

## Validation

- 14/14 adversarial Block 18 tests
- 10/10 independent Block 18 mathematical checks
- 12/12 installed Block 18 plus frozen-stable CLI checks
- 174/174 cumulative installed mathematical checks through Block 15
- Clean wheel installation passed outside the source tree
- Fresh sdist extraction, wheel rebuild, installation, and verification passed

Windows and CUDA execution were not available on the build host and are not
claimed by this report.
