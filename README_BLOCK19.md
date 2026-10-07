# Komorebi Math Block 19

Block 19 closes the largest remaining finite calculus gap: integration against
typed finite measures.

It provides exact rational polynomial integration, positive atomic and interval
measures, products and pushforwards, signed quadrature functionals, explicit
quadrature error strength, exact FTC and integration-by-parts checks, finite
Fubini, affine change of variables, and coordinate integration covectors.

The release is `1.0.0b3` and remains research-only.  The 131-name stable API,
83-name beta API, 51 portable codecs, and 10 stable semantic cases remain frozen.

Rigorous composite Simpson certification is deliberately narrow: it requires a
provenance-matched `ExactPolynomial` and exact rational `IntervalMeasure`, with
exact rational nodes, weights, evaluation, summation, derivative bound, and
error bound.  Arbitrary floating callables receive an estimated Richardson
statement without a supplied bound, or a conditional statement when a
derivative bound is explicitly assumed; neither path is labeled rigorous.

Run the installed verifier:

```powershell
python -m komorebi_math verify integration
python -m komorebi_math verify block19
```

The committed adversarial runner exercises 47 cases, and the independent
mathematical verifier exercises 16 checks.

Run the worked example from the source tree:

```powershell
$env:PYTHONPATH="$PWD\src"; python .\examples\block19_measure_integration_worked_example.py
```

See `docs/BLOCK_19_WORKED_EXAMPLE.md` for the full derivation and
`docs/FINITE_MEASURE_INTEGRATION.md` for the API and scope boundary.
