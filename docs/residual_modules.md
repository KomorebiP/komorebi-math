# Exact residual modules over prime-power rings

`komorebi_math.generating.residual_modules` implements the finite constructions
in Cartier Compilers, Chapter 24, from the standardized publication edition of
8 October 2026. The coefficient ring is `Z/p**r Z`, with prime `p` and positive
precision `r`. Matrices act on column states. Words are read left to right;
appending a digit applies its matrix after the preceding word.

```python
from komorebi_math.generating.residual_modules import (
    RingLinearRealization,
    canonical_residual_module,
    finite_word_comparison,
)

# The paper uses row states; this matrix is its transpose.
realization = RingLinearRealization(
    2, 2, (((1, 0), (2, 1)),), (1, 0), (2, 0)
)
residual = canonical_residual_module(realization)
assert residual.ambient_cardinality == 16
assert len(residual.reachable_states) == 8
assert residual.carrier_cardinality == 2
assert residual.additive_p_exponents() == (1,)
assert residual.verify_laws().valid
assert finite_word_comparison(realization, residual).equivalent
```

This is Example 24.4. The quotient is the two-element module `Z/2Z`, which is
nonfree over `Z/4Z`. The seed has order two in the quotient, and multiplication
by scalar two kills it. Every seed iterate has output two, so a deterministic
output automaton would have one state. The residual linear carrier has two
elements: deterministic automaton size and linear module cardinality are
different comparison classes.

The construction enumerates the complete seed orbit, forms its ring span, and
refines output classes until every digit preserves the equivalence relation.
Stable equivalence means equal outputs after every future word. The resulting
classes are the reachable/unobservable quotient from Theorem 24.1.
`project` exposes its quotient map, and `unobservable_states` exposes its kernel
within the reachable module. Explicit addition, scalar, digit-action and output
tables preserve the arithmetic without presuming a basis. Local class indices
are deterministic labels; the source's canonical isomorphism statement does
not imply identical coordinates in different presentations.

`additive_p_exponents()` determines the cyclic factors `Z/p**e Z` by counting
elements killed by each power of `p`. It distinguishes, for example, the
eight-element module `Z/4Z` times `Z/2Z` from a free module. `verify_laws()`
exhaustively checks the additive group, scalar ring, action linearity and output
linearity laws. Its cubic associativity check has a separate work preflight.

`finite_word_comparison(left, right)` accepts free realizations and these
nonfree quotients over the same ring and ordered finite alphabet. When the
carrier sizes are `p**b` and `p**c`, Theorem 24.5 says that equality on every word
of length strictly below `b+c` decides complete behavior. The implementation
uses product-orbit breadth-first search, checking each distinct reachable pair
once. A mismatch returns a shortest distinguishing word, its two outputs and
the strict theorem horizon. Termination without a mismatch certifies equality
of the supplied finite behaviors. The test includes the empty word and all
noncanonical digit strings; selected integer-index outputs alone do not decide
this question without a separately proved zero-padding law.

`ResidualModuleBudget` bounds the scalar count, ambient carrier, dimensions,
alphabet, input bit sizes, word length, operation tables and estimated work.
Exponents and ambient cardinalities are bounded before exponentiation or
enumeration. Table size is checked before table allocation, and product size
before comparison. Exceeding a cap raises `ResourceLimitError`; it never returns
an incomplete result as equality. Defaults admit at most 4,096 ambient elements,
1,000,000 operation entries and 10,000,000 estimated work units. These algorithms
are intentionally suitable for small finite carriers. They do not provide
polynomial complexity, a free basis, cross-precision morphisms, infinite-module
constructions, an adapter for every external Cartier representation, or a
minimum deterministic automaton.

The regression suite independently closes state sets under all pairwise sums
and actions, then compares kernels and classes against all words inside the
source's proven finite horizon. It covers free, nonfree and mixed modules,
noncommuting matrices, canonical digit strings versus extra padding, empty
alphabets, zero dimensions, scalar laws, resource refusal and malformed inputs.
These bounded software checks supply implementation evidence; they do not
replace the paper's general proofs or establish formal Python verification.

Run the worked example with:

```bash
PYTHONPATH=src python examples/residual_modules.py
```
