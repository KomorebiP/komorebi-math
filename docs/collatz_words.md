# Exact finite Collatz words

`komorebi_math.arithmetic.collatz` implements the valuation-word machinery in
the Collatz reader edition 1.1, Sections 2, 3 and 6. Integers have arbitrary
precision. Inputs are positive odd seeds and positive integral valuations.

```python
from komorebi_math.arithmetic.collatz import CollatzWord, first_odd_descent

word = CollatzWord((1, 3))
assert (word.residue, word.modulus) == (19, 32)
assert word.orbit(51) == (51, 77, 29)
assert word.descends(51)
word.realization_certificate(51).require()

exit_word = CollatzWord((1, 1, 10))
assert exit_word.orbit(8343) == (8343, 12515, 18773, 55)
assert first_odd_descent(8343, max_steps=3).word == exit_word
```

For a word of length `k`, total depth `S` and offset `C`, the endpoint is
`(3**k*n+C)/2**S`. Its exact integer cylinder retains the terminal parity bit:
`3**k*n+C == 2**S (mod 2**(S+1))`. `apply` rejects a seed outside that cylinder.
`affine_value` evaluates the rational expression without claiming realization.
The word `(1,3)` has an integral affine value at `n=3`, but that seed realizes
`(1,4)`. This is a built-in negative control.

`concatenate` preserves word order and the composed offset. `descends` compares
the endpoint with the original seed using the exact inequality
`(2**S-3**k)*n > C`. A contracting coefficient alone cannot certify this:
the paper's seventeen-step word from 165 ends at 167. Its first actual descent
already occurs at 31, so it is not a candidate nonterminating orbit.

`uniform_descent_start` returns the first positive cylinder member strictly
above the integer fixed-point floor, when the coefficient contracts. This
arithmetic formula applies to that fixed word's progression; it does not say
that every starting integer belongs to a descending word.

`first_odd_descent` stops at the first comparison below its initial source.
Exhausting `max_steps` gives `status == 'inconclusive'` and withholds the primary
certificate. Exceeding the explicit depth or bit-size cap raises
`ResourceLimitError`. Seed one has separate terminal status. Finite successful
searches do not establish universal descent or the Collatz conjecture.

The regression suite compares all 85 words of length zero through three with
valuations one through four against direct integer division, checks the exact
progressions and strict fixed-point boundary, and tests exhaustion and input
ownership. The installed stress framework can replay these finite controls.
