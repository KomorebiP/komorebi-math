# Special-function machinery

`komorebi_math.special_functions` integrates selected machinery from the
Special Functions I–VIII releases and their corrected October 2026
consolidation. The namespace adds numerical Gamma shape transforms and exact
rational residue/scalar functions. It imports without loading mpmath, SciPy,
or SymPy. Numerical evaluation needs the existing `scientific` extra;
the exact rational functions need no additional dependency.

```python
from fractions import Fraction
from komorebi_math.special_functions import (
    bernoulli_packet_integer,
    gamma_sech_ambiguity,
    beta_binomial_refinement,
    meixner_pollaczek,
    q_racah_scalar,
)

# Exact rational Bernoulli polynomial packet, with B_1(x)=x-1/2.
packet = bernoulli_packet_integer(4, Fraction(3, 7), 3, 1)

# Unit log-Gamma windows with two positive shapes.
overlap = gamma_sech_ambiguity("1.25", "2.75", ".8", ".4", dps=60)

# p_(n,alpha+gamma)(x+y) expands into these normalized child amplitudes.
coefficients = beta_binomial_refinement(4, "1.25", "2.75", dps=60)
polynomial = meixner_pollaczek(4, 4, ".6", dps=60)

# A terminating scalar value with explicit exact rational parameters.
scalar = q_racah_scalar(2, 3, "2/7", "3/5", "4/9", "5/8", "2/3")
```

## Exact rational functions

`bernoulli_packet_integer(n,a,M,r)` evaluates
`M**(n-1)*B_n((a+r)/M)` with rational arithmetic. It accepts any rational
translation, including zero and negative values. It reuses the existing
Bernoulli-number implementation rather than adding a second number sequence.
Summing the `M` residues reconstructs `B_n(a)`; parent packets also reconstruct
from their child residues. Degree and residue must be nonnegative integers,
`M>=1`, and `r<M`.

`q_pochhammer`, `q_racah_lattice`, `q_racah_scalar`, and `racah_scalar` accept
integers, `Fraction` values, and exact rational strings. They reject floating
inputs. The q functions require `0<q<1`. The scalar sums retain terms through
`min(n,y)` and reject a zero denominator in every retained term, before any
cancellation. No value is assigned to a singular retained term.

These functions evaluate scalar formulas. They do not assert positive
orthogonality or identify those formulas with the finite matrix recoupling
construction. Correction C11 supplies an exact negative control: for
`q=1/2`, `alpha=beta=1/2`, `gamma=4`, `delta=1`, and `n,y` in `{0,1}`,
both lattice sites equal 3 and every table entry equals 1. The termination
condition `gamma*q=q**(-1)` therefore does not establish a positive
orthogonal family.

## Numerical families and conventions

Numerical functions return mpmath scalar values. Each call creates an isolated
precision context; `dps` is the requested decimal precision and must be at
least 15. The process-wide mpmath precision is unchanged. Shapes must be
positive finite real values. Real arguments must be finite, and booleans are
excluded from all scalar and integer domains.

Shape functions use extra isolated working digits to resolve cancellation at
large parameters, then round the returned result to the requested precision.
More than 10,000 working decimal digits, or excessive positive window-exponent
work, raises `ResourceLimitError`. This work limit is not a uniform bound on
elapsed time, and extra precision does not make the output a directed enclosure.

| Functions | Object and convention |
| --- | --- |
| `log_gamma_window`, `log_gamma_fourier` | Unit window `g_s(y)=Gamma(s)**(-1/2)*exp(s*y/2-exp(y)/2)`; unitary Fourier transform with `exp(-i*omega*y)` |
| `gamma_sech_ambiguity`, `shape_affinity` | Centered overlap `integral g_alpha(y+h/2)*g_gamma(y-h/2)*exp(i*eta*y) dy`; zero-shift affinity |
| `meixner_density`, `meixner_characteristic` | Fourier energy density and characteristic function `sech(h/2)**s` |
| `beta_logit_density` | Density of `log(U/(1-U))` for `U~Beta(alpha,gamma)` |
| `meixner_pollaczek` | Normalized polynomial with generating function `(1-i*z)**(-s/2+i*x)*(1+i*z)**(-s/2-i*x)` before normalization |
| `beta_binomial_refinement`, `shape_refinement_coefficient` | Positive two-child and many-child refinement amplitudes; degree vectors and shape vectors must have the same positive length |
| `shape_transfer_singular_value` | `sqrt((s)_n/(s+t)_n)` for `s>0` and increment `t>0` |
| `bernoulli_hurwitz`, `bernoulli_hurwitz_packet` | Entire-order `-z*zeta(1-z,a)` and residue refinement for real `a>0`; exact removable values at `z=0` |

The Bernoulli evaluator retains tiny nonzero orders instead of substituting
the zero-order value. It temporarily increases its isolated working precision
to resolve the Hurwitz pole cancellation, with a 10000-digit working cap.
The integer-order polynomial function has a broader translation domain than
the complex-order Hurwitz function.

These are numerical evaluations, not interval enclosures. `dps` alone does
not certify an error bound. Integration tests compare independent quadrature,
generating-function coefficients, exact residue reconstruction, polynomial
addition, and refinement-tree composition in a stated finite range. They do
not certify arithmetic depth asymptotics, unresolved signed estimates, or
all proofs in the source manuscripts.

## Source provenance and integration scope

The authoritative mathematical restrictions are the consolidation's
`manuscript/arithmetic_core.tex`, `shape_core.tex`, `recoupling_core.tex`,
and correction register C01–C12. The older standalone scripts combine
mathematics with plotting, mutable global precision, and fixed numerical
cutoffs; these scripts are not imported into the package.

| Source archive | SHA-256 |
| --- | --- |
| `Komorebi_Special_Functions_Consolidation_v0.1.zip` | `060b5559c14b3d9c02f2f028d3efea78663b6763ff0b23dacef6da6f58942a42` |
| `Komorebi_Special_Functions_Expanded_Visual_Guide_v0.4_Source.zip` | `df775d10c1940bf333848025379c3303fc136d51e417b4a5768cebdce73a965a` |

The expanded visual guide was compared with the consolidation, including its
shape and recoupling conventions. Existing `coordinates`, `analytic`,
`generating`, and `sequences` machinery already provides residue packets,
finite analysis/synthesis, normalized generating functions, and classical
polynomials. Those APIs remain the entry points for that existing work.
This addition provides the absent Gamma/Meixner shape family,
Bernoulli–Hurwitz order interpolation, and finite scalar q-Racah bridge.
Infinite arithmetic transfer estimates, the full representation-theoretic
matrix recoupling engine, and a continuous-series spectral transform are
outside this namespace's implemented contract.

The seven exact correction witnesses are reproducible separately from the
implementation regression suite. Their evidence distinguishes corrected
conventions; it does not turn bounded numerical tests into theorem proofs.
