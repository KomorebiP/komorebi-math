# Episode 015 Literature Audit

## Question

The audit needed external context for two conventions only:

1. how finite p-adic precision differs from exact p-adic zero;
2. the standard Mahler/binomial expansion of exponential modes.

No novelty claim depends on these sources.

## Finite p-adic precision

SageMath's p-adic documentation describes a finite approximation in `Z/p^N Z` as an element known modulo `p^N`, with absolute precision `N`. It also distinguishes several fixed/capped precision models. Its p-adic element API follows the conventional exact valuation `v_p(0)=+infinity` for the zero p-adic element.

This matters because Komorebi's repaired result is a *project contract*, not a claim that every p-adic library must return `N` for zero. Komorebi's method is explicitly documented as a **visible valuation capped by the declared precision**. For that contract, a zero residue records only divisibility through the precision cap, so returning `N` avoids asserting information the truncated representation does not carry. Exact `padic_valuation(0,p)=+infinity` remains unchanged.

Sources checked 2026-09-23:

- SageMath 10.8 p-adic reference/tutorial, finite approximation and absolute precision: https://doc.sagemath.org/html/en/reference/padics/sage/rings/padics/tutorial.html
- SageMath p-adic generic element documentation, exact zero valuation convention: https://doc.sagemath.org/html/en/reference/padics/sage/rings/padics/padic_generic_element.html

Overlap classification: **known standard precision semantics; project-specific API-contract choice**.

## Mahler/binomial expansion

Keith Conrad's notes on Mahler expansions state the classical representation

`f(x)=sum a_n binom(x,n)`

and, for power functions in the p-adically appropriate regime,

`a^x = sum (a-1)^n binom(x,n)`.

For the finite nonnegative-integer identity used by Komorebi, no p-adic convergence assumption is needed: it is simply the ordinary binomial theorem, because `binom(x,n)=0` for `n>x`.

Source checked 2026-09-23:

- Keith Conrad, *Mahler expansions*, especially the binomial-basis theorem and power example: https://kconrad.math.uconn.edu/blurbs/gradnumthy/mahlerexpansions.pdf

Overlap classification: **known verbatim after notation translation**.

## Novelty boundary

Episode 015 does not propose new p-adic or Mahler mathematics. Its contribution is a project-specific semantic correction: exact mathematical objects and finite representations must not be assigned the same certificate semantics when the finite representation has discarded information.
