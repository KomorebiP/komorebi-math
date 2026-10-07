# Block 19 worked example

This example follows one integral through measure transport, polynomial pullback,
coordinate change, exact evaluation, and certified numerical evaluation.

## 1. Base measure and chart

Let \(\mu=dx\) on \([0,1]\), and define

\[
\phi(x)=2x+1.
\]

The pushforward \(\nu=\phi_\#\mu\) is supported on \([1,3]\).  Relative to
ordinary Lebesgue measure \(dy\), its density is

\[
d\nu(y)=\frac12\,dy.
\]

The factor is \(1/|\phi'|=1/2\).  This density statement uses the Jacobian.
The abstract pushforward identity itself does not:

\[
\int g\,d(\phi_\#\mu)=\int g\circ\phi\,d\mu.
\]

## 2. Integrand and exact pullback

Take

\[
f(y)=y^4-2y+1.
\]

The exact affine pullback is

\[
f(2x+1)=16x^4+32x^3+24x^2+4x.
\]

Therefore

\[
\begin{aligned}
\int f\,d\nu
&=\frac12\int_1^3 (y^4-2y+1)\,dy\\
&=\int_0^1 (16x^4+32x^3+24x^2+4x)\,dx\\
&=\frac{16}{5}+8+8+2\\
&=\boxed{\frac{106}{5}}.
\end{aligned}
\]

Block 19 evaluates both sides with rational arithmetic and certifies their exact
equality.

## 3. The integral as a coordinate covector

Let \(c\) be the coefficient vector of \(f\), and let \(\ell\) be the vector of
basis-function integrals.  Then

\[
\int f\,d\nu=\ell c.
\]

The same physical polynomial gives the following exact pairings.

| Basis | Coordinates of \(f\) | Integration covector | Pairing |
|---|---:|---:|---:|
| Monomial | \((1,-2,0,0,1)\) | \((1,2,13/3,10,121/5)\) | \(106/5\) |
| Euler | \((1/2,0,3,2,1)\) | \((1,3/2,7/3,15/4,31/5)\) | \(106/5\) |
| Fibonacci | \((3,-2,-3,0,1)\) | \((1,2,16/3,14,191/5)\) | \(106/5\) |

If \(C\) maps old coefficients to new coefficients, the covector law is

\[
\ell_{\mathrm{new}}C=\ell_{\mathrm{old}}.
\]

The example certifies this contragredient transport exactly.  It also checks the
basis-independent fundamental theorem identity

\[
\ell D=\frac12(e_3-e_1).
\]

## 4. Certified numerical evaluation

The example passes the exact polynomial itself, its exact rational interval, and
the fourth-derivative bound derived from that same polynomial and interval to
composite Simpson quadrature.  With 16 subintervals, exact rational nodes,
weights, evaluations, and summation return

\[
Q=\frac{130253}{6144}=21.200032552083333\ldots.
\]

The exact integral is \(106/5=21.2\).  Since \(f^{(4)}=24\), the
provenance-matched exact calculation derives the rigorous error bound

\[
|I-Q|\le \frac1{30720}
=3.255208333333333\ldots\times10^{-5}.
\]

Here the observed error is exactly \(1/30720\), so it lies inside the bound.  The
rigorous label depends on the complete provenance chain: a matching
`ExactPolynomial`, the same exact rational `IntervalMeasure`, and exact rational
evaluation and bound arithmetic.  If no derivative bound is supplied, the
fine/coarse statement is estimated.  If an arbitrary floating callable is used
with an assumed derivative bound, the result is conditional rather than
rigorous because bound applicability and floating-point roundoff are not proved.

## 5. Why sampled agreement is not a proof

The polynomial

\[
h(x)=x^2\left(x-\frac12\right)^2(x-1)^2
\]

vanishes at every node of the three-node Simpson rule on \([0,1]\).  The sampled
quadrature value is zero, but

\[
\int_0^1 h(x)\,dx=\frac1{840}.
\]

The exactness checker finds that Simpson is certified only through degree three,
while \(h\) has degree six.  It rejects the attempted exact-integral certificate.

## Run it

From the unpacked handoff:

```powershell
$env:PYTHONPATH="$PWD\src"; python .\examples\block19_measure_integration_worked_example.py
```

Machine-readable output:

```powershell
$env:PYTHONPATH="$PWD\src"; python .\examples\block19_measure_integration_worked_example.py --json
```
