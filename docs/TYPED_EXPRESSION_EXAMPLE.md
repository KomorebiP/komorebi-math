# Exact typed expressions across two bases

This example follows one exact polynomial through antiderivation, two coordinate
changes, differentiation, and definite integration. Every vector and matrix is
over \(\mathbb Q\). The complete operator compiles to \(I_4\), while the final
scalar is \(5/4\).

The second half expresses the affine integration result \(106/5\) inside
the finite expression language and records three failure cases that must not be
rewritten away.

## 1. The two graded bases

Let the standard Fibonacci polynomials satisfy

\[
F_0(x)=0,
\qquad F_1(x)=1,
\qquad F_{n+1}(x)=xF_n(x)+F_{n-1}(x).
\]

Komorebi Math uses the graded basis \(G_k=F_{k+1}\). Through degree four,

\[
\begin{aligned}
G_0&=1,\\
G_1&=x,\\
G_2&=x^2+1,\\
G_3&=x^3+2x,\\
G_4&=x^4+3x^2+1.
\end{aligned}
\]

The Euler basis is defined by

\[
\frac{2e^{xt}}{e^t+1}=\sum_{n\ge0}E_n(x)\frac{t^n}{n!}.
\]

The required prefix is

\[
\begin{aligned}
E_0&=1,\\
E_1&=x-\frac12,\\
E_2&=x^2-x,\\
E_3&=x^3-\frac32x^2+\frac14,\\
E_4&=x^4-2x^3+x.
\end{aligned}
\]

These coefficient lists are the type witnesses. The words "Fibonacci" and
"Euler" are only display labels.

## 2. Input in Fibonacci coordinates

Take the degree-three Fibonacci basis element

\[
p(x)=G_3(x)=x^3+2x.
\]

In the ordered basis \((G_0,G_1,G_2,G_3)\), its coordinate vector is

\[
v_0=
\begin{bmatrix}
0\\0\\0\\1
\end{bmatrix}_{F_3}.
\]

Here the subscript \(F_3\) means the Fibonacci coordinate space through degree
three. It does not mean the standard Fibonacci polynomial \(F_3(x)\).

## 3. Basepoint antiderivative in the Fibonacci basis

Let \(J_{0,F}^{(4)}:\mathcal P_3^F\to\mathcal P_4^F\) be the primitive that
vanishes at zero. Its exact matrix is

\[
J_{0,F}^{(4)}=
\begin{bmatrix}
0&-\frac12&0&-\frac12\\
1&0&\frac13&0\\
0&\frac12&0&\frac14\\
0&0&\frac13&0\\
0&0&0&\frac14
\end{bmatrix}.
\]

Applying it gives

\[
v_1=J_{0,F}^{(4)}v_0
=
\begin{bmatrix}
-\frac12\\0\\\frac14\\0\\\frac14
\end{bmatrix}_{F_4}.
\]

Reconstruction confirms

\[
-\frac12G_0+\frac14G_2+\frac14G_4
=\frac14x^4+x^2.
\]

This is exactly the primitive of \(x^3+2x\) with value zero at the basepoint.

## 4. Change from Fibonacci to Euler coordinates

The exact degree-four coordinate change is

\[
C_{F_4\to E_4}=
\begin{bmatrix}
1&\frac12&\frac32&\frac32&3\\
0&1&1&\frac72&5\\
0&0&1&\frac32&6\\
0&0&0&1&2\\
0&0&0&0&1
\end{bmatrix}.
\]

Therefore

\[
v_2=C_{F_4\to E_4}v_1
=
\begin{bmatrix}
\frac58\\[2pt]
\frac32\\[2pt]
\frac74\\[2pt]
\frac12\\[2pt]
\frac14
\end{bmatrix}_{E_4}.
\]

The physical polynomial has not changed. The vector \(v_2\) still reconstructs
to \(x^4/4+x^2\).

## 5. Differentiate in Euler coordinates

Euler polynomials are an Appell family, so the tight derivative has the simple
matrix

\[
D_E^{(4)}=
\begin{bmatrix}
0&1&0&0&0\\
0&0&2&0&0\\
0&0&0&3&0\\
0&0&0&0&4
\end{bmatrix}.
\]

It maps the degree-four space to the degree-three space. The next vector is

\[
v_3=D_E^{(4)}v_2
=
\begin{bmatrix}
\frac32\\[2pt]
\frac72\\[2pt]
\frac32\\[2pt]
1
\end{bmatrix}_{E_3}.
\]

Indeed,

\[
\frac32E_0+\frac72E_1+\frac32E_2+E_3=x^3+2x.
\]

## 6. Return to Fibonacci coordinates

The degree-three change back to Fibonacci coordinates is

\[
C_{E_3\to F_3}=
\begin{bmatrix}
1&-\frac12&-1&\frac74\\
0&1&-1&-2\\
0&0&1&-\frac32\\
0&0&0&1
\end{bmatrix}.
\]

It returns the original vector exactly:

\[
v_4=C_{E_3\to F_3}v_3
=
\begin{bmatrix}
0\\0\\0\\1
\end{bmatrix}_{F_3}
=v_0.
\]

No tolerance or floating comparison appears in this calculation.

## 7. The compiled operator is exactly \(I_4\)

The whole map on \(\mathcal P_3^F\) is

\[
\mathcal O=
C_{E_3\to F_3}
D_E^{(4)}
C_{F_4\to E_4}
J_{0,F}^{(4)}.
\]

Exact matrix multiplication gives

\[
\mathcal O=
\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}
=I_4.
\]

This is the concrete naturality form of \(DJ_0=I\). The coordinate changes
do not alter the physical operator.

## 8. Integrate the recovered polynomial

The exact integration covector on \([0,1]\) in the Fibonacci degree-three
basis is

\[
\ell_{F_3,[0,1]}=
\begin{bmatrix}
1&\frac12&\frac43&\frac54
\end{bmatrix}.
\]

The entries are the integrals of \(G_0,G_1,G_2,G_3\). Hence

\[
\ell_{F_3,[0,1]}v_4
=\frac54.
\]

Directly,

\[
\int_0^1(x^3+2x)\,dx
=\frac14+1
=\boxed{\frac54}.
\]

## 9. Affine and weighted integration regression

Let \(\mu=dx\) on \([0,1]\), let \(y=2x+1\), and let
\(\nu=\phi_\#\mu\). Then

\[
d\nu(y)=\frac12\,dy
\quad\text{on }[1,3].
\]

For

\[
f(y)=y^4-2y+1,
\]

exact affine substitution gives

\[
f(2x+1)=16x^4+32x^3+24x^2+4x.
\]

The monomial coefficient vector is therefore

\[
c_x=
\begin{bmatrix}
0&4&24&32&16
\end{bmatrix}^{\mathsf T}.
\]

Pairing with the \([0,1]\) moment row gives

\[
\begin{bmatrix}
1&\frac12&\frac13&\frac14&\frac15
\end{bmatrix}c_x
=\frac{106}{5}.
\]

The direct weighted moment row on \([1,3]\) is

\[
\ell_{[1,3],1/2}
=
\begin{bmatrix}
1&2&\frac{13}{3}&10&\frac{121}{5}
\end{bmatrix}.
\]

With \(c_y=(1,-2,0,0,1)^{\mathsf T}\),

\[
\ell_{[1,3],1/2}c_y
=\frac{106}{5}.
\]

Thus the affine pullback and pushforward-density paths agree exactly.

## 10. Adversaries

### Same label, different basis

Consider

\[
B=(1,x,x^2),
\qquad
B'=(1,x,x^2+x).
\]

They may both be displayed as `custom basis`, and both have dimension three,
but their exact witnesses differ. Their structural fingerprints differ, so a
coordinate map certified for \(B\) cannot be reused for \(B'\).

### The false reverse identity

The equation \(DJ_0=I\) is valid on the smaller source space. The tempting
reverse equation \(J_0D=I\) is false. Apply it to the constant polynomial:

\[
J_0D(1)=J_0(0)=0\ne1.
\]

The correct formula is

\[
J_0D=I-u e_0.
\]

The rank-one term removes the constant value at the basepoint.

### Omitted pushforward density

If the factor \(1/2\) is omitted from the integral on \([1,3]\), every moment
doubles and the result becomes

\[
\int_1^3(y^4-2y+1)\,dy=\frac{212}{5}.
\]

This is not another representation of \(106/5\). It is a different measure.
The density is therefore part of the integration primitive's exact structural
data.

### Singular inverse

The matrix

\[
S=
\begin{bmatrix}
1&1&0\\
0&0&0\\
0&0&1
\end{bmatrix}
\]

has determinant zero. An exact inverse does not exist, regardless of its label
or square shape. The basis constructor's graded leading-coefficient condition
prevents such a matrix from becoming a valid polynomial basis witness, and the
exact inverse routine rejects it. No cancellation rewrite may be certified from
this matrix.

## Independent replay

The standalone verifier implements the basis formulas and all matrix arithmetic
locally with `Fraction`. It imports neither the expression engine nor the
existing polynomial-family implementation.

```powershell
python verification/verify_expression_math.py
```

Machine-readable output can be written with

```powershell
python verification/verify_expression_math.py --quiet --output qualification/expression-math.json
```

Its 15 independent checks form a focused mathematical oracle. It is not, by itself, a wheel, platform, or full-package
qualification claim.
