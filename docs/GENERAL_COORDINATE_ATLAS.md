# General Coordinate Atlas

`komorebi_math.coordinates` is a general coordinate calculus for finite mathematical objects. It separates the object being represented from the coordinates used to represent it. A coordinate change may therefore connect arbitrary vector bases, sequence realizations, polynomial families, formal parameters, finite fields, state-space models, tensor components, or numerical spectral systems.

Euler numbers in Fibonacci companion coordinates remain available as one worked bridge. They do not define the scope of the namespace.

## Coordinate systems and atlases

`FiniteBasis` represents any linearly independent finite basis in a rational ambient space. Square bases span the complete ambient space. Rectangular bases describe exact subspaces.

```python
from komorebi_math.coordinates import CoordinateAtlas, basis_from_matrix, standard_basis

standard = standard_basis(3, name="standard")
shear = basis_from_matrix(
    "shear",
    (
        (1, 1, 0),
        (0, 1, 1),
        (0, 0, 1),
    ),
)

change = standard.connection_to(shear)
coordinates = change.apply((3, 5, 7))
assert change.inverse().apply(coordinates) == (3, 5, 7)

atlas = CoordinateAtlas("three dimensional atlas")
atlas.register_basis(standard)
atlas.register_basis(shear)
atlas.register_transform(change, include_inverse=True)
atlas.resolve("standard", "shear").transform.certificate().require()
```

An atlas stores named systems and directed transformations. It composes a shortest path, compares declared paths, and certifies commuting diagrams exactly.

## Normalized generating-function coordinates

A normalized series basis has elements

\[
e_n(z)=\frac{z^n}{\gamma_n}.
\]

Changing from normalization \(\gamma\) to \(\eta\) is diagonal:

\[
b_n=\frac{\eta_n}{\gamma_n}a_n.
\]

The coordinate namespace reuses the normalized bases from `komorebi_math.generating`. Ordinary, exponential, factorial-power, Fibonacci-factorial, Pochhammer, and custom declared normalizations can therefore be connected without introducing a second basis abstraction.

## Combinatorial coordinates

The namespace provides exact triangular changes among common sequence coordinates:

* monomial and falling-factorial coordinates
* monomial and rising-factorial coordinates
* signed and unsigned Stirling transforms
* Lah transforms
* binomial and inverse-binomial transforms
* sample values and Newton forward-difference coordinates
* divisor zeta and Möbius coordinates
* subset zeta and Möbius coordinates

These are ordinary coordinate transforms with explicit inverses. They can be inserted into an atlas or composed with any other compatible exact transform.

```python
from komorebi_math.coordinates import forward_difference_transform

values = (0, 1, 4, 9, 16, 25)
newton = forward_difference_transform(len(values))
assert newton.apply(values) == (0, 1, 2, 0, 0, 0)
assert newton.inverse().apply(newton.apply(values)) == values
```

## Polynomial-family coordinates

A `PolynomialFamily` is a graded basis of the polynomial ring through a declared finite degree. The catalog includes:

* monomials
* falling and rising factorials
* binomial polynomials
* Euler polynomials
* Bernoulli polynomials
* Fibonacci and Lucas polynomials
* Pell polynomials
* Chebyshev families of the first and second kind
* Legendre polynomials
* probabilists' and physicists' Hermite polynomials
* Laguerre polynomials
* Bernstein bases at a fixed degree
* Newton and Lagrange interpolation bases
* custom Appell families

```python
from komorebi_math.coordinates import get_polynomial_family, connection_transform

legendre = get_polynomial_family("legendre")
chebyshev = get_polynomial_family("chebyshev-t")
connection = connection_transform(legendre, chebyshev, 8)
connection.certificate().require()
```

The same machinery compiles derivative, integral, multiplication, affine-substitution, and finite-difference operators in arbitrary polynomial coordinates. Operators can then be transported between families by matrix conjugation or rectangular transport.

## Recurrence coordinates

A constant-coefficient recurrence of order \(d\) gives finite coordinates consisting of:

1. the first \(d\) values
2. the unique recurrence-defect stream

For

\[
a_{n+d}=c_0a_n+\cdots+c_{d-1}a_{n+d-1}+\delta_n,
\]

the coordinate vector is

\[
(a_0,\ldots,a_{d-1},\delta_0,\ldots).
\]

This representation is exact for every finite sequence, whether or not the sequence satisfies the recurrence homogeneously. It produces invertible finite changes between arbitrary recurrence models of possibly different orders.

Named constructors include Fibonacci, Lucas, Pell, Jacobsthal, Tribonacci, and Padovan models. `custom_recurrence_model` accepts any finite constant-coefficient rule.

```python
from komorebi_math.coordinates import get_recurrence_model

pell = get_recurrence_model("pell")
tribonacci = get_recurrence_model("tribonacci")
change = pell.coordinate_transform_to(tribonacci, 20)
change.certificate().require()
```

Euler-to-Fibonacci coordinates are one specialization of this general defect calculus. The forcing stream is retained because Euler numbers are not a homogeneous Fibonacci trajectory.

## Realization and state-space coordinates

The coordinate calculus supports finite linear realizations and input-output state-space models. It provides:

* companion realizations
* Green kernels and variation of constants
* observability matrices and exact state recovery
* controllability and observability coordinate systems
* cyclic Krylov coordinates
* intertwiner spaces solving \(BS=SA\)
* exact similarity transforms
* Markov-parameter invariance under state changes

For a state transform \(z=Tx\), the system

\[
x'=Ax+Bu,\qquad y=Cx+Du
\]

becomes

\[
z'=(TAT^{-1})z+(TB)u,\qquad y=(CT^{-1})z+Du.
\]

The implementation certifies the transformed dynamics and input-output equivalence.

## Affine, barycentric, and formal coordinates

`AffineCoordinateTransform` represents

\[
y=Ax+b
\]

with exact rational data. `BarycentricCoordinateSystem` converts points to and from the coordinates of a nondegenerate simplex.

`FormalCoordinateChange` represents an invertible finite jet

\[
w=\phi(z)=a_1z+a_2z^2+\cdots+a_Nz^N,
\qquad a_1\ne0.
\]

It supports exact truncation, composition, compositional inversion, pullback of scalar series, pushforward, and substitution matrices.

## Tensor and operator transport

The namespace distinguishes vectors, covectors, operators, bilinear forms, and higher tensor components. Under \(y=Tx\):

* vectors transform by \(T\)
* covectors transform by \(T^{-\mathsf T}\)
* square operators transform by \(TAT^{-1}\)
* bilinear forms transform by \(T^{-\mathsf T}GT^{-1}\)
* mixed tensors transform according to each covariant or contravariant index

Direct sums and tensor products of coordinate changes are also available. Pairing and bilinear-form invariance can be checked exactly.

## Finite-field coordinates

`FiniteFieldCoordinateTransform` performs exact linear coordinate changes over a prime field. Rational transforms can be reduced modulo a prime when every denominator is invertible. Inversion, composition, application, and certificates use modular Gauss-Jordan elimination.

## Spectral and packet coordinates

The numerical layer provides finite transforms for:

* the discrete Fourier transform
* the type-II discrete cosine transform
* Walsh-Hadamard coordinates
* complete Haar wavelet coordinates
* modal coordinates of diagonalizable matrices

These transforms carry numerical inverse and conditioning certificates.

The packet layer provides exact permutation coordinates for:

* residue or polyphase ordering
* digit-reversal ordering
* Chinese-remainder ordering
* tensor-axis permutations
* arbitrary stable sorted orders

This connects the coordinate calculus directly to `komorebi_math.signals`, packet systems, arithmetic residue decompositions, and multidimensional layouts.

## Persistence

The trusted persistence layer stores data-only records for:

* coordinate systems and atlases
* finite bases
* exact and numerical transforms
* affine and barycentric charts
* formal coordinate changes
* finite-field transforms
* recurrence and state-space models
* realization and recovery records
* polynomial coordinate operators
* coordinate budgets

Files use the package's versioned canonical envelope and SHA-256 payload integrity. Loading never imports arbitrary classes or reconstructs executable callbacks.

## Command line

```powershell
python -m komorebi_math coordinates catalog
python -m komorebi_math coordinates basis --source-matrix "1,0;0,1" --target-matrix "1,1;0,1" --vector "3,5"
python -m komorebi_math coordinates polynomial --source legendre --target chebyshev-t --degree 8
python -m komorebi_math coordinates model --source pell --target tribonacci --count 20
python -m komorebi_math coordinates combinatorial --kind forward-difference --size 8 --values "0,1,4,9,16,25,36,49"
python -m komorebi_math coordinates spectral --kind dft --size 8
python -m komorebi_math coordinates packet --kind residue --length 12 --modulus 3
python -m komorebi_math coordinates formal --coordinate "0,1,1/2,1/6" --series "1,2,3,4"
python -m komorebi_math coordinates euler-fibonacci --count 20
python -m komorebi_math verify coordinates
```

The final command is retained as a named example. The general API does not privilege Euler or Fibonacci coordinates.
