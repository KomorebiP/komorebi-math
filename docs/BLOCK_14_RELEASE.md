# Block 14 release

Version `0.14.0a2` completes the general coordinate atlas under `komorebi_math.coordinates`.

The corrected scope covers arbitrary exact finite bases, coordinate graphs and commuting diagrams, normalized generating-function bases, recurrence and state-space realizations, classical and custom polynomial families, combinatorial transforms, affine and barycentric charts, formal changes of variable, tensor and operator transport, finite-field maps, numerical spectral transforms, packet permutations, and trusted persistence.

Euler-to-Fibonacci conversion remains a named test case inside the recurrence and polynomial layers. It is no longer presented as the organizing purpose of the subsystem.

## Installation

```powershell
python -m pip install .\komorebi_math-0.14.0a2-py3-none-any.whl
```

## Verification

```powershell
python -m komorebi_math verify coordinates
python -m komorebi_math verify block14
python -m komorebi_math golden
python -m komorebi_math api check --strict-signatures
```

## Representative commands

```powershell
python -m komorebi_math coordinates catalog
python -m komorebi_math coordinates basis --source-matrix "1,0;0,1" --target-matrix "1,1;0,1" --vector "3,5"
python -m komorebi_math coordinates polynomial --source legendre --target chebyshev-t --degree 8
python -m komorebi_math coordinates model --source pell --target tribonacci --count 20
python -m komorebi_math coordinates combinatorial --kind forward-difference --size 8 --values "0,1,4,9,16,25,36,49"
python -m komorebi_math coordinates spectral --kind haar --size 8
python -m komorebi_math coordinates packet --kind residue --length 12 --modulus 3
python -m komorebi_math coordinates formal --coordinate "0,1,1/2,1/6" --series "1,2,3,4"
```

The mathematical feature scope remains frozen after this corrected Block 14 release. Further work toward 1.0 is stabilization, platform validation, documentation, API curation, and release engineering.

## Source evidence

The corrected source release passed 896 unit tests, 170 aggregate verifier checks, 24 coordinate verifier checks, 15 behavioral golden fixtures, 52 packaged examples, and 454,486 independent coordinate comparisons with no recorded failures. The coordinate namespace contains 229 documented and annotated public names.
