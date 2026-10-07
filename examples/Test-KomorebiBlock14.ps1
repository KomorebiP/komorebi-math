param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python -m komorebi_math info
& $Python -m komorebi_math verify coordinates
& $Python -m komorebi_math verify block14
& $Python -m komorebi_math coordinates catalog
& $Python -m komorebi_math coordinates basis --source-matrix "1,0;0,1" --target-matrix "1,1;0,1" --vector "3,5"
& $Python -m komorebi_math coordinates model --source pell --target tribonacci --count 20
& $Python -m komorebi_math coordinates polynomial --source legendre --target chebyshev-t --degree 10
& $Python -m komorebi_math coordinates combinatorial --kind forward-difference --size 8 --values "0,1,4,9,16,25,36,49"
& $Python -m komorebi_math coordinates spectral --kind haar --size 8
& $Python -m komorebi_math coordinates packet --kind residue --length 12 --modulus 3
& $Python -m komorebi_math coordinates formal --coordinate "0,1,1/2,1/6" --series "1,2,3,4"
& $Python -m komorebi_math coordinates euler-fibonacci --count 20
& $Python -m komorebi_math golden
& $Python -m komorebi_math api check --strict-signatures
& $Python -m komorebi_math api check --beta
