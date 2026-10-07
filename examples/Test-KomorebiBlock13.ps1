$ErrorActionPreference = "Stop"

python -m komorebi_math info
python -m komorebi_math verify generating
python -m komorebi_math verify block13
python -m komorebi_math api check --strict-signatures
python -m komorebi_math api check --beta
python -m komorebi_math generating mask --mask "1,1,1" --radix 2 --count 20
python -m komorebi_math generating cartier --numerator "1,1" --denominator "1,-1" --prime 5 --count 20
python -m komorebi_math generating kernel --index 100 --subindex 37 --prime 3 --exponent 5 --power 2
python -m komorebi_math generating bell --index 137 --prime 5 --power 9
python .\examples\custom_generating_functions.py
python .\examples\mahler_cartier_compilers.py
python .\examples\prime_power_generating.py
