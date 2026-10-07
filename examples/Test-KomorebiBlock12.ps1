$ErrorActionPreference = "Stop"
python -m komorebi_math verify dsp
python -m komorebi_math verify block12
python -m komorebi_math api check --strict-signatures
python -m komorebi_math api check --beta
python .\examples\production_dsp.py
