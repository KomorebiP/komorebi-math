from __future__ import annotations
import json
from pathlib import Path
from komorebi_math.arithmetic import TruncatedPadic, UnitModeMahler
from komorebi_math.generating import TruncatedPadicDomain

result = {
    'schema':'komorebi-math.episode015-installed-block19-probe',
    'schema_version':1,
    'import_path': str(Path(__import__('komorebi_math').__file__).resolve()),
    'arithmetic_zero_visible': TruncatedPadic(3,4).valuation(0),
    'arithmetic_modulus_visible': TruncatedPadic(3,4).valuation(81),
    'generating_zero_visible': TruncatedPadicDomain(3,4).valuation(0),
    'undertruncated_valid': UnitModeMahler(2,1,0).certificate(5, degree=0).valid,
    'full_degree_valid': UnitModeMahler(2,1,0).certificate(5, degree=5).valid,
}
result['passed'] = (
    result['arithmetic_zero_visible'] == 4
    and result['arithmetic_modulus_visible'] == 4
    and result['generating_zero_visible'] == 4
    and result['undertruncated_valid'] is False
    and result['full_degree_valid'] is True
)
print(json.dumps(result,indent=2,sort_keys=True))
raise SystemExit(0 if result['passed'] else 1)
