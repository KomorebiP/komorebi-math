import json
import numpy as np
from pathlib import Path

rng = np.random.default_rng(20260922)
records = []
max_upper_ratio = 0.0
max_lower_ratio = 0.0
violations = 0
composition_violations = 0
cases = 0

for dim in (2, 3, 4):
    I = np.eye(dim)
    for _ in range(1000):
        # Build invertible gauges with a broad singular-value range while avoiding overflow.
        q1, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        q2, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        scales = 10.0 ** rng.uniform(-3.0, 3.0, size=dim)
        g = q1 @ np.diag(scales) @ q2

        q3, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        q4, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        scales_h = 10.0 ** rng.uniform(-3.0, 3.0, size=dim)
        h = q3 @ np.diag(scales_h) @ q4

        E = rng.normal(size=(dim, dim))
        r = np.linalg.norm(E, 'fro')
        gin = np.linalg.inv(g)
        transformed = g @ E @ gin
        rt = np.linalg.norm(transformed, 'fro')
        k = np.linalg.cond(g, 2)
        lower = r / k
        upper = r * k
        tol = 2e-10 * max(1.0, upper, rt)
        if rt < lower - tol or rt > upper + tol:
            violations += 1
        max_upper_ratio = max(max_upper_ratio, rt / upper if upper else 0.0)
        max_lower_ratio = max(max_lower_ratio, lower / rt if rt else 0.0)

        combined = h @ g
        combined_rt = np.linalg.norm(combined @ E @ np.linalg.inv(combined), 'fro')
        kh = np.linalg.cond(h, 2)
        product_k = k * kh
        comp_lower = r / product_k
        comp_upper = r * product_k
        comp_tol = 5e-10 * max(1.0, comp_upper, combined_rt)
        if combined_rt < comp_lower - comp_tol or combined_rt > comp_upper + comp_tol:
            composition_violations += 1
        cases += 1

result = {
    'schema': 'aster-inquiry-episode004-conditioning-stress',
    'seed': 20260922,
    'cases': cases,
    'dimensions': [2, 3, 4],
    'single_gauge_violations': violations,
    'composed_envelope_violations': composition_violations,
    'max_actual_over_upper': max_upper_ratio,
    'max_lower_over_actual': max_lower_ratio,
    'statement_tested': 'r/kappa(g) <= ||g E g^-1||_F <= kappa(g) r; sequential envelope uses kappa(g)kappa(h)',
}
path = Path('/mnt/data/komorebi_ep004/episode004_conditioning_stress.json')
path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True))
