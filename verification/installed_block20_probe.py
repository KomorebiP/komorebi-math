"""Installed-wheel regression probe for R4 Block 20 finite Frobenius separation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import komorebi_math
from komorebi_math.deformation import finite_frobenius_separator
from komorebi_math.exceptions import ValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-installed", action="store_true")
    args = parser.parse_args()
    module_path = Path(komorebi_math.__file__).resolve()
    if args.require_installed and "site-packages" not in module_path.parts:
        raise SystemExit(f"probe did not import installed wheel: {module_path}")

    good = finite_frobenius_separator({2: (1, 1), 3: (1, -1)})
    good.certificate.require()
    bad = finite_frobenius_separator({2: (1, 1), 3: (2, 2)})
    bad_rejected = False
    try:
        bad.certificate.require()
    except ValidationError:
        bad_rejected = True

    payload = {
        "schema": "komorebi-math.episode017-installed-block20-probe",
        "schema_version": 1,
        "module_path": str(module_path),
        "good": {
            "rank": good.rank,
            "constituent_count": good.constituent_count,
            "separating": good.separating,
            "certificate_valid": good.certificate.valid,
        },
        "bad": {
            "rank": bad.rank,
            "constituent_count": bad.constituent_count,
            "separating": bad.separating,
            "certificate_valid": bad.certificate.valid,
            "require_rejected": bad_rejected,
        },
        "passed": good.separating and good.certificate.valid and (not bad.separating) and (not bad.certificate.valid) and bad_rejected,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
