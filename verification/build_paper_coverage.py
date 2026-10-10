"""Build the installed source-to-machinery inventory from reviewed fragments.

Run with PYTHONPATH=src. Original papers are optional inputs; when supplied,
their bytes (including named archive members) must match the recorded hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from komorebi_math.audit.coverage import SCHEMA, validate_coverage_catalog

STATUS_ALIASES = {
    "integrated_finite": "implemented",
    "existing_related": "partial",
    "theory_boundary": "theory_only",
    "implemented_existing": "implemented",
    "implemented_new": "implemented",
    "partially_implemented": "partial",
    "paper_only": "theory_only",
    "external_research_companion": "separate_reference",
    "research_boundary": "external",
    "research_only": "theory_only",
}

REQUIRED_FRAGMENTS = frozenset(
    {
        "core.json",
        "middle.json",
        "late.json",
        "signal_machine.json",
        "special_functions.json",
        "companion_calculus.json",
        "companion_dynamics.json",
        "companion_arithmetic.json",
        "conjecture_dynamics.json",
        "conjecture_arithmetic.json",
        "conjecture_geometry.json",
    }
)


def assemble(root: Path) -> dict:
    sources, mechanisms, fragments = [], [], []
    directory = root / "verification/paper_coverage"
    missing = sorted(name for name in REQUIRED_FRAGMENTS if not (directory / name).is_file())
    if missing:
        raise ValueError("Required reviewed fragments are missing: " + ", ".join(missing))
    for path in sorted(directory.glob("*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        if (
            path.name not in REQUIRED_FRAGMENTS
            and isinstance(data, dict)
            and data.get("kind") == "finite_mathematical_stress_report"
        ):
            continue  # Explicitly identified run outputs are not inventory fragments.
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("mechanisms"), list)
            or not isinstance(data.get("sources"), list)
            or not data["mechanisms"]
            or not data["sources"]
        ):
            raise ValueError(f"Malformed coverage fragment: {path.name}")
        fragments.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "scope": data.get("audit_scope", data.get("scope", "Reviewed source crosswalk")),
            }
        )
        sources.extend(data["sources"])
        for mechanism in data["mechanisms"]:
            item = dict(mechanism)
            previous = item["status"]
            item["status"] = STATUS_ALIASES.get(previous, previous)
            if item["status"] != previous:
                item["original_status"] = previous
            item["fragment"] = path.relative_to(root).as_posix()
            mechanisms.append(item)
    return {
        "schema": SCHEMA,
        "audit_scope": "Current authorized paper-source snapshot: finite Python mechanisms and explicit mathematical boundaries.",
        "interpretation": "A source crosswalk, not a proof audit, theorem approval, or declaration that every paper mechanism is integrated.",
        "status_definitions": {
            "implemented": "The stated finite mechanism has an API and executable controls; read its boundaries.",
            "partial": "A related or restricted implementation leaves part of this source mechanism unimplemented.",
            "missing": "A dedicated finite construction is not present in the Python runtime.",
            "theory_only": "The recorded mechanism is a theorem or requires infinite/abstract hypotheses, without a general finite evaluator.",
            "external": "This mechanism depends on external mathematical objects, data, or an unresolved research boundary.",
            "separate_reference": "Source/research/formal machinery exists separately and is not integrated in this Python package.",
        },
        "fragments": fragments,
        "sources": sorted(sources, key=lambda x: x["id"]),
        "mechanisms": sorted(mechanisms, key=lambda x: x["key"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="check that the installed catalog matches its fragments"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        catalog = assemble(root)
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"Cannot assemble paper coverage: {error}")
        return 1
    report = validate_coverage_catalog(catalog, repository_root=root, paper_source_root=args.source_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.valid:
        return 1
    serialized = json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    destination = root / "src/komorebi_math/data/paper_machinery_coverage.json"
    if args.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != serialized:
            print("Installed catalog differs from reviewed fragments; rebuild it.")
            return 1
    else:
        destination.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
