"""Reassemble the technical manual's source archive and verify its checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += len(chunk)
            digest.update(chunk)
    return count, digest.hexdigest()


def validate_record(record: dict) -> None:
    name = record["name"]
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or "/" in name or "\\" in name or Path(name).name != name):
        raise ValueError("The manifest contains an unsafe filename")
    if type(record["bytes"]) is not int or record["bytes"] < 0:
        raise ValueError("The manifest contains an invalid file size")
    if not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError("The manifest contains an invalid SHA-256 checksum")


def join(directory: Path, output: Path | None = None) -> Path:
    manifest = json.loads((directory / "SOURCE_PARTS.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "komorebi-math.source-archive-parts" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported source-parts manifest")
    archive = manifest["archive"]
    parts = manifest["parts"]
    validate_record(archive)
    if not isinstance(parts, list) or len(parts) != manifest["part_count"] or not parts:
        raise ValueError("The manifest contains an invalid part list")
    for record in parts:
        validate_record(record)
    if len({record["name"] for record in parts}) != len(parts):
        raise ValueError("The manifest repeats a part filename")
    if sum(record["bytes"] for record in parts) != archive["bytes"]:
        raise ValueError("Part sizes do not sum to the complete archive size")
    output = (output or directory / archive["name"]).resolve()
    if output.exists():
        if digest_file(output) == (archive["bytes"], archive["sha256"]):
            print(f"Existing archive verified: {output}")
            return output
        raise ValueError(f"An existing file has a different checksum: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=archive["name"] + ".", suffix=".joining", dir=output.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            for record in parts:
                path = directory / record["name"]
                if not path.is_file() or path.stat().st_size != record["bytes"]:
                    raise ValueError(f"Missing part or incorrect file size: {record['name']}")
                part_digest = hashlib.sha256()
                count = 0
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(chunk)
                        part_digest.update(chunk)
                        digest.update(chunk)
                        count += len(chunk)
                        total += len(chunk)
                if count != record["bytes"] or part_digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Part checksum mismatch: {record['name']}")
        if total != archive["bytes"] or digest.hexdigest() != archive["sha256"]:
            raise ValueError("The complete source archive checksum does not match")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created and verified: {output}")
    print(f"SHA-256: {archive['sha256']}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent,
                        help="folder containing all source parts and SOURCE_PARTS.json")
    parser.add_argument("--output", type=Path, help="optional path for the complete ZIP")
    arguments = parser.parse_args()
    try:
        join(arguments.directory.resolve(), arguments.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
