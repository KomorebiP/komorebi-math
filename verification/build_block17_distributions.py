"""Build Block 17 wheel and source distribution with the declared backend."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from hashlib import sha256
from pathlib import Path

import setuptools.build_meta as backend


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(project: Path, destination: Path, *, clean: bool = True) -> dict[str, object]:
    project = project.resolve()
    destination = destination.resolve()
    if clean and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(project)
        wheel_name = backend.build_wheel(str(destination))
        sdist_name = backend.build_sdist(str(destination))
    finally:
        os.chdir(previous)
    wheel = destination / wheel_name
    sdist = destination / sdist_name
    return {
        "schema": "komorebi-math.distribution-build",
        "schema_version": 1,
        "package_version": "1.0.0b1",
        "wheel": {"path": str(wheel), "size": wheel.stat().st_size, "sha256": _hash(wheel)},
        "sdist": {"path": str(sdist), "size": sdist.stat().st_size, "sha256": _hash(sdist)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dist")
    parser.add_argument("--output")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args(argv)
    project = Path(args.project)
    destination = Path(args.dist) if args.dist else project / "release" / "dist"
    payload = build(project, destination, clean=not args.no_clean)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
