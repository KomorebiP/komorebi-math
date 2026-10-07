from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from setuptools.build_meta import build_sdist, build_wheel

SOURCE_DATE_EPOCH = 1787356800


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(root: Path) -> None:
    for rel in ('build', 'src/komorebi_math.egg-info'):
        p = root / rel
        if p.exists():
            shutil.rmtree(p)


def canonicalize(root: Path, sdist: Path) -> None:
    import sys
    sys.path.insert(0, str(root / 'verification'))
    try:
        from canonicalize_block20_sdist import canonicalize as canon
        canon(sdist, epoch=SOURCE_DATE_EPOCH)
    finally:
        sys.path.pop(0)


def build_once(root: Path, destination: Path) -> tuple[Path, Path]:
    identity = json.loads((root / 'RELEASE_ID.json').read_text())
    tag = identity['wheel_build_tag']
    clean(root)
    destination.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    try:
        os.chdir(root)
        wheel_name = build_wheel(str(destination), config_settings={'--build-option': [f'--build-number={tag}']})
        sdist_name = build_sdist(str(destination))
    finally:
        os.chdir(old)
    wheel = destination / wheel_name
    sdist = destination / sdist_name
    canonicalize(root, sdist)
    return wheel, sdist


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    release = root / 'release'
    dist = release / 'dist'
    report_path = release / 'REPAIR_R4_BLOCK16_DISTRIBUTION_BUILD.json'
    previous = os.environ.get('SOURCE_DATE_EPOCH')
    os.environ['SOURCE_DATE_EPOCH'] = str(SOURCE_DATE_EPOCH)
    try:
        with tempfile.TemporaryDirectory(prefix='komorebi-r4b16-build-') as td:
            temp = Path(td)
            w1, s1 = build_once(root, temp / 'first')
            w2, s2 = build_once(root, temp / 'replay')
            wheel_repro = w1.read_bytes() == w2.read_bytes()
            sdist_repro = s1.read_bytes() == s2.read_bytes()
            if not wheel_repro or not sdist_repro:
                raise SystemExit('distribution build is not byte reproducible')
            if dist.exists():
                for p in dist.iterdir():
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
            else:
                dist.mkdir(parents=True)
            final_wheel = dist / w1.name
            final_sdist = dist / s1.name
            shutil.copy2(w1, final_wheel)
            shutil.copy2(s1, final_sdist)
            report = {
                'schema': 'komorebi-math.repair-r4-block16-distribution-build',
                'schema_version': 1,
                'source_date_epoch': SOURCE_DATE_EPOCH,
                'wheel_build_tag': json.loads((root / 'RELEASE_ID.json').read_text())['wheel_build_tag'],
                'wheel_byte_reproducible': wheel_repro,
                'sdist_byte_reproducible': sdist_repro,
                'artifacts': [
                    {'name': final_wheel.name, 'sha256': sha256(final_wheel), 'size': final_wheel.stat().st_size},
                    {'name': final_sdist.name, 'sha256': sha256(final_sdist), 'size': final_sdist.stat().st_size},
                ],
            }
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if previous is None:
            os.environ.pop('SOURCE_DATE_EPOCH', None)
        else:
            os.environ['SOURCE_DATE_EPOCH'] = previous
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
