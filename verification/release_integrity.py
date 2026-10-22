"""Fail-closed parity checks for the actual wheel and source distribution.

Standard-library only. It does not import the package under test. An intact
wheel RECORD alone is insufficient: every package byte must match the source.
"""
from __future__ import annotations

import argparse
import base64
import csv
import configparser
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from unicodedata import normalize
from typing import Any


try:
    from .release_metadata import check_build_configuration, check_metadata, check_wheel_install_contract, wheel_filename
except ImportError:
    from release_metadata import check_build_configuration, check_metadata, check_wheel_install_contract, wheel_filename


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '\\' in name
            or any(part in ('', '.', '..') for part in name.rstrip('/').split('/'))
            or any(':' in part or part.endswith(('.', ' ')) for part in path.parts)
            or any(re.fullmatch(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part, re.I) for part in path.parts)
            or normalize('NFC', name) != name
            or any(ord(c) < 32 or ord(c) == 127 for c in name)):
        raise ValueError(f'unsafe archive member: {name!r}')
    return name.rstrip('/')


def package_files(root: Path) -> dict[str, bytes]:
    base = root / 'src'
    package = base / 'komorebi_math'
    files: dict[str, bytes] = {}
    for path in sorted(package.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'symlink in package: {path}')
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'):
            files[path.relative_to(base).as_posix()] = path.read_bytes()
    if not files:
        raise ValueError('no package source files found')
    return files


def compare_files(expected: dict[str, bytes], actual: dict[str, bytes]) -> dict[str, Any]:
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    different = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
    return {'passed': not (missing or unexpected or different), 'source_files': len(expected),
            'artifact_files': len(actual), 'missing': missing, 'unexpected': unexpected,
            'mismatched': different}


def _zip_files(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_name(info.filename)
            if name.casefold() in seen:
                raise ValueError(f'duplicate or case-colliding wheel member: {name}')
            seen.add(name.casefold())
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f'wheel contains link: {name}')
            if not info.is_dir():
                files[name] = archive.read(info)  # Also validates CRC.
    return files


def canonicalize_wheel(path: Path) -> None:
    """Normalize ZIP transport metadata without changing any wheel member byte.

    Store members uncompressed so the envelope does not depend on the host zlib
    version. RECORD remains byte-for-byte intact and is checked before and after.
    """
    if path.is_symlink():
        raise ValueError('wheel must not be a symlink')
    path = path.resolve(strict=True)
    payloads = _zip_files(path)
    _record_check(payloads)
    with zipfile.ZipFile(path) as source:
        directories = {info.filename for info in source.infolist() if info.is_dir()}
        if any(source.read(name) for name in directories):
            raise ValueError('wheel directory contains data')
    handle, temporary_name = tempfile.mkstemp(prefix='komorebi-wheel-', suffix='.whl', dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as target:
            for name in sorted(set(payloads) | directories):
                info = zipfile.ZipInfo(name, date_time=(2026, 8, 22, 0, 0, 0))
                info.create_system = 3
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = ((stat.S_IFDIR | 0o755) if name in directories else
                                      (stat.S_IFREG | 0o644)) << 16
                if name in directories:
                    info.external_attr |= 0x10
                target.writestr(info, b'' if name in directories else payloads[name])
        normalized = _zip_files(temporary)
        if normalized != payloads:
            raise ValueError('wheel canonicalization changed member bytes')
        _record_check(normalized)
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tar_files(path: Path) -> tuple[dict[str, bytes], dict[str, int]]:
    files: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    seen: set[str] = set()
    roots: set[str] = set()
    with tarfile.open(path, 'r:gz') as archive:
        for info in archive.getmembers():
            name = safe_name(info.name)
            if name.casefold() in seen:
                raise ValueError(f'duplicate or case-colliding sdist member: {name}')
            seen.add(name.casefold())
            roots.add(PurePosixPath(name).parts[0])
            if not (info.isfile() or info.isdir()):
                raise ValueError(f'sdist contains a link or special entry: {name}')
            if info.isfile():
                stream = archive.extractfile(info)
                if stream is None:
                    raise ValueError(f'unreadable sdist member: {name}')
                files[name] = stream.read()
                modes[name] = stat.S_IMODE(info.mode)
    if len(roots) != 1:
        raise ValueError(f'sdist must have one root: {sorted(roots)}')
    prefix = next(iter(roots)) + '/'
    return ({k.removeprefix(prefix): v for k, v in files.items()},
            {k.removeprefix(prefix): v for k, v in modes.items()})


def _record_check(files: dict[str, bytes]) -> dict[str, Any]:
    records = [n for n in files if n.endswith('.dist-info/RECORD')]
    if len(records) != 1:
        raise ValueError('wheel must have exactly one RECORD')
    record = records[0]
    seen: set[str] = set()
    count = 0
    for row in csv.reader(io.StringIO(files[record].decode('utf-8'))):
        if len(row) != 3:
            raise ValueError('malformed wheel RECORD row')
        name, encoded, size = row
        safe_name(name)
        if name in seen or name not in files:
            raise ValueError(f'duplicate or missing RECORD member: {name}')
        seen.add(name)
        if name == record:
            if encoded or size:
                raise ValueError('RECORD self-entry must not carry a hash or size')
            continue
        if not encoded.startswith('sha256=') or not size.isdecimal():
            raise ValueError(f'unhashed or unsupported RECORD entry: {name}')
        digest = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).rstrip(b'=').decode()
        if encoded != 'sha256=' + digest or int(size) != len(files[name]):
            raise ValueError(f'wheel RECORD mismatch: {name}')
        count += 1
    if seen != files.keys():
        raise ValueError(f'wheel contains unrecorded files: {sorted(files.keys() - seen)}')
    return {'passed': True, 'hashed_members': count}


def check_source_modes(root: Path) -> dict[str, Any]:
    scripts = sorted(root.glob('*.sh'))
    if os.name == 'nt':
        return {'passed': True, 'executed': False, 'reason': 'POSIX permissions are not meaningful on Windows',
                'scripts': len(scripts)}
    bad = [p.name for p in scripts if not p.stat().st_mode & stat.S_IXUSR]
    return {'passed': not bad, 'executed': True, 'scripts': len(scripts), 'non_executable': bad}


def support_files(root: Path) -> set[str]:
    try:
        from .source_layout import support_files as selected_support_files
    except ImportError:
        from source_layout import support_files as selected_support_files
    return selected_support_files(root)


def resolve_distribution_directory(root: Path) -> Path:
    """Use the current output directory, accepting legacy fixtures when present."""
    current = root / 'dist'
    legacy = root / 'release/dist'
    return current if current.is_dir() or not legacy.is_dir() else legacy


# Short import name used by the qualification commands.
distribution_directory = resolve_distribution_directory


def verify_release(root: Path, distribution_directory: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    directory = (distribution_directory or resolve_distribution_directory(root)).resolve()
    report: dict[str, Any] = {'schema': 'komorebi-math.release-integrity', 'schema_version': 1,
                             'passed': False, 'failures': []}
    failures = report['failures']
    try:
        report['build_configuration'] = check_build_configuration(root)
        text = (root / 'pyproject.toml').read_text(encoding='utf-8')
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        if not match:
            raise ValueError('missing package version')
        version = match.group(1)
        identity = json.loads((root / 'RELEASE_ID.json').read_text(encoding='utf-8'))
        if identity['package_version'] != version:
            raise ValueError('release identity version mismatch')
        build = identity['wheel_build_tag']
        expected_wheel_name = wheel_filename(version, build)
        report.update(release_id=identity['release_id'], package_version=version, wheel_build_tag=build)
        wheels = sorted(directory.glob('*.whl'))
        sdists = sorted(directory.glob('*.tar.gz'))
        if len(wheels) != 1 or len(sdists) != 1:
            raise ValueError(f'exactly one current wheel and sdist required, found {len(wheels)} and {len(sdists)}')
        wheel, sdist = wheels[0], sdists[0]
        if wheel.name != expected_wheel_name:
            raise ValueError('wheel filename does not identify the declared release')
        if sdist.name != f'komorebi_math-{version}.tar.gz':
            raise ValueError('sdist filename/version mismatch')
        source = package_files(root)
        wheel_files = _zip_files(wheel)
        report['wheel_record'] = _record_check(wheel_files)
        report['wheel_install_contract'] = check_wheel_install_contract(root, wheel_files, build)
        metadata = [n for n in wheel_files if n.endswith('.dist-info/METADATA')]
        wheel_metadata = [n for n in wheel_files if n.endswith('.dist-info/WHEEL')]
        if len(metadata) != 1 or len(wheel_metadata) != 1:
            raise ValueError('ambiguous wheel metadata')
        if BytesParser().parsebytes(wheel_files[metadata[0]])['Version'] != version:
            raise ValueError('wheel METADATA version mismatch')
        if BytesParser().parsebytes(wheel_files[wheel_metadata[0]]).get_all('Build', []) != ([build] if build else []):
            raise ValueError('wheel WHEEL build identifier mismatch')
        wp = {k: v for k, v in wheel_files.items() if k.startswith('komorebi_math/')}
        report['wheel_package_parity'] = compare_files(source, wp)
        tar_files, modes = _tar_files(sdist)
        report['sdist_metadata'] = check_metadata(root, tar_files['PKG-INFO'], label='sdist PKG-INFO')
        if identity.get('release_channel') in {'final', 'stable'} or re.fullmatch(r'\d+\.\d+\.\d+', version):
            try:
                from .source_layout import audit_sdist_layout
            except ImportError:
                from source_layout import audit_sdist_layout
            report['sdist_source_layout'] = audit_sdist_layout(root, tar_files)
        sp = {k.removeprefix('src/'): v for k, v in tar_files.items() if k.startswith('src/komorebi_math/')}
        report['sdist_package_parity'] = compare_files(source, sp)
        # Also bind test and release-tool sources, not just importable package files.
        required = support_files(root)
        absent = sorted(required - tar_files.keys())
        changed = sorted(n for n in required & tar_files.keys() if tar_files[n] != (root / n).read_bytes())
        report['sdist_support_parity'] = {'passed': not (absent or changed), 'files': len(required),
                                         'missing': absent, 'mismatched': changed}
        shell_modes = {n: f'{m:04o}' for n, m in modes.items() if n.endswith('.sh')}
        report['sdist_shell_modes'] = {'passed': all(m == '0755' for m in shell_modes.values()),
                                      'scripts': len(shell_modes), 'modes': shell_modes}
        report['source_shell_modes'] = check_source_modes(root)
        for name, value in report.items():
            if isinstance(value, dict) and value.get('passed') is False:
                failures.append(name)
        report['package_source_sha256'] = sha256(json.dumps(
            {k: sha256(v) for k, v in sorted(source.items())}, separators=(',', ':'), sort_keys=True).encode())
        report['artifacts'] = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p.read_bytes())}
                               for p in (wheel, sdist)]
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, tarfile.TarError, configparser.Error) as exc:
        failures.append(f'{type(exc).__name__}: {exc}')
    report['passed'] = not failures
    return report


def release_binding(root: Path) -> dict[str, Any]:
    """Content-address the executed source, qualification tools and current artifacts."""
    root=root.resolve()
    source={f'src/{name}': sha256(data) for name,data in package_files(root).items()}
    for name in sorted(support_files(root)):
        path=root/name
        if path.is_symlink():
            raise ValueError(f'symlink in bound release source: {name}')
        source[name]=sha256(path.read_bytes())
    identity=json.loads((root/'RELEASE_ID.json').read_text(encoding='utf-8'))
    directory = resolve_distribution_directory(root)
    artifacts=sorted([*directory.glob('*.whl'), *directory.glob('*.tar.gz')])
    return {'release_id':identity['release_id'],'source_files':len(source),
            'source_sha256':sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()),
            'artifacts':[{'name':p.name,'sha256':sha256(p.read_bytes()),'bytes':p.stat().st_size} for p in artifacts]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--distribution-directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify_release(args.root, args.distribution_directory)
    payload = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
