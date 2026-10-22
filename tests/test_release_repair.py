"""Adversarial release checks: valid internal hashes must not conceal stale code."""
from __future__ import annotations
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tomllib
import zipfile

import pytest
from verification.release_integrity import verify_release, package_files, check_source_modes

VERSION = '1.0.0'
WHEEL = f'komorebi_math-{VERSION}-py3-none-any.whl'
SDIST = f'komorebi_math-{VERSION}.tar.gz'
INFO = f'komorebi_math-{VERSION}.dist-info'


def metadata_bytes(root: Path) -> bytes:
    project = tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']
    lines = ['Metadata-Version: 2.4', f"Name: {project['name']}", f"Version: {project['version']}"]
    for key, field in [('requires-python', 'Requires-Python'), ('license', 'License-Expression')]:
        if key in project:
            lines.append(f'{field}: {project[key]}')
    lines += [f'License-File: {name}' for name in project.get('license-files', [])]
    lines += [f'Requires-Dist: {item}' for item in project.get('dependencies', [])]
    for extra, dependencies in project.get('optional-dependencies', {}).items():
        lines.append(f'Provides-Extra: {extra}')
        lines += [f'Requires-Dist: {item}; extra == "{extra}"' for item in dependencies]
    description = (root/project['readme']).read_text(encoding='utf-8') if 'readme' in project else ''
    return ('\n'.join(lines)+'\n\n'+description).encode('utf-8')


def write_wheel(root: Path, overrides: dict[str, bytes | None] | None = None, *, bad_record: bool = False) -> None:
    files = package_files(root)
    files[f'{INFO}/METADATA'] = metadata_bytes(root)
    files[f'{INFO}/top_level.txt'] = b'komorebi_math\n'
    files[f'{INFO}/entry_points.txt'] = b'[console_scripts]\nkomorebi = komorebi_math.cli.main:main\n'
    files[f'{INFO}/licenses/LICENSE.md'] = (root/'LICENSE.md').read_bytes()
    files[f'{INFO}/WHEEL'] = b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'
    for key, value in (overrides or {}).items():
        if value is None:
            files.pop(key, None)
        else:
            files[key] = value
    record = io.StringIO()
    writer = csv.writer(record, lineterminator='\n')
    for key, value in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b'=').decode()
        writer.writerow([key, 'sha256=' + digest, len(value)])
    writer.writerow([f'{INFO}/RECORD', '', ''])
    files[f'{INFO}/RECORD'] = record.getvalue().encode()
    if bad_record:
        files['komorebi_math/__init__.py'] = b'corruption after RECORD construction\n'
    with zipfile.ZipFile(root/'dist'/WHEEL, 'w') as z:
        for key, value in files.items():
            z.writestr(key, value)


def write_sdist(root: Path, overrides: dict[str, bytes | None] | None = None, *, executable: bool = True,
                symlink: bool = False) -> None:
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*')
             if p.is_file() and p.relative_to(root).parts[0] != 'dist'}
    for key, value in (overrides or {}).items():
        if value is None:
            files.pop(key, None)
        else:
            files[key] = value
    if 'PKG-INFO' not in files and 'PKG-INFO' not in (overrides or {}):
        files['PKG-INFO'] = metadata_bytes(root)
    with tarfile.open(root/'dist'/SDIST, 'w:gz') as t:
        for key, value in files.items():
            info = tarfile.TarInfo(f'komorebi_math-{VERSION}/{key}')
            info.size = len(value)
            info.mode = 0o755 if key.endswith('.sh') and executable else 0o644
            t.addfile(info, io.BytesIO(value))
        if symlink:
            info = tarfile.TarInfo(f'komorebi_math-{VERSION}/external-link')
            info.type = tarfile.SYMTYPE
            info.linkname = '/tmp/not-a-package-file'
            t.addfile(info)


@pytest.fixture
def release(tmp_path: Path) -> Path:
    root = tmp_path/'repo'; root.mkdir()
    files = {
        'pyproject.toml': f'[build-system]\nrequires=["setuptools>=77.0.3","wheel"]\nbuild-backend="setuptools.build_meta"\n[project]\nname="komorebi-math"\nversion="{VERSION}"\nrequires-python=">=3.11"\nreadme="README.md"\nlicense="MIT"\nlicense-files=["LICENSE.md"]\ndependencies=["numpy>=1.26"]\n[project.scripts]\nkomorebi="komorebi_math.cli.main:main"\n',
        'RELEASE_ID.json': json.dumps({'package_version': VERSION, 'wheel_build_tag': '', 'release_id': VERSION, 'release_channel': 'stable'}),
        'src/komorebi_math/__init__.py': 'value = 42\n',
        'src/komorebi_math/py.typed': '',
        'src/komorebi_math/data/a.json': '{}\n',
        'verification/probe.py': 'print("probe")\n',
        'tests/test_a.py': 'def test_a():\n    assert 1 == 1\n',
        '.github/workflows/ci.yml': 'name: test\n',
        'LICENSE.md': 'MIT License\n', 'README.md': '# Fixture\n',
        'mkdocs.yml': 'site_name: Fixture\n', 'docs/index.md': '# Fixture docs\n',
        'examples/probe.py': 'print(42)\n',
        'MANIFEST.in': 'include README.md\n',
        'check_release.sh': '#!/bin/sh\necho test\n',
    }
    for name, value in files.items():
        p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(value,encoding='utf-8')
        p.chmod(0o755 if name.endswith('.sh') else 0o644)
    (root/'dist').mkdir(parents=True)
    write_wheel(root); write_sdist(root)
    return root


def test_clean_artifacts_pass(release: Path) -> None:
    report = verify_release(release)
    assert report['passed'], report
    assert report['wheel_package_parity']['source_files'] == 3


def test_valid_record_does_not_hide_stale_wheel(release: Path) -> None:
    write_wheel(release, {'komorebi_math/__init__.py': b'value = 0\n'})
    report = verify_release(release)
    assert report['wheel_record']['passed']
    assert not report['passed']
    assert report['wheel_package_parity']['mismatched'] == ['komorebi_math/__init__.py']


def test_stale_sdist_rejected(release: Path) -> None:
    write_sdist(release, {'src/komorebi_math/__init__.py': b'value = -1\n'})
    report = verify_release(release)
    assert not report['passed']
    assert report['sdist_package_parity']['mismatched'] == ['komorebi_math/__init__.py']


def test_missing_wheel_data_rejected(release: Path) -> None:
    write_wheel(release, {'komorebi_math/data/a.json': None})
    report = verify_release(release)
    assert not report['passed']
    assert report['wheel_package_parity']['missing'] == ['komorebi_math/data/a.json']


def test_extra_wheel_code_rejected(release: Path) -> None:
    write_wheel(release, {'komorebi_math/extra.py': b'x = 1\n'})
    assert not verify_release(release)['passed']


def test_corrupt_record_rejected(release: Path) -> None:
    write_wheel(release, bad_record=True)
    report = verify_release(release)
    assert not report['passed']
    assert 'RECORD mismatch' in str(report['failures'])


def test_missing_distribution_rejected(release: Path) -> None:
    (release/'dist'/WHEEL).unlink()
    assert not verify_release(release)['passed']


def test_duplicate_current_wheels_rejected(release: Path) -> None:
    p=release/'dist'/WHEEL
    p.with_name('second.whl').write_bytes(p.read_bytes())
    assert not verify_release(release)['passed']


def test_wrong_metadata_version_rejected(release: Path) -> None:
    write_wheel(release, {f'{INFO}/METADATA': b'Metadata-Version: 2.4\nVersion: 0.0.0\n'})
    assert not verify_release(release)['passed']


def test_wrong_wheel_build_rejected(release: Path) -> None:
    write_wheel(release, {f'{INFO}/WHEEL': b'Wheel-Version: 1.0\nBuild: 2\n'})
    assert not verify_release(release)['passed']


def test_unsafe_wheel_path_rejected(release: Path) -> None:
    write_wheel(release, {'../escape.py': b'not executable\n'})
    assert not verify_release(release)['passed']


def test_sdist_links_rejected(release: Path) -> None:
    write_sdist(release, symlink=True)
    assert not verify_release(release)['passed']


def test_sdist_modes_rejected_even_on_windows(release: Path) -> None:
    write_sdist(release, executable=False)
    report = verify_release(release)
    assert not report['passed']
    assert not report['sdist_shell_modes']['passed']


@pytest.mark.skipif(os.name == 'nt', reason='POSIX execute permissions only')
def test_source_shell_mode_loss_rejected(release: Path) -> None:
    (release/'check_release.sh').chmod(0o644)
    assert not check_source_modes(release)['passed']
    assert not verify_release(release)['passed']


def test_stale_verifier_in_sdist_rejected(release: Path) -> None:
    write_sdist(release, {'verification/probe.py': b'print("outdated")\n'})
    report = verify_release(release)
    assert not report['passed']
    assert report['sdist_support_parity']['mismatched'] == ['verification/probe.py']


def test_missing_test_in_sdist_rejected(release: Path) -> None:
    write_sdist(release, {'tests/test_a.py': None})
    assert not verify_release(release)['passed']


def test_missing_license_in_sdist_rejected(release: Path) -> None:
    write_sdist(release, {'LICENSE.md': None})
    assert not verify_release(release)['passed']


def test_postbuild_source_change_rejected(release: Path) -> None:
    (release/'src/komorebi_math/__init__.py').write_text('value = 99\n')
    report = verify_release(release)
    assert not report['passed']
    assert not report['wheel_package_parity']['passed']
    assert not report['sdist_package_parity']['passed']


def test_source_handoff_compatibility_import_preserves_archive_safety_api() -> None:
    from verification import build_source_release as current
    from verification import build_block21_handoff as previous
    assert previous.PayloadEntry is current.PayloadEntry
    assert previous._write_zip is current._write_zip
    assert previous._replay is current._replay


@pytest.mark.skipif(os.name == 'nt', reason='POSIX execute permission replay')
def test_handoff_replay_restores_verified_zip_modes(tmp_path: Path, monkeypatch) -> None:
    from verification import build_source_release as handoff
    script = tmp_path/'run.sh'; script.write_bytes(b'#!/bin/sh\nexit 0\n'); script.chmod(0o644)
    entry = handoff.PayloadEntry(path='run.sh', size=script.stat().st_size,
                                 sha256=hashlib.sha256(script.read_bytes()).hexdigest(), mode='0755')
    manifest = {'archive_root': 'fixture', 'payload': [entry.to_dict()]}
    archive = tmp_path/'fixture.zip'
    handoff._write_zip(tmp_path, 'fixture', (entry,), json.dumps(manifest).encode(), archive)

    def fake_gate(command, **kwargs):
        from types import SimpleNamespace
        extracted = Path(kwargs['cwd'])
        assert (extracted/'run.sh').stat().st_mode & 0o111 == 0o111
        output = Path(command[command.index('--output')+1])
        output.write_text(json.dumps({'passed': True, 'report_sha256': 'fixture-only',
                                      'capability_record': {'record_sha256': 'fixture-only'}}))
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(handoff.subprocess, 'run', fake_gate)
    assert handoff._replay(archive, manifest, python='unused-test-interpreter')['passed']
