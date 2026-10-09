"""Common candidate wheels retain payload identity across host ZIP metadata."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from verification.release_artifact_gate import require_identical_wheels
from verification.release_integrity import _zip_files, canonicalize_wheel
from verification.reproducible_wheel import reproducible_metadata


def wheel(path: Path, *, windows: bool, code: bytes = b'value = 7\n') -> dict[str, bytes]:
    files = {'example/__init__.py': code,
             'example-1.0.dist-info/METADATA': b'Metadata-Version: 2.4\nName: example\nVersion: 1.0\n'}
    record_name = 'example-1.0.dist-info/RECORD'
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([name, 'sha256=' + digest, len(data)])
    writer.writerow([record_name, '', ''])
    files[record_name] = stream.getvalue().encode()
    with zipfile.ZipFile(path, 'w') as archive:
        archive.comment = b'Windows build' if windows else b'POSIX build'
        names = list(files)
        if windows:
            names.reverse()
        for name in names:
            info = zipfile.ZipInfo(name, (2025, 1, 1, 0, 0, 0) if windows else (2026, 8, 22, 0, 0, 0))
            info.create_system = 0 if windows else 3
            info.external_attr = (0o100666 if windows else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED if windows else zipfile.ZIP_STORED
            info.comment = b'host metadata' if windows else b''
            archive.writestr(info, files[name])
    return files


def test_canonical_envelope_preserves_members_and_record_across_hosts(tmp_path: Path) -> None:
    posix, windows = tmp_path / 'posix.whl', tmp_path / 'windows.whl'
    expected = wheel(posix, windows=False)
    assert wheel(windows, windows=True) == expected
    assert posix.read_bytes() != windows.read_bytes()
    for path in (posix, windows):
        canonicalize_wheel(path)
        assert _zip_files(path) == expected
    assert require_identical_wheels(posix, windows)['passed']
    before = posix.read_bytes()
    canonicalize_wheel(posix)
    assert posix.read_bytes() == before


def test_canonicalization_does_not_hide_changed_payload_with_valid_record(tmp_path: Path) -> None:
    original, changed = tmp_path / 'original.whl', tmp_path / 'changed.whl'
    wheel(original, windows=False)
    wheel(changed, windows=True, code=b'value = 8\n')
    for path in (original, changed):
        canonicalize_wheel(path)
    with pytest.raises(RuntimeError, match='not byte-identical'):
        require_identical_wheels(original, changed)


def test_canonicalization_rejects_invalid_record_without_replacing_input(tmp_path: Path) -> None:
    path = tmp_path / 'invalid.whl'
    files = wheel(path, windows=True)
    files['example/__init__.py'] = b'value = 9\n'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='RECORD mismatch'):
        canonicalize_wheel(path)
    assert path.read_bytes() == original


def test_metadata_writer_uses_lf_before_record_and_restores_backend(tmp_path: Path, monkeypatch) -> None:
    import setuptools._core_metadata as metadata

    original = metadata.NamedTemporaryFile

    def simulated_windows_file(*args, **kwargs):
        kwargs.setdefault('newline', '\r\n')
        return original(*args, **kwargs)

    monkeypatch.setattr(metadata, 'NamedTemporaryFile', simulated_windows_file)
    payload = 'Name: example\nVersion: 1.0\n\nA description.\n'
    distribution = SimpleNamespace(write_pkg_file=lambda handle: handle.write(payload))
    metadata.write_pkg_info(distribution, str(tmp_path))
    assert (tmp_path / 'PKG-INFO').read_bytes() == payload.replace('\n', '\r\n').encode()
    with reproducible_metadata():
        metadata.write_pkg_info(distribution, str(tmp_path))
    assert (tmp_path / 'PKG-INFO').read_bytes() == payload.encode()
    assert metadata.NamedTemporaryFile is simulated_windows_file
    with pytest.raises(RuntimeError, match='bounded failure'):
        with reproducible_metadata():
            raise RuntimeError('bounded failure')
    assert metadata.NamedTemporaryFile is simulated_windows_file
