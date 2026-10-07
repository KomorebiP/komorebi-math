"""Reject internally consistent artifacts whose install contract was changed."""
from pathlib import Path
import pytest
from tests.test_release_repair import release, write_wheel, write_sdist, INFO, metadata_bytes
from verification.release_integrity import verify_release, safe_name
from verification.release_metadata import check_build_configuration


@pytest.mark.parametrize('old,new', [
    (b'Name: komorebi-math', b'Name: unrelated-package'),
    (b'Requires-Python: >=3.11', b'Requires-Python: >=3.14'),
    (b'Requires-Dist: numpy>=1.26', b'Requires-Dist: numpy<1'),
    (b'License-Expression: LicenseRef-Proprietary', b'License-Expression: MIT'),
    (b'License-File: LICENSE.md', b'License-File: WRONG.md'),
    (b'# Fixture', b'# Stale description'),
])
def test_metadata_fields_bound_to_project(release: Path, old: bytes, new: bytes) -> None:
    original = metadata_bytes(release)
    assert old in original
    write_wheel(release, {f'{INFO}/METADATA': original.replace(old,new)})
    assert not verify_release(release)['passed']


@pytest.mark.parametrize('name,payload', [
    (f'{INFO}/entry_points.txt', b'[console_scripts]\nkomorebi = missing.module:main\n'),
    (f'{INFO}/entry_points.txt', b'[console_scripts]\nkomorebi = komorebi_math.cli.main:main\nextra = missing.module:main\n'),
    (f'{INFO}/top_level.txt', b'komorebi_math\nextra_module\n'),
    (f'{INFO}/licenses/LICENSE.md', b'A different license\n'),
    ('unexpected.py', b'UNEXPECTED = 1\n'),
    ('unexpected.pth', b'../unexpected\n'),
    (f'{INFO}/unrecognized.txt', b'not a declared wheel payload\n'),
])
def test_nonpackage_install_payload_bound(release: Path, name: str, payload: bytes) -> None:
    write_wheel(release, {name: payload})
    result=verify_release(release)
    assert not result['passed']
    assert result['wheel_record']['passed'], 'Negative control must have valid internal hashes'


def test_duplicate_metadata_name_rejected(release: Path) -> None:
    data=metadata_bytes(release).replace(b'Name: komorebi-math\n',b'Name: komorebi-math\nName: other\n')
    write_wheel(release,{f'{INFO}/METADATA':data})
    assert not verify_release(release)['passed']


def test_sdist_metadata_rejected(release: Path) -> None:
    write_sdist(release,{'PKG-INFO':metadata_bytes(release).replace(b'Name: komorebi-math',b'Name: different')})
    assert not verify_release(release)['passed']


@pytest.mark.parametrize('path',['docs/index.md','examples/probe.py','mkdocs.yml'])
def test_sdist_docs_and_examples_bound(release: Path, path: str) -> None:
    write_sdist(release,{path:b'outdated\n'})
    result=verify_release(release)
    assert not result['passed']
    assert path in result['sdist_support_parity']['mismatched']


@pytest.mark.parametrize('name',['x/CON.txt','x/NUL','x/LPT1.txt','x/file:stream','x/file.','x/file ','x/../file'])
def test_windows_unsafe_paths_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        safe_name(name)


def test_pep639_build_floor(release: Path) -> None:
    assert check_build_configuration(release)['passed']
    p=release/'pyproject.toml';p.write_text(p.read_text().replace('setuptools>=77.0.3','setuptools>=70'))
    with pytest.raises(ValueError,match='77.0.3'):
        check_build_configuration(release)


@pytest.mark.parametrize('default_encoding', ['utf-8', 'cp1252'])
def test_utf8_readme_metadata_roundtrip(release: Path, monkeypatch, default_encoding: str) -> None:
    original_read_text = Path.read_text

    def read_text(path, encoding=None, errors=None, **kwargs):
        return original_read_text(
            path,
            encoding=default_encoding if encoding is None else encoding,
            errors=errors,
            **kwargs,
        )

    monkeypatch.setattr(Path, 'read_text', read_text)
    (release/'README.md').write_text('# Exact π and café\n',encoding='utf-8')
    write_wheel(release);write_sdist(release)
    report=verify_release(release)
    assert report['passed'],report
