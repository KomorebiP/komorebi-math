"""Bind installable metadata and entry points to the declared project (stdlib only)."""
from __future__ import annotations

from collections import Counter
import configparser
from email.parser import BytesParser
from pathlib import Path
import re
import tomllib
from typing import Any


def normalized_name(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def _requirement(text: str) -> str:
    """Normalize formatting, not meaning, for this project's static requirements."""
    text = re.sub(r'\s+', '', text).replace("'", '"')
    match = re.match(r'[A-Za-z0-9][A-Za-z0-9_.-]*', text)
    if match is None:
        raise ValueError(f'invalid dependency entry: {text!r}')
    return normalized_name(match.group()) + text[match.end():]


def read_project(root: Path) -> dict[str, Any]:
    return tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))


def check_build_configuration(root: Path) -> dict[str, Any]:
    config = read_project(root)
    build = config.get('build-system', {})
    requirements = build.get('requires', [])
    # PEP 639 expressions and project.license-files require setuptools >=77.
    # 77.0.3 is the documented PyPA minimum and avoids early 77.x defects.
    bounds = [re.search(r'^setuptools\s*>=\s*(\d+(?:\.\d+){0,2})(?:,|$)', item)
              for item in requirements]
    bounds = [match.group(1) for match in bounds if match]
    lower = tuple(int(v) for v in bounds[0].split('.')) if len(bounds) == 1 else ()
    padded = lower + (0,) * (3 - len(lower)) if lower else ()
    uses_pep639 = isinstance(config['project'].get('license'), str) or 'license-files' in config['project']
    if uses_pep639 and (not padded or padded < (77, 0, 3)):
        raise ValueError('PEP 639 project metadata requires an explicit setuptools>=77.0.3 build floor')
    if build.get('build-backend') != 'setuptools.build_meta':
        raise ValueError('unrecognized build backend for this release verifier')
    return {'passed': True, 'requirements': requirements, 'pep639_floor': '77.0.3'}


def check_metadata(root: Path, payload: bytes, *, label: str) -> dict[str, Any]:
    project = read_project(root)['project']
    metadata = BytesParser().parsebytes(payload)
    checks: list[str] = []
    fields = {'Name': project['name'], 'Version': project['version']}
    for key, field in [('description', 'Summary'), ('requires-python', 'Requires-Python'),
                       ('license', 'License-Expression')]:
        if key in project:
            fields[field] = project[key]
    for field, expected in fields.items():
        values = metadata.get_all(field, [])
        if len(values) != 1 or values[0] != expected:
            raise ValueError(f'{label} {field} differs from pyproject.toml: {values!r}')
        checks.append(field)
    declared = list(project.get('dependencies', []))
    extras = project.get('optional-dependencies', {})
    for extra, dependencies in extras.items():
        for dependency in dependencies:
            # Setuptools canonicalizes the extra names according to PEP 685.
            normalized = normalized_name(extra)
            if ';' in dependency:
                base, marker = dependency.split(';', 1)
                declared.append(f'{base}; ({marker.strip()}) and extra == "{normalized}"')
            else:
                declared.append(f'{dependency}; extra == "{normalized}"')
    if Counter(map(_requirement, metadata.get_all('Requires-Dist', []))) != Counter(map(_requirement, declared)):
        raise ValueError(f'{label} Requires-Dist differs from declared dependencies/extras')
    if Counter(metadata.get_all('Provides-Extra', [])) != Counter(normalized_name(x) for x in extras):
        raise ValueError(f'{label} Provides-Extra differs from declared extras')
    license_files = project.get('license-files', [])
    # This repository uses explicit license filenames, not glob patterns.
    if any(any(ch in filename for ch in '*?[') for filename in license_files):
        raise ValueError('license globs need explicit expansion before release qualification')
    if Counter(metadata.get_all('License-File', [])) != Counter(license_files):
        raise ValueError(f'{label} License-File differs from declared license files')
    readme = project.get('readme')
    if isinstance(readme, str):
        expected = (root / readme).read_text(encoding='utf-8').replace('\r\n', '\n').rstrip('\n')
        parts = re.split(rb'\r?\n\r?\n', payload, maxsplit=1)
        actual = (parts[1] if len(parts) == 2 else b'').decode('utf-8').replace('\r\n', '\n').rstrip('\n')
        if actual != expected:
            raise ValueError(f'{label} long description differs from current README')
    return {'passed': True, 'identity_fields': checks, 'dependency_entries': len(declared),
            'extras': len(extras), 'license_files': license_files, 'readme_bound': isinstance(readme, str)}


def check_wheel_install_contract(root: Path, files: dict[str, bytes], build: str) -> dict[str, Any]:
    project = read_project(root)['project']
    info = f"komorebi_math-{project['version']}.dist-info"
    required = {f'{info}/{name}' for name in ('METADATA', 'WHEEL', 'RECORD', 'top_level.txt', 'entry_points.txt')}
    license_files = project.get('license-files', [])
    required.update(f'{info}/licenses/{name}' for name in license_files)
    actual = {name for name in files if not name.startswith('komorebi_math/')}
    if actual != required:
        raise ValueError(f'wheel non-package inventory mismatch: missing={sorted(required-actual)}, unexpected={sorted(actual-required)}')
    wheel = BytesParser().parsebytes(files[f'{info}/WHEEL'])
    for key, expected in [('Wheel-Version', '1.0'), ('Build', build), ('Root-Is-Purelib', 'true'), ('Tag', 'py3-none-any')]:
        if wheel.get_all(key, []) != [expected]:
            raise ValueError(f'wheel {key} differs from the pure-Python release contract')
    if files[f'{info}/top_level.txt'].decode('utf-8').splitlines() != ['komorebi_math']:
        raise ValueError('wheel top_level.txt must expose only komorebi_math')
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(files[f'{info}/entry_points.txt'].decode('utf-8'))
    observed = {section: dict(parser.items(section)) for section in parser.sections() if parser.items(section)}
    expected_groups = {k: dict(v) for k, v in project.get('entry-points', {}).items() if v}
    if project.get('scripts'):
        expected_groups['console_scripts'] = project['scripts']
    if project.get('gui-scripts'):
        expected_groups['gui_scripts'] = project['gui-scripts']
    if observed != expected_groups:
        raise ValueError(f'wheel entry points differ from project declarations: {observed!r}')
    for name in license_files:
        if files[f'{info}/licenses/{name}'] != (root/name).read_bytes():
            raise ValueError(f'wheel license payload differs from source: {name}')
    metadata = check_metadata(root, files[f'{info}/METADATA'], label='wheel METADATA')
    return {'passed': True, 'metadata': metadata, 'entry_points': observed,
            'non_package_members': len(actual), 'wheel_tags': wheel.get_all('Tag')}
