"""Build wheel metadata with LF newlines before its RECORD is generated."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def reproducible_metadata():
    """Limit newline normalization to setuptools' temporary metadata writer.

    Source and package payloads are untouched. Setuptools copies PKG-INFO into
    METADATA, so the writer must use LF on Windows as well as POSIX before the
    wheel backend hashes the file into RECORD.
    """
    import setuptools._core_metadata as metadata

    original = metadata.NamedTemporaryFile

    def metadata_file(*args, **kwargs):
        mode = args[0] if args else kwargs.get('mode', 'w+b')
        if 'b' not in mode:
            kwargs['newline'] = '\n'
        return original(*args, **kwargs)

    metadata.NamedTemporaryFile = metadata_file
    try:
        yield
    finally:
        metadata.NamedTemporaryFile = original


def build_canonical_wheel(destination: str, build_tag: str) -> str:
    from setuptools.build_meta import build_wheel
    try:
        from .release_integrity import canonicalize_wheel
    except ImportError:
        from release_integrity import canonicalize_wheel

    with reproducible_metadata():
        name = build_wheel(destination, config_settings={'--build-option': ['--build-number=' + build_tag]})
    canonicalize_wheel(Path(destination) / name)
    return name
