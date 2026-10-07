"""Platform-independent version-header checks for built wheels."""

import zipfile
from pathlib import Path

import pytest

from verification.build_block21_distributions import _audit_wheel


INFO = "komorebi_math-1.0.0b4.dist-info"
VERSION = "1.0.0b4"


def _wheel(tmp_path: Path, metadata: bytes) -> Path:
    path = tmp_path / "fixture.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{INFO}/METADATA", metadata)
        archive.writestr(f"{INFO}/RECORD", b"")
    return path


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_wheel_version_header_accepts_platform_line_endings(tmp_path: Path, newline: bytes) -> None:
    metadata = newline.join(
        [b"Metadata-Version: 2.4", b"Name: komorebi-math", b"Version: 1.0.0b4", b"", b"Exact math"]
    )
    report = _audit_wheel(_wheel(tmp_path, metadata), VERSION)
    assert report["metadata"] == f"{INFO}/METADATA"


@pytest.mark.parametrize(
    "metadata",
    [
        b"Metadata-Version: 2.4\nName: komorebi-math\n\n",
        b"Metadata-Version: 2.4\nVersion: 0.0.0\n\n",
        b"Metadata-Version: 2.4\nVersion: 1.0.0b4\nVersion: 1.0.0b4\n\n",
        b"Metadata-Version: 2.4\nVersion: 1.0.0b4\nVersion: 0.0.0\n\n",
        b"Metadata-Version: 2.4\nName: komorebi-math\n\nVersion: 1.0.0b4\n",
    ],
    ids=["missing", "wrong", "duplicate", "conflicting", "body-only"],
)
def test_wheel_version_header_rejects_missing_or_ambiguous_identity(tmp_path: Path, metadata: bytes) -> None:
    with pytest.raises(SystemExit, match="wheel metadata version mismatch"):
        _audit_wheel(_wheel(tmp_path, metadata), VERSION)
