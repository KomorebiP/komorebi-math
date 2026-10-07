"""Installed release-qualification checks for the path to 1.0."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from .. import __version__
from ..compatibility import create_beta_api_manifest, create_public_api_manifest, create_stable_api_manifest
from ..persistence_contract import load_portable_codec_contract
from ..release import check_release_contract, check_upgrade_contract, check_validation_evidence, check_stable_documentation
from ..stable_semantics import load_stable_semantics_fixture
from ..support import load_support_matrix, runtime_environment
from .core import VerificationCheck, VerificationReport, _capture

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")


def run_release_qualification_verification() -> VerificationReport:
    """Run metadata and policy gates that are independent of source checkout."""

    checks: list[VerificationCheck] = []

    def version_identity() -> None:
        assert _VERSION_PATTERN.fullmatch(__version__)
        assert create_public_api_manifest().package_version == __version__
        assert create_beta_api_manifest().package_version == __version__
        assert create_stable_api_manifest().package_version == __version__
        assert load_portable_codec_contract().package_version == __version__

    checks.append(_capture("release version identity", version_identity))

    def unbound_manifest_signatures() -> None:
        from ..compatibility import (
            create_beta_api_manifest as beta,
            create_public_api_manifest as public,
            create_stable_api_manifest as stable,
        )

        for function in (public, beta, stable):
            parameter = inspect.signature(function).parameters["package_version"]
            assert parameter.default is None

    checks.append(_capture("version-independent API factory signatures", unbound_manifest_signatures))

    def packaged_contract_data() -> None:
        source, fixture, digest = load_stable_semantics_fixture()
        assert source.is_file()
        assert fixture["cases"]
        assert len(digest) == 64
        codec = load_portable_codec_contract()
        assert codec.entries

    checks.append(_capture("packaged semantic and persistence contracts", packaged_contract_data))

    def support_policy() -> None:
        matrix = load_support_matrix()
        assert matrix.release_line
        assert matrix.python["minimum"] == "3.11"
        assert "numpy" in matrix.base_dependencies
        assert matrix.validated_environments

    checks.append(_capture("machine-readable support policy", support_policy))

    def runtime_snapshot() -> None:
        snapshot = runtime_environment(probe_ad=False)
        assert snapshot["python"]
        assert snapshot["numpy"]
        assert Path(snapshot["python_executable"]).name

    checks.append(_capture("runtime environment snapshot", runtime_snapshot))

    def stable_surface_hygiene() -> None:
        from .. import stable

        assert stable.__all__
        assert len(stable.__all__) == len(set(stable.__all__))
        assert all(hasattr(stable, name) for name in stable.__all__)
        assert not any(name.startswith("_") and name != "__version__" for name in stable.__all__)

    checks.append(_capture("stable namespace hygiene", stable_surface_hygiene))

    def release_contract() -> None:
        check_release_contract().certificate.require()

    checks.append(_capture("current beta release contract", release_contract))
    checks.append(_capture("predecessor upgrade contract", lambda: check_upgrade_contract().certificate.require()))
    checks.append(_capture("platform evidence contract", lambda: check_validation_evidence().certificate.require()))
    checks.append(_capture("stable documentation contract", lambda: check_stable_documentation().certificate.require()))

    return VerificationReport(tuple(checks))


__all__ = ["run_release_qualification_verification"]
