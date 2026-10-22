"""Block 15 stabilization contract for the curated 1.0-candidate API."""

from pathlib import Path

import komorebi_math
import komorebi_math.api.v1 as api_v1
import komorebi_math.stable as stable
import komorebi_math.v1 as v1

from komorebi_math.compatibility import (
    STABLE_API_STABILITY_LEVEL,
    STABLE_PUBLIC_NAMESPACES,
    _public_symbol,
    check_stable_api,
    create_stable_api_manifest,
    load_stable_api_manifest,
)
from komorebi_math.export.persistence import portable_codecs
from komorebi_math.refinement import PolyphaseRefinement


def test_block15_version() -> None:
    from komorebi_math.compatibility import load_stable_api_manifest
    assert komorebi_math.__version__ == load_stable_api_manifest().package_version


def test_stable_namespace_has_unique_explicit_exports() -> None:
    assert stable.__all__
    assert len(stable.__all__) == len(set(stable.__all__))
    assert all(hasattr(stable, name) for name in stable.__all__)


def test_versioned_alias_modules_match_stable_namespace() -> None:
    assert v1.__all__ == stable.__all__
    assert api_v1.__all__ == stable.__all__
    assert v1.PolyphaseRefinement is stable.PolyphaseRefinement
    assert api_v1.refine_polyphase is stable.refine_polyphase


def test_stable_manifest_is_deterministic() -> None:
    left = create_stable_api_manifest()
    right = create_stable_api_manifest()
    assert left.digest == right.digest
    assert left.namespaces == STABLE_PUBLIC_NAMESPACES
    assert left.stability_level == STABLE_API_STABILITY_LEVEL


def test_committed_stable_manifest_matches_current_api() -> None:
    reference = load_stable_api_manifest()
    current = create_stable_api_manifest()
    assert reference.to_dict() == current.to_dict()


def test_strict_stable_api_check_passes() -> None:
    report = check_stable_api()
    assert report.compatible
    assert report.removed == ()
    assert report.kind_changed == ()
    assert report.signature_changed == ()


def test_refinement_portable_codecs_are_registered() -> None:
    tags = {codec.tag for codec in portable_codecs()}
    assert {
        "refinement.polyphase",
        "refinement.polyphase_tower",
        "refinement.lattice_polyphase",
        "signals.polyphase_coordinates",
        "signals.lattice_polyphase_coordinates",
        "refinement.residue_coordinates",
        "arithmetic.polyphase_coordinates",
    } <= tags


def test_refinement_api_is_available_from_domain_and_stable_namespaces() -> None:
    assert stable.PolyphaseRefinement is PolyphaseRefinement
    assert stable.PolyphaseCoordinateRecord is not None
    assert Path(stable.__file__).name == "stable.py"


def test_path_constants_have_platform_neutral_manifest_identity() -> None:
    from pathlib import PurePosixPath, PureWindowsPath

    posix = _public_symbol("test", "path", PurePosixPath("a/b"))
    windows = _public_symbol("test", "path", PureWindowsPath(r"a\b"))
    assert posix.implementation_module == windows.implementation_module == "pathlib"
    assert posix.qualname == windows.qualname == "Path"
