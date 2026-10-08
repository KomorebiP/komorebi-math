# GitHub readiness

Reviewed on 2026-10-08. The current Python library is a qualified research beta in `KomorebiP/komorebi-math`. Its package version remains `1.0.0b4`, its repair identity is `1.0.0b4-repair-r4-block21`, and its wheel build tag is `4b21`. The frozen stable surface has 131 symbols. A qualified beta checkpoint does not change the release channel to final 1.0.

## Qualified repository state

| Item | Recorded value |
|---|---|
| Repository | [KomorebiP/komorebi-math](https://github.com/KomorebiP/komorebi-math) |
| Default branch | `main` |
| Qualified commit | `50447a253c2d00d4a7a4108440e090844b7148b2` |
| Git tree | `faf5d17d4ab43090ab02bb4b2b53d2d5b87f2081` |
| Workflow | [CPU release qualification, run 37698654422](https://github.com/KomorebiP/komorebi-math/actions/runs/37698654422) |
| Workflow result | All eight jobs passed on 2026-10-07 |
| Visibility at review | Private |
| Release designation | Research beta and development checkpoint |

The repository contains the canonical Block 21 source and three subsequent repository repairs. These repairs fix workflow YAML, preserve hash-sensitive checkout bytes, produce Windows metadata with consistent newlines, and make enum signature reflection portable to Python 3.11 while retaining custom enum behavior. Regression tests accompany the implementation repairs.

## Executed qualification

| Check | Evidence and scope |
|---|---|
| Full source suite | Passed on Linux and Windows with Python 3.11, 3.12, and 3.13. These jobs install the test, exact, scientific, and YAML extras. |
| Representative source result | The Linux Python 3.13 job reported 1,725 passed, 7 skipped, and 4 warnings. |
| API and finite semantics | Every CPU matrix job passed strict research signatures, the stable API contract, stable verification, Block 20 verification, release qualification, and dependency consistency. |
| Documentation | Strict MkDocs build and the documentation-required publication gate passed on Linux Python 3.13. |
| Distributions | Reproducible wheel and source-distribution construction, source parity, and isolated installed-artifact regressions passed on Linux Python 3.13. |
| Base wheel | A separate clean environment installed the wheel and passed research and stable API checks, stable verification, Block 20 verification, and release qualification. |

The seven skipped tests in the representative source job were four JAX or PyTorch integration tests because those packages were absent, and three LaTeX compilation tests because LaTeX was absent. This CPU run supplies no fresh CUDA, JAX, PyTorch, or LaTeX qualification. Older platform reports retain their original dates and scopes. The run also does not test a NumPy-only installed wheel separately on every Python minor and platform.

The successful distribution job uploaded artifact `11516153327`, named `komorebi-math-50447a253c2d00d4a7a4108440e090844b7148b2`. Its recorded archive SHA-256 is `79a77f120ec44a7838b3f23054d61ff69aa58ca760f9e6c6ad8ed8a25affb292`. The artifact contains the built distributions, the distribution-build report, and the installed-artifact gate report. Access follows the repository's permissions.

These results apply to the named commit. Any subsequent change needs the corresponding checks rerun before its own readiness is asserted. Finite verification establishes the recorded finite semantics and certificate hypotheses, not stronger infinite-dimensional or unconditional mathematical claims.

## Install and verify

### Readiness repairs prepared on 2026-10-08

The next repository change corrects dependency capabilities that the
latest-dependency matrix did not cover. It raises the optional SymPy floor to
1.14, restores NumPy 1.26 quadrature and pseudoinverse support, and normalizes
NumPy scalar default representations in API snapshots. The callable signatures
and committed API manifests remain unchanged. Smith certificates now check
integrality, shape compatibility, the transformation identity, unimodularity,
diagonal form and divisibility, and invariant factor agreement independently.

New regressions exercise these failure modes, and a ninth CI job targets the
minimum CPU dependency combination. The eight-job run above predates these
repairs; qualification of this change must use its own test and artifact records.
The package remains a research beta under the existing license.

### Local commands

From the root of an authorized checkout, using Python 3.11 through 3.13:

```bash
python -m pip install .
python -m komorebi_math api check --stable
python -m komorebi_math verify stable
python -m komorebi_math verify block20
```

For the full CPU source and documentation checks:

```bash
python -m pip install -e ".[test,exact,scientific,yaml]" "setuptools==82.0.1" wheel mkdocs
python verification/build_block21_distributions.py --root . --output-directory release/dist --report release/LOCAL_DISTRIBUTION_BUILD.json
python -m pytest -q --tb=short tests
python -m mkdocs build --strict
python verification/block21_publication_gate.py --full-tests --require-mkdocs --output release/LOCAL_PUBLICATION_GATE.json --markdown release/LOCAL_PUBLICATION_GATE.md
python verification/episode018_certificate_consistency_probe.py --root . --output release/LOCAL_CERTIFICATE_CONSISTENCY_PROBE.json --seed 20260924 --trials 600
python -m pip check
```

The independent Episode 018 probe is an additional audit command. It is not a separate step in the eight-job CI run cited above.

To install the generated wheel:

```bash
python -m pip install --force-reinstall release/dist/komorebi_math-1.0.0b4-4b21-py3-none-any.whl
```

The optional `exact`, `scientific`, `yaml`, `jax`, and `torch` extras can be selected from local source, for example `python -m pip install ".[exact,scientific]"`. CuPy requires a package compatible with the local CUDA installation. See [supported platforms](SUPPORTED_PLATFORMS.md), [optional backend isolation](OPTIONAL_BACKEND_ISOLATION.md), and [GPU validation](GPU_VALIDATION.md).

## Release decisions still open

The existing license permits private evaluation and reproducibility testing only. `pyproject.toml` records `LicenseRef-Proprietary`, and `RELEASE_ID.json` preserves those terms. Publishing the repository would not by itself grant open-source reuse rights. An open-source release needs an explicitly selected replacement license applied consistently to the license file, metadata, and release identity.

No GitHub Release was listed at this review. The tag workflow builds and uploads Actions artifacts when the tag matches `v1.0.0b4-repair-r4-block21`; it does not create a GitHub Release or publish to PyPI. The current installation instructions therefore use an authorized source checkout or a local wheel.

Public visibility, license replacement, a tagged GitHub Release, PyPI publication, and promotion from beta to final 1.0 are distinct release decisions. Optional hardware qualification also remains distinct from the passing CPU suite.
