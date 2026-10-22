# Reproducibility

Komorebi Math separates functional checks from release qualification. A
mathematical certificate reports the obligations of its operation. A passing
CPU check does not establish CUDA execution or a result on another platform.

## Prepare an environment

Use one supported Python 3.11–3.13 interpreter for every command:

```console
python -m pip install -e ".[test,exact,scientific,yaml]" "setuptools==82.0.1" wheel mkdocs "pymdown-extensions==10.16.1"
python verification/build_distributions.py --root .
```

The builder writes the wheel and source distribution to `dist/`. Rebuild them
after a source or documentation change. Generated builds, qualification reports
and development history are excluded from the source payload.

## Functional and source checks

```console
python -m komorebi_math verify all
python -m komorebi_math release check
python examples/library_tour.py --json
python verification/check_release.py --full-tests --require-mkdocs
python verification/check_release_manual.py --profile verification/release_manual_profile.json --output qualification/examples.json
```

The source check runs the complete test suite, API and release contracts,
a deterministic library tour, local documentation links and a strict MkDocs
build. Its JSON record retains the exact test output and JUnit report, including
failed or malformed results. Optional backend exclusions are recorded explicitly.
The manual checker executes each selected example in a fresh interpreter and
temporary directory. Reference fragments and optional tools have explicit
exclusion reasons.

## Installed distributions

Installed tests use clean virtual environments for the original wheel and a
wheel rebuilt from the source distribution. A local wheelhouse permits
dependency-isolated offline checks. See the complete commands in the
[release process](docs/RELEASE_PROCESS.md).

## Documentation and source archive

```console
python -m mkdocs build --strict
python verification/check_math_docs.py --site-dir site --output qualification/math_docs.json
python verification/build_source_release.py --help
```

Documentation includes a local MathJax renderer and its upstream license. Math
rendering requires no CDN or network font. The source-archive builder validates
paths, file modes, hashes and reproducible archive bytes. Its command-line help
describes the required source-check report and replay options.

## Interpreting records

Exact algebraic outputs and canonical JSON hashes should be reproducible.
Backend versions, executable paths, operating-system strings and optional native
libraries describe an environment. Checksums establish byte identity rather
than authorship or independent mathematical review.

Final publication requires the current candidate's platform, hardware and tagged
predecessor review described in the [release process](docs/RELEASE_PROCESS.md).
