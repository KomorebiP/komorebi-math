# Contributing

Komorebi Math is open-source software under the MIT license. See [LICENSE.md](LICENSE.md) for the grant and third-party notices.

For a substantial contribution, describe the mathematical statement, accepted
inputs, expected API tier and compatibility impact in an issue or pull request.
A bug report should include a small reproducer and the observed result.

Contributions should provide meaningful regression tests and document the scope
of any certificate, numerical tolerance or conditional result. Keep the versioned
API in `komorebi_math.v1` compatible. New experimental interfaces belong in the
subject modules until their contracts are ready for the stable namespace.

Use one Python interpreter for installation and validation:

```console
python -m pip install -e ".[test,exact,scientific,yaml]" "setuptools==82.0.1" wheel mkdocs "pymdown-extensions==10.16.1"
python verification/build_distributions.py --root .
python verification/check_release.py --full-tests --require-mkdocs
```

[Reproducibility](REPRODUCIBILITY.md) describes the checks and their limits.
The [release process](docs/RELEASE_PROCESS.md) covers installed distributions,
platform qualification and publication.
