# Block 36 release repair R2

Current delivery: `1.0.0b4-block36-r2`. Public package version: `1.0.0b4`.
The wheel build tag is `2`. The R1 and older reports are historical.

This repair binds package metadata, declared dependencies, entry points, license
bytes, documentation and examples to the actual source. It expands installed
artifact qualification to the named profile in
`verification/installed_runtime_profile.json`, checks every loaded package module
comes from the installed virtual environment, and requires independent sdist
wheel reconstruction to be byte-identical. Source mathematical APIs and the
manual are unchanged. Existing license rights and restrictions are unchanged.

## Local verification

Run `python verification/release_artifact_gate.py --root . --runtime-tests`
from this directory, or use `VERIFY_REPAIR.ps1` / `verify_repair.sh`.
The default artifact gate uses package-isolated virtual environments with
inherited dependencies. This is not dependency-isolated qualification.

The `--dependency-mode isolated --wheelhouse PATH` option creates clean build
and runtime virtual environments and installs only from a local wheelhouse.
It refuses to fall back to inherited packages. A populated wheelhouse with
NumPy, SciPy, SymPy, mpmath, pytest, PyYAML, wheel, and the same setuptools
version as the direct build is required, including transitive dependencies.
The gate never downloads dependencies. The clean mode and strict MkDocs
require execution evidence before they can be claimed to have passed.

The setuptools build floor is now 77.0.3, matching the PEP 639 metadata used
by this project. This does not alter the project's license terms.

## Evidence binding and test portability

Publication reports now bind the current package source, qualification tests,
release tools, documentation, and distribution hashes. The handoff builder
verifies the report self-digest, rejects any failing check, and rejects an old
passing report after the bound inputs change. This is an integrity check, not
an authenticated publisher signature. R1 handoff records are preserved rather
than overwritten by the current R2 builder.

The installed profile retains its tampered-semantics negative control. That
test now loads its fixture using `importlib.resources`, so it exercises the
installed resource without requiring a `src` checkout directory. The inherited
mode explicitly records parent site-package directories when nested virtual
environments need them. Isolated mode never adds those inherited paths.

The handoff collector also excludes generated `.venv-*` and `.venv_*`
directories. Tagged-release workflows create such environments before building
the handoff; their interpreter links and installed dependencies are not source
payload and must not break or contaminate the source archive.
