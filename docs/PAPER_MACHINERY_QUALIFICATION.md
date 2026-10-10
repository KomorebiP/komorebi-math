# Paper machinery validation

This page records the initial mathematical snapshot published at
`5ef999f2618958e0e2691cfbd677e1083c0364b2`. Its counts and artifact hashes
are historical. See [the continuation](PAPER_MACHINERY_CONTINUATION.md)
for the additional constructions and their separate qualification records.

The mathematical patch passed **3,414 source tests**, with seven optional
JAX/PyTorch checks skipped because those backends were unavailable. Six
warnings came from deliberately nonfinite negative controls. No test failed
in the final full run.

The installed wheel was tested outside the checkout using a fresh virtual
environment with shared scientific dependencies. Its import origin was
verified inside that environment. Source and wheel each passed all **21**
stress checks with **16 samples**, seed **20261010**, no skipped, timed-out or
unexecuted checks, and unchanged source inputs. Both runs have the same
scientific replay digest:

`15c57fb0a8a6ede69da696f06a3282186dafd1f4ac5b9ef349b84fedd17d8d39`

The source campaign additionally pins 424 code/data inputs in a validated
bounded execution envelope: 240 seconds wall time, 2 GiB address space,
180 seconds CPU per process, two CPU cores and one numerical-library thread.
The full regression suite is a separate run, not included in that cap.

Wheel SHA-256:

`209330f98c2e9d80fc175ea24ff82b16182f9d04faf27b4291e84e2a1d3476e8`

The wheel and canonical source distribution were rebuilt byte reproducibly
from one frozen source snapshot in private build directories. Artifact hashes
remained unchanged throughout the final full regression.
Strict MkDocs and the offline mathematical rendering audit passed: 446
expressions across 34 mathematical pages. Ruff passed for new Python files.
The frozen 131-symbol stable root API remains unchanged.

The existing artifact CI gate now runs all 21 checks at sample count one for
both direct and source-rebuilt wheels, requiring installed catalog validation
and no skipped checks. The local isolated-import replay initially omitted
SymPy (19 passes, two explicit skips), which the gate rejected. Declaring the
existing exact dependency directory gave 21 passes; this uses shared
dependencies and is not a clean dependency installation.

## Active exact examples

The installed Gaussian example computes posterior variance `1/2`, candidate
Joseph variance `5/9`, complete residual packet energy `1/9`, and squared-law
upper score `2/9`. Precision zero refuses approval. At precision eight, the
upper score is `57/256` and the mean/covariance interval radius is `1/512`.
These parameter intervals enclose the candidate Joseph law; its law-error
claim separately depends on the declared Gaussian model and residual bound.

The installed predictive example finds return-path defect `1/4`, completes
the observable row space to dimension two, computes the exact delay map
`[[0,1],[1/12,5/6]]`, predicts withheld output `61/144`, and reconstructs memory
`11/24`. Its finite data fit leaves unknown-process coverage unestablished.

```bash
python examples/finite_gaussian_verifier.py
python examples/finite_predictive_foundations.py
python -m komorebi_math.audit --papers --coverage --samples 16 \
    --seed 20261010 --timeout 30 --total-timeout 180 --output math_stress.json
```

## Scope and evidence

The source crosswalk validates 55 source objects, 528 mechanism records and
646 API references against the original source bytes and repository test
paths. Its statuses are **159 implemented**, **177 partial**, **36 missing**,
**65 theory only**, **70 external**, and **21 separate reference**.
This inventory does not declare the entire mathematical program implemented.
See [coverage and remaining work](paper_machinery_coverage.md).

Machine-readable evidence is retained in the repository:

- `verification/PAPER_MACHINERY_QUALIFICATION.json`
- `verification/PAPER_MACHINERY_COVERAGE_CHECK.json`
- `verification/PAPER_MACHINERY_CONTRACT.json`
- `verification/paper_runs/source/manifest.json` and `result.json`
- `verification/PAPER_MACHINERY_WHEEL_STRESS.json`
- `verification/PAPER_MACHINERY_PYTEST.txt`

Initial setup commands did fail: unpinned imports selected the old installed
package; four publication tests lacked historical files/distributions; the
bounded runner refused unpinned inputs; and one example command had a wrong
virtualenv path. These were corrected before the results above. Independent
review also exposed actual arithmetic, occupancy, assertion-worker and
resource-limit bugs; the patch includes their fixes and regression controls.
A final packaging rebuild also caught transient synchronization files entering
one wheel and rejected stale source-distribution evidence. Packaging exclusions
protect the source tree; a private frozen build snapshot also prevents
transients introduced into build output after source filtering. A later recheck
found stale distribution files (3,406 passes, two publication failures); the
final rebuild and 3,414-pass regression use verified, unchanged artifacts.

No new Lean compiler run, universal proof, conjecture completion, GPU test, or
fresh cross-platform matrix is claimed by this local evidence. The Gaussian
direct rational/dyadic evaluator is distinct from the source's complete
formal rounded graph, cache and fuel program. This patch is reviewable code
under the existing beta identity, not a merged or tagged release.
