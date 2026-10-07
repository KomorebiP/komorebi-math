# Evidence precedence: R4 Block 7

The current implementation is `1.0.0b4-repair-r4-block7`, wheel build `4b7`.
The enclosing Block 7 verification record selects the fresh scoped persistence
and revalidation evidence. Production files are identified by hashes, not by the
package version alone.

The frozen R4 Block 6 archive is the immediate predecessor. Its mutation/lifecycle
evidence and all earlier repair records remain historical for their exact source
bytes. A persisted `Certificate` is itself historical evidence: its recorded
validity is retained, but loading does not rerun the producer and does not create
a fresh live certificate.

Development attempts, expected failing baseline regressions and superseded
intermediate distributions are not final verification gates. No upstream
dependency rebuild, all-platform qualification, whole-library soundness or
publication permission is implied by archive integrity or a passing scoped suite.
