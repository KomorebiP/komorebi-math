# Security Policy

## Supported release

Security reports are accepted for the current `1.0.0b4` beta. Historical alpha and beta artifacts under `release/predecessors` are retained only for compatibility testing.

## Reporting

Do not open a public issue for a suspected vulnerability that could expose credentials, execute unintended code, corrupt portable records, or bypass validation. Use GitHub's private vulnerability reporting feature when it is enabled for the repository. Otherwise contact the repository owner through a private channel listed on the owner's GitHub profile.

Include the affected version, operating system, Python version, a minimal reproducer, the observed impact, and whether untrusted input is required.

## Scope

The most relevant security boundaries are portable JSON decoding, archive creation and extraction, extension discovery, LaTeX invocation, optional native backends, and release scripts. Mathematical disagreement without an implementation or validation defect is handled as a research issue rather than a security report.

No response-time or disclosure-time guarantee is asserted for this research beta.
