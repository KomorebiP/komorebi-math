# Security policy

Security reports are accepted for the current 1.0 release line.

For a suspected vulnerability that could execute unintended code, expose
credentials, corrupt portable records or bypass validation, use GitHub private
vulnerability reporting when available. Otherwise contact the repository owner
through a private channel listed on their profile. Do not publish sensitive
input or credentials in an issue.

Include the affected version, operating system, Python version, a minimal
reproducer and the observed impact. Identify whether untrusted input is needed.

Relevant boundaries include portable JSON decoding, archive creation and
extraction, extension discovery, LaTeX invocation and optional native backends.
A mathematical disagreement without an implementation or validation defect
should be reported as a normal issue.

No response-time or disclosure-time guarantee is offered.
