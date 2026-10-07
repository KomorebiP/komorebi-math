# Known issues after R4 Block 7

- The monolithic whole-library source and installed-wheel regression gate remains
  pending for Block 9; scoped evidence is not a substitute for that run.
- A historical certificate can report `recorded_valid=True`; this means the
  producer recorded success before serialization, not that the claim was rerun
  after loading. Use a fresh producer result for live validity.
- Portable SHA-256 envelopes detect payload modification only when the digest is
  not replaced with it. They do not authenticate publisher identity or defend
  against an attacker able to replace both payload and digest.
- Registered custom codec decoders and migration functions are trusted
  application code and can impose semantics outside the built-in audit.
- General file provenance, signatures, remote artifact trust and release-channel
  authentication remain outside this block.
- Resource exhaustion and malformed-input allocation limits are Block 8.
- CuPy/GPU, Windows/PowerShell, CUDA and broad platform qualification remain open.
- The inherited static comparison-review queue remains 139 locations; this
  persistence block does not promote unrelated numerical comparison rows.
