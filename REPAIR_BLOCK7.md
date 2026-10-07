# Komorebi Math R4 Block 7

Checkpoint: `1.0.0b4-repair-r4-block7`. Wheel build: `4b7`.

This scoped checkpoint separates portable evidence integrity from mathematical
revalidation. `core.certificate` advances from codec version 1 to version 2 with
an explicit adjacent migration. A loaded certificate retains the producer's
recorded validity through `recorded_valid`, but its live `valid` flag is false,
`revalidation_required` is true, and `require()` rejects it until the relevant
producer is rerun and returns a new certificate.

The version-2 decoder also enforces an exact field schema, boolean recorded
validity, finite numeric residuals, finite nonnegative tolerance, string
assumptions, text-or-null theorem IDs, and mapping diagnostics. Unknown fields
and coercions such as string `nan`/`inf` fail closed.

Packet-jet snapshots, spectral-sequence records and stream manifests therefore
retain their persisted mathematical data while nested certificates are correctly
identified as historical records. Fresh stream digest verification remains a new
live integrity certificate and does not revalidate the original mathematical
capture theorem.

This block does not provide authenticity or authorship: SHA-256 remains an
unkeyed integrity mechanism. It does not rerun arbitrary producers, verify custom
codec decoder semantics, or close serialization of every object in the package.
