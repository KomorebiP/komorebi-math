# Resumable packet-jet streams

## Purpose

A finite packet-jet snapshot stores every retained standard-basis coefficient in one in-memory record. That representation is appropriate for small and moderate derivatives. Its dense capture cost is still

\[
N_{\mathrm{basis}} = d_{\mathrm{source}}^r,
\]

where `r` is the derivative order. A high-order snapshot can therefore be too large to capture, serialize, or reload as one object.

Block 11 adds a directory-backed stream format. It writes coefficients in independently digested chunks, commits progress after each chunk, and supports resuming an interrupted capture. A reader verifies and decodes at most one chunk at a time during iteration or evaluation.

This changes the peak storage pattern. It does not remove the exponential basis count.

## Two write modes

### Direct capture

`capture_packet_jet_stream` evaluates a live `PacketJet` on the ambient standard basis. `evaluation_budget` limits the additional basis evaluations performed by one call.

```python
from komorebi_math.nonlinear import capture_packet_jet_stream

partial = capture_packet_jet_stream(
    jet,
    "quadratic.kmstream",
    chunk_size=256,
    evaluation_budget=500,
)
assert not partial.completed

complete = capture_packet_jet_stream(
    jet,
    "quadratic.kmstream",
    chunk_size=256,
    resume=True,
)
complete.certificate.require()
```

An incomplete write lives at `<destination>.partial`. The requested destination appears only after a complete manifest has been written and revalidated.

A direct resume compares the packet address, dimensions, provider name, exactness, point, tolerance, and chunk size with the recorded metadata. Arbitrary Python callables do not have a portable semantic hash. The caller must therefore keep the provider implementation fixed across a resume. The certificate records that limitation.

### Existing snapshot

`save_packet_jet_snapshot_stream` divides an already captured snapshot into coefficient chunks. `coefficient_budget` limits the additional coefficients written by one call.

```python
from komorebi_math.nonlinear import save_packet_jet_snapshot_stream

result = save_packet_jet_snapshot_stream(
    snapshot,
    "snapshot.kmstream",
    chunk_size=1024,
)
result.certificate.require()
```

Snapshot mode fingerprints the complete coefficient record and capture certificate. Resuming with a different snapshot is rejected.

## Directory format

A complete stream has the following structure:

```text
snapshot.kmstream/
  manifest.json
  chunks/
    chunk-00000000.json
    chunk-00000001.json
    ...
```

Each JSON file has its own envelope:

```json
{
  "schema": "komorebi-math.packet-jet-stream.chunk",
  "schema_version": 1,
  "payload_sha256": "...",
  "payload": {}
}
```

The manifest records:

* packet address and dimensions
* provider label and exactness
* base point and tolerance
* direct or snapshot capture mode
* source fingerprint
* dense basis count
* completed basis and coefficient counts
* canonical chunk records
* omitted-zero residual bound
* final capture certificate for complete streams
* a stream content digest

The content digest covers metadata, progress, chunk payload digests, and the final certificate digest. Chunk payload digests cover coefficient values and ordinal ranges.

## Atomic progress

A chunk is written to a temporary file in the target directory, flushed, and atomically renamed. Only after that rename succeeds is the manifest advanced. Before resume appends any work, it re-reads and digest-checks every committed prefix chunk. An interrupted write can leave an orphan chunk, but resume removes chunks beyond the committed manifest length only after the committed prefix has passed verification.

Completion uses a staging directory. Promotion to the requested destination is atomic when the source and destination are on the same filesystem. Explicit overwrite uses a temporary backup and restores it if promotion fails.

Filesystem durability still depends on operating-system and storage semantics. The implementation does not claim transactional guarantees across hardware failure, network filesystems, or an adversarial storage layer.

## Bounded reads

```python
from komorebi_math.nonlinear import open_packet_jet_snapshot_stream

reader = open_packet_jet_snapshot_stream(
    "snapshot.kmstream",
    verify_chunks=True,
)

for record, coefficients in reader.iter_chunks():
    process(record, coefficients)

selection = reader.select(start=100, stop=200, max_coefficients=100)
selection.certificate.require()
```

`iter_chunks` and `iter_coefficients` retain one decoded chunk at a time. `select` and `load_snapshot` require explicit coefficient guards because they materialize data in memory.

## Complete and partial evaluation

```python
result = reader.evaluate(directions)
result.certificate.require()

left = reader.evaluate(directions, start=0, stop=100)
right = reader.evaluate(directions, start=100)
```

A complete evaluation receives a complete-snapshot certificate. A bounded ordinal range is a declared partial sum. Its certificate lists coefficients outside the selected range as unchecked. The API never labels a partial range as the complete multilinear map.

## Integrity and authenticity

SHA-256 digests detect accidental modification and any modification that does not also replace the affected digest chain. They are not signatures. An attacker who can replace the payloads, manifests, and digests can construct another internally consistent stream.

Use an authenticated filesystem, an external signature, or a signed release manifest when provenance or authorship matters.

## Resource boundaries

The stream layer has explicit guards for:

* maximum dense basis evaluations during direct capture
* per-call evaluation or coefficient budgets
* positive chunk size
* materialized coefficient count
* selection ranges and dimensions

The complete manifest still grows linearly with the number of chunks. Choose a chunk size that balances restart granularity, manifest size, and filesystem overhead.

## Command line inspection

```powershell
komorebi stream inspect .\snapshot.kmstream --verify-chunks
komorebi stream verify .\snapshot.kmstream
```

Use `--allow-incomplete` only when intentionally inspecting a `.partial` capture. An incomplete stream cannot be materialized as a complete `PacketJetSnapshot`.
