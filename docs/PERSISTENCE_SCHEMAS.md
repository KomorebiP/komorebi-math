# Portable persistence schemas

## Purpose

Block 8 export normalization is designed for reports and broad data export. Block 10 introduced narrower executable persistence under an explicit schema and codec registry. Block 11 adds deterministic adjacent migrations and a separate chunked stream format for large finite packet-jet snapshots.

Use `to_serializable` for presentation-oriented export. Use `dumps_portable` when a supported object must be reconstructed as the same declared mathematical record.

## Envelope

A portable document contains:

```json
{
  "schema": "komorebi-math.portable",
  "schema_version": 1,
  "payload": {},
  "payload_sha256": "..."
}
```

The digest is computed over canonical JSON for `payload`. Object keys are sorted, separators are fixed, non-ASCII text is retained, and non-finite JSON numbers are rejected.

The digest detects accidental or unkeyed payload modification. It is not a digital signature and does not establish authorship, authorization, confidentiality, or resistance to an attacker who can replace both payload and digest.

## Codec records

Objects registered in the codec table use:

```json
{
  "__komorebi_type__": "nonlinear.packet_jet_snapshot",
  "__version__": 2,
  "data": {}
}
```

Decoding accepts only a registered tag. Current versions decode directly. Older versions must traverse a complete registered chain of adjacent migrations from `n` to `n + 1`; missing steps and future versions fail closed. The decoder never resolves Python import paths from the input.

## Built-in scalar and container forms

The encoder supports:

* `None`, booleans, integers, strings, and finite floats
* exact fractions
* finite complex values
* NumPy scalar values that convert to supported Python scalar types
* NumPy arrays with shape and dtype metadata
* tuples, lists, sets, frozensets, and mappings
* bytes and `pathlib.Path` values
* explicitly registered portable codecs

Mappings are represented by sorted key-value pairs so heterogeneous key types remain deterministic.

Object arrays are recursively encoded, preserving nested lists, tuples, and
arrays as individual cells. Numeric arrays use finite scalar data and restore
the declared shape. Structured and subarray dtypes are unsupported. The decoder
validates dimensions and recursion depth before reconstruction.

Integer array payloads must contain genuine integers within the declared
dtype's range; fractional truncation and integer overflow are rejected.
Floating and complex conversions must remain finite. Extended NumPy scalar
precision that cannot convert to a supported Python scalar is rejected
explicitly, rather than rounded silently or serialized recursively.

## Built-in mathematical codecs

The current registry includes codecs for:

* `Certificate`
* `OrderedRootedTree`
* `PacketJetAddress`
* `PacketJetCoefficient`
* `PacketJetSnapshot`
* `SparseSymmetricTensor`
* `LiftBranch`
* `LiftTreeNode`
* `FiniteLiftTree`
* `SpectralSequenceEntry`
* `SpectralSequencePage`
* `FilteredSpectralSequenceResult`

The packet-jet snapshot codec is version 2. Its provider field is normalized and coefficient ordering is explicit. Version 1 remains readable through the registered deterministic migration. A changed representation must receive a new version and an explicit migration path. Silent reinterpretation of old payloads is prohibited.

### Certificate lifecycle after loading

`core.certificate` uses codec version 2. Version 1 remains readable through an
explicit adjacent migration. A portable certificate is a historical record of a
producer's earlier check, not a command to rerun that check. On load:

* `recorded_valid` preserves the boolean recorded by the producer;
* `valid` is false;
* `revalidation_required` is true;
* `require()` fails until the producer is actually rerun and produces a fresh
  in-memory certificate.

This distinction is intentional. A valid payload digest proves that the stored
record matches its envelope; it does not prove that mutable external inputs,
software assumptions, external theorems, or producer logic still justify the
recorded mathematical conclusion. Fresh integrity checks may themselves return
new live certificates, but they certify only the integrity properties they rerun.

The version-2 certificate decoder also rejects unknown decoded fields, nonboolean
recorded validity, nonfinite or nonnumeric residuals, nonfinite/negative tolerance,
non-string assumptions, and malformed theorem/message fields rather than coercing
them.

## Codec migrations

```python
from komorebi_math.export import register_portable_migration

register_portable_migration(
    tag="my_project.record",
    from_version=1,
    to_version=2,
    migrator=lambda data: {**data, "schema_name": "record-v2"},
)
```

A migration advances exactly one version. The persisted payload selects only a registered type tag and declared version; it cannot provide code or an import path. Migration functions are trusted application code and must validate their own schema.

## Packet-jet stream format

`capture_packet_jet_stream` and `save_packet_jet_snapshot_stream` use a directory format rather than the single-document portable envelope. Each chunk and the manifest have independent canonical-JSON SHA-256 envelopes. Incomplete direct captures remain at `<destination>.partial` until final validation and atomic promotion.

The stream reader can decode one chunk at a time, but the complete basis count remains exponential in derivative order. See [Resumable packet-jet streams](STREAMING_PACKET_JETS.md) for the format, recovery procedure, partial-evaluation semantics, and integrity boundary.

## File writes

`save_portable` writes and flushes a temporary file in the destination directory,
then publishes the complete file atomically. With the default `overwrite=False`,
publication rejects an existing destination even when another writer creates
it concurrently. This uses a no-replacement rename on Windows and a hard link
on POSIX, so the POSIX destination filesystem must support hard links. With
`overwrite=True`, publication atomically replaces the destination. Temporary
files are cleaned after failure where the platform permits.

## Custom codecs

```python
from dataclasses import dataclass
from komorebi_math.export import register_portable_codec

@dataclass(frozen=True)
class Record:
    value: int

register_portable_codec(
    Record,
    tag="my_project.record",
    version=1,
    encoder=lambda value: {"value": value.value},
    decoder=lambda data: Record(int(data["value"])),
)
```

Codec decoders are trusted application code. Register a codec only for a type whose schema, validation, and versioning policy are controlled by the caller. `replace=True` is intended for tests and deliberate migrations, not routine import-time replacement.

## Safe marker decoding

`from_serializable` reverses only the built-in Block 8 markers for fractions, complex numbers, and NumPy arrays. Unknown `__type__` dictionaries remain ordinary dictionaries. This prevents the broad export decoder from becoming a class-instantiation mechanism.
