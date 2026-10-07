# Upgrade Contract

The upgrade contract is stored in `src/komorebi_math/data/upgrade_contract_1_0.json`.

## Supported predecessors

```text
0.15.0a2
0.16.0a2
```

The `0.16.0a2` stable API is required to be exactly identical to the `1.0.0b1` stable API. The earlier `0.15.0a2` manifest is checked under its documented migration allowance.

## Portable records

Seven committed files are decoded by the current reader:

- two polyphase records from `0.15.0a2`
- analysis–synthesis, arithmetic polyphase, finite basis, polyphase refinement, and refinement-tower records from `0.16.0a2`

Each record binds:

- producer version
- source-file SHA-256
- portable payload SHA-256
- expected reconstructed Python type
- certificate execution when the object exposes a zero-argument certificate

## Executable wheel path

`Run_Block17_Upgrade_Validation.ps1` creates an isolated environment for each predecessor. It installs the predecessor wheel, verifies its version, replaces it with the beta wheel, and executes the full current upgrade contract.

The contract does not cover pickle, undocumented internal objects, or third-party codecs.
