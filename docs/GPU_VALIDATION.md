# GPU validation

The CUDA checks execute PyTorch and CuPy in separate child interpreters. A
native abort, timeout, import failure or nonzero worker exit produces a failed
record while the parent interpreter remains alive.

For the package installed in the selected environment:

```bash
python -m komorebi_math verify gpu
```

## Checked computations

The PyTorch worker checks CUDA availability and evaluates a cubic directional
JVP at `2`, with value `8` and tangent `12`. It also checks a device matrix-vector
product with expected result `[14, 38, 62, 86]`.

The CuPy worker checks device availability, executes a device-resident
`PacketIR`, and compares the result with `[0, 1, 4, 6]` using Euclidean residual
norm at most `1e-12`.

These are bounded equivalence controls for the recorded environment. They do
not establish unrestricted GPU performance, stress coverage or compatibility
with every driver and framework build.

## Release records

The hardware producer verifies the installed wheel inventory before and after
execution, clears source-path overrides, records interpreter/device/runtime
versions and retains the nested worker results. Its temporary installation
uses the exact candidate wheel and inherits the selected host's CUDA runtimes.

Follow the [release guide](RELEASE_PROCESS.md#stress-and-hardware-qualification)
for that artifact-bound command. Keep the complete report, including failed
runs. An installed-package diagnostic and a release-bound record answer
different questions.
