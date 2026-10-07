# Block 16: API Freeze Candidate and Release Qualification

Version `0.16.0a2` is the second stabilization block after canonical polyphase refinement. It does not attempt to freeze the complete research namespace. It defines a deliberately smaller versioned surface and adds independent gates for structure, semantics, persistence, platforms, and optional hardware.

## Release objective

The Block 16 execution chain is

\[
\text{construct}
\longrightarrow
\text{analyze}
\longrightarrow
\text{decompose}
\longrightarrow
\text{refine}
\longrightarrow
\text{certify}
\longrightarrow
\text{persist}.
\]

A release is not qualified merely because imports work. The candidate must satisfy

\[
C_{16}
=
C_{\mathrm{API}}
\wedge
C_{\mathrm{semantic}}
\wedge
C_{\mathrm{persistence}}
\wedge
C_{\mathrm{support}}
\wedge
C_{\mathrm{distribution}}.
\]

Hardware qualification is an additional gate when requested:

\[
C_{16}^{\mathrm{GPU}}
=
C_{16}\wedge C_{\mathrm{CUDA}}.
\]

## Stable namespace

The three imports below are identity aliases of one curated namespace:

```python
import komorebi_math.stable as stable
import komorebi_math.v1 as v1
import komorebi_math.api.v1 as api_v1

assert stable.__all__ == v1.__all__ == api_v1.__all__
```

The committed stable manifest records every exported name, broad object kind, implementation module, qualified name, and reflected signature. Stable compatibility is strict: removals, kind changes, and signature changes are release failures. Additions are also rejected by the Block 16 deterministic-snapshot gate because the committed candidate surface must be updated intentionally.

## Semantic contract

Structural compatibility cannot detect a function whose signature remains unchanged while its mathematical behavior changes. Block 16 therefore ships a closed semantic fixture covering ten representative laws:

1. general analysis and synthesis with transfer-block reconstruction
2. signal polyphase refinement and direct-analysis equivalence
3. exact arithmetic polyphase refinement
4. rectangular lattice polyphase refinement
5. typed operator-relation realization
6. exact finite-basis coordinate round trips
7. factorial-power normalized-series products
8. portable Markdown and LaTeX rendering
9. scoped runtime settings and restoration
10. portable refinement-tower round trips

The fixture is canonical JSON. The runner compares canonical observed values with committed expected values and records a SHA-256 digest of the complete fixture.

## Persistence contract

The portable registry remains extensible. The 1.0 candidate separately records every codec owned by `komorebi_math`. For each tag, the contract fixes

- the tag
- the represented Python type
- the current codec version

A compatible release may add codecs. It may not remove a committed codec, silently change the represented type, decrease a codec version, or increase a version without every adjacent migration edge.

For a tag advancing from version \(r\) to \(s\), the required path is

\[
r\to r+1\to\cdots\to s.
\]

## General analysis--synthesis runtime

Block 16 adds a common finite runtime for a matrix analysis map

\[
A:\mathbb{F}^{n}\to\mathbb{F}^{m}
\]

and synthesis map

\[
S:\mathbb{F}^{m}\to\mathbb{F}^{n}
\]

subject to the checked left-inverse law

\[
SA=I_n.
\]

It supports square invertible coordinate systems and redundant full-column-rank frames. Given a coordinate selector \(E_I\), the source-space reconstruction component is

\[
C_I=SE_IA.
\]

For any complete coordinate partition \(I_1,\ldots,I_k\),

\[
\sum_{j=1}^{k}C_{I_j}=SA=I_n.
\]

In a square invertible system, these components are algebraic projectors. In a redundant system, they need not be idempotent. The canonical methods are therefore `component()` and `components()`. The older discoverability aliases `projector()` and `projectors()` remain, with this limitation documented.

An operator \(T\) has the finite transfer expansion

\[
T
=
\left(\sum_{\beta}C_{\beta}\right)
T
\left(\sum_{\alpha}C_{\alpha}\right)
=
\sum_{\beta,\alpha}C_{\beta}TC_{\alpha}.
\]

`AnalysisSynthesisBlocks` stores and certifies this reconstruction.

## Installed verification

```powershell
komorebi release contract
komorebi verify stable
komorebi verify release
komorebi verify block16
komorebi api check --stable
komorebi release qualify
```

`komorebi verify block16` is cumulative. It executes every installed verification block through Block 16. `komorebi release qualify` produces a machine-readable record with runtime inventory, isolated JAX and PyTorch status, stable semantic checks, support-policy checks, and an optional GPU report.

## Hardware qualification

```powershell
komorebi release qualify --require-gpu --output .\gpu-qualification.json --force
```

PyTorch and CuPy run in separate child interpreters. The hardware gate checks

- a CUDA-visible PyTorch device
- `torch.func.jvp` on `cuda:0`
- a CUDA-visible CuPy device
- a device-resident Komorebi `PacketIR` execution
- exact expected output within the declared finite tolerance

A native import abort or other abnormal worker exit is contained to the affected backend worker and reported as a failed hardware gate rather than terminating the parent release process.

## Position before 1.0

Version `0.16.0a2` is a freeze candidate, not the final freeze. The next public sequence is:

1. `1.0.0b1`: freeze the curated stable namespace and persistence contract
2. `1.0.0rc1`: rebuild from a clean tag and execute the full supported matrix
3. `1.0.0`: change only approved version and release metadata from the successful release candidate

Any code change after a release candidate requires another release candidate.

## Distribution and batch gates

The complete source suite can be executed in bounded fresh-process batches:

```powershell
.\Run_Block16_Batched_Validation.ps1
```

Wheel and source-distribution construction and normalized rebuild comparison use:

```powershell
.\Build_Block16_Distributions.ps1
```

The distribution audit rejects unsafe archive paths, missing release assets, forbidden wheel payloads, malformed wheel records, and source distributions that cannot rebuild an equivalent wheel payload.

## Companion C review

`verification/COMPANION_C_BLOCK16_CROSSWALK.json` records the finite Companion C identities that map to package-native stable or research interfaces. The crosswalk does not automatically promote duplicate implementations into the stable namespace.
