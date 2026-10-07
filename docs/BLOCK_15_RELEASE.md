# Block 15: Polyphase Refinement and 1.0-Candidate Stabilization

Version `0.15.0a2` consolidates the package's finite polyphase refinement machinery and establishes a curated API boundary for the path to 1.0.


## `0.15.0a2` validation hardening

The second alpha patch closes a Windows validation failure caused by optional PyTorch import. The former availability check used `find_spec`, which only established that a package name could be located. It did not establish that the native runtime could be imported safely or that `torch.func.jvp` existed.

The corrected contract is:

\[
\operatorname{available}(B)
=
\operatorname{discoverable}(B)
\land
\operatorname{isolated\_import}(B)
\land
\operatorname{required\_capability}(B).
\]

Discovery and capability checks now run in child interpreters. JAX and PyTorch adapter tests also run in isolated workers. A native abort is reduced to a recorded nonzero return code, after which the corresponding optional tests are skipped without affecting the NumPy-only suite.

## Canonical law

For coarse order `M`, refinement ratio `K`, coarse residue `r`, and inner digit `s`, the fine address is

\[
a=r+Ms,
\qquad
0\le r<M,
\qquad
0\le s<K.
\]

Thus

\[
\left(x_r^{(M)}\right)_s^{(K)}=x_{r+Ms}^{(MK)}
\]

and each coarse projector splits into its fine descendants:

\[
P_r^{(M)}=\sum_{s=0}^{K-1}P_{r+Ms}^{(MK)}.
\]

## Implemented records

The canonical implementation is in `komorebi_math.refinement.polyphase`:

* `PolyphaseRefinement`
* `PolyphaseRefinementTower`
* `LatticePolyphaseRefinement`
* `refine_polyphase`
* `coarsen_polyphase`
* `polyphase_refinement_certificate`
* `refine_lattice_polyphase`
* `coarsen_lattice_polyphase`

It accepts finite signal coordinates, residue coordinates, exact arithmetic coordinates, and rectangular lattice coordinates. Refinement certificates verify both coarsening after refinement and agreement with direct analysis at the fine order.

## Stable API boundary

The curated 1.0-candidate surface is available from each of the following equivalent namespaces:

```python
import komorebi_math.stable as km
import komorebi_math.v1 as km_v1
import komorebi_math.api.v1 as versioned
```

The committed manifest `src/komorebi_math/data/public_api_stable_0_15.json` records exported names, implementation modules, symbol kinds, and callable signatures. `check_stable_api()` performs strict signature comparison.

## Persistence

Portable codecs cover refinement plans, towers, lattice refinement plans, and all supported finite coordinate records. The existing digest-checked portable envelope rejects modified payloads before reconstruction.

## Boundaries

This block certifies finite coordinate identities. It does not infer infinite inverse limits, convergence of an unbounded refinement tower, or analytic regularity from finite data. Such claims require separate hypotheses.
