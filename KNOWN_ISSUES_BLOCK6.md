# Known issues after R4 Block 6

- The monolithic whole-library regression gate is still pending; earlier blocks
  recorded execution-cap timeouts rather than passes.
- Serialization, loading, stored certificate validity, cache authentication and
  revalidation are not closed here.
- Callable providers are deliberately live dependencies and cannot be snapshotted
  as arrays; their semantic stability remains an explicit assumption.
- SymPy and other mutable non-NumPy payloads receive owned copies at the repaired
  backend-matrix boundaries, but this block specifically enforces read-only
  behavior only for NumPy arrays.
- CuPy/GPU, Windows/PowerShell, CUDA and broad platform qualification remain open.
- 139 inherited static comparison-review locations remain unreviewed; Block 6 is
  a lifecycle audit and does not promote additional rows from that comparison queue.
