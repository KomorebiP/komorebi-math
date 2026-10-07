# Known issues after R4 Block 4

Block 4 closes only its recorded analytic scope. Remaining release work includes:

- producer-wide prerequisite/dependency audit;
- certificate mutation, caching, serialization and revalidation lifecycle;
- resource/allocation guards and malformed-input audit;
- optional GPU/backend paths not exercised here;
- full advertised Windows/PowerShell/CUDA/dependency qualification;
- independent mathematical/software review.

Numerical SVD/Newton/Picard computations remain floating-point models. They are not
interval proofs of real-input rank, existence, uniqueness, or convergence. Declared
Lipschitz and derivative bounds remain external unless separately certified.
