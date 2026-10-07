# API compatibility for 0.14

Version `0.14.0a2` expands `komorebi_math.coordinates` into a general coordinate atlas while preserving every public name introduced in `0.14.0a1`.

The broader research API follows the additive pre-beta policy. New general coordinate names may be added during the 0.14 alpha line. Existing names are not removed. The curated `komorebi_math.beta` namespace remains unchanged and retains strict signature checking.

The coordinate surface now includes arbitrary bases, atlases, combinatorial transforms, classical polynomial families, recurrence and state-space models, affine and formal charts, finite-field maps, tensor transport, numerical spectral systems, packet permutations, and portable records.

Euler-to-Fibonacci helpers remain public for compatibility, but they are ordinary named constructors within the larger namespace.
