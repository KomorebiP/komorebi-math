# Komorebi Math R4 Block 6

Checkpoint: `1.0.0b4-repair-r4-block6`. Wheel build: `4b6`.

This scoped checkpoint repairs mutation and stale-evidence hazards in selected
certified finite objects. Constructor matrices are owned snapshots rather than
caller-controlled live NumPy buffers. NumPy snapshots that underpin cached SVD,
inverse, packet, or realization state are read-only.

Covered surfaces:

- `PacketSystem` projectors;
- `OperatorRealization` assigned matrices;
- direct `PacketOperator` blocks;
- `HilbertObstructionProjector` correction maps;
- `RightInverseCorrectionProblem` linear/right-inverse matrices and target;
- `LyapunovSchmidtReduction` base points;
- `WeightedTransferGraph` diagonals and edge operators.

This does not claim that arbitrary callbacks are immutable, that every package
object has snapshot semantics, or that serialized/reloaded certificates are
revalidated. Serialization and stored-validity handling are Block 7.
