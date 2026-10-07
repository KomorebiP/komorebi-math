# Komorebi Math 1.0.0b2 — Block 18 Research Beta

Block 18 adds frame algebra at the research tier. It supplies coefficient-space tools for redundant analysis/synthesis systems without promoting new names into the frozen stable or beta namespaces.

For analysis `A`, synthesis `S`, and a signal-space operator `T`, the induced coefficient operator is

\[
\widehat T = A T S, \qquad G = A S.
\]

The research model exposes the frame projection and closure relation

\[
G\widehat T G = \widehat T,
\]

with composition agreeing with signal-space composition when the required exact-reconstruction hypotheses hold. These facilities are experimental: their names, signatures, and behavior may evolve before promotion.

## Compatibility boundary

`1.0.0b2` preserves the Block 17 contract exactly:

| Contract | Frozen size |
|---|---:|
| Stable API | 131 symbols |
| Beta API | 83 symbols |
| Portable codecs | 51 codecs |
| Stable semantic fixtures | 10 cases |

The stable aliases remain object-identical, persistence remains compatible, and Block 18 does not change stable signatures or semantics. Research exports may grow only additively relative to the saved `1.0.0b1` research baseline.

Block 18 remains a beta. Frame algebra must remain behind the research boundary until it has a separately reviewed promotion contract.
