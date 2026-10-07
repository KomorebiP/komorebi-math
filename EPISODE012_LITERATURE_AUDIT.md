# Episode 012 Literature Audit — Weighted Realization and Hankel Rank

## Source question

Does finite-dimensional linear compression of future residuals go beyond
classical weighted-automata/rational-series theory, and is Hankel rank already
the standard minimal linear realization dimension?

## Verified modern source 1

Tianyu Li, Guillaume Rabusseau, Doina Precup,
**Nonlinear Weighted Finite Automata**, AISTATS 2018, PMLR 84.

Checked section 2.1--2.2 in the primary PMLR PDF.

The paper defines a `k`-state weighted finite automaton by initial/final vectors
and symbol transition matrices, defines `rank(f)` as the number of states in a
minimal WFA, defines the bi-infinite Hankel matrix

`H_f[u,v]=f(uv)`,

and states Theorem 1:

`rank(f)=rank(H_f)`.

It attributes the classical theorem to Carlyle--Paz (1971) and Fliess (1974).
The paper also gives the factorization

`H_f = P S`,

with prefix rows `alpha^T A_u` and suffix columns `A_v omega`, exactly matching
the factorization used in Episode 012.

Stable source:
https://proceedings.mlr.press/v84/li18a/li18a.pdf

## Verified modern source 2

Raphael Bailly,
**Quadratic Weighted Automata: Spectral Algorithm and Likelihood Maximization**,
AISTATS 2011, PMLR 20.

Checked the preliminaries in the primary PMLR PDF.

It states that a series is rational iff its residual space is finite-dimensional
(equivalently, iff it is computed by a weighted automaton), and defines the rank
of a rational series as the dimension of its residual space.  A rank-`d` series
has a `d`-state weighted automaton.

Stable source:
https://proceedings.mlr.press/v20/bailly11/bailly11.pdf

## Translation to Episode 012

Komorebi Block 14 evaluated polyphase matrices provide symbol transition
matrices.  A scalar source/target phase query has the form

`f(w)=alpha^T M_w omega`.

Therefore it is already a weighted linear representation in the classical
sense.  The finite Hankel matrix used in the audit is exactly the standard
prefix/suffix Hankel block.

Block 13's discrete observer identifies prefixes only by exact equality of their
future query functions.  Weighted realization instead stores coordinates in the
**linear span** of those residual functions.  Bailly's residual-space language
makes this distinction explicit.

## Classification

* Minimal WFA dimension = Hankel rank: **known verbatim in the checked modern source**.
* Finite residual-space dimension = weighted rationality: **known verbatim in the checked modern source**.
* Applying the theorem to evaluated Komorebi polyphase route matrices:
  **formal translation / application, not a new general theorem**.
* The explicit two-tap route family with infinite discrete residual classes and
  exact rank `N`: a project-specific worked example/proof; no novelty claim was
  established or searched globally for that exact example.

## Search boundary and limitations

Search terms covered weighted automata, rational series, Hankel rank, residual
space, minimal linear representation, and spectral learning.  Modern primary
PMLR sources were inspected directly.  The original Carlyle--Paz (1971) and
Fliess (1974) sources cited by Li et al. were **not independently authenticated**
in this episode.  Consequently attribution to those originals is secondary-source
attribution, while the theorem itself is verified from the modern primary
papers above.
