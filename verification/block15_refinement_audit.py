"""Deterministic independent audit for Block 15 finite refinement identities."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from komorebi_math.arithmetic import polyphase_analyze as arithmetic_analyze
from komorebi_math.refinement import (
    LatticePolyphaseRefinement,
    PolyphaseRefinement,
    PolyphaseRefinementTower,
    residue_analyze,
)
from komorebi_math.signals import lattice_polyphase_analyze, polyphase_analyze

SEED = 20260818


def main() -> int:
    rng = np.random.default_rng(SEED)
    comparisons = 0
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        nonlocal comparisons
        comparisons += 1
        if not condition:
            failures.append(label)

    for trial in range(320):
        ndim = int(rng.integers(1, 4))
        shape = tuple(int(rng.integers(1, 16)) for _ in range(ndim))
        axis = int(rng.integers(-ndim, ndim))
        values = rng.normal(size=shape)
        coarse_order = int(rng.integers(1, 9))
        ratio = int(rng.integers(1, 6))
        fine_order = coarse_order * ratio
        coarse = polyphase_analyze(values, coarse_order, axis=axis)
        plan = PolyphaseRefinement(coarse_order, fine_order)
        fine = plan.refine(coarse)
        restored = plan.coarsen(fine)
        direct = polyphase_analyze(values, fine_order, axis=axis)
        check(np.array_equal(fine.reconstruct(), values), f"signal reconstruct {trial}")
        check(np.array_equal(restored.reconstruct(), values), f"signal coarsen {trial}")
        check(fine.axis == coarse.axis, f"signal axis {trial}")
        check(plan.certificate(coarse).valid, f"signal certificate {trial}")
        for address in range(fine_order):
            check(
                np.array_equal(fine.channels[address], direct.channels[address]),
                f"signal direct {trial}:{address}",
            )

    for length in range(0, 96):
        values = np.arange(length, dtype=np.int64)
        for coarse_order in (1, 2, 3, 5, 8):
            for ratio in (1, 2, 4):
                fine_order = coarse_order * ratio
                coarse = residue_analyze(values, coarse_order)
                plan = PolyphaseRefinement(coarse_order, fine_order)
                fine = plan.refine(coarse)
                restored = plan.coarsen(fine)
                direct = residue_analyze(values, fine_order)
                check(np.array_equal(fine.reconstruct(), values), "residue reconstruct")
                check(np.array_equal(restored.reconstruct(), values), "residue coarsen")
                check(plan.certificate(coarse).valid, "residue certificate")
                for address in range(fine_order):
                    check(
                        np.array_equal(fine.channels[address], direct.channels[address]),
                        "residue direct",
                    )

    for length in range(0, 48):
        values = tuple(Fraction(index + 1, index + 2) for index in range(length))
        for coarse_order, ratio in ((1, 1), (1, 5), (2, 3), (3, 4), (5, 2)):
            fine_order = coarse_order * ratio
            coarse = arithmetic_analyze(values, coarse_order)
            plan = PolyphaseRefinement(coarse_order, fine_order)
            fine = plan.refine(coarse)
            restored = plan.coarsen(fine)
            check(fine.reconstruct() == values, "arithmetic reconstruct")
            check(restored.reconstruct() == values, "arithmetic coarsen")
            check(plan.certificate(coarse).valid, "arithmetic certificate")

    lattice_cases = (
        ((1, 1), (1, 1), (1, 1)),
        ((2, 3), (2, 3), (2, 2)),
        ((5, 7), (2, 3), (3, 2)),
        ((4, 3, 2), (2, 2, 1), (2, 3, 4)),
        ((3, 5, 7), (3, 2, 2), (2, 2, 3)),
    )
    for repeat in range(50):
        for shape, coarse_strides, ratios in lattice_cases:
            values = np.arange(int(np.prod(shape))).reshape(shape)
            fine_strides = tuple(
                coarse * ratio
                for coarse, ratio in zip(coarse_strides, ratios, strict=True)
            )
            coarse = lattice_polyphase_analyze(values, coarse_strides)
            plan = LatticePolyphaseRefinement(coarse_strides, fine_strides)
            fine = plan.refine(coarse)
            restored = plan.coarsen(fine)
            direct = lattice_polyphase_analyze(values, fine_strides)
            check(np.array_equal(fine.reconstruct(), values), "lattice reconstruct")
            check(np.array_equal(restored.reconstruct(), values), "lattice coarsen")
            check(plan.certificate(coarse).valid, "lattice certificate")
            for address in fine.channels:
                check(np.array_equal(fine.channels[address], direct.channels[address]), "lattice direct")

    for orders in ((1, 2), (2, 6, 12), (3, 6, 24), (1, 3, 12, 24)):
        values = np.arange(53)
        tower = PolyphaseRefinementTower(orders)
        coarse = polyphase_analyze(values, orders[0])
        fine = tower.refine(coarse)
        direct = polyphase_analyze(values, orders[-1])
        check(np.array_equal(fine.reconstruct(), values), "tower reconstruct")
        check(np.array_equal(tower.coarsen(fine).reconstruct(), values), "tower coarsen")
        check(tower.coherence_certificate(coarse).valid, "tower certificate")
        for left, right in zip(fine.channels, direct.channels, strict=True):
            check(np.array_equal(left, right), "tower direct")

    result = {
        "schema": "komorebi-math.block15-refinement-audit",
        "schema_version": 1,
        "seed": SEED,
        "comparisons": comparisons,
        "failures": failures,
        "passed": not failures,
    }
    destination = Path(__file__).with_name("BLOCK15_REFINEMENT_AUDIT.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
