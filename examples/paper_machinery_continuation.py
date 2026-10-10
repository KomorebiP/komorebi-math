"""Five exact finite worked constructions; install komorebi-math[exact]."""

import json
from fractions import Fraction

from komorebi_math.algorithms.boolean_synthesis import BooleanExpr, bounded_boolean_search
from komorebi_math.arithmetic.lehmer_shells import enumerate_five_core_shell, tau_from_q_product
from komorebi_math.decision.avoc import FiniteChannel
from komorebi_math.generating.residual_modules import RingLinearRealization, canonical_residual_module
from komorebi_math.signals.tikhonov import bracket_discrepancy


def main():
    x, y = BooleanExpr.variable(1), BooleanExpr.variable(2)
    reference = BooleanExpr.negate(BooleanExpr.binary("or", BooleanExpr.negate(x), BooleanExpr.negate(y)))
    synthesis = bounded_boolean_search(reference, 1, budget=6890, trial_cap=85)
    assert synthesis.certificate.valid and synthesis.selected_nodes == 3 and synthesis.modeled_charge == 6890

    channel = FiniteChannel(((1, 0), (0, 1)))
    value = channel.optimal_policy((Fraction(1, 2), Fraction(1, 2)), ((1, 0), (0, 1))).value
    garbled = channel.garble(FiniteChannel(((1,), (1,))))
    garbled_value = garbled.optimal_policy((Fraction(1, 2), Fraction(1, 2)), ((1, 0), (0, 1))).value
    assert value == 1 and garbled_value == Fraction(1, 2)

    model = RingLinearRealization(2, 2, (((1, 0), (2, 1)),), (1, 0), (2, 0))
    quotient = canonical_residual_module(model)
    assert quotient.verify_laws().valid and quotient.additive_p_exponents() == (1,)

    bracket = bracket_discrepancy(((1, 0), (0, 0)), (3, 4), Fraction(18), width=Fraction(1, 1024))
    assert bracket.converged and bracket.certificate.valid
    shell = enumerate_five_core_shell(121)
    assert shell.weighted_sum == tau_from_q_product(128)[121]

    print(json.dumps({
        "boolean": {"before_nodes": synthesis.reference_nodes, "after_nodes": synthesis.selected_nodes,
                    "modeled_search_charge": synthesis.modeled_charge},
        "channel": {"value": str(value), "garbled_value": str(garbled_value)},
        "ring": {"ambient": quotient.ambient_cardinality, "reachable": len(quotient.reachable_states),
                 "residual": quotient.carrier_cardinality, "additive_p_exponents": quotient.additive_p_exponents()},
        "discrepancy": {"target_squared": 18, "attainable_floor_squared": 16,
                        "parameter_interval": [str(bracket.lower), str(bracket.upper)]},
        "shell_121": {"packets": len(shell.packets), "signed_total": shell.weighted_sum,
                      "content_masses": shell.content_masses},
        "scope": "Exact supplied finite objects; no universal limit, paper-completion or conjecture claim.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
