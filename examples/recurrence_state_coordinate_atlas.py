"""Use named recurrence and state-space coordinate systems."""

from fractions import Fraction

from komorebi_math.coordinates import (
    certify_state_coordinate_change,
    controllability_coordinate_transform,
    get_recurrence_model,
    observability_coordinate_transform,
    sequence_coordinates_between_models,
    siso_state_space,
)

pell = get_recurrence_model("Pell")
tribonacci = get_recurrence_model("Tribonacci")
values = pell.sequence(16)
source, target, bridge = sequence_coordinates_between_models(values, pell, tribonacci)
assert bridge.apply(source.coordinates()) == target.coordinates()

model = siso_state_space(
    ((0, 1, 0), (0, 0, 1), (2, -1, 3)),
    (0, 0, 1),
    (1, 2, -1),
    feedthrough=2,
    name="third-order model",
)
controllability = controllability_coordinate_transform(model)
certify_state_coordinate_change(model, controllability).certificate().require()

observability = observability_coordinate_transform(model)
state = (Fraction(2), Fraction(-3), Fraction(5))
assert observability.inverse().apply(observability.apply(state)) == state

print("recurrence bridge:", bridge.source, "->", bridge.target)
print("observable state coordinates:", observability.apply(state))
