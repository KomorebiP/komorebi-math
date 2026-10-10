"""Worked exact singular-prior Gaussian candidate and dyadic final check."""

from fractions import Fraction as F

from komorebi_math.dynamics.gaussian_verifier import (
    FactorGaussianModel,
    evaluate_gaussian_gain,
    exact_gaussian_posterior,
    gaussian_packet_energy,
    verify_gaussian_gain,
)


def main():
    model = FactorGaussianModel(
        prior_factor=[[1], [0]],
        observation_operator=[[1, 1]],
        noise_factor=[[]],
        noise_floor=1,
        prior_mean=[0, 0],
        observation=[1],
    )
    gain = [[F(1, 3)], [0]]
    candidate = evaluate_gaussian_gain(model, gain)
    target = exact_gaussian_posterior(model)
    packets = gaussian_packet_energy(
        model,
        gain,
        output_packets=[[[1, 0], [0, 0]], [[0, 0], [0, 1]]],
        input_packets=[[[1]]],
    )
    packets.certificate.require()
    coarse = verify_gaussian_gain(model, gain, tolerance=F(1, 2), precision=0)
    checked = verify_gaussian_gain(model, gain, tolerance=F(1, 2), precision=8, output_radius=F(1, 100))
    checked.certificate.require()
    assert not coarse.certificate.valid
    print("Exact posterior gain:", target.gain)
    print("Candidate Joseph variance:", candidate.covariance[0][0])
    print("Exact posterior variance:", target.covariance[0][0])
    print("Complete packet energy:", packets.total)
    print("Squared-law upper score:", candidate.score)
    print("Coarse interval passes:", coarse.certificate.valid)
    print("Dyadic upper score:", checked.score_interval.upper)
    print("Mean/covariance radius:", checked.mean_intervals[0].radius)


if __name__ == "__main__":
    main()
