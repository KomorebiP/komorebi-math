"""Build a standard window and refine one Mahler series."""

from komorebi_math.arithmetic import MahlerSeries, StandardWindow


def main() -> None:
    window = StandardWindow(prime=7, compression=2)
    window.certificate(address=3, periods=4).certificate.require()
    print("period", window.period)
    print("window", window.residues(address=3, degree=12))

    series = MahlerSeries.from_values([n**4 + 2 * n + 1 for n in range(9)])
    refined = series.affine_subsequence(multiplier=3, shift=2)
    print("refined", refined.first(8))


if __name__ == "__main__":
    main()
