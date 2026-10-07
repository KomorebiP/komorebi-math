"""Compare the odd and dyadic Euler coordinate systems."""

from komorebi_math.arithmetic import EulerLocalCoordinates


def main() -> None:
    for prime in (2, 5):
        coordinates = EulerLocalCoordinates(prime)
        result = coordinates.truncate(index=20, depth=2)
        result.certificate.require()
        print(
            {
                "prime": prime,
                "exact": coordinates.reconstruct(20),
                "truncated": str(result.value),
                "modulus": result.modulus,
                "error": str(result.error),
            }
        )


if __name__ == "__main__":
    main()
