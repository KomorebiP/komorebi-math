"""Execute recurrence packets and modular terms without large integers."""

from komorebi_math.arithmetic import fibonacci_recurrence


def main() -> None:
    recurrence = fibonacci_recurrence()
    packet = recurrence.residue_packet(modulus=7, address=3)
    packet.certificate(count=12).certificate.require()
    print("packet", packet.first(12))
    print("F(1_000_000) mod 1_000_003", recurrence.term_mod(1_000_000, 1_000_003))
    print("period mod 32", recurrence.period_mod(32).period)


if __name__ == "__main__":
    main()
