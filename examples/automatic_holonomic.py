"""Use automatic and holonomic sequence models directly."""

from komorebi_math.sequences import apery_recurrence, thue_morse_automaton


def main() -> None:
    automatic = thue_morse_automaton()
    automatic.certificate(256).certificate.require()
    residue = automatic.residue_model(power=4, address=7)
    print("Thue Morse residue", residue.first(20))

    holonomic = apery_recurrence()
    holonomic.certificate(15).certificate.require()
    print("Apery", holonomic.first(10))


if __name__ == "__main__":
    main()
