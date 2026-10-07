"""Compile several exact sequence families to their strongest declared models."""

from komorebi_math.sequences import (
    apery_sequence,
    compile_sequence,
    fibonacci_sequence,
    thue_morse_sequence,
)


def main() -> None:
    for sequence in (fibonacci_sequence(), thue_morse_sequence(), apery_sequence()):
        compiled = compile_sequence(sequence)
        compiled.certificate.require()
        print(sequence.name, compiled.strategy, compiled.first(12))


if __name__ == "__main__":
    main()
