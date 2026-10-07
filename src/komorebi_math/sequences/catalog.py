"""A small exact catalog of classical integer sequences."""

from __future__ import annotations

from fractions import Fraction
from math import comb

from ..arithmetic.euler import bernoulli_numbers, euler_zigzag_number, signed_euler_number
from ..arithmetic.recurrence import fibonacci_recurrence, lucas_recurrence
from .automatic import thue_morse_automaton
from .holonomic import apery_recurrence, motzkin_recurrence
from .base import Sequence


def fibonacci_sequence() -> Sequence:
    """Return the Fibonacci sequence."""

    recurrence = fibonacci_recurrence()
    return Sequence(recurrence.term, name="Fibonacci", metadata={"recurrence": recurrence, "recurrence_certified": True})


def lucas_sequence() -> Sequence:
    """Return the Lucas sequence."""

    recurrence = lucas_recurrence()
    return Sequence(recurrence.term, name="Lucas", metadata={"recurrence": recurrence, "recurrence_certified": True})


def signed_euler_sequence() -> Sequence:
    """Return the signed Euler sequence from ``sech``."""

    return Sequence(signed_euler_number, name="signed Euler")


def euler_zigzag_sequence() -> Sequence:
    """Return the Euler zigzag sequence."""

    return Sequence(euler_zigzag_number, name="Euler zigzag")


def bernoulli_sequence() -> Sequence:
    """Return Bernoulli numbers with ``B_1 = -1/2``."""

    cache: list[Fraction] = []

    def evaluate(index: int) -> Fraction:
        nonlocal cache
        if len(cache) <= index:
            cache = list(bernoulli_numbers(index))
        return cache[index]

    return Sequence(evaluate, name="Bernoulli")


def catalan_sequence() -> Sequence:
    """Return Catalan numbers."""

    return Sequence(lambda n: comb(2 * n, n) // (n + 1), name="Catalan")


def bell_sequence() -> Sequence:
    """Return Bell numbers."""

    cache = [1]

    def evaluate(index: int) -> int:
        while len(cache) <= index:
            n = len(cache) - 1
            cache.append(sum(comb(n, k) * cache[k] for k in range(n + 1)))
        return cache[index]

    return Sequence(evaluate, name="Bell")


def motzkin_sequence() -> Sequence:
    """Return Motzkin numbers through an exact holonomic recurrence."""

    return motzkin_recurrence().to_sequence()


def partition_sequence() -> Sequence:
    """Return integer partition numbers."""

    cache = [1]
    sigma = [0]

    def ensure(index: int) -> None:
        while len(cache) <= index:
            n = len(cache)
            sigma.append(sum(divisor for divisor in range(1, n + 1) if n % divisor == 0))
            cache.append(sum(sigma[k] * cache[n - k] for k in range(1, n + 1)) // n)

    def evaluate(index: int) -> int:
        ensure(index)
        return cache[index]

    return Sequence(evaluate, name="partition")


def apery_sequence() -> Sequence:
    """Return the Apéry numbers for ``zeta(3)`` through their exact recurrence."""

    return apery_recurrence().to_sequence()


def thue_morse_sequence() -> Sequence:
    """Return the zero one Thue Morse automatic sequence."""

    return thue_morse_automaton().to_sequence()


def sequence_catalog() -> dict[str, Sequence]:
    """Return a new dictionary of built in sequence objects."""

    return {
        "apery": apery_sequence(),
        "bell": bell_sequence(),
        "bernoulli": bernoulli_sequence(),
        "catalan": catalan_sequence(),
        "euler": signed_euler_sequence(),
        "fibonacci": fibonacci_sequence(),
        "lucas": lucas_sequence(),
        "motzkin": motzkin_sequence(),
        "partition": partition_sequence(),
        "thue-morse": thue_morse_sequence(),
        "zigzag": euler_zigzag_sequence(),
    }


def get_sequence(name: str) -> Sequence:
    """Return one built in sequence by a normalized name."""

    key = name.strip().lower().replace("_", "-")
    catalog = sequence_catalog()
    aliases = {
        "apery-zeta3": "apery",
        "euler-zigzag": "zigzag",
        "signed-euler": "euler",
        "thue-morse": "thue-morse",
        "thuemorse": "thue-morse",
    }
    key = aliases.get(key, key)
    if key not in catalog:
        raise KeyError(f"unknown built in sequence: {name}")
    return catalog[key]


__all__ = [
    "apery_sequence",
    "bell_sequence",
    "bernoulli_sequence",
    "catalan_sequence",
    "euler_zigzag_sequence",
    "fibonacci_sequence",
    "get_sequence",
    "lucas_sequence",
    "motzkin_sequence",
    "partition_sequence",
    "sequence_catalog",
    "signed_euler_sequence",
    "thue_morse_sequence",
]
