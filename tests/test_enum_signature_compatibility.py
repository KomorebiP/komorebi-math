"""Enum reflection must retain the frozen API across Python versions."""

from __future__ import annotations

import inspect
from enum import EnumType, StrEnum

from komorebi_math import compatibility
from komorebi_math.certificates import Exactness


def test_inherited_enum_constructor_ignores_interpreter_signature(monkeypatch) -> None:
    reflected = inspect.signature

    def python311_signature(value):
        if isinstance(value, EnumType):
            return inspect.Signature(
                [
                    inspect.Parameter("value", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter(
                        "names", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                    ),
                ]
            )
        return reflected(value)

    monkeypatch.setattr(compatibility.inspect, "signature", python311_signature)
    assert compatibility._public_signature(Exactness) == "(*values)"
    compatibility.check_public_api(strict_signatures=True).certificate.require()
    compatibility.check_stable_api().certificate.require()


def test_explicit_enum_signature_is_preserved() -> None:
    class Explicit(StrEnum):
        MEMBER = "member"

    Explicit.__signature__ = inspect.Signature(
        [inspect.Parameter("member", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    assert compatibility._public_signature(Explicit) == "(member)"


def test_custom_metaclass_signature_with_inherited_constructor_is_preserved() -> None:
    class SignatureEnumType(EnumType):
        @property
        def __signature__(cls):
            return inspect.Signature(
                [inspect.Parameter("member", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            )

    class Custom(StrEnum, metaclass=SignatureEnumType):
        MEMBER = "member"

    assert type(Custom).__call__ is EnumType.__call__
    assert str(inspect.signature(Custom)) == "(member)"
    assert compatibility._public_signature(Custom) == "(member)"


def test_custom_enum_metaclass_constructor_is_reflected(monkeypatch) -> None:
    class CustomEnumType(EnumType):
        def __call__(cls, member, *, strict=True):
            return super().__call__(member)

    class Custom(StrEnum, metaclass=CustomEnumType):
        MEMBER = "member"

    reflected = inspect.signature

    def custom_signature(value):
        if value is Custom:
            return reflected(CustomEnumType.__call__).replace(
                parameters=list(reflected(CustomEnumType.__call__).parameters.values())[1:]
            )
        return reflected(value)

    monkeypatch.setattr(compatibility.inspect, "signature", custom_signature)
    assert compatibility._public_signature(Custom) == "(member, *, strict=True)"
