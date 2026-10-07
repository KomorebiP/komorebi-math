"""Adapters that turn mathematical values into readable documents."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..certificates import Certificate, ComputationResult, ConditionalResult
from .model import DocumentMetadata, MathDocument
from .serialization import table_rows, to_serializable


def certificate_document(certificate: Certificate, *, title: str = "Certificate") -> MathDocument:
    """Build a compact document describing one certificate."""

    document = MathDocument(DocumentMetadata(title=title))
    document.table(
        ("Field", "Value"),
        (
            ("Valid", certificate.valid),
            ("Exactness", certificate.exactness.value),
            ("Tolerance", certificate.tolerance),
            ("Theorem", certificate.theorem_id),
            ("Message", certificate.message),
        ),
    )
    if certificate.residuals:
        document.section("Residuals")
        document.table(("Name", "Value"), certificate.residuals.items())
    if certificate.assumptions_checked:
        document.section("Assumptions checked")
        document.bullet_list(certificate.assumptions_checked)
    if certificate.assumptions_unchecked:
        document.section("Assumptions not checked")
        document.bullet_list(certificate.assumptions_unchecked)
    if certificate.diagnostics:
        document.section("Diagnostics")
        document.code(json.dumps(to_serializable(certificate.diagnostics), indent=2, sort_keys=True), language="json")
    return document


def document_from_data(
    value: Any,
    *,
    title: str = "Komorebi Math Export",
    author: str | None = None,
    date: str | None = None,
    abstract: str | None = None,
    include_certificate: bool = True,
) -> MathDocument:
    """Create a readable mathematical document from common Python data."""

    if isinstance(value, MathDocument):
        return value

    metadata = DocumentMetadata(title=title, author=author, date=date, abstract=abstract)
    document = MathDocument(metadata)

    certificate: Certificate | None = None
    payload = value
    if isinstance(value, (ComputationResult, ConditionalResult)):
        payload = value.value
        certificate = value.certificate
    elif isinstance(value, Certificate):
        certificate = value
        payload = None

    if payload is not None:
        if isinstance(payload, np.ndarray) and payload.ndim <= 2:
            headers, rows = table_rows(payload)
            document.table(headers, rows, caption="Exported numerical data")
        elif isinstance(payload, Mapping):
            headers, rows = table_rows(payload)
            document.table(headers, rows, caption="Exported mapping")
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            headers, rows = table_rows(payload)
            document.table(headers, rows, caption="Exported sequence")
        elif is_dataclass(payload) and not isinstance(payload, type):
            normalized = to_serializable(payload, preserve_types=False)
            headers, rows = table_rows(normalized)
            document.table(headers, rows, caption=type(payload).__name__)
        elif isinstance(payload, str):
            document.paragraph(payload)
        else:
            document.code(
                json.dumps(to_serializable(payload), indent=2, sort_keys=True),
                language="json",
            )

    if certificate is not None and include_certificate:
        document.section("Certificate")
        document.table(
            ("Field", "Value"),
            (
                ("Valid", certificate.valid),
                ("Exactness", certificate.exactness.value),
                ("Tolerance", certificate.tolerance),
                ("Theorem", certificate.theorem_id),
                ("Message", certificate.message),
            ),
        )
        if certificate.residuals:
            document.section("Residuals", level=2)
            document.table(("Name", "Value"), certificate.residuals.items())
        if certificate.assumptions_checked:
            document.section("Assumptions checked", level=2)
            document.bullet_list(certificate.assumptions_checked)
        if certificate.assumptions_unchecked:
            document.section("Assumptions not checked", level=2)
            document.bullet_list(certificate.assumptions_unchecked)

    return document


def sequence_document(
    name: str,
    values: Sequence[Any],
    *,
    strategy: str | None = None,
    modulus: int | None = None,
    author: str | None = None,
) -> MathDocument:
    """Build a table report for one executable sequence."""

    title = f"{name} sequence"
    document = MathDocument(DocumentMetadata(title=title, author=author))
    if strategy:
        document.paragraph(f"Execution strategy: {strategy}")
    if modulus is not None:
        document.paragraph(f"Values are reduced modulo {modulus}.")
    document.table(
        ("Index", "Value"),
        tuple((index, value) for index, value in enumerate(values)),
        caption=f"First {len(values)} values of {name}",
    )
    return document


__all__ = ["certificate_document", "document_from_data", "sequence_document"]
