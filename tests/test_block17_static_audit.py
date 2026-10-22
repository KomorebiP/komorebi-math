"""Compatibility imports resolve to the current static source audit."""
from verification import block16_static_audit, block17_static_audit, source_audit


def test_previous_static_audit_imports_use_current_checks() -> None:
    assert block16_static_audit.run_audit is source_audit.run_audit
    assert block17_static_audit.run_audit is source_audit.run_audit
    assert block16_static_audit._shell_mode_diagnostics is source_audit._shell_mode_diagnostics
