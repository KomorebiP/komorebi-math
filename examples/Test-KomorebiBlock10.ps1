param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

& $Python "$PSScriptRoot\scalable_nonlinear_runtime.py"
& $Python "$PSScriptRoot\portable_nonlinear_state.py"
& $Python -m komorebi_math api check --strict-signatures
& $Python -m komorebi_math verify block10
& $Python "$Root\verification\block10_deep_audit.py"
