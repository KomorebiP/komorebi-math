param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

& $Python "$PSScriptRoot\nonlinear_jets_composition.py"
& $Python "$PSScriptRoot\nonlinear_trees_elimination.py"
& $Python "$PSScriptRoot\filtered_transport.py"
& $Python -m komorebi_math verify block9
& $Python "$Root\verification\block9_deep_audit.py"
