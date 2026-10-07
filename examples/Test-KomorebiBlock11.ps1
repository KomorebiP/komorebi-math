param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

& $Python "$PSScriptRoot\resumable_packet_jet_stream.py"
& $Python "$PSScriptRoot\block11_semantic_profile.py"
& $Python -m komorebi_math golden
& $Python -m komorebi_math api check --strict-signatures
& $Python -m komorebi_math api check --beta
& $Python -m komorebi_math verify block11
& $Python "$Root\verification\block11_deep_audit.py"
