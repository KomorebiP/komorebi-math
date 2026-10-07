$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python verification/release_artifact_gate.py --runtime-tests --root . --output release/REPAIR_R2_ARTIFACT_GATE.json
    if ($LASTEXITCODE -ne 0) { throw 'Installed artifact verification failed.' }
    python verification/block21_publication_gate.py --full-tests --skip-mkdocs --output release/REPAIR_R2_PUBLICATION_GATE.json --markdown release/REPAIR_R2_PUBLICATION_GATE.md
    if ($LASTEXITCODE -ne 0) { throw 'Source publication gate failed.' }
} finally { Pop-Location }
