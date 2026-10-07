[CmdletBinding()]
param(
    [ValidateSet("Quick", "Full", "Maximum")]
    [string]$Mode = "Full",
    [string]$Python = "py",
    [switch]$UseSystemSitePackages,
    [string]$OfflineWheelhouse = "",
    [switch]$NoNetwork,
    [switch]$InstallOptionalAD,
    [switch]$RunCuda,
    [switch]$RunBuildReproducibility,
    [switch]$KeepWorkDirectory,
    [string]$ResultsDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$BundleRoot = Split-Path -Parent $PSScriptRoot
if (Test-Path (Join-Path $PSScriptRoot "komorebi_math-0.16.0a2-py3-none-any.whl")) {
    $BundleRoot = $PSScriptRoot
}

if ([string]::IsNullOrWhiteSpace($ResultsDirectory)) {
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $ResultsDirectory = Join-Path $BundleRoot "validation-results-$Stamp"
}
$ResultsDirectory = [System.IO.Path]::GetFullPath($ResultsDirectory)
$LogsDirectory = Join-Path $ResultsDirectory "logs"
$WorkDirectory = Join-Path $ResultsDirectory "work"
$SummaryPath = Join-Path $ResultsDirectory "validation-summary.json"
$EnvironmentPath = Join-Path $ResultsDirectory "environment.json"
New-Item -ItemType Directory -Force -Path $LogsDirectory, $WorkDirectory | Out-Null

$script:Results = [System.Collections.Generic.List[object]]::new()
$script:RequiredFailures = 0
$script:OptionalFailures = 0

function Get-SafeName {
    param([Parameter(Mandatory = $true)][string]$Name)
    return ($Name -replace '[^A-Za-z0-9._-]+', '_').Trim('_')
}

function Add-Result {
    param(
        [string]$Name,
        [string]$Status,
        [bool]$Required,
        [double]$Seconds,
        [string]$Log,
        [string]$Detail
    )
    $script:Results.Add([pscustomobject]@{
        name = $Name
        status = $Status
        required = $Required
        seconds = [Math]::Round($Seconds, 3)
        log = $Log
        detail = $Detail
    })
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [switch]$Optional,
        [switch]$Critical
    )
    $Safe = Get-SafeName $Name
    $LogPath = Join-Path $LogsDirectory "$Safe.log"
    $Watch = [System.Diagnostics.Stopwatch]::StartNew()
    $Passed = $false
    $Detail = ""
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    try {
        $global:LASTEXITCODE = 0
        & $Action *>&1 | Tee-Object -FilePath $LogPath
        $Code = $global:LASTEXITCODE
        if ($null -eq $Code) {
            $Code = 0
        }
        if ($Code -ne 0) {
            throw "Native command exited with code $Code"
        }
        $Passed = $true
        $Detail = "completed"
    }
    catch {
        $Detail = $_.Exception.Message
        $Detail | Out-File -FilePath $LogPath -Append -Encoding utf8
        if ($Optional) {
            $script:OptionalFailures += 1
            Write-Warning "$Name failed and was recorded as optional: $Detail"
        }
        else {
            $script:RequiredFailures += 1
            Write-Error "$Name failed: $Detail" -ErrorAction Continue
        }
    }
    finally {
        $Watch.Stop()
        Add-Result -Name $Name -Status $(if ($Passed) { "passed" } elseif ($Optional) { "optional-failure" } else { "failed" }) -Required (-not $Optional) -Seconds $Watch.Elapsed.TotalSeconds -Log $LogPath -Detail $Detail
    }
    if (-not $Passed -and $Critical) {
        throw "Critical validation step failed: $Name"
    }
    return $Passed
}

function Test-PythonModule {
    param([string]$PythonPath, [string]$ModuleName)
    & $PythonPath -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Invoke-PythonInDirectory {
    param(
        [string]$Directory,
        [string]$PythonPath,
        [string[]]$Arguments,
        [string]$PythonPathValue = ""
    )
    $OldPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $PythonPathValue
        Push-Location $Directory
        try {
            & $PythonPath @Arguments
        }
        finally {
            Pop-Location
        }
    }
    finally {
        $env:PYTHONPATH = $OldPythonPath
    }
}

$Wheel = Get-ChildItem -Path $BundleRoot -Recurse -File -Filter "komorebi_math-0.16.0a2-py3-none-any.whl" | Select-Object -First 1
$Sdist = Get-ChildItem -Path $BundleRoot -Recurse -File -Filter "komorebi_math-0.16.0a2.tar.gz" | Select-Object -First 1
$GitBundle = Get-ChildItem -Path $BundleRoot -Recurse -File -Filter "Komorebi_Math_Wheel_Block_16.bundle" | Select-Object -First 1
$ChecksumFile = Get-ChildItem -Path $BundleRoot -File -Filter "SHA256SUMS.txt" | Select-Object -First 1

if ($null -eq $Wheel) {
    throw "The Block 16 wheel was not found under $BundleRoot"
}
if ($null -eq $Sdist) {
    throw "The Block 16 source distribution was not found under $BundleRoot"
}

$Environment = [ordered]@{
    schema = "komorebi-math.local-validation-environment.v1"
    started_utc = (Get-Date).ToUniversalTime().ToString("o")
    mode = $Mode
    host = $Host.Name
    powershell = $PSVersionTable.PSVersion.ToString()
    os = [System.Environment]::OSVersion.VersionString
    machine = [System.Environment]::MachineName
    processor_count = [System.Environment]::ProcessorCount
    base_python = $Python
    wheel = $Wheel.FullName
    sdist = $Sdist.FullName
    git_bundle = if ($null -ne $GitBundle) { $GitBundle.FullName } else { $null }
    use_system_site_packages = [bool]$UseSystemSitePackages
    no_network = [bool]$NoNetwork
    offline_wheelhouse = $OfflineWheelhouse
    install_optional_ad = [bool]$InstallOptionalAD
    run_cuda = [bool]$RunCuda
    run_build_reproducibility = [bool]$RunBuildReproducibility
}
$Environment | ConvertTo-Json -Depth 8 | Set-Content -Path $EnvironmentPath -Encoding utf8

try {
    if ($null -ne $ChecksumFile) {
        Invoke-Step -Name "Bundle SHA256 manifest" -Critical -Action {
            $Failures = 0
            foreach ($Line in Get-Content $ChecksumFile.FullName) {
                if ([string]::IsNullOrWhiteSpace($Line)) {
                    continue
                }
                if ($Line -notmatch '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
                    throw "Malformed checksum line: $Line"
                }
                $Expected = $Matches[1].ToLowerInvariant()
                $Relative = $Matches[2].Trim()
                $Path = Join-Path $BundleRoot $Relative
                if (-not (Test-Path $Path -PathType Leaf)) {
                    Write-Host "Missing: $Relative"
                    $Failures += 1
                    continue
                }
                $Actual = (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($Actual -ne $Expected) {
                    Write-Host "Mismatch: $Relative"
                    $Failures += 1
                }
                else {
                    Write-Host "Passed: $Relative"
                }
            }
            if ($Failures -ne 0) {
                throw "$Failures checksum validations failed"
            }
        } | Out-Null
    }

    Invoke-Step -Name "Base Python identity" -Critical -Action {
        & $Python -c "import json,platform,sys; print(json.dumps({'executable':sys.executable,'version':sys.version,'platform':platform.platform()},indent=2))"
    } | Out-Null

    $SourceRoot = Join-Path $WorkDirectory "repository"
    $PackageRoot = $null
    $Git = Get-Command git -ErrorAction SilentlyContinue
    if ($null -ne $GitBundle -and $null -ne $Git) {
        Invoke-Step -Name "Clone tagged Git bundle" -Critical -Action {
            if (Test-Path $SourceRoot) {
                Remove-Item -Recurse -Force $SourceRoot
            }
            & git clone $GitBundle.FullName $SourceRoot
            & git -C $SourceRoot checkout wheel-v0.16.0a2
            & git -C $SourceRoot fsck --full --strict
        } | Out-Null
        $PackageRoot = Join-Path $SourceRoot "packages\komorebi-math"
    }
    else {
        $ExtractHelper = Join-Path $WorkDirectory "extract_sdist.py"
        @'
from __future__ import annotations
import sys
import tarfile
from pathlib import Path
source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
destination.mkdir(parents=True, exist_ok=True)
with tarfile.open(source, "r:gz") as archive:
    members = archive.getmembers()
    for member in members:
        target = (destination / member.name).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(f"unsafe archive path: {member.name}") from exc
        if member.issym() or member.islnk():
            raise RuntimeError(f"archive links are not accepted: {member.name}")
    if sys.version_info >= (3, 12):
        archive.extractall(destination, filter="data")
    else:
        archive.extractall(destination)
roots = [path for path in destination.iterdir() if path.is_dir()]
if len(roots) != 1:
    raise RuntimeError(f"expected one source root, found {len(roots)}")
print(roots[0])
'@ | Set-Content -Path $ExtractHelper -Encoding utf8
        $ExtractRoot = Join-Path $WorkDirectory "sdist"
        Invoke-Step -Name "Safely extract source distribution" -Critical -Action {
            & $Python $ExtractHelper $Sdist.FullName $ExtractRoot
        } | Out-Null
        $PackageRoot = (Get-ChildItem -Path $ExtractRoot -Directory | Select-Object -First 1).FullName
    }

    if (-not (Test-Path (Join-Path $PackageRoot "pyproject.toml"))) {
        throw "The package source root is invalid: $PackageRoot"
    }

    $Venv = Join-Path $WorkDirectory "venv"
    Invoke-Step -Name "Create validation virtual environment" -Critical -Action {
        $Arguments = @("-m", "venv")
        if ($UseSystemSitePackages) {
            $Arguments += "--system-site-packages"
        }
        $Arguments += $Venv
        & $Python @Arguments
    } | Out-Null

    $Py = Join-Path $Venv "Scripts\python.exe"
    $Komorebi = Join-Path $Venv "Scripts\komorebi.exe"
    if (-not (Test-Path $Py)) {
        throw "Virtual environment Python was not created: $Py"
    }

    Invoke-Step -Name "Install Block 16 wheel" -Critical -Action {
        $Arguments = @("-m", "pip", "install")
        if (-not [string]::IsNullOrWhiteSpace($OfflineWheelhouse)) {
            $Arguments += @("--no-index", "--find-links", $OfflineWheelhouse)
        }
        elseif ($NoNetwork -or $UseSystemSitePackages) {
            $Arguments += "--no-deps"
        }
        $Arguments += $Wheel.FullName
        & $Py @Arguments
    } | Out-Null

    if (-not $NoNetwork -or -not [string]::IsNullOrWhiteSpace($OfflineWheelhouse)) {
        $Dependencies = @("pytest>=8", "sympy>=1.12", "scipy>=1.11", "mpmath>=1.3", "matplotlib>=3.8", "PyYAML>=6.0")
        if ($Mode -eq "Maximum" -or $RunBuildReproducibility) {
            $Dependencies += @("build>=1.2", "mypy>=1.10", "ruff>=0.6", "mkdocs>=1.6")
        }
        if ($InstallOptionalAD) {
            $Dependencies += @("jax>=0.4", "torch>=2.1")
        }
        Invoke-Step -Name "Install validation dependencies" -Critical -Action {
            $Arguments = @("-m", "pip", "install")
            if (-not [string]::IsNullOrWhiteSpace($OfflineWheelhouse)) {
                $Arguments += @("--no-index", "--find-links", $OfflineWheelhouse)
            }
            $Arguments += $Dependencies
            & $Py @Arguments
        } | Out-Null
    }

    Invoke-Step -Name "Installed package identity" -Action { & $Komorebi info } | Out-Null
    Invoke-Step -Name "Installed dependency doctor" -Action { & $Komorebi doctor } | Out-Null
    Invoke-Step -Name "Installed feature registry" -Action { & $Komorebi features } | Out-Null
    Invoke-Step -Name "Installed inherited Block 14 verifier" -Action { & $Komorebi verify block14 } | Out-Null
    Invoke-Step -Name "Installed coordinate verifier" -Action { & $Komorebi verify coordinates } | Out-Null
    Invoke-Step -Name "Installed generating verifier" -Action { & $Komorebi verify generating } | Out-Null
    Invoke-Step -Name "Installed DSP verifier" -Action { & $Komorebi verify dsp } | Out-Null
    Invoke-Step -Name "Installed semantic golden fixtures" -Action { & $Komorebi golden } | Out-Null
    Invoke-Step -Name "Installed research API compatibility" -Action { & $Komorebi api check --strict-signatures } | Out-Null
    Invoke-Step -Name "Installed beta API compatibility" -Action { & $Komorebi api check --beta } | Out-Null
    Invoke-Step -Name "Installed stable API compatibility" -Action {
        & $Py -c "from komorebi_math.compatibility import check_stable_api; r=check_stable_api(); print(r.to_dict()); raise SystemExit(0 if r.compatible else 1)"
    } | Out-Null
    Invoke-Step -Name "Installed complete stable contract" -Action { & $Komorebi verify stable } | Out-Null
    Invoke-Step -Name "Installed release contract" -Action { & $Komorebi release contract } | Out-Null
    Invoke-Step -Name "Installed release policy" -Action { & $Komorebi verify release } | Out-Null
    Invoke-Step -Name "Installed cumulative Block 16 verifier" -Action { & $Komorebi verify block16 } | Out-Null
    Invoke-Step -Name "Installed release qualification" -Action {
        & $Komorebi release qualify --output (Join-Path $ResultsDirectory "BLOCK16_INSTALLED_QUALIFICATION.json") --force
    } | Out-Null

    Invoke-Step -Name "Coordinate catalog CLI" -Action {
        & $Komorebi coordinates catalog
    } | Out-Null
    Invoke-Step -Name "Arbitrary basis coordinate CLI" -Action {
        & $Komorebi coordinates basis --source-matrix "1,0;0,1" --target-matrix "1,1;0,1" --vector "3,5"
    } | Out-Null
    Invoke-Step -Name "Recurrence model coordinate CLI" -Action {
        & $Komorebi coordinates model --source pell --target tribonacci --count 24
    } | Out-Null
    Invoke-Step -Name "Classical polynomial coordinate CLI" -Action {
        & $Komorebi coordinates polynomial --source legendre --target chebyshev-t --degree 14
    } | Out-Null
    Invoke-Step -Name "Combinatorial coordinate CLI" -Action {
        & $Komorebi coordinates combinatorial --kind forward-difference --size 8 --values "0,1,4,9,16,25,36,49"
    } | Out-Null
    Invoke-Step -Name "Spectral coordinate CLI" -Action {
        & $Komorebi coordinates spectral --kind haar --size 8
    } | Out-Null
    Invoke-Step -Name "Packet coordinate CLI" -Action {
        & $Komorebi coordinates packet --kind residue --length 12 --modulus 3
    } | Out-Null
    Invoke-Step -Name "Formal coordinate CLI" -Action {
        & $Komorebi coordinates formal --coordinate "0,1,1/2,1/6" --series "1,2,3,4"
    } | Out-Null
    Invoke-Step -Name "Named Euler to Fibonacci regression CLI" -Action {
        & $Komorebi coordinates euler-fibonacci --count 32
    } | Out-Null

    $SourcePythonPath = Join-Path $PackageRoot "src"
    Invoke-Step -Name "Source byte compilation" -Action {
        Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("-m", "compileall", "-q", "src", "tests", "examples", "verification") -PythonPathValue $SourcePythonPath
    } | Out-Null

    if ($Mode -ne "Quick") {
        Invoke-Step -Name "Installed wheel unit tests" -Action {
            Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("verification/run_isolated_test_batches.py", "--output", (Join-Path $ResultsDirectory "INSTALLED_WHEEL_TEST_BATCHES.json")) -PythonPathValue ""
        } | Out-Null
    }
    else {
        Invoke-Step -Name "Focused Block 16 and compatibility unit tests" -Action {
            Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("-m", "pytest", "-q", "tests/test_block15_polyphase_refinement.py", "tests/test_block15_persistence.py", "tests/test_block15_stable_api.py", "tests/test_block16_analysis_synthesis.py", "tests/test_block16_contracts.py", "tests/test_block16_release_qualification.py", "tests/test_block16_cli.py", "tests/test_block16_release_contract.py", "tests/test_block16_gpu_qualification.py", "tests/test_block16_companion_crosswalk.py", "tests/test_block16_release_layout.py", "tests/test_block16_static_audit.py", "tests/test_block16_package_audit.py", "tests/test_optional_backend_probe.py") -PythonPathValue ""
        } | Out-Null
    }

    if ($Mode -eq "Maximum") {
        Invoke-Step -Name "Source tree unit tests" -Action {
            Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("verification/run_isolated_test_batches.py", "--output", (Join-Path $ResultsDirectory "SOURCE_TEST_BATCHES.json")) -PythonPathValue $SourcePythonPath
        } | Out-Null
    }

    if ($Mode -ne "Quick") {
        $ExampleFiles = Get-ChildItem -Path (Join-Path $PackageRoot "examples") -File -Filter "*.py" | Sort-Object Name
        foreach ($Example in $ExampleFiles) {
            Invoke-Step -Name "Example $($Example.Name)" -Action {
                Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @($Example.FullName) -PythonPathValue $SourcePythonPath
            } | Out-Null
        }
    }
    else {
        foreach ($ExampleName in @("quickstart.py", "general_coordinate_atlas.py", "classical_coordinate_families.py", "recurrence_state_coordinate_atlas.py", "spectral_packet_coordinates.py", "formal_affine_coordinate_changes.py")) {
            $Example = Join-Path $PackageRoot "examples\$ExampleName"
            Invoke-Step -Name "Example $ExampleName" -Action {
                Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @($Example) -PythonPathValue $SourcePythonPath
            } | Out-Null
        }
    }

    Invoke-Step -Name "Block 14 independent deep audit" -Action {
        Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("verification/block14_deep_audit.py") -PythonPathValue $SourcePythonPath
    } | Out-Null
    Invoke-Step -Name "Block 14 independent reference audit" -Action {
        Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("verification/block14_reference_audit.py") -PythonPathValue $SourcePythonPath
    } | Out-Null
    Invoke-Step -Name "Block 14 general coordinate atlas audit" -Action {
        Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("verification/block14_general_coordinate_audit.py") -PythonPathValue $SourcePythonPath
    } | Out-Null
    Invoke-Step -Name "Block 15 polyphase refinement audit" -Action {
        Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("verification/block15_refinement_audit.py") -PythonPathValue $SourcePythonPath
    } | Out-Null

    Invoke-Step -Name "Coordinate persistence and tamper rejection" -Action {
        $Smoke = Join-Path $WorkDirectory "coordinate_persistence_smoke.py"
        @'
from fractions import Fraction
from pathlib import Path
import json
from komorebi_math.coordinates import (
    CoordinateAtlas, CoordinateSystem, CoordinateTransform,
    load_coordinate_object, save_coordinate_object,
)
root = Path(__file__).resolve().parent
path = root / "coordinate-atlas.json"
atlas = CoordinateAtlas("persistence smoke atlas")
atlas.register_system(CoordinateSystem("source", 2))
atlas.register_system(CoordinateSystem("target", 2))
transform = CoordinateTransform("source", "target", ((1, 2), (0, 1)), "smoke")
atlas.register_transform(transform, include_inverse=True)
save_coordinate_object(path, atlas, overwrite=True)
loaded = load_coordinate_object(path)
assert loaded.resolve("source", "target").transform == transform
payload = json.loads(path.read_text(encoding="utf-8"))
payload["payload_sha256"] = "0" * 64
path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
try:
    load_coordinate_object(path)
except Exception:
    print("round trip and tamper rejection passed")
else:
    raise AssertionError("tampered coordinate record was accepted")
'@ | Set-Content -Path $Smoke -Encoding utf8
        & $Py $Smoke
    } | Out-Null

    $Block14PowerShell = Join-Path $PackageRoot "examples\Test-KomorebiBlock14.ps1"
    Invoke-Step -Name "Block 14 PowerShell smoke workflow" -Action {
        & $Block14PowerShell -Python $Py
    } | Out-Null

    if ($null -ne $Git -and (Test-Path (Join-Path $SourceRoot ".git"))) {
        Invoke-Step -Name "Git object verification" -Action {
            & git -C $SourceRoot fsck --full --strict
            & git -C $SourceRoot diff --check
        } | Out-Null
        Invoke-Step -Name "Static source and public API audit" -Action {
            Invoke-PythonInDirectory -Directory $SourceRoot -PythonPath $Py -Arguments @("tools/static_python_audit.py", "--write") -PythonPathValue $SourcePythonPath
        } | Out-Null
    }
    else {
        Add-Result -Name "Git and static repository audit" -Status "skipped" -Required $false -Seconds 0 -Log "" -Detail "No Git bundle or Git executable was available"
    }

    if ($InstallOptionalAD) {
        if ((Test-PythonModule $Py "jax") -and (Test-PythonModule $Py "torch")) {
            Invoke-Step -Name "Optional JAX and PyTorch adapter tests" -Action {
                Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("-m", "pytest", "-q", "tests/test_block10_sparse_autodiff.py", "tests/test_nonlinear_sparse_autodiff.py") -PythonPathValue ""
            } | Out-Null
        }
        else {
            Add-Result -Name "Optional JAX and PyTorch adapter tests" -Status "skipped" -Required $false -Seconds 0 -Log "" -Detail "JAX or PyTorch was unavailable"
        }
    }

    if ($RunCuda) {
        Invoke-Step -Name "Block 16 crash-isolated CUDA qualification" -Action {
            Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("-m", "komorebi_math", "verify", "gpu") -PythonPathValue ""
        } | Out-Null
    }

    if ($RunBuildReproducibility -or $Mode -eq "Maximum") {
        Invoke-Step -Name "Block 16 wheel and source distribution build" -Action {
            $BuildOutput = Join-Path $ResultsDirectory "BLOCK16_DISTRIBUTION_BUILD.json"
            Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("verification/build_block16_distributions.py", "--output", $BuildOutput) -PythonPathValue $SourcePythonPath
        } | Out-Null
        Invoke-Step -Name "Block 16 distribution reconstruction audit" -Action {
            $AuditOutput = Join-Path $ResultsDirectory "BLOCK16_DISTRIBUTION_VALIDATION.json"
            Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("verification/block16_distribution_audit.py", "--output", $AuditOutput) -PythonPathValue $SourcePythonPath
        } | Out-Null
    }

    if (Test-PythonModule $Py "ruff") {
        Invoke-Step -Name "Ruff source check" -Optional -Action {
            Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("-m", "ruff", "check", "src", "tests", "verification") -PythonPathValue $SourcePythonPath
        } | Out-Null
    }
    else {
        Add-Result -Name "Ruff source check" -Status "skipped" -Required $false -Seconds 0 -Log "" -Detail "Ruff was unavailable"
    }

    if (Test-PythonModule $Py "mypy") {
        Invoke-Step -Name "Mypy coordinate and generating surfaces" -Optional -Action {
            Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("-m", "mypy", "src/komorebi_math/coordinates", "src/komorebi_math/generating", "src/komorebi_math/compatibility.py", "src/komorebi_math/export/persistence.py") -PythonPathValue $SourcePythonPath
        } | Out-Null
    }
    else {
        Add-Result -Name "Mypy coordinate and generating surfaces" -Status "skipped" -Required $false -Seconds 0 -Log "" -Detail "mypy was unavailable"
    }

    if (Test-PythonModule $Py "mkdocs") {
        Invoke-Step -Name "Strict documentation build" -Optional -Action {
            Invoke-PythonInDirectory -Directory $PackageRoot -PythonPath $Py -Arguments @("-m", "mkdocs", "build", "--strict") -PythonPathValue $SourcePythonPath
        } | Out-Null
    }
    else {
        Add-Result -Name "Strict documentation build" -Status "skipped" -Required $false -Seconds 0 -Log "" -Detail "MkDocs was unavailable"
    }
}
catch {
    $script:RequiredFailures += 1
    Add-Result -Name "Validation orchestration" -Status "failed" -Required $true -Seconds 0 -Log "" -Detail $_.Exception.Message
    Write-Error $_.Exception.Message -ErrorAction Continue
}
finally {
    $Completed = (Get-Date).ToUniversalTime().ToString("o")
    $Summary = [ordered]@{
        schema = "komorebi-math.local-validation-summary.v1"
        package_version = "0.16.0a2"
        completed_utc = $Completed
        mode = $Mode
        passed = ($script:RequiredFailures -eq 0)
        required_failures = $script:RequiredFailures
        optional_failures = $script:OptionalFailures
        results_directory = $ResultsDirectory
        environment_file = $EnvironmentPath
        results = $script:Results
    }
    $Summary | ConvertTo-Json -Depth 10 | Set-Content -Path $SummaryPath -Encoding utf8
    Write-Host "`nValidation summary: $SummaryPath" -ForegroundColor Green
    Write-Host "Required failures: $($script:RequiredFailures)"
    Write-Host "Optional failures: $($script:OptionalFailures)"
    if (-not $KeepWorkDirectory -and $script:RequiredFailures -eq 0) {
        Remove-Item -Recurse -Force $WorkDirectory -ErrorAction SilentlyContinue
    }
}

if ($script:RequiredFailures -ne 0) {
    exit 1
}
exit 0
