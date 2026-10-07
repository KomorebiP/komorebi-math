# Komorebi Math Block 21: Public Reproducibility

Block 21 closes the gap between an internally verified research package and an externally reproducible publication candidate.

The block adds:

1. a deterministic public capability tour spanning the stable and Block 19/20 surfaces
2. one publication gate that binds API, mathematical, release, documentation, and source-test evidence
3. self-digesting JSON and Markdown evidence records
4. Linux and PowerShell launchers
5. tag-triggered GitHub artifact construction without automatic package-registry publication
6. repository security, contribution, and reproducibility policies

Quick execution:

```powershell
$env:PYTHONPATH="$PWD\src"
python .\examples\block21_public_capability_tour.py --json
python .\verification\block21_publication_gate.py --skip-mkdocs
```

Complete execution:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

Block 21 does not alter the 131-symbol stable namespace and does not convert inherited Windows evidence into current Windows evidence. The release-candidate boundary therefore remains explicit.

Deterministic release handoff:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Build_Block21_Handoff.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

This command runs the complete publication gate, rebuilds the wheel and source distribution, constructs a path-safe deterministic repository ZIP, rebuilds it byte for byte, extracts it into a clean directory, and replays the quick publication gate from that extraction.
