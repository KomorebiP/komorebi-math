# Public Beta to Release Candidate

The beta gate and release-candidate gate are intentionally different.

## Beta gate

```powershell
python -m komorebi_math release readiness
```

The beta gate requires current execution on at least one required platform and may accept a frozen predecessor record for another platform when the upgrade contract authorizes it.

## Release-candidate gate

```powershell
python -m komorebi_math release readiness --final
```

The release-candidate gate requires current-version evidence for every required platform family:

```text
windows-x86_64
linux-x86_64
```

The current package is `1.0.0b4`. Its embedded registry still inherits the
Windows record from `0.16.0a2`. The Linux record matches the package version but
predates the October repairs. The gate defines "current" by version equality,
and each digest checks the referenced JSON record, not the present Git commit.
Its known failing condition is the missing current-version Windows record.
Closing that condition alone would not establish qualification of later code.

Revision-specific Linux and Windows CPU CI results are recorded separately in
[GitHub readiness](GITHUB_READINESS.md). They do not replace registry entries
automatically. Both platforms need evidence tied to the release revision and
artifacts before promotion. GPU qualification is a separate hardware
requirement in the [release process](RELEASE_PROCESS.md), and cannot be inferred
from CPU CI or `--final` alone. On the CUDA host, require both checks with:

```powershell
python -m komorebi_math release readiness --final --require-gpu
```

## Freeze discipline

After beta publication, changes permitted before `1.0.0rc1` are limited to defect corrections, documentation corrections, platform qualification, packaging corrections, and validation expansion. Any stable API or persistence change restarts beta qualification.
