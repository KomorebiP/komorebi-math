# Public Beta to Release Candidate

The beta gate and release-candidate gate are intentionally different.

## Beta gate

```powershell
komorebi release readiness
```

The beta gate requires current execution on at least one required platform and may accept a frozen predecessor record for another platform when the upgrade contract authorizes it.

## Release-candidate gate

```powershell
komorebi release readiness --final
```

The release-candidate gate requires current-version evidence for every required platform family:

```text
windows-x86_64
linux-x86_64
```

The current package is `1.0.0b4`. Its embedded registry still inherits the
Windows record from `0.16.0a2`. Current Linux and Windows CPU CI results are
recorded separately in [GitHub readiness](GITHUB_READINESS.md); they do not
replace that registry entry automatically. Final readiness requires a
current-version Windows execution record committed with its digest. GPU
qualification is a separate hardware requirement in the
[release process](RELEASE_PROCESS.md), and cannot be inferred from CPU CI.

## Freeze discipline

After beta publication, changes permitted before `1.0.0rc1` are limited to defect corrections, documentation corrections, platform qualification, packaging corrections, and validation expansion. Any stable API or persistence change restarts beta qualification.
