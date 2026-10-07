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

For `1.0.0b1`, the expected open item is current Windows evidence. Once the exact beta passes the Windows source, wheel, upgrade, and GPU scripts, that result can replace inherited Windows evidence.

## Freeze discipline

After beta publication, changes permitted before `1.0.0rc1` are limited to defect corrections, documentation corrections, platform qualification, packaging corrections, and validation expansion. Any stable API or persistence change restarts beta qualification.
