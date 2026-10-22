"""Canonicalize a ``.tar.gz`` source distribution in place.

Setuptools does not promise byte-identical gzip and tar metadata across two
otherwise identical builds.  This standard-library-only postprocessor keeps
the member bytes and names while fixing their order, ownership, modes,
timestamps, PAX metadata, and gzip header.  It rejects links, special entries,
unsafe paths, duplicates, and case-fold collisions before replacing the input.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import io
import os
import shutil
import stat
import tarfile
from pathlib import Path, PurePosixPath
from unicodedata import normalize


def _safe_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise SystemExit(f"unsafe sdist member name: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
        or normalize("NFC", name) != name
    ):
        raise SystemExit(f"unsafe sdist member name: {name!r}")
    return path.as_posix()


def canonicalize(path: Path, *, epoch: int) -> None:
    if path.is_symlink():
        raise SystemExit(f"sdist must not be a symlink: {path}")
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"could not resolve sdist: {path}: {exc}") from exc
    if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise SystemExit(f"sdist is not a regular file: {path}")
    if epoch < 0:
        raise SystemExit("epoch must be nonnegative")

    entries: list[tuple[tarfile.TarInfo, bytes | None]] = []
    names: set[str] = set()
    folded_names: set[str] = set()
    with tarfile.open(path, mode="r:gz") as source:
        for original in source.getmembers():
            name = _safe_name(original.name.rstrip("/"))
            folded = name.casefold()
            if name in names or folded in folded_names:
                raise SystemExit(f"duplicate or case-colliding sdist member: {name}")
            names.add(name)
            folded_names.add(folded)
            if not (original.isfile() or original.isdir()):
                raise SystemExit(f"unsupported sdist member type: {name}")
            payload: bytes | None = None
            if original.isfile():
                stream = source.extractfile(original)
                if stream is None:
                    raise SystemExit(f"could not read sdist member: {name}")
                payload = stream.read()
                if len(payload) != original.size:
                    raise SystemExit(f"truncated sdist member: {name}")
            info = copy.copy(original)
            info.name = name
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = epoch
            info.pax_headers = {}
            info.linkname = ""
            info.devmajor = 0
            info.devminor = 0
            if info.isdir():
                info.mode = 0o755
                info.size = 0
            else:
                info.mode = 0o755 if name.endswith(".sh") else 0o644
                info.size = len(payload or b"")
            entries.append((info, payload))

    temporary_tar = path.with_name(f".{path.name}.canonical.tar")
    temporary_gzip = path.with_name(f".{path.name}.canonical.gz")
    for temporary in (temporary_tar, temporary_gzip):
        if temporary.exists():
            temporary.unlink()
    try:
        with tarfile.open(temporary_tar, mode="w", format=tarfile.PAX_FORMAT) as target:
            for info, payload in sorted(entries, key=lambda item: item[0].name):
                target.addfile(info, None if payload is None else io.BytesIO(payload))
        with temporary_tar.open("rb") as source, temporary_gzip.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw,
                mtime=epoch,
            ) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
        os.replace(temporary_gzip, path)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        for temporary in (temporary_tar, temporary_gzip):
            if temporary.exists():
                temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--epoch", type=int, required=True)
    arguments = parser.parse_args()
    canonicalize(arguments.sdist, epoch=arguments.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
