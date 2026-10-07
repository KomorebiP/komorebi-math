"""Safe LaTeX compilation with explicit engine discovery."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, ValidationError


class LatexEngine(StrEnum):
    """Supported LaTeX compiler choices."""

    AUTO = "auto"
    PDFLATEX = "pdflatex"
    XELATEX = "xelatex"
    LUALATEX = "lualatex"
    LATEXMK = "latexmk"


@dataclass(frozen=True, slots=True)
class LatexCompileOptions:
    """Options for one isolated LaTeX compilation."""

    engine: LatexEngine | str = LatexEngine.AUTO
    passes: int = 2
    timeout: float = 120.0
    shell_escape: bool = False
    keep_auxiliary: bool = False
    jobname: str | None = None

    def __post_init__(self) -> None:
        engine = LatexEngine(self.engine)
        object.__setattr__(self, "engine", engine)
        if self.passes <= 0:
            raise ValidationError("LaTeX passes must be positive")
        if self.timeout <= 0:
            raise ValidationError("LaTeX timeout must be positive")
        if self.jobname is not None:
            if not self.jobname.strip() or any(character in self.jobname for character in "/\\"):
                raise ValidationError("LaTeX jobname must be a simple file name")


@dataclass(frozen=True, slots=True)
class LatexCompilation:
    """Paths, logs, and certificate returned by a LaTeX compilation."""

    source_path: Path
    pdf_path: Path | None
    log_path: Path
    engine: LatexEngine
    commands: tuple[tuple[str, ...], ...]
    return_codes: tuple[int, ...]
    certificate: Certificate

    @property
    def valid(self) -> bool:
        return self.certificate.valid

    def require(self) -> Path:
        """Return the PDF path when compilation succeeded."""

        self.certificate.require()
        if self.pdf_path is None:
            raise ValidationError("LaTeX compilation did not produce a PDF path")
        return self.pdf_path


def available_latex_engines() -> dict[str, bool]:
    """Report which supported LaTeX commands are available on PATH."""

    return {engine.value: shutil.which(engine.value) is not None for engine in LatexEngine if engine is not LatexEngine.AUTO}


def resolve_latex_engine(engine: LatexEngine | str = LatexEngine.AUTO) -> LatexEngine:
    """Resolve an explicit or automatic LaTeX engine choice."""

    requested = LatexEngine(engine)
    if requested is not LatexEngine.AUTO:
        if shutil.which(requested.value) is None:
            raise BackendUnavailableError(f"LaTeX engine is not installed: {requested.value}")
        return requested
    for candidate in (LatexEngine.LUALATEX, LatexEngine.XELATEX, LatexEngine.PDFLATEX, LatexEngine.LATEXMK):
        if shutil.which(candidate.value) is not None:
            return candidate
    raise BackendUnavailableError("no supported LaTeX engine was found on PATH")


def _direct_command(
    engine: LatexEngine,
    source: Path,
    output_dir: Path,
    options: LatexCompileOptions,
) -> tuple[str, ...]:
    command = [
        engine.value,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={output_dir}",
    ]
    command.append("-shell-escape" if options.shell_escape else "-no-shell-escape")
    if options.jobname:
        command.append(f"-jobname={options.jobname}")
    command.append(str(source))
    return tuple(command)


def _latexmk_command(source: Path, output_dir: Path, options: LatexCompileOptions) -> tuple[str, ...]:
    command = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={output_dir}",
    ]
    if options.jobname:
        command.append(f"-jobname={options.jobname}")
    command.append(str(source))
    return tuple(command)


def _auxiliary_paths(output_dir: Path, stem: str) -> Iterable[Path]:
    for suffix in (
        ".aux",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".toc",
        ".synctex.gz",
        ".bbl",
        ".blg",
    ):
        yield output_dir / f"{stem}{suffix}"


def compile_latex(
    source: str | Path,
    output_dir: str | Path,
    *,
    options: LatexCompileOptions | None = None,
    source_name: str = "document.tex",
) -> LatexCompilation:
    """Compile LaTeX source to PDF without enabling shell escape by default."""

    options = options or LatexCompileOptions()
    engine = resolve_latex_engine(options.engine)
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    source_is_path = isinstance(source, Path)
    if isinstance(source, str) and "\n" not in source and "\r" not in source and len(source) < 4096:
        candidate = Path(source).expanduser()
        source_is_path = candidate.suffix.lower() == ".tex" and candidate.exists()
    if source_is_path:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise ValidationError(f"LaTeX source file does not exist: {source_path}")
    else:
        source_path = target_dir / source_name
        source_path.write_text(str(source), encoding="utf-8")

    stem = options.jobname or source_path.stem
    log_path = target_dir / f"{stem}.komorebi.log"
    commands: list[tuple[str, ...]] = []
    return_codes: list[int] = []
    chunks: list[str] = []
    environment = dict(os.environ)
    environment.setdefault("max_print_line", "1000")
    environment.setdefault("SOURCE_DATE_EPOCH", "0")
    environment.setdefault("FORCE_SOURCE_DATE", "1")

    try:
        if engine is LatexEngine.LATEXMK:
            if options.shell_escape:
                raise ValidationError("shell escape is not supported through the latexmk wrapper")
            command = _latexmk_command(source_path, target_dir, options)
            commands.append(command)
            completed = subprocess.run(
                command,
                cwd=source_path.parent,
                env=environment,
                capture_output=True,
                text=True,
                timeout=options.timeout,
                check=False,
            )
            return_codes.append(completed.returncode)
            chunks.extend([completed.stdout, completed.stderr])
        else:
            command = _direct_command(engine, source_path, target_dir, options)
            for _ in range(options.passes):
                commands.append(command)
                completed = subprocess.run(
                    command,
                    cwd=source_path.parent,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=options.timeout,
                    check=False,
                )
                return_codes.append(completed.returncode)
                chunks.extend([completed.stdout, completed.stderr])
                if completed.returncode != 0:
                    break
    except subprocess.TimeoutExpired as exc:
        chunks.append(f"LaTeX compilation timed out: {exc}")
        return_codes.append(124)

    log_path.write_text("\n".join(chunks), encoding="utf-8", errors="replace")
    pdf_path = target_dir / f"{stem}.pdf"
    valid = bool(return_codes) and all(code == 0 for code in return_codes) and pdf_path.is_file() and pdf_path.stat().st_size > 0
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        assumptions_checked=(
            "selected LaTeX engine is installed",
            "shell escape policy is explicit",
            "compiler return codes were inspected",
            "PDF existence and size were inspected",
        ),
        diagnostics={
            "engine": engine.value,
            "source": str(source_path),
            "output": str(pdf_path),
            "return_codes": tuple(return_codes),
            "commands": commands,
            "log": str(log_path),
        },
        message="LaTeX compilation succeeded" if valid else f"LaTeX compilation failed, inspect {log_path}",
    )

    if valid and not options.keep_auxiliary:
        for path in _auxiliary_paths(target_dir, stem):
            if path == log_path:
                continue
            path.unlink(missing_ok=True)


    return LatexCompilation(
        source_path=source_path,
        pdf_path=pdf_path if pdf_path.is_file() else None,
        log_path=log_path,
        engine=engine,
        commands=tuple(commands),
        return_codes=tuple(return_codes),
        certificate=certificate,
    )


__all__ = [
    "LatexCompilation",
    "LatexCompileOptions",
    "LatexEngine",
    "available_latex_engines",
    "compile_latex",
    "resolve_latex_engine",
]
