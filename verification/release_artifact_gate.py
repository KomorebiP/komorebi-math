"""Offline installed-artifact qualification with an optional clean dependency mode.

The default inherits installed dependencies and says so. The isolated mode needs
an explicit local wheelhouse, never downloads, and never falls back to inheritance.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tempfile
import time
from typing import Any

try:
    from .release_integrity import _tar_files, compare_files, package_files, release_binding, sha256, verify_release, _zip_files
except ImportError:
    from release_integrity import _tar_files, compare_files, package_files, release_binding, sha256, verify_release, _zip_files


def require_identical_wheels(direct: Path, rebuilt: Path) -> dict[str, Any]:
    left=sha256(direct.read_bytes()); right=sha256(rebuilt.read_bytes())
    if left != right:
        raise RuntimeError(f'sdist-rebuilt wheel is not byte-identical: {left} != {right}')
    return {'passed':True,'direct_sha256':left,'rebuilt_sha256':right}


def validate_options(dependency_mode: str, wheelhouse: Path | None) -> Path | None:
    if dependency_mode not in ('inherited','isolated'):
        raise ValueError('dependency mode must be inherited or isolated')
    if dependency_mode == 'inherited' and wheelhouse is not None:
        raise ValueError('--wheelhouse requires --dependency-mode isolated')
    if dependency_mode == 'isolated':
        if wheelhouse is None:
            raise ValueError('isolated qualification requires a local wheelhouse; no fallback is allowed')
        wheelhouse=wheelhouse.resolve()
        if not wheelhouse.is_dir() or not any(wheelhouse.glob('*.whl')):
            raise ValueError('wheelhouse is missing or contains no wheel files')
    return wheelhouse


def checked_run(command: list[str], cwd: Path, environment: dict[str,str], report: dict,
                timeout: int=180) -> str:
    start=time.monotonic()
    row={'command':command,'cwd':str(cwd),'timeout_seconds':timeout}
    try:
        result=subprocess.run(command,cwd=cwd,env=environment,capture_output=True,text=True,timeout=timeout)
        row.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,timed_out=False)
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(errors='replace') if isinstance(value,bytes) else value or ''
        row.update(returncode=None,stdout=text(exc.stdout),stderr=text(exc.stderr),timed_out=True)
        raise RuntimeError(f'command timed out after {timeout}s: {command}') from exc
    finally:
        row['seconds']=round(time.monotonic()-start,3)
        report['commands'].append(row)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {command}\n{result.stdout[-2500:]}\n{result.stderr[-2500:]}')
    return result.stdout


def qualify(root: Path, *, runtime_tests: bool=False, dependency_mode: str='inherited',
            wheelhouse: Path | None=None) -> dict[str,Any]:
    root=root.resolve()
    report: dict[str,Any]={'schema':'komorebi-math.installed-artifact-gate','schema_version':2,
        'passed':False,'dependency_mode_requested':dependency_mode,'dependency_isolated':False,
        'dependencies_inherited':dependency_mode=='inherited','network_used':False,
        'runtime_tests_requested':runtime_tests,'commands':[],
        'environment':{'platform':platform.system(),'machine':platform.machine(),'python':platform.python_version()}}
    try:
        wheelhouse=validate_options(dependency_mode,wheelhouse)
        if sys.flags.optimize:
            raise RuntimeError('qualification cannot run with Python assertions disabled')
        integrity=verify_release(root);report['integrity']=integrity
        if not integrity['passed']:
            raise RuntimeError('Refusing to qualify distributions that fail source parity/integrity.')
        report['binding_before']=release_binding(root)
        environment=os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE','PYTEST_ADDOPTS'):
            environment.pop(key,None)
        environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
            SOURCE_DATE_EPOCH='1787356800',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        identity=json.loads((root/'RELEASE_ID.json').read_text(encoding='utf-8'))
        report['release_id']=identity['release_id']
        run=lambda command,cwd,timeout=180: checked_run(command,cwd,environment,report,timeout)
        with tempfile.TemporaryDirectory(prefix='komorebi-r2-install-') as temporary:
            work=Path(temporary)
            def make_venv(name: str, *, build_environment: bool=False) -> Path:
                target=work/name
                args=[sys.executable,'-I','-B','-m','venv']
                if dependency_mode=='inherited':
                    args.append('--system-site-packages')
                run([*args,str(target)],work)
                python=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
                cfg=(target/'pyvenv.cfg').read_text(encoding='utf-8').lower()
                if dependency_mode=='isolated':
                    if 'include-system-site-packages = false' not in cfg:
                        raise RuntimeError('clean venv unexpectedly inherits system packages')
                    packages=([f"setuptools=={importlib.metadata.version('setuptools')}",'wheel']
                              if build_environment else ['numpy'])
                    if runtime_tests and not build_environment:
                        packages += ['pytest==8.4.2','sympy','scipy','mpmath','PyYAML']
                    run([str(python),'-I','-B','-m','pip','--isolated','install','--no-index',
                         '--find-links',str(wheelhouse),'--disable-pip-version-check',*packages],work)
                    run([str(python),'-I','-B','-m','pip','--isolated','check'],work)
                else:
                    # Nested venvs inherit the base interpreter's sites, not necessarily
                    # the parent venv's sites. Add only the parent's site directories,
                    # never arbitrary PYTHONPATH or an editable checkout. The runtime
                    # import guard still requires every package module inside target.
                    parent_sites=[str(Path(path).resolve()) for path in site.getsitepackages()
                                  if Path(path).is_dir()]
                    purelib=Path(run([str(python),'-I','-B','-c',
                                      'import sysconfig; print(sysconfig.get_path("purelib"))'],work).strip())
                    if not purelib.resolve().is_relative_to(target.resolve()):
                        raise RuntimeError('venv purelib path escaped its own environment')
                    (purelib/'komorebi_validation_parent_deps.pth').write_text(
                        '\n'.join(parent_sites)+'\n',encoding='utf-8')
                    report.setdefault('inherited_parent_site_paths',{})[name]=parent_sites
                return python
            dist=root/'release/dist'
            direct=next(dist.glob('*.whl'));sdist=next(dist.glob('*.tar.gz'))
            extracted=work/'sdist';extracted.mkdir()
            files,modes=_tar_files(sdist)
            for name,content in files.items():
                target=extracted/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(content);target.chmod(modes[name])
            rebuilt_dir=work/'rebuilt';rebuilt_dir.mkdir()
            builder=make_venv('build_environment',build_environment=True) if dependency_mode=='isolated' else Path(sys.executable)
            code=('import sys; sys.path.insert(0,sys.argv[1]); '
                  'from reproducible_wheel import build_canonical_wheel; '
                  'build_canonical_wheel(sys.argv[2],sys.argv[3])')
            run([str(builder),'-I','-B','-c',code,str(root/'verification'),str(rebuilt_dir),identity['wheel_build_tag']],extracted)
            candidates=list(rebuilt_dir.glob('*.whl'))
            if len(candidates)!=1:
                raise RuntimeError('sdist rebuild must produce exactly one wheel')
            rebuilt=candidates[0];rebuilt_files=_zip_files(rebuilt)
            report['sdist_rebuilt_package_parity']=compare_files(package_files(root),{
                k:v for k,v in rebuilt_files.items() if k.startswith('komorebi_math/')})
            report['sdist_rebuilt_wheel_sha256']=sha256(rebuilt.read_bytes())
            report['direct_wheel_sha256']=sha256(direct.read_bytes())
            report['sdist_rebuilt_wheel_byte_identical']=direct.read_bytes()==rebuilt.read_bytes()
            report['byte_reproducibility']=require_identical_wheels(direct,rebuilt)
            if not report['sdist_rebuilt_package_parity']['passed']:
                raise RuntimeError('sdist-rebuilt wheel differs from source')
            for label,wheel in [('installed_wheel',direct),('installed_sdist_rebuilt_wheel',rebuilt)]:
                python=make_venv(label)
                run([str(python),'-I','-B','-m','pip','--isolated','install','--no-index','--no-deps',
                     '--force-reinstall','--disable-pip-version-check',str(wheel)],work)
                if dependency_mode=='isolated':
                    run([str(python),'-I','-B','-m','pip','--isolated','check'],work)
                probe=json.loads(run([str(python),'-I','-B',str(root/'verification/installed_artifact_regressions.py'),
                                      '--require-installed'],work))
                if probe.get('passed') is not True:
                    raise RuntimeError(f'{label} regression probes failed')
                report[label]=probe
                cli=python.parent/('komorebi.exe' if os.name=='nt' else 'komorebi')
                if not cli.is_file():
                    raise RuntimeError(f'installed console entry point missing: {cli}')
                report[label]['cli_help_passed']='usage' in run([str(cli),'--help'],work).lower()
                if not report[label]['cli_help_passed']:
                    raise RuntimeError('installed CLI did not print usage')
                if runtime_tests:
                    suite=work/(label+'_suite');suite.mkdir()
                    shutil.copytree(root/'tests',suite/'tests',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
                    result_file=suite/'runtime-results.json'
                    run([str(python),'-I','-B',str(root/'verification/installed_runtime_suite.py'),
                         '--profile',str(root/'verification/installed_runtime_profile.json'),
                         '--workspace',str(suite),'--output',str(result_file)],suite,timeout=600)
                    result=json.loads(result_file.read_text(encoding='utf-8'))
                    if result.get('passed') is not True:
                        raise RuntimeError(f'{label} installed runtime suite failed')
                    report[label]['runtime_suite']=result
            report['dependency_isolated']=dependency_mode=='isolated'
            report['binding_after']=release_binding(root)
            if report['binding_before'] != report['binding_after']:
                raise RuntimeError('release source or artifacts changed during qualification')
            report['passed']=True
    except (OSError,RuntimeError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired,StopIteration) as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--runtime-tests',action='store_true')
    parser.add_argument('--dependency-mode',choices=['inherited','isolated'],default='inherited')
    parser.add_argument('--wheelhouse',type=Path)
    args=parser.parse_args()
    report=qualify(args.root,runtime_tests=args.runtime_tests,dependency_mode=args.dependency_mode,wheelhouse=args.wheelhouse)
    text=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    print(text,end='')
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
