"""Run a named mathematical pytest profile against installed modules, not src/."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def installed_module_audit() -> dict:
    prefix=Path(sys.prefix).resolve()
    imported=[]; wrong=[]
    for name, module in tuple(sys.modules.items()):
        if name == 'komorebi_math' or name.startswith('komorebi_math.'):
            location=getattr(module, '__file__', None)
            if location:
                resolved=Path(location).resolve()
                imported.append({'name':name,'path':str(resolved)})
                if not resolved.is_relative_to(prefix):
                    wrong.append({'name':name,'path':str(resolved)})
    return {'passed':bool(imported) and not wrong, 'module_count':len(imported),
            'wrong_locations':wrong, 'prefix':str(prefix)}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report={'schema':'komorebi-math.installed-runtime-suite','passed':False}
    capture=io.StringIO()
    try:
        if sys.flags.optimize:
            raise RuntimeError('pytest qualification must not run with Python assertions disabled')
        import komorebi_math
        import pytest
        first=installed_module_audit()
        if not first['passed']:
            raise RuntimeError('package was not imported from this virtual environment')
        profile=json.loads(args.profile.read_text(encoding='utf-8'))
        work=args.workspace.resolve()
        files=profile['test_files']
        if len(files) != len(set(files)) or not files:
            raise ValueError('empty or duplicate installed-test profile')
        for name in files:
            if Path(name).name != name or not name.startswith('test_') or not name.endswith('.py'):
                raise ValueError(f'invalid profile test filename: {name!r}')
            if not (work/'tests'/name).is_file():
                raise ValueError(f'missing installed-profile test: {name}')
        if (work/'src').exists():
            raise RuntimeError('installed-test workspace must not contain a src directory')
        audits=[]
        class ImportGuard:
            def pytest_runtest_setup(self, item):
                audit=installed_module_audit()
                if not audit['passed']:
                    pytest.fail(f'source import leakage detected: {audit}')
            def pytest_sessionfinish(self, session, exitstatus):
                audit=installed_module_audit();audits.append(audit)
                if not audit['passed']:
                    session.exitstatus=1
        previous=Path.cwd();os.chdir(work)
        try:
            with redirect_stdout(capture),redirect_stderr(capture):
                code=pytest.main(['-q','--tb=short','--strict-markers','--import-mode=importlib',
                                  '-p','no:cacheprovider',f'--junitxml={work/"installed-tests.xml"}',
                                  *[str(work/'tests'/name) for name in files]],plugins=[ImportGuard()])
        finally:
            os.chdir(previous)
        xml=ET.parse(work/'installed-tests.xml')
        suites=list(xml.getroot().iter('testsuite'))
        counts={key:sum(int(node.attrib.get(key,0)) for node in suites)
                for key in ('tests','errors','failures','skipped')}
        report.update(profile=profile['name'],test_files=len(files),counts=counts,
                      returncode=int(code),module_audit=installed_module_audit(),
                      package_version=komorebi_math.__version__,
                      dependency_versions={name:importlib.metadata.version(name)
                                           for name in ('numpy','pytest','sympy','scipy','mpmath')})
        report['passed']=(int(code)==0 and counts['tests']>0 and counts['skipped']==0
                          and report['module_audit']['passed'])
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    report['pytest_output']=capture.getvalue()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
