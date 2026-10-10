from pathlib import Path
import subprocess
import pytest
from verification.release_artifact_gate import require_identical_wheels, validate_options, checked_run, qualify, require_installed_math_campaign


def test_rebuilt_wheel_difference_is_failure(tmp_path: Path) -> None:
    direct=tmp_path/'direct.whl';rebuilt=tmp_path/'rebuilt.whl'
    direct.write_bytes(b'one');rebuilt.write_bytes(b'two')
    with pytest.raises(RuntimeError,match='not byte-identical'):
        require_identical_wheels(direct,rebuilt)
    rebuilt.write_bytes(b'one')
    assert require_identical_wheels(direct,rebuilt)['passed']


@pytest.mark.parametrize('mode,local',[('isolated',None),('unknown',None),('inherited',Path('.'))])
def test_invalid_dependency_mode_rejected(mode,local) -> None:
    with pytest.raises(ValueError):
        validate_options(mode,local)


def test_empty_wheelhouse_does_not_fall_back(tmp_path: Path) -> None:
    report=qualify(tmp_path,dependency_mode='isolated',wheelhouse=tmp_path)
    assert not report['passed']
    assert not report['dependency_isolated']
    assert report['commands']==[]
    assert 'wheelhouse' in report['error']


def test_timeout_is_recorded(tmp_path: Path,monkeypatch) -> None:
    def timeout(*args,**kwargs):
        raise subprocess.TimeoutExpired(args[0],1,output=b'partial output',stderr=b'partial error')
    monkeypatch.setattr(subprocess,'run',timeout)
    report={'commands':[]}
    with pytest.raises(RuntimeError,match='timed out'):
        checked_run(['fake'],tmp_path,{},report,timeout=1)
    row=report['commands'][0]
    assert row['timed_out'] and row['returncode'] is None
    assert row['stdout']=='partial output'


def mathematical_campaign():
    return {
        'kind': 'finite_mathematical_stress_report', 'status': 'passed', 'stress_status': 'passed',
        'seed': 20261010, 'source_inputs_unchanged': True,
        'counts': {'passed': 21, 'failed': 0, 'error': 0, 'skipped': 0, 'timeout': 0, 'not_run': 0},
        'checks': [{'identifier': f'finite-check-{index}', 'status': 'passed'} for index in range(21)],
        'resource_limits': {'per_check_wall_seconds': 30, 'total_wall_seconds': 180},
        'paper_coverage': {'valid': True, 'api_links_checked': True,
                           'all_catalogued_mechanisms_implemented': False},
    }


def test_partial_installed_catalog_with_every_executed_check_can_qualify():
    require_installed_math_campaign(mathematical_campaign())


@pytest.mark.parametrize('outcome', ['failed', 'error', 'skipped', 'timeout', 'not_run'])
def test_nonpassing_installed_check_cannot_hide_under_passing_summary(outcome):
    result = mathematical_campaign()
    result['checks'][0]['status'] = outcome
    with pytest.raises(RuntimeError, match='nonpassing check evidence'):
        require_installed_math_campaign(result)
    result = mathematical_campaign()
    result['counts']['passed'] = 20
    result['counts'][outcome] = 1
    with pytest.raises(RuntimeError, match='all 21 checks'):
        require_installed_math_campaign(result)


@pytest.mark.parametrize('mutation', [
    lambda report: report['checks'].pop(),
    lambda report: report['checks'][0].update(identifier=report['checks'][1]['identifier']),
    lambda report: report.update(seed=0),
    lambda report: report.update(source_inputs_unchanged=False),
    lambda report: report.update(paper_coverage=None),
    lambda report: report['paper_coverage'].update(valid=False),
    lambda report: report['paper_coverage'].update(api_links_checked=False),
    lambda report: report['resource_limits'].update(total_wall_seconds=None),
])
def test_omitted_or_unverified_installed_campaign_is_failure(mutation):
    result = mathematical_campaign()
    mutation(result)
    with pytest.raises(RuntimeError):
        require_installed_math_campaign(result)


def test_publication_gate_conflicting_doc_options() -> None:
    from verification.block21_publication_gate import run_gate
    with pytest.raises(ValueError,match='mutually exclusive'):
        run_gate(require_mkdocs=True,skip_mkdocs=True)


def test_handoff_rejects_tampered_self_digest(tmp_path: Path) -> None:
    import json
    from verification.build_block21_handoff import _load_passing_gate, _canonical_sha256
    payload={'schema':'komorebi-math.block21-publication-gate','passed':True,'mode':'full',
             'checks':[{'name':'fixture','passed':True}], 'full_test_record':{'returncode':0}}
    payload['report_sha256']=_canonical_sha256(payload)
    file=tmp_path/'gate.json';file.write_text(json.dumps(payload))
    assert _load_passing_gate(file)['passed']
    payload['checks'][0]['name']='modified'
    file.write_text(json.dumps(payload))
    with pytest.raises(SystemExit,match='self-digest'):
        _load_passing_gate(file)


def test_handoff_rejects_failing_checks_even_with_valid_digest(tmp_path: Path) -> None:
    import json
    from verification.build_block21_handoff import _load_passing_gate, _canonical_sha256
    payload={'schema':'komorebi-math.block21-publication-gate','passed':True,'mode':'full',
             'checks':[{'name':'fixture','passed':False}], 'full_test_record':{'returncode':0}}
    payload['report_sha256']=_canonical_sha256(payload)
    file=tmp_path/'gate.json';file.write_text(json.dumps(payload))
    with pytest.raises(SystemExit,match='failing result'):
        _load_passing_gate(file)


def test_handoff_rejects_stale_source_binding(tmp_path: Path,monkeypatch) -> None:
    import json
    from verification import release_integrity
    from verification.build_block21_handoff import _load_passing_gate, _canonical_sha256
    monkeypatch.setattr(release_integrity,'release_binding',lambda root:{'source_sha256':'current'})
    payload={'schema':'komorebi-math.block21-publication-gate','passed':True,'mode':'full',
             'checks':[{'name':'fixture','passed':True}], 'full_test_record':{'returncode':0},
             'environment':{'release_binding':{'source_sha256':'old'}}}
    payload['report_sha256']=_canonical_sha256(payload)
    file=tmp_path/'gate.json';file.write_text(json.dumps(payload))
    with pytest.raises(SystemExit,match='stale'):
        _load_passing_gate(file,tmp_path)


@pytest.mark.parametrize('name',['.venv-release','.venv-dist','.venv_custom'])
def test_handoff_excludes_generated_virtual_environments(tmp_path: Path,name: str) -> None:
    from verification.build_block21_handoff import _collect
    (tmp_path/'README.md').write_text('# fixture\n')
    environment=tmp_path/name
    (environment/'bin').mkdir(parents=True)
    (environment/'bin/python').write_text('not release source\n')
    (environment/'large-cache').write_text('generated dependency files\n')
    assert [entry.path for entry in _collect(tmp_path)] == ['README.md']
