"""Run one bounded operand diagnostic on all existing saved operator cases.

This is not a training implementation. Reuse the original finite FA callback,
public FA and original artifacts. No model, DT or optimizer is constructed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import ENTRY, ROOT, SSH, SCP


def ref(path, remote_path=None):
    raw = path.read_bytes()
    return dict(path=remote_path or str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    local = HERE/'FA-source-reference-v1'
    local.mkdir(exist_ok=True)
    out = ROOT+'/receipts/credit-FA-source-reference-20261009-v1'
    original = ROOT+'/receipts/credit-attention-pv-textcraft-20261009-v2'
    files = [HERE/'attention-pv-textcraft-v2'/f'rank{r}.json' for r in (0, 1)]
    points = []
    for f in files:
        report = json.loads(f.read_bytes())
        assert report['phase'] == 'complete'
        for batch in report['batches']:
            for p in batch['points']:
                entry = {k:p[k] for k in ('traj_uid','packed_slot','token_id','initial_state_sha256','native_single_d','cohorts','previously_examined')}
                # The original uniform and census query records have distinct
                # documented scalar fields; neither is inferred from text.
                entry['saved_d'] = p['saved_d'] if 'saved_d' in p else p['d']
                points.append(entry)
    assert len(points) == len({(p['traj_uid'],p['packed_slot']) for p in points}) == 165
    protocol = dict(
        scope='Original frozen TextCraft operator collection, all 128 uniformly sampled sources and 37 predicted-tail census sources. '
            'This is not the new unified recall frame; no recall or task-wide prevalence will be inferred from it.',
        observable_gap='Existing native operand contractions show joint-background PV and softmax residuals. '
            'Test whether the source-key V reference accounts for those residuals, before proposing any finite-rule change.',
        intervention='Only the source-key row of joint reference V0 becomes the saved native single-deletion V at that row. '
            'Q0/Q1, K0/K1, upstream, dtype, causal coordinates and original finite callback remain unchanged.',
        inference='Local operator diagnostic only. An oracle operand is not a deployable rule. '
            'No global d/advantage, faithfulness improvement or training repair is claimed from this substitution.',
        aggregation='Keep uniform and census cohorts, predicted/native ratio bins, examined identity and states separate. '
            'Report paired local-residual direction/counts and medians within crossed cells, then state-level summaries. '
            'Do not pool unbounded raw advantage moments or use this old collection to estimate unified recall.',
        numerical_policy='Record baseline replay drift rather than invent a tolerance. No pass/fail threshold for cause research. '
            'No correction multiplier; both variants use exactly the same replayed original public-FA LSE.',
        budget=dict(timeout_seconds=600,expected_original_public_FA=96,expected_original_finite_FA=72,
            expected_model=0,expected_DT=0,expected_optimizer=0,expected_rollout=0,expected_checkpoint_restore=0),
        original_rank_reports=[ref(f,original+'/results/'+f.name) for f in files],points=points)
    p = local/'protocol.json'
    raw = (json.dumps(protocol,indent=2)+'\n').encode()
    if p.exists():
        assert p.read_bytes() == raw, 'Do not change a frozen diagnostic protocol'
    else:
        p.write_bytes(raw)
    if args.prepare_only:
        print(json.dumps(dict(protocol=ref(p),points=len(points))))
        return
    assert not (local/'launch.json').exists(), 'Do not launch the same diagnostic twice'
    source = HERE/'inspect_saved_FA_source_reference.py'
    hashes = {f.name:ref(f)['sha256'] for f in (source,p)}
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    subprocess.run(SSH+['mkdir','-p',out],check=True,timeout=40)
    subprocess.run(SCP+[str(source),str(p),SSH[-1]+':'+out+'/'],check=True,timeout=40)
    code = '''import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(%r); hashes=%r
assert not (out/'launch.json').exists(),'Do not duplicate diagnostic'
for name,digest in hashes.items():
 path=out/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 if name.endswith('.py'):ast.parse(path.read_bytes())
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\\|\\s*4\\s+\\d+\\s+\\S',physical,re.M),'GPU4 occupied'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
env.pop('MACA_VISIBLE_DEVICES',None)
env['PYTHONPATH']=%r
argv=['timeout','600',env['VENV_PYTHON'],'-u',str(out/'inspect_saved_FA_source_reference.py'),
 '--directory',%r,'--owner',%r,'--library',%r,'--protocol',str(out/'protocol.json'),'--output',str(out/'result.json')]
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),source_commit=%r,
 files=hashes,argv=argv,devices=[4],physical_before=physical,
 host_available_bytes=psutil.virtual_memory().available,production_modified=False,training_candidate=False)
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))
'''%(out,hashes,
        ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/deltatrace/clean/qwen35',
        original,
        ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/deltatrace/clean/qwen35/vendor_fa_finite_bf16_d256.py',
        ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/libfinite_row_query_starts.so',commit)
    shell='set -eu\nsource '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    (local/'launch-command.sh').write_text(shell,encoding='utf-8',newline='\n')
    run=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=45)
    (local/'launch.stdout').write_bytes(run.stdout)
    (local/'launch.stderr').write_bytes(run.stderr)
    run.check_returncode()
    (local/'launch.json').write_bytes(run.stdout)
    print(json.dumps(dict(local=str(local),launch=json.loads(run.stdout))))


if __name__=='__main__':
    main()
