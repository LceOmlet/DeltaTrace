"""Bind the already frozen sample to the existing native-head diagnostic."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ENTRY,ROOT,SSH,SCP
from stage_tail_probability_sample import LOCAL,REMOTE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sample = json.loads((LOCAL/'sample.json').read_bytes())
    overlay = json.loads((REPO/'experiments/rl/results_fla_seed_range_deployment_20261009.json').read_bytes())
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    helpers = HERE.parents[1]/'direct-target-action-author-curve-20261007/v1'
    endpoint = HERE.parents[1]/'direct-target-extreme-token-endpoint-20261007/v1'
    files = [HERE/'inspect_native_head_collection.py',HERE/'launch_native_head_collection.py',
        helpers/'inspect_action_curve.py',endpoint/'inspect_extreme_endpoint.py']
    assert sha(files[2]) == '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6'
    assert sha(files[3]) == '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'
    prepared = dict(scope=__doc__,diagnostic_source_commit=commit,
        sample_sha256=sha(LOCAL/'sample.json'),sample_source_commit=sample['source_commit'],
        numerical_tolerances_changed=False,original_native_model_unchanged=True,
        accepted_FP16_version=overlay['version'],formal_training_started=False,
        tasks={},jobs=[])
    for task,spec in sample['tasks'].items():
        measured = []
        for rank in (0,1):
            data = json.loads((HERE/f'native-head-{task}-final/rank{rank}.json').read_bytes())
            measured.extend(x['seconds'] for b in data['batches'] for x in b['native_phases'])
        measured.sort()
        median = measured[len(measured)//2]
        prepared['tasks'][task] = dict(sampled_sources=spec['sampled_sources'],
            sampled_trajectories=spec['sampled_trajectories'],sampled_states=spec['sampled_states'],
            original_native_head_median_seconds=median,
            estimated_two_card_model_seconds=sum(c['native_B8_calls_per_rank'] for c in spec['chunks'])*median,
            estimate_scope='Measured earlier native-head B8 calls; widths and targets differ. Not a throughput promise. Model initialization excluded.')
        for chunk in spec['chunks']:
            name = f"{task}-chunk{chunk['chunk']}"
            local = LOCAL/name
            local.mkdir(exist_ok=False)
            directory = REMOTE+'/'+name
            plan = dict(scope='Exact pre-frozen probability sample queries; no new sampling or outcome-dependent addition.',
                frozen_probability_sample_sha256=prepared['sample_sha256'],
                tasks={task:dict(entries=spec['entries'],batches=chunk['batches'])})
            (local/'layer-collection-inputs.json').write_text(json.dumps(plan,indent=2)+'\n')
            hashes = {p.name:sha(p) for p in files}
            hashes['layer-collection-inputs.json'] = sha(local/'layer-collection-inputs.json')
            runtime = overlay['version_mapping'][task]
            source = dict(files=hashes,diagnostic_source_commit=commit,
                accepted_fla_seed_range_overlay=dict(version=overlay['version'],
                    sha256=runtime['fixed_sha256'],original_sha256=runtime['base_sha256']),
                frozen_probability_sample_sha256=prepared['sample_sha256'])
            (local/'native-head-source-hashes.json').write_text(json.dumps(source,indent=2)+'\n')
            prepared['jobs'].append(dict(task=task,chunk=chunk['chunk'],directory=directory,
                queries=chunk['queries'],native_B8_calls_per_rank=chunk['native_B8_calls_per_rank'],
                max_width=max(b['padded_width'] for b in chunk['batches']),
                max_head_union_predictor_rows=max(b['head_union_predictor_rows'] for b in chunk['batches']),
                original_worker_wall_budget_seconds=1500))
    (LOCAL/'native-prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
    bundle = LOCAL/'native-source.tar'
    with tarfile.open(bundle,'w') as tar:
        tar.add(LOCAL/'native-prepared.json',arcname='native-prepared.json')
        tar.add(HERE/'run_tail_probability_chunks.py',arcname='run_tail_probability_chunks.py')
        for job in prepared['jobs']:
            name = f"{job['task']}-chunk{job['chunk']}"
            for p in files:
                tar.add(p,arcname=name+'/'+p.name)
            for n in ('layer-collection-inputs.json','native-head-source-hashes.json'):
                tar.add(LOCAL/name/n,arcname=name+'/'+n)
    subprocess.run(SCP+[str(bundle),SSH[-1]+':'+REMOTE+'/native-source.tar'],check=True,timeout=45)
    shell = ('set -eu\nsource '+ENTRY+'/metax-entry.env.sh\ncd '+REMOTE+'\n'
        'tar -xf native-source.tar\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'
        'import hashlib,json,os,psutil,subprocess,time\nfrom pathlib import Path\n'
        'out=Path('+repr(REMOTE)+')\n'
        'assert not (out/"native-controller-launch.json").exists(), "Do not duplicate collection"\n'
        'with (out/"native-controller.log").open("xb") as log:\n'
        ' p=subprocess.Popen([os.environ["VENV_PYTHON"],str(out/"run_tail_probability_chunks.py"),str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n'
        'receipt=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),prepared_sha256=hashlib.sha256((out/"native-prepared.json").read_bytes()).hexdigest())\n'
        '(out/"native-controller-launch.json").write_text(json.dumps(receipt,indent=2)+"\\n")\nprint(json.dumps(receipt))\nPY\n')
    (LOCAL/'native-launch-command.sh').write_text(shell,encoding='utf-8',newline='\n')
    run = subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=45)
    (LOCAL/'native-launch.stdout').write_bytes(run.stdout)
    (LOCAL/'native-launch.stderr').write_bytes(run.stderr)
    run.check_returncode()
    (LOCAL/'native-controller-launch.json').write_bytes(run.stdout)
    print(json.dumps(dict(prepared=prepared,launch=json.loads(run.stdout)),ensure_ascii=False))


if __name__ == '__main__':
    main()
