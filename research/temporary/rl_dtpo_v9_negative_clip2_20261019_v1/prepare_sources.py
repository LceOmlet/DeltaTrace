"""Isolate the approved negative2 setting from the deployed clip1 sources."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
PARENT=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'
AUDIT=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v3'
MODE='reward_target_exponential_clipped_v9'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    AUDIT.mkdir(exist_ok=True);(AUDIT/'source').mkdir(exist_ok=True)
    assert not list(AUDIT.glob('*-source-prepared.json')), 'Inspect any prior preparation before resubmission'
    incoming=Path(__file__).parent
    for task in ('textcraft','appworld'):
        for method in ('dt','ppo','grpo'):
            parent=PARENT/f'{task}-{method}-source-prepared.json'
            row=json.loads(parent.read_bytes())
            assert all(sha(path)==expected for path,expected in row['source_bindings'].items())
            before_entry,before_verl=Path(row['entry']),Path(row['verl_root'])
            entry,verl=before_entry,before_verl
            changes={};env=dict(row['effective_environment'])
            bindings=dict(row['source_bindings'])
            if method=='dt':
                candidate=ROOT/'candidates'/f'{task}-dt-v9-negative-clip2-20261019-v3'
                candidate.mkdir(exist_ok=False)
                entry,verl=candidate/'entry',candidate/'verl-owner'
                ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc')
                shutil.copytree(before_entry,entry,ignore=ignore)
                shutil.copytree(before_verl,verl,ignore=ignore)
                for name in ('counterfactual.py','dt_training_batch.py','reward_readout.py'):
                    source=incoming/name;ast.parse(source.read_text())
                    shutil.copyfile(source,entry/name)
                    changes[str(entry/name)]=dict(before=sha(before_entry/name),after=sha(entry/name))
                for path in bindings:
                    mapped=path.replace(str(before_entry),str(entry)).replace(str(before_verl),str(verl))
                    if mapped not in changes:assert sha(mapped)==sha(path)
                bindings={p.replace(str(before_entry),str(entry)).replace(str(before_verl),str(verl)):
                    sha(p.replace(str(before_entry),str(entry)).replace(str(before_verl),str(verl))) for p in bindings}
                bindings.update({p:sha(p) for p in changes})
                env={k:(v.replace(str(before_entry),str(entry)).replace(str(before_verl),str(verl)) if isinstance(v,str) else v)
                     for k,v in env.items()}
                env.update(DT_ENTRY_ROOT=str(entry),VERL_ROOT=str(verl),DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP='2')
            result=dict(row,status='isolated_negative2_sources_prepared_not_verified_not_deployed',
                parent=str(parent),parent_SHA256=sha(parent),entry=str(entry),verl_root=str(verl),
                effective_environment=env,source_bindings=bindings,changes=changes,
                event_u_clip=[-2,1] if method=='dt' else None,
                method_version='DTPO-v5-exponential-u-clipped-minus2-plus1-v9' if method=='dt' else method,
                prior_clip1_source_receipt=str(parent),extra_model_samples=0,extra_environment_samples=0)
            (AUDIT/f'{task}-{method}-source-prepared.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(task=task,method=method,entry=str(entry),changed_files=len(changes))))


if __name__=='__main__':main()
