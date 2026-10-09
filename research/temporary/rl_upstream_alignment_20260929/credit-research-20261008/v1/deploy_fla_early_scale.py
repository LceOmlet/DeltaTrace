"""Promote verified scale-order source through the two existing leaf symlinks.

Historical owners and stopped startup manifests remain immutable. Only the
new accepted overlay is added to current runtime metadata; no job is started.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ENTRY,ROOT,SSH,SCP


def main():
    local=HERE/'fla-early-scale-review-v1'
    receipt=REPO/'experiments/rl/results_fla_early_output_scale_20261009.json'
    verified=json.loads(receipt.read_bytes())
    assert verified['status']=='verified_for_user_authorized_deployment'
    assert verified['official_checks']['checks_passed']==22 and verified['noncoincident_equivalence']['checks_passed']==5
    version=ROOT+'/releases/'+verified['version']
    targets=dict(textcraft=ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/deltatrace',
        appworld=ROOT+'/candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for task,owner in verified['owners'].items():
        path=Path(owner['candidate_path']);raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==owner['candidate_sha256']
        assert subprocess.check_output(['git','show',commit+':'+path.relative_to(REPO).as_posix()])==raw
        subprocess.run(SSH+['mkdir','-p',version+'/'+task],check=True,timeout=40)
        subprocess.run(SCP+[str(path),SSH[-1]+':'+version+'/'+task+'/qwen35_gdn_finite.py'],check=True,timeout=40)
    subprocess.run(SCP+[str(receipt),SSH[-1]+':'+version+'/verification.json'],check=True,timeout=40)
    unchanged=json.loads((HERE/'accepted-fla-seed-range-v1/post-deploy-snapshot.json').read_bytes())['unchanged_numerical_files']
    code='''import ast,hashlib,json,os,psutil,re,subprocess,sys,time
from pathlib import Path
version=Path(%r);targets=%r;unchanged=%r
assert not (version/'deployed.json').exists(),'Do not duplicate promotion'
v=json.loads((version/'verification.json').read_bytes());rows=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\\|\\s*[45]\\s+\\d+\\s+\\S',physical,re.M),'Wait for the diagnostic owners to exit'
for original in unchanged:
 assert sha(original['path'])==original['sha256'],'Unrelated numerical owner changed'
for task,root in targets.items():
 old=Path(root)/'clean/qwen35/qwen35_gdn_finite.py';new=version/task/'qwen35_gdn_finite.py';owner=v['owners'][task]
 assert old.is_symlink() and sha(old)==owner['base_sha256'] and sha(new)==owner['candidate_sha256']
 base=old.resolve();before=base.read_bytes();after=new.read_bytes()
 fn=lambda raw:next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='gdn_finite_pullback')
 assert ast.dump(fn(before).args)==ast.dump(fn(after).args)
 assert ('consume_captures' in {n.arg for n in fn(after).args.kwonlyargs})==(task=='appworld')
 rows.append(dict(task=task,root=root,path=str(old),parent_resolved=str(base),parent_sha256=sha(base),
  versioned_path=str(new),new_sha256=sha(new),public_signature_unchanged=True))
def probe(row,preview):
 source_path=Path(%r)/'runs/direct-target-prefix-runtime-20261007-v1'/row['task']/(row['task']+'-dt')/'source.json'
 source=json.loads(source_path.read_bytes());env=dict(os.environ,**source['environment']);env['CUDA_VISIBLE_DEVICES']='-1'
 qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35'];root=row['root']
 official=env.get('DT_OFFICIAL_ROOT') or qwen['official_root']
 env['PYTHONPATH']=':'.join([root,official,root+'/clean/qwen35',source['pythonpath'],qwen['ft_extension_root']])
 prelude='import inspect,importlib.util,hashlib,json,torch;from pathlib import Path;'
 if preview:code=prelude+"spec=importlib.util.spec_from_file_location('preview_scale_owner',"+repr(row['versioned_path'])+");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
 else:code=prelude+'import qwen35_gdn_finite as m;'
 code+="p=Path(m.__file__);print(json.dumps(dict(path=str(p),resolved=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),signature=str(inspect.signature(m.gdn_finite_pullback)),cuda_initialized=torch.cuda.is_initialized())))"
 run=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
 label='preview' if preview else 'active'
 (version/row['task']/(label+'-stdout.txt')).write_text(run.stdout);(version/row['task']/(label+'-stderr.txt')).write_text(run.stderr)
 assert run.returncode==0,run.stderr
 p=json.loads(run.stdout.splitlines()[-1]);assert p['sha256']==row['new_sha256'] and not p['cuda_initialized']
 p['startup_source_sha256']=sha(source_path);return p
for row in rows:row['preview_import']=probe(row,True)
for row in rows:
 path=Path(row['path']);tmp=path.with_name(path.name+'.scale-order-link');assert not tmp.exists() and not tmp.is_symlink()
 tmp.symlink_to(row['versioned_path']);os.replace(tmp,path)
 row['fresh_import']=probe(row,False)
 assert sha(row['parent_resolved'])==row['parent_sha256']
r=dict(unix=time.time(),version=v['version'],parent_version=v['parent_version'],source_commit=%r,
 status='deployed_to_both_DT_owner_paths',owners=rows,verification_sha256=sha(version/'verification.json'),
 unchanged_numerical_files=unchanged,physical_before=physical,official_tolerances_changed=False,
 formal_restart=False,QVA_whitening_PPO_changed=False,credit_correction=False,extreme_credit_accuracy_repaired=False)
(version/'deployed.json').write_text(json.dumps(r,indent=2)+'\\n')
for path in (Path(%r)/'active-training.json',Path(%r)/'active-source.json',Path(%r)/'verified_runtime.json'):
 d=json.loads(path.read_bytes());d['accepted_fla_early_output_scale_overlay']=r
 tmp=path.with_name(path.name+'.scale-order.tmp');tmp.write_text(json.dumps(d,indent=2)+'\\n');os.replace(tmp,path)
print(json.dumps(r))
'''%(version,targets,unchanged,ROOT,commit,ROOT,ROOT,ENTRY)
    shell='set -eu\nsource '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    (local/'deploy-command.sh').write_text(shell,encoding='utf-8',newline='\n')
    run=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=120)
    (local/'deploy.stdout').write_bytes(run.stdout);(local/'deploy.stderr').write_bytes(run.stderr)
    run.check_returncode();(local/'deployed.json').write_bytes(run.stdout);print(run.stdout.decode())


if __name__=='__main__':
    main()
