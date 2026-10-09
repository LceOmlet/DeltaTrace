"""Deploy the accepted dtype repair to the two exact active owner seams.

Preserve historical resolved owners. Version new files separately, then replace
only each current leaf symlink. No training restart or historical rewrite.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH,SCP


def main():
    prepared=json.loads((HERE/'accepted-fla-seed-range-v1/prepared.json').read_bytes())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    targets=dict(textcraft=ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/deltatrace',
                 appworld=ROOT+'/candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace')
    version=ROOT+'/releases/fla-seed-range-20261009-v1'
    sources={task:Path(prepared[task]['source']) for task in targets}
    for task,source in sources.items():
        raw=source.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==prepared[task]['repaired_sha256']
        assert subprocess.check_output(['git','show',commit+':'+source.relative_to(REPO).as_posix()],cwd=REPO)==raw
    subprocess.run(SSH+['mkdir','-p',version+'/textcraft',version+'/appworld'],check=True)
    for task,source in sources.items():
        subprocess.run(SCP+[str(source),f'{SSH[-1]}:{version}/{task}/qwen35_gdn_finite.py'],check=True)
    receipt=REPO/'experiments/rl/results_fla_range_owner_20261009.json'
    subprocess.run(SCP+[str(receipt),f'{SSH[-1]}:{version}/verified-range-receipt.json'],check=True)
    binding=HERE/'fla-range-deploy-base-precheck.jsonl'
    subprocess.run(SCP+[str(binding),f'{SSH[-1]}:{version}/baseline-binding.jsonl'],check=True)
    script=f'''set -e
CUDA_VISIBLE_DEVICES=-1 /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python - <<'PY'
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
targets={targets!r};prepared={prepared!r};version=Path({version!r})
assert not (version/'deployed.json').exists()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
verified=json.loads((version/'verified-range-receipt.json').read_bytes())
binding={{r['task']:r for r in map(json.loads,(version/'baseline-binding.jsonl').read_text().splitlines()) if 'task' in r}}
assert verified['official_checks']['checks_passed']==22 and verified['official_checks']['checks_failed']==0
assert verified['repair']['owner_sha256']==prepared['appworld']['repaired_sha256']
rows=[]
for task,root in targets.items():
 root=Path(root);path=root/'clean/qwen35/qwen35_gdn_finite.py';new=version/task/'qwen35_gdn_finite.py'
 assert path.is_symlink() and not any(p.is_symlink() for p in [path.parent,root/'clean',root])
 assert sha(path) in (prepared[task]['base_sha256'],prepared[task]['repaired_sha256']) and sha(new)==prepared[task]['repaired_sha256']
 original=Path(binding[task]['resolved']);oldraw=original.read_bytes();assert sha(original)==prepared[task]['base_sha256']
 backup=version/task/'original-GDN.py';backup.write_bytes(oldraw)
 before=ast.parse(oldraw);after=ast.parse(new.read_bytes())
 fn=lambda tree:next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='gdn_finite_pullback')
 assert ast.dump(fn(before).args)==ast.dump(fn(after).args)
 assert any(x.arg=='consume_captures' for x in fn(after).args.kwonlyargs)==prepared[task]['consume_captures']
 rows.append(dict(task=task,root=str(root),path=str(path),original_resolved=str(original),old_link=binding[task]['symlink_chain'][0]['link'],
   original_sha256=sha(original),versioned_path=str(new),new_sha256=sha(new),signature_unchanged=True))
def probe(row,preview):
 source_path=Path({ROOT!r})/'runs/direct-target-prefix-runtime-20261007-v1'/row['task']/(row['task']+'-dt')/'source.json'
 source=json.loads(source_path.read_bytes());env=dict(os.environ,**source['environment']);env['CUDA_VISIBLE_DEVICES']='-1'
 qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35'];root=row['root']
 official=env.get('DT_OFFICIAL_ROOT') or qwen['official_root']
 env['PYTHONPATH']=':'.join([root,official,root+'/clean/qwen35',source['pythonpath'],qwen['ft_extension_root']])
 prelude='import hashlib,inspect,importlib.util,json,torch;from pathlib import Path;'
 if preview:
  code=prelude+"spec=importlib.util.spec_from_file_location('accepted_range_preview',"+repr(row['versioned_path'])+");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
 else:code=prelude+'import qwen35_gdn_finite as m;'
 code+="p=Path(m.__file__);print(json.dumps(dict(path=str(p),resolved=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),signature=str(inspect.signature(m.gdn_finite_pullback)),CUDA_initialized=torch.cuda.is_initialized())))"
 r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
 label='preview' if preview else 'active'
 (version/row['task']/(label+'-import-stdout.txt')).write_text(r.stdout)
 (version/row['task']/(label+'-import-stderr.txt')).write_text(r.stderr)
 assert r.returncode==0,r.stderr
 imported=json.loads(r.stdout.splitlines()[-1]);assert imported['sha256']==row['new_sha256'] and not imported['CUDA_initialized']
 imported['startup_source_sha256']=sha(source_path);imported['source_environment_reused']=True
 return imported
for row in rows:row['preview_import']=probe(row,True)
# Both exact bases are checked before either active leaf is changed.
for row in rows:
 path=Path(row['path']);link=path.with_name(path.name+'.accepted-range-link')
 assert not link.exists() and not link.is_symlink()
 link.symlink_to(row['versioned_path']);os.replace(link,path)
 assert sha(path)==row['new_sha256']
 assert sha(Path(row['original_resolved']))==row['original_sha256']
 row['fresh_import']=probe(row,False);row['historical_owner_unchanged']=True
record=dict(unix=time.time(),version='fla-seed-range-20261009-v1',source_commit={commit!r},
 status='deployed_to_both_active_DT_owner_paths_and_local_default',owners=rows,
 verification_receipt_sha256=sha(version/'verified-range-receipt.json'),official_checks_passed=22,
 original_AppWorld_B4_replay_complete=True,TextCraft_edits_reused_verbatim_from_verified_patch=True,
 TextCraft_new_whole_DT_replay=False,training_restart=False,extreme_attribution_repaired=False,
 original_PPO_NaN_repaired=False,QVA_whitening_PPO_changed=False,official_tolerance_changed=False,
 old_unaccepted_candidates_promoted=False,stopped_startup_source_json_rewritten=False)
(version/'deployed.json').write_text(json.dumps(record,indent=2)+'\\n')
# Keep original startup/verification fields. Publish the completed overlay separately.
for path in [Path({ROOT!r})/'active-training.json',Path({ROOT!r})/'active-source.json',Path({ENTRY!r})/'verified_runtime.json']:
 d=json.loads(path.read_bytes());d['accepted_fla_seed_range_overlay']=record
 tmp=path.with_name(path.name+'.fla-range-version.tmp');tmp.write_text(json.dumps(d,indent=2)+'\\n');os.replace(tmp,path)
print(json.dumps(record))
PY
'''
    folder=HERE/'accepted-fla-seed-range-v1'
    (folder/'deploy-command.sh').write_text(script,encoding='utf-8',newline='\n')
    run=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True)
    (folder/'deploy.stdout').write_bytes(run.stdout);(folder/'deploy.stderr').write_bytes(run.stderr)
    run.check_returncode()
    record=json.loads(run.stdout)
    (folder/'deployed.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
