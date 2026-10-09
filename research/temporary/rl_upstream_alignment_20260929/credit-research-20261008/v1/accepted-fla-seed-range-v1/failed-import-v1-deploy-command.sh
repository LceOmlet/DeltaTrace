set -e
CUDA_VISIBLE_DEVICES=-1 /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python - <<'PY'
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
targets={'textcraft': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/deltatrace', 'appworld': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace'};prepared={'status': 'prepared_for_user_authorized_deployment', 'version': 'fla-seed-range-20261009-v1', 'source': 'D:\\Users\\Administrator\\Documents\\ChatGPT\\DeltaTrace\\research\\temporary\\rl_upstream_alignment_20260929\\credit-research-20261008\\v1\\promote_fla_seed_range.py', 'edits_reused_verbatim_from_verified_AppWorld': 3, 'inverse_patch_restores_exact_base': True, 'textcraft': {'base_sha256': 'ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9', 'repaired_sha256': 'bc1a11d95255f8990f312696235c50da219a91d2d3f026155084f2f56273bb15', 'source': 'D:\\Users\\Administrator\\Documents\\ChatGPT\\DeltaTrace\\deltatrace\\clean\\qwen35\\qwen35_gdn_finite.py', 'consume_captures': False}, 'appworld': {'base_sha256': '448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72', 'repaired_sha256': '33b169b3fb660eb8ce57a6bda6ecb04c7f7235a029aa04038ff447b2faddf6fd', 'source': 'D:\\Users\\Administrator\\Documents\\ChatGPT\\DeltaTrace\\research\\temporary\\rl_upstream_alignment_20260929\\credit-research-20261008\\v1\\native-fla-range-owner-prepared\\qwen35_gdn_finite.py', 'consume_captures': True}, 'verification_receipt': 'experiments/rl/results_fla_range_owner_20261009.json', 'default_norm_gate_selector_changed': False, 'finite_FLA_callback_count_changed': False, 'QVA_or_whitening_or_PPO_changed': False, 'formal_training_restart': False, 'historical_candidate_files_modified': False};version=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/releases/fla-seed-range-20261009-v1')
assert not (version/'deployed.json').exists()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
verified=json.loads((version/'verified-range-receipt.json').read_bytes())
assert verified['official_checks']['checks_passed']==22 and verified['official_checks']['checks_failed']==0
assert verified['repair']['owner_sha256']==prepared['appworld']['repaired_sha256']
rows=[]
for task,root in targets.items():
 root=Path(root);path=root/'clean/qwen35/qwen35_gdn_finite.py';new=version/task/'qwen35_gdn_finite.py'
 assert path.is_symlink() and not any(p.is_symlink() for p in [path.parent,root/'clean',root])
 assert sha(path)==prepared[task]['base_sha256'] and sha(new)==prepared[task]['repaired_sha256']
 original=path.resolve();oldraw=path.read_bytes();backup=version/task/'original-GDN.py';backup.write_bytes(oldraw)
 before=ast.parse(oldraw);after=ast.parse(new.read_bytes())
 fn=lambda tree:next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='gdn_finite_pullback')
 assert ast.dump(fn(before).args)==ast.dump(fn(after).args)
 assert any(x.arg=='consume_captures' for x in fn(after).args.kwonlyargs)==prepared[task]['consume_captures']
 rows.append(dict(task=task,root=str(root),path=str(path),original_resolved=str(original),old_link=os.readlink(path),
   original_sha256=sha(path),versioned_path=str(new),new_sha256=sha(new),signature_unchanged=True))
# Both exact bases are checked before either active leaf is changed.
for row in rows:
 path=Path(row['path']);link=path.with_name(path.name+'.accepted-range-link')
 assert not link.exists() and not link.is_symlink()
 link.symlink_to(row['versioned_path']);os.replace(link,path)
 assert sha(path)==row['new_sha256']
 assert sha(Path(row['original_resolved']))==row['original_sha256']
 # Fresh process per root prevents module-cache cross-contamination.
 probe="import hashlib,inspect,json,sys,torch;from pathlib import Path;root="+repr(row['root'])+";sys.path[:0]=[root+'/clean/qwen35',root+'/clean',root];import qwen35_gdn_finite as m;p=Path(m.__file__);print(json.dumps(dict(path=str(p),resolved=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),signature=str(inspect.signature(m.gdn_finite_pullback)),CUDA_initialized=torch.cuda.is_initialized())))"
 r=subprocess.run([sys.executable,'-c',probe],capture_output=True,text=True,check=True)
 imported=json.loads(r.stdout.splitlines()[-1]);assert imported['sha256']==row['new_sha256'] and not imported['CUDA_initialized']
 row['fresh_import']=imported;row['historical_owner_unchanged']=True
record=dict(unix=time.time(),version='fla-seed-range-20261009-v1',source_commit='538d19dc9700d92506db13ce95f9819f35f90a59',
 status='deployed_to_both_active_DT_owner_paths_and_local_default',owners=rows,
 verification_receipt_sha256=sha(version/'verified-range-receipt.json'),official_checks_passed=22,
 original_AppWorld_B4_replay_complete=True,TextCraft_edits_reused_verbatim_from_verified_patch=True,
 TextCraft_new_whole_DT_replay=False,training_restart=False,extreme_attribution_repaired=False,
 original_PPO_NaN_repaired=False,QVA_whitening_PPO_changed=False,official_tolerance_changed=False,
 old_unaccepted_candidates_promoted=False,stopped_startup_source_json_rewritten=False)
(version/'deployed.json').write_text(json.dumps(record,indent=2)+'\n')
# Keep original startup/verification fields. Publish the completed overlay separately.
for path in [Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')/'active-training.json',Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')/'active-source.json',Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry')/'verified_runtime.json']:
 d=json.loads(path.read_bytes());d['accepted_fla_seed_range_overlay']=record
 tmp=path.with_name(path.name+'.fla-range-version.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,path)
print(json.dumps(record))
PY
