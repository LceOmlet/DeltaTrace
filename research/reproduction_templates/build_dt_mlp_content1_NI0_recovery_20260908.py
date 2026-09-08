"""Separately freeze one evidence-saving recovery; preserve failed v1 unchanged."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
p=json.loads((A/'dt_mlp_content1_NI0_protocol_20260908.json').read_bytes())
old=(R/'research/reproduction_templates/dt_mlp_content1_NI0_20260908.py').read_text(encoding='utf-8')
needle=" assert r['symmetric_vs_saved_score_max_abs']<1e-6 and r['methods']['symmetric']['needle']==old_result['needle']['recovery']"
assert old.count(needle)==1
new=old.replace(needle,""" r['control_comparison']={'score_relative_L2':float(np.linalg.norm(vectors['symmetric']-old_signed)/max(np.linalg.norm(old_signed),1e-30)),
  'needle_unchanged':r['methods']['symmetric']['needle']==old_result['needle']['recovery'],
  'top31_set_unchanged':set(r['methods']['symmetric']['selected'])==set(old_result['needle']['selected']),
  'eligible_sign_changes':int(np.sum(np.sign(vectors['symmetric'][keep])!=np.sign(old_signed[keep]))),
  'interpretation':'Report native default-precision drift; same-run rules share captures and seed. No retroactive pass of v1 and no universal1e-6 threshold.'}""")
new=new.replace("  vectors[method]=signed.cpu().numpy()", """  vectors[method]=signed.cpu().numpy()
  # Persist every completed vector before metric/comparison assertions.
  with (A/'signed_vectors.partial').open('wb') as handle:np.savez_compressed(handle,**vectors)
  (A/'signed_vectors.partial').replace(A/'signed_vectors.npz')
  r['artifacts']['signed_vectors.npz']={'sha256':sha((A/'signed_vectors.npz').read_bytes()),'bytes':(A/'signed_vectors.npz').stat().st_size};save()""")
public=R/'research/reproduction_templates/dt_mlp_content1_NI0_recovery_20260908.py';assert not public.exists();public.write_text(new,encoding='utf-8')
files={'study.py':public.read_bytes()}
oldroot=A/'snapshot${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1'
for name,want in p['files_sha256'].items():
 if name=='study.py':continue
 raw=((R/'research/runtime'/name).read_bytes() if name=='mlp_content1_finite.py' else (oldroot/name).read_bytes())
 assert sha(raw)==want,name;files[name]=raw
for raw in files.values():ast.parse(raw)
p.update(recovery={'original_failed_directory':'${ARTIFACT_ROOT}/codex_dt_mlp_content1_NI0_20260908_v1',
 'original_protocol_sha256':'33a3424cc0ae996f16da3bb12db97a56b328312d7d3f2ce71fbed0e6f2befe44',
 'original_results_sha256':'64bb566406f0f5a3a5ea4f2d4d319e8a810cbc7f0a4509e444f5905c80ba02108',
 'reason':'Both complete finite passes ran, but a driver1e-6 comparison assertion fired before full-vector export. This one recovery preserves vectors before comparison and reports ordinary native precision drift. v1 remains failed.',
 'algorithm_and_candidate_unchanged':True,'one_additional_model_load':1,'additional_finite_calls':64,'further_automatic_retry':False},
 stop='One separately frozen evidence recovery only. Persist complete vectors before metric/comparison checks. Source/input/nonfinite errors terminate; native default-precision drift is measured rather than treated as universal1e-6 equivalence. No candidate tuning, FT edits, new samples, generation or deletion.',
 files_sha256={k:sha(v) for k,v in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();path=A/'dt_mlp_content1_NI0_recovery_protocol_20260908.json';assert not path.exists();path.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_mlp_content1_NI0_20260908_v2'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_mlp_content1_NI0_recovery_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'candidate_sha256':p['files_sha256']['mlp_content1_finite.py']}))
