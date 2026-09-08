"""Use the actual current exp2 input default, with no FT source modifications."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
prior_path=A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_pristine_20260908_v1/results.json'
prior=json.loads(prior_path.read_bytes());assert prior['root_forwards_completed']==0 and 'AssertionError' in prior['error']
contract_path=A/'snapshot/tmp/qwen35_official_input_contract_20260908.json';contract=json.loads(contract_path.read_bytes())
assert contract['current_keep_equals_historical'] and contract['current_gold_equals_historical']
assert contract['modes']['current_public_chat_true']['input_ids_sha256']==prior['actual_input']['sha256']
tree=json.loads((A/'official_FT_e81_source_tree_20260908.json').read_bytes())
up=A/'snapshot${PRIVATE_MOUNT_PATH}'
verified={}
for name in ['exp/exp2/run_exp.py','exp/exp2/dataset_utils.py','ft_ifr_improve.py','llm_attr.py']:
 raw=(up/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
 assert blob==next(x['sha'] for x in tree['tree'] if x['path']==name)
 verified[name]=sha(raw)
run_tree=ast.parse((up/'exp/exp2/run_exp.py').read_bytes())
constructor=next(n for n in ast.walk(run_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='LLMIFRAttributionBoth')
assert 'use_chat_template' not in [kw.arg for kw in constructor.keywords]
p=json.loads((A/'official_ft_pristine_protocol_20260908.json').read_bytes())
p.update(expected_input=contract['modes']['current_exp2_default'],
 input_contract_sha256=sha(contract_path.read_bytes()),current_exp2_source_sha256=verified,
 prior_driver_stop={'result_sha256':sha(prior_path.read_bytes()),'model_loads':1,'model_body_forwards':0,
  'cause':'Driver incorrectly assumed public chat=True matched old075 formatter. The input guard stopped before model body; original FT was not the source of this failure.'},
 scope='Standalone current official exp2-default reference on NI0; original fixed prompt/target and gold/eligible preserved. Input588 differs from old DT605, so no paired DT comparison or quality advantage claim. Future DT must use this same official input before comparison.',
 current_input_policy='Use the actual current exp2 constructor default use_chat_template=False; do not edit FT/tokenizer template to restore historical075 Context/Query/thinking=False wrapper.',
 invocation_family={'model_load_ceiling_including_prior_driver_stop':2,'complete_model_body_forwards_ceiling':1,'quality_call_ceiling':4})
p['selection']['use_chat_template']=False
study=(R/'research/reproduction_templates/official_ft_pristine_run_20260908.py').read_bytes();ast.parse(study)
p['study_sha256']=sha(study)
files={'study.py':study,'protocol.json':json.dumps(p,indent=2).encode()}
(A/'official_ft_exp2_default_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode();python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1");d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_official_ft_exp2_default_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':sha(study),'budget':p['budget'],'expected_input':p['expected_input']}))
