"""Correct a checkpoint-vs-loaded-parameter dtype mismatch; no new root pass."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
bad=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_gdn_saved_recovery_20260908_v1'
r=json.loads((bad/'results.json').read_bytes())
assert r['status']=='full_GDN_vectors_and_native_gradient_limit_recovered'
wrong=[k for k,v in r['weight_tensor_receipts'].items() if v['dtype']=='torch.float32']
assert sorted(k.rsplit('.',1)[-1] for k in wrong)==['A_log','weight']
source=(A/'qwen35_gdn_saved_recovery_20260908.py').read_text(encoding='utf-8')
def replace(old,new):
    global source
    assert source.count(old)==1,old;source=source.replace(old,new)
replace('    from transformers import AutoConfig',
    '    from transformers import AutoConfig\n    from transformers.utils import ContextManagers\n    from transformers.core_model_loading import _materialize_copy')
replace("    cfg=AutoConfig.from_pretrained(cp,local_files_only=True,trust_remote_code=False).text_config\n    r['native_module_load_attempts']+=1;save()\n    with torch.device('meta'):module=native.Qwen3_5GatedDeltaNet(cfg,layer_idx=0)",
"""    cfg=AutoConfig.from_pretrained(cp,local_files_only=True,trust_remote_code=False)
    cfg._attn_implementation='flash_attention_2'
    cls=native.Qwen3_5ForConditionalGeneration
    r['meta_model_constructions']=1;r['native_module_load_attempts']+=1;save()
    with ContextManagers(cls.get_init_context(torch.bfloat16,False,False,False)):
        container=cls(cfg)
    plan=container._get_dtype_plan(torch.bfloat16);assert plan=={}
    module=container.model.language_model.layers[0].linear_attn
    desired_dtypes={k:v.dtype for k,v in module.state_dict().items()}
    assert set(desired_dtypes.values())=={torch.bfloat16}
    r['official_loading_policy']={'requested_dtype':'torch.bfloat16','dtype_plan':plan,
        'empty_parameter_dtypes':{k:str(v) for k,v in desired_dtypes.items()},
        'context':'Original model class get_init_context; metadata only, all parameters remain meta until nine GDN tensors are loaded.',
        'materialization':'Original Transformers _materialize_copy with the initialized parameter dtype.'}""")
replace("                value=sf.get_tensor(name);state[name.removeprefix(prefix)]=value",
"""                raw_value=sf.get_tensor(name)
                value=_materialize_copy(sf.get_slice(name),device='cpu',dtype=desired_dtypes[name.removeprefix(prefix)])
                state[name.removeprefix(prefix)]=value""")
replace("'tensor_bytes_sha256':sha(value.view(torch.uint8).numpy().tobytes())}",
    "'tensor_bytes_sha256':sha(value.view(torch.uint8).numpy().tobytes()),\n                    'checkpoint_dtype':str(raw_value.dtype),'checkpoint_tensor_sha256':sha(raw_value.view(torch.uint8).numpy().tobytes())}")
replace("module.eval().requires_grad_(False).to('cuda');del state", "module.eval().requires_grad_(False).to('cuda');del state,container")
replace("    persist('native',{'gradient':native_gradient[1::2],'replay_output':y.detach()})",
"""    def relative(a,b):
        a=a.float().numpy().astype(np.float64);b=b.float().numpy().astype(np.float64)
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        norm=float(np.linalg.norm(a.ravel()));error=float(np.linalg.norm((b-a).ravel()))
        return {'relative_L2':error/norm if norm else None,'max_abs':float(np.abs(a-b).max()),'exact_equal':bool(np.array_equal(a,b))}
    r['native_replay_vs_real_root']={group:{k:relative(saved[group][k],v) for k,v in values.items() if v is not None}
        for group,values in [('values',capture.values),('endpoints',capture.endpoints)]}
    save()
    persist('native',{'gradient':native_gradient[1::2],'replay_output':y.detach()})""")
replace("r['status']='full_GDN_vectors_and_native_gradient_limit_recovered'",
    "r['status']='dtype_matched_GDN_vectors_and_native_gradient_limit_recovered'")
ast.parse(source);study=A/'qwen35_gdn_dtype_matched_20260908.py';study.write_text(source,encoding='utf-8')
p=json.loads((A/'qwen35_gdn_saved_recovery_protocol_20260908.json').read_bytes())
site=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/lib/python3.12/site-packages/transformers'
p.update(unmatched_recovery_sha256=sha((bad/'results.json').read_bytes()),
    loading_source_sha256={n:sha((site/n).read_bytes()) for n in ['modeling_utils.py','core_model_loading.py']},
    correction='Checkpoint F32 norm.weight/A_log must follow original BF16 from_pretrained initialization/materialization policy. Previous mixed-dtype recovery is invalid for the actual model and retained as a negative result.',
    maximum_meta_model_constructions=1,maximum_model_loads=0,maximum_root_forward_attempts=0,
    combined_family_budget='Including observer failures and wrong-dtype recovery: two full-model loads/root attempts, one completed root; two standalone GDN tensor loads; six finite GDN calls and three local ordinary GDN forward/backward calls. No new model root or quality pass.',
    stop='One dtype-matched saved-state correction. Inspect all replayed native endpoints against the actual root; no further rerun or precision scan in this stage.')
files={'study.py':study.read_bytes(),'qwen35_gdn_finite.py':(R/'research/runtime/qwen35_gdn_finite.py').read_bytes(),
    'finite_fla_gpu.py':(R/'research/runtime/finite_fla_gpu.py').read_bytes(),'signed_secant_rules.py':(R/'core/signed_secant_rules.py').read_bytes()}
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'qwen35_gdn_dtype_matched_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_gdn_dtype_matched_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_gdn_dtype_matched_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'source_sha256':p['files_sha256']}))
