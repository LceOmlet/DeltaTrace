"""Continue from saved native attention operands after a None-buffer observer error."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
previous=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_fa_actual_20260908_v2'
r=json.loads((previous/'results.json').read_bytes());assert r['status']=='failed' and r['finite_attempts']==0
assert "'NoneType' object has no attribute 'numel'" in r['error']
capture=next(a for a in r['artifacts'] if a['file']=='native_decoder3_attention_capture.pt')
s=(previous/'study.py').read_text(encoding='utf-8')
begin=s.index('    cfg=AutoConfig.from_pretrained');end=s.index('    q,k,v=(c[',begin)
s=s[:begin]+'''    # No new decoder construction, weight load, forward, or endpoint capture.
    previous_dir=Path(p['saved_capture_directory']);raw=(previous_dir/'results.json').read_bytes()
    assert sha(raw)==p['saved_capture_result_sha256'];record=json.loads(raw)
    assert record['status']=='failed' and record['finite_attempts']==0
    source=previous_dir/'native_decoder3_attention_capture.pt'
    assert sha(source.read_bytes())==p['saved_capture_sha256']
    saved=torch.load(source,map_location='cpu',weights_only=True)
    B,T,C=saved['decoder_input'].shape;assert (B,T,C)==(4,605,4096)
    mask=saved['mask'].to('cuda');assert mask.sum(1).tolist()==[605,605,368,368]
    c={k:v.to('cuda') for k,v in saved['values'].items()};del saved
    for key in ('decoder_replay_vs_root','decoder_replay_valid_vs_root','capture_calls',
                'actual_interface_arguments','actual_public_FA_arguments','weight_tensor_receipts','official_loading_policy'):
        r[key]=record[key]
    r['inherited_decoder_evidence']={'directory':str(previous_dir),'result_sha256':p['saved_capture_result_sha256'],
        'capture_sha256':p['saved_capture_sha256'],'new_decoder_calls':0}
    original_forward=native.Qwen3_5Attention.forward
    save()
'''+s[end:]
assert s.count('output,lse,unused=result;assert unused.numel()==0')==1
s=s.replace('output,lse,unused=result;assert unused.numel()==0',
    'output,lse,unused=result;assert unused is None or unused.numel()==0')
s=s.replace("'unused_probability_buffer_elements':unused.numel()", "'unused_probability_buffer_elements':0 if unused is None else unused.numel(),'unused_probability_buffer_is_None':unused is None")
ast.parse(s);study=A/'qwen35_finite_fa_saved_20260908.py';study.write_text(s,encoding='utf-8',newline='\n')
p=dict(r['protocol']);p.update(saved_capture_directory='${ARTIFACT_ROOT}/codex_qwen35_finite_fa_actual_20260908_v2',
    saved_capture_result_sha256=sha((previous/'results.json').read_bytes()),saved_capture_sha256=capture['sha256'],
    correction='The public native FA returns None for omitted probabilities at dropout0. Accept None or zero-element tensor without reading either; use saved real decoder operands, no new decoder load/replay.',
    budget={'full_model_loads':0,'full_model_forwards':0,'meta_model_constructions':0,'decoder3_weight_loads':0,
        'decoder3_forward_attempts':0,'auxiliary_public_FA_forward_attempts':1,'public_FA_backward_attempts':1,
        'finite_calls':3,'finite_kernel_launches':9,'generation_calls':0,'quality_queries':0},
    family_budget='Two original decoder loads/replays, two public auxiliary FA forwards including the None-buffer observer abort, one public FA backward, three finite calls. No full-model pass, generation or quality query.',
    stop='Continue three finite calls from saved actual operands. Do not replay the decoder or repeat for numerical disagreement. Preserve both observer failures and their costs.')
files={'study.py':study.read_bytes(),
    'vendor_fa_finite_bf16_d256.py':(R/'research/runtime/vendor_fa_finite_bf16_d256.py').read_bytes(),
    'native_attention_capture.py':(R/'research/runtime/native_attention_capture.py').read_bytes(),
    'vendor_fa_finite_p1_bf16_d256.cu':(R/'research/prototypes/vendor_fa_finite_p1_bf16_d256.cu').read_bytes()}
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'qwen35_finite_fa_saved_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_finite_fa_saved_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_finite_fa_saved_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'study_sha256':p['files_sha256']['study.py'],'protocol_sha256':sha(files['protocol.json'])}))
