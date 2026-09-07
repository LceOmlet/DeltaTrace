"""Freeze one native capture and a full-GDN finite integration screen."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
source=(A/'qwen35_paired_fla_capture_20260908.py').read_text(encoding='utf-8').split('    original={n:type(m).forward')[0]
source=source.replace('One real B4 EOS/input forward and native adjoint stages; no shadow model.',
    'One real B4 capture and complete finite GDN propagation; default model unchanged.')
source=source.replace("'whole_model_attributions':0,'adjoint_stage_attempts':0,\n   'adjoint_stages_completed':0,'artifacts':[]}",
    "'whole_model_attributions':0,'artifacts':[]}")
source=source.replace("os.environ['TRITON_CACHE_DIR']=p['compiler_cache']", "os.environ['TRITON_CACHE_DIR']=p['compiler_cache']\nos.environ['TORCHINDUCTOR_CACHE_DIR']=str(HERE/'inductor_cache')\nos.environ['OPENBLAS_NUM_THREADS']='1'")
source+=(A/'qwen35_gdn_integration_tail_20260908.txt').read_text(encoding='utf-8')
ast.parse(source);study=A/'qwen35_gdn_integration_20260908.py';study.write_text(source,encoding='utf-8')
old=json.loads((A/'qwen35_paired_fla_capture_protocol_20260908.json').read_bytes())
keys=['checkpoint','verified_shard_sizes','weight_identity_receipt_sha256','checkpoint_config_tokenizer_sha256',
    'native_model_sha256','isolated_site','fla_mapped_utils_sha256','fla_scheduled_chunk_sha256','wy_backported_sha256',
    'compiler_cache','wheels','author_root','author_source_sha256','cache_sha256','native_stage_source_sha256',
    'spans_directory','spans_sha256','source_tree_sha256']
p={k:old[k] for k in keys}
norm=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py'
conv=A/'snapshot${SITE_PACKAGES}/causal_conv1d/causal_conv1d_interface.py'
p.update(boundary_source_sha256={'gated_norm':sha(norm.read_bytes()),'causal_conv':sha(conv.read_bytes())},
    maximum_model_loads=1,maximum_root_forward_attempts=1,maximum_native_GDN_forward_calls=1,
    maximum_native_GDN_backward_calls=1,maximum_finite_GDN_calls=3,maximum_quality_queries=0,
    maximum_auxiliary_paired_conv_forward_calls=3,maximum_auxiliary_paired_conv_backward_calls=3,
    workload='Official NI0/MH1 fixed texts and original gold mapping. B4 EOS/input, true right-padding605/368; first complete GDN layer, no generation.',
    finite_rule='FLA content1 recurrence; output normalized content uses SiLU(z1), gate uses normalized baseline content. RMS/L2 use existing symmetric finite rules; scalar gates use actual endpoint secants.',
    convolution='One public native paired activation=None convolution and one autograd backward per finite call. Fused native SiLU outputs are actual endpoints. Record equal rounded preactivation/different fused output cases. Discarded weight gradients still counted.',
    validation='One cold and one warm full-GDN finite call, one equal-endpoint finite call versus one ordinary native GDN forward/backward. Per-boundary and per-head effects, padding coefficients and complete input vector retained. No new precision gate.',
    cost_scope='Root forward has passive CPU captures, saved layer checkpoints and actual default model backends. Finite-only cold/warm calls exclude endpoint generation. Ordinary gradient diagnostic includes local native forward and passive capture; not a matched speed comparison.',
    stop='One integration implementation, no automatic retry after failure, precision scan or full quality sweep. Analyze failure at the boundary; preserve source and results before the next decision.',
    whole_model_status='Complete GDN local propagation is this screen target; full decoder/MLP/full-attention propagation and original answer objective remain unimplemented.',
    compiler='Existing official compiled finite FLA with scalar reuse disabled; new boundaries initially use installed torch ops and native conv autograd. Generated sources/cold and default tuning costs retained.')
files={'study.py':study.read_bytes(),
    'official_fixed_text_inputs.py':(R/'research/runtime/official_fixed_text_inputs.py').read_bytes(),
    'qwen35_gdn_finite.py':(R/'research/runtime/qwen35_gdn_finite.py').read_bytes(),
    'finite_fla_gpu.py':(R/'research/runtime/finite_fla_gpu.py').read_bytes(),
    'signed_secant_rules.py':(R/'core/signed_secant_rules.py').read_bytes()}
for raw in files.values():ast.parse(raw)
p['files_sha256']={k:sha(v) for k,v in files.items()}
files['protocol.json']=json.dumps(p,indent=2).encode();(A/'qwen35_gdn_integration_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_gdn_integration_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_gdn_integration_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'files_sha256':p['files_sha256'],'budget':[1,1,1,1,3,0]}))
