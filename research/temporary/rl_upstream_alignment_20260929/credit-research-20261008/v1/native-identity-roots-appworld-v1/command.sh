source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-native-identity-roots-appworld-20261009-v1'
HASHES={'inspect_native_identity_roots.py': '22621c4752e0b7fd255841697a45746e7dba1f4118bcc72ae3e6e71d9c76dc23', 'layer-collection-inputs.json': 'e835680d4b527829fae2b2c116c58b198fb3eda7bf12da137e11871e411f87a5', 'inspect_action_curve.py': '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'}
GDN={'task': 'appworld', 'root': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace', 'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace/clean/qwen35/qwen35_gdn_finite.py', 'parent_resolved': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/releases/fla-seed-range-20261009-v1/appworld/qwen35_gdn_finite.py', 'parent_sha256': '33b169b3fb660eb8ce57a6bda6ecb04c7f7235a029aa04038ff447b2faddf6fd', 'versioned_path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/releases/fla-early-output-scale-20261009-v1/appworld/qwen35_gdn_finite.py', 'new_sha256': '3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1', 'public_signature_unchanged': True, 'preview_import': {'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/releases/fla-early-output-scale-20261009-v1/appworld/qwen35_gdn_finite.py', 'resolved': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/releases/fla-early-output-scale-20261009-v1/appworld/qwen35_gdn_finite.py', 'sha256': '3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1', 'signature': "(module, values, endpoints, upstream, scale, fla_pullback, diagnostics=False, *, norm_gate_rule='content1', key_norm_pullback=None, offload_endpoints=False, consume_captures=False, fla_head_batch_size=None, norm_gate_pullback=None, conv_silu_pullback=None, input_shape=None, fla_coefficient_start=0, capture_start=0)", 'cuda_initialized': False, 'startup_source_sha256': '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'}, 'fresh_import': {'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace/clean/qwen35/qwen35_gdn_finite.py', 'resolved': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/releases/fla-early-output-scale-20261009-v1/appworld/qwen35_gdn_finite.py', 'sha256': '3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1', 'signature': "(module, values, endpoints, upstream, scale, fla_pullback, diagnostics=False, *, norm_gate_rule='content1', key_norm_pullback=None, offload_endpoints=False, consume_captures=False, fla_head_batch_size=None, norm_gate_pullback=None, conv_silu_pullback=None, input_shape=None, fla_coefficient_start=0, capture_start=0)", 'cuda_initialized': False, 'startup_source_sha256': '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'}}
VERSION='fla-early-output-scale-20261009-v1'
COMMIT='fd4d4ab26eb0ccd0ef194325db5fc1f733eb3281'
import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(OUT);root=Path(ROOT)
for name,h in HASHES.items():
 p=out/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
 if p.suffix=='.py':ast.parse(p.read_text())
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source=json.loads(source_path.read_bytes())
names=['qwen35_dense_finite_runner','qwen35_answer_finite','qwen35_gdn_finite','qwen35_decoder_finite','qwen35_native_prefix_artifacts']
owners={n:dict(source['actual_CPU_imports'][n]) for n in names}
owners['qwen35_gdn_finite']['sha256']=GDN['new_sha256']
for name,v in owners.items():assert hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'],name
assert Path(owners['qwen35_gdn_finite']['path']).resolve()==Path(GDN['versioned_path'])
spec=json.loads((out/'layer-collection-inputs.json').read_bytes())['tasks']['appworld']
assert len(spec['batches'])==12 and len(spec['entries'])==48
protocol=dict(scope='Original 48 AppWorld trajectories / 16 states, twelve original B4 roots, no finite propagation',
 numerical_version=VERSION,numerical_owners=owners,plan_sha256=HASHES['layer-collection-inputs.json'],code_commit=COMMIT)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Diagnostic devices occupied'
env=dict(os.environ,**source['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None)
env.update(CUDA_VISIBLE_DEVICES='4,5',DT_TASK=source['startup_options']['env.env_name'],
 DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
env['PYTHONPATH']=':'.join([str(out),source['pythonpath'],str(Path(source['dt_root'])/'clean/qwen35')])
argv=[env['VENV_PYTHON'],str(out/'inspect_native_identity_roots.py'),'--source',str(source_path),'--output',str(out/'results'),'--case','appworld']
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),launched_unix=time.time(),code_commit=COMMIT,
 scripts=HASHES,argv=argv,devices=[4,5],protocol=protocol,native_root_B4_calls_per_rank=6,
 operations_excluded=['finite_seed','DT','optimizer','backward','rollout','checkpoint_restore'],
 formal_release=False,physical_before=physical)
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

PY
