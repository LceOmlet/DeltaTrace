"""Freeze an eleven-call native FA operator study; only write launch payload."""
import ast
import base64
import hashlib
import json
import shlex
import zipfile
import zlib
from pathlib import Path


A = Path(__file__).resolve().parent
R = A.parent / 'DeltaTrace'
source_dir = A / 'snapshot${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1'
sha = lambda value: hashlib.sha256(value).hexdigest()
with zipfile.ZipFile(source_dir / 'review_bundle.zip') as bundle:
    source_protocol_raw = bundle.read('protocol.json')
    source_results_raw = bundle.read('results.json')
    helper_raw = bundle.read('decoder19_conditional_decomposition_20260908.py')
source_protocol = json.loads(source_protocol_raw)
source_results = json.loads(source_results_raw)
assert source_results['status'] == 'decoder19_6_actual_conditional_observation_complete'
artifact = source_results['private_artifact']
assert artifact['file'] == 'MH1_decoder19_6_actual_private.pt'
assert artifact['sha256'] == '0b33ce76ca28fd00375e931a4769fbf4868cf2ff0cddd7040785bc42a6a2672a'
assert artifact['bytes'] == 1378988743
helper_name = 'decoder19_conditional_decomposition_20260908.py'
assert sha(helper_raw) == source_protocol['files_sha256'][helper_name]
assert sha((R / 'research/reproduction_templates' / helper_name).read_bytes()) == sha(helper_raw)
files = {'study.py': (R / 'research/reproduction_templates/dt_fa19_native_hybrid_20260908.py').read_bytes(),
         helper_name: helper_raw}
for name, value in files.items():
    ast.parse(value, filename=name)
source_remote = '${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1'
directory = '${ARTIFACT_ROOT}/codex_dt_fa19_native_hybrid_20260908_v1'
python = '${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
schedule = ['replay_0', 'replay_1', 'replay_10', 'replay_20', 'X_C0']
for step in ('1', '10', '20'):
    schedule.extend(['X_A0_' + step, 'X_CA_' + step])
assert len(schedule) == 11
protocol = {
    'scope': 'Fixed MH1 decoder19 native FA hybrid operator contrasts. No model, DT, scorer, FT, backward or explicit probability matrix.',
    'source_results_path': source_remote + '/results.json', 'source_results_sha256': sha(source_results_raw),
    'source_protocol_path': source_remote + '/protocol.json', 'source_protocol_sha256': sha(source_protocol_raw),
    'private_artifact_path': source_remote + '/' + artifact['file'],
    'private_artifact_sha256': artifact['sha256'], 'private_artifact_bytes': artifact['bytes'],
    'input_sha256': source_results['input']['input_sha256'],
    'case': 'morehopqa_1', 'decoder_index': 19, 'fixed_steps': ['1', '10', '20'],
    'frozen_input_receipts': {step: source_results['points'][step]['input_receipt'] for step in ('0', '1', '10', '20')},
    'original_core_errors_including_seed_cast': {step: source_results['points'][step]['layer_decompositions']['19']['terms']['finite_FA_core_including_seed_cast'] for step in ('1', '10', '20')},
    'installed_FA_interface_sha256': source_protocol['installed_FA_interface_sha256'],
    'native_model_sha256': source_protocol['native_model_sha256'],
    'checkpoint_config_path': source_protocol['checkpoint'] + '/config.json',
    'checkpoint_config_sha256': source_protocol['checkpoint_config_tokenizer_sha256']['config.json'],
    'scaling_verification': 'Read pinned native Attention __init__ AST self.scaling=self.head_dim**-0.5 and original config head_dim=256. No model construction; capture did not retain scalar call arguments separately.',
    'native_FA_kwargs': {'dropout_p': 0.0, 'softmax_scale': 0.0625, 'causal': True,
                         'window_size': [-1, -1], 'softcap': 0.0, 'alibi_slopes': None,
                         'return_attn_probs': False},
    'native_call_schedule': schedule,
    'budget': {'native_FA_calls': 11, 'actual_output_replays': 4, 'hybrid_operator_calls': 7,
               'model_calls': 0, 'DT_calls': 0, 'scorer_calls': 0, 'FT_calls': 0,
               'backward_calls': 0, 'generation_calls': 0, 'new_samples': 0,
               'extra_warmup_calls': 0, 'wall_time_seconds': 180},
    'contrasts': {'X_C0': 'FA(q_clean,k_clean,V0), computed once',
                  'X_A0': 'FA(q_A,k_A,V0)', 'X_CA': 'FA(q_clean,k_clean,V_A)',
                  'R0': '<m,X_C0-X_A0>', 'VCVA': '<m,O_clean-X_CA>', 'RA': '<m,X_CA-O_A>',
                  'exact_three_term_ledger': '(dq*delta_q+dk*delta_k-R0)+(dv*delta_v-VCVA)+(R0-RA)',
                  'interaction_ideal': 'm(P_clean-P_A)(V0-V_A), a reference-value routing interaction, including native rounding in this experiment'},
    'precision_contract': 'Main ledger uses actual stored FP32 mcontent and captured native BF16 O_clean/O_A. Four replays report drift only, never replace endpoints. Also report E_stored=E_BF16+<m_BF16-m_stored,O_clean-O_A>; no double counting.',
    'causal_scope': 'Hybrid attention-operator inputs only, not independently executed model counterfactuals or an EOS independent causal contribution. Q/K mismatch may include off-pair/B1-B2 transfer and quantization; no component alone proves a kernel bug.',
    'group_contract': 'Signed statistics of token-net sums. Contrast output-query locations and finite-coefficient operand locations are explicitly different roles; grouping does not create original source-token causal effects.',
    'acceptance': 'All source/config/artifact hashes and actual tensor layouts hold. CPU recomputed original ledger matches saved core. Three-term/seed/group closures <1e-7; exactly11 entered/returned native FA calls. Report every replay drift; no tuned numerical or bitwise admission threshold.',
    'stop': 'First source/layout/nonfinite/algebra failure or180seconds; preserve partial results and entered/returned calls. No alternate kernel, higher precision, retry, extra model or extra warmup.',
    'artifact_contract': 'Actual private1.379GB capture remains in its source directory and is never copied into the launch payload or public review archive. Only signed scalar/token contrasts and source metadata are exported.',
    'files_sha256': {name: sha(value) for name, value in files.items()}}
files['protocol.json'] = json.dumps(protocol, indent=2).encode()
protocol_path = A / 'dt_fa19_native_hybrid_protocol_20260908.json'
launch_path = A / 'launch_dt_fa19_native_hybrid_20260908.json'
assert not protocol_path.exists() and not launch_path.exists(), 'Refuse to overwrite frozen protocol or payload.'
blob = base64.b64encode(zlib.compress(json.dumps({name: base64.b64encode(value).decode()
                                               for name, value in files.items()}).encode())).decode()
loader = ('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path(' + repr(directory) + ');d.mkdir(exist_ok=False);'
          'files=json.loads(zlib.decompress(base64.b64decode(' + repr(blob) + ')));'
          '[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
          'f=(d/"driver.log").open("w");j=subprocess.Popen([' + repr(python) + ',"-B",str(d/"study.py")],'
          'stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
          '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
protocol_path.write_bytes(files['protocol.json'])
launch_path.write_text(json.dumps({'cmd': python + ' -c ' + shlex.quote(loader), 'timeout': 10}))
print(json.dumps({'protocol_sha256': sha(files['protocol.json']), 'study_sha256': protocol['files_sha256']['study.py'],
                  'launch_payload': str(launch_path), 'remote_directory': directory, 'budget': protocol['budget']}))
