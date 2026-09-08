"""Freeze separate build/run payloads; do not compile or execute remotely."""
import ast
import base64
import hashlib
import json
import shlex
import zlib
from pathlib import Path


A = Path(__file__).resolve().parent
R = A.parent / 'DeltaTrace'
sha = lambda data: hashlib.sha256(data).hexdigest()
old = A / 'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_fa_build_20260908_v2'
source = A / 'snapshot${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1'
hybrid = A / 'snapshot${ARTIFACT_ROOT}/codex_dt_fa19_native_hybrid_20260908_v2'
bp = json.loads((old / 'protocol.json').read_bytes())
sp = json.loads((source / 'protocol.json').read_bytes())
sr = json.loads((source / 'results.json').read_bytes())
hr = json.loads((hybrid / 'results.json').read_bytes())
hp = json.loads((hybrid / 'protocol.json').read_bytes())
assert sr['status'] == 'decoder19_6_actual_conditional_observation_complete'
assert hr['status'] == 'eleven_native_FA_hybrid_contrasts_complete'
assert hr['native_FA_calls_entered'] == hr['native_FA_calls_returned'] == 11
assert sha((hybrid / 'results.json').read_bytes()) == json.loads((hybrid / 'terminal_receipt.json').read_bytes())['files']['results.json']['sha256']
assert sr['private_artifact']['sha256'] == '0b33ce76ca28fd00375e931a4769fbf4868cf2ff0cddd7040785bc42a6a2672a'
extension_name = 'vendor_fa_finite_p1_bf16_d256_conditional_diag_20260908.cu'
library_name = 'libdeltatrace_fa19_conditional_diag_20260908.so'
original_extension = R / 'research/prototypes/vendor_fa_finite_p1_bf16_d256.cu'
assert sha(original_extension.read_bytes()) == bp['extension_sha256']
files = {'study.py': (R / 'research/reproduction_templates/dt_fa19_qk_softmax_diagnostic_20260908.py').read_bytes(),
         'build.py': (R / 'research/reproduction_templates/dt_fa19_qk_softmax_build_20260908.py').read_bytes(),
         extension_name: (R / 'research/prototypes' / extension_name).read_bytes(),
         'vendor_fa_conditional_diag_20260908.py': (R / 'research/runtime/vendor_fa_conditional_diag_20260908.py').read_bytes(),
         'vendor_fa_finite_bf16_d256.py': (R / 'research/runtime/vendor_fa_finite_bf16_d256.py').read_bytes()}
assert sha(files['vendor_fa_finite_bf16_d256.py']) == sp['files_sha256']['vendor_fa_finite_bf16_d256.py']
for name, value in files.items():
    if name.endswith('.py'):
        ast.parse(value, filename=name)
remote = '${ARTIFACT_ROOT}/codex_dt_fa19_qk_softmax_diagnostic_20260908_v1'
python = '${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
source_remote = '/tmp/' + source.name
hybrid_remote = '/tmp/' + hybrid.name
old_remote = '/tmp/' + old.name
protected = []
for folder, names in ((source, ('results.json', 'protocol.json')),
                       (hybrid, ('results.json', 'protocol.json', 'signed_token_contrasts.npz')),
                       (old, ('study.py', 'protocol.json', 'vendor_fa_finite_p1_bf16_d256.cu'))):
    for name in names:
        protected.append({'path': '/tmp/' + folder.name + '/' + name,
                          'sha256': sha((folder / name).read_bytes())})
protected.append({'path': sp['finite_FA_library'], 'sha256': sp['finite_FA_library_sha256']})
kwargs = dict(hp['native_FA_kwargs'])
kwargs['return_attn_probs'] = True
protocol = {
    'purpose': 'One saved FA19 MH1 operator. Split frozen route error into replay, QK allocation plus multiplier rounding, weight cast, and finite softmax conditional response. No model/candidate/metric run.',
    'fixed_steps': ['1', '10', '20'], 'case': 'morehopqa_1', 'layer': 19,
    'source_results_path': source_remote + '/results.json',
    'hybrid_results_path': hybrid_remote + '/results.json',
    'hybrid_vectors_path': hybrid_remote + '/signed_token_contrasts.npz',
    'private_artifact_path': source_remote + '/' + sr['private_artifact']['file'],
    'private_artifact_sha256': sr['private_artifact']['sha256'],
    'private_artifact_bytes': sr['private_artifact']['bytes'],
    'input_sha256': sr['input']['input_sha256'],
    'original_route_error': {step: hr['points'][step]['fields']['routing_at_baseline_values_prediction_error']['net']
                             for step in ('1', '10', '20')},
    'protected_sources': protected,
    'vendor_source_parent': bp['vendor_source_parent'], 'vendor_parent_sha256': bp['vendor_parent_sha256'],
    'vendor_git_commit': bp['vendor_git_commit'], 'vendor_source_version': bp['vendor_source_version'],
    'installed_model_FA_version': bp['installed_model_FA_version'], 'same_version_source_claim': False,
    'production_extension_path': old_remote + '/vendor_fa_finite_p1_bf16_d256.cu',
    'production_extension_sha256': bp['extension_sha256'],
    'production_library_path': sp['finite_FA_library'], 'production_library_sha256': sp['finite_FA_library_sha256'],
    'build_script_derived_from_path': old_remote + '/study.py',
    'build_script_derived_from_sha256': sha((old / 'study.py').read_bytes()),
    'diagnostic_extension_name': extension_name, 'diagnostic_library_name': library_name,
    'diagnostic_ABI': '20 pointers (original14,actual_qc/kc/qa/ka,FP32_row/BF16_row),4 int(B,H,Hkv,T),float scale,stream pointer. Separate conditional_diag C symbol; no production library overwrite.',
    'traits': {'head_dim': 256, 'tile_m': 32, 'tile_n': 32, 'warps': 2, 'shared_bytes': 32768},
    'extension_change': 'Keep original three phases and dq/dk/dv arithmetic. Phase1 only: reuse original pair with endpoint=-1 to preserve frozen midpoint, flash::gemm for C/A scores, exact original CONVERT_TENSOR_TYPE for W_BF16, original row reductions for two O(BHT) outputs. No alternative attention/softmax implementation, T-by-T output, tile search or production change.',
    'installed_FA_interface_sha256': hp['installed_FA_interface_sha256'],
    'native_FA_kwargs': kwargs,
    'native_scaling_provenance': hp['scaling_verification'],
    'build_budget': {'compiler_attempts': 1, 'compiler_seconds': 150, 'wall_seconds': 180,
                     'tile_candidates': 1, 'GPU_kernel_calls': 0, 'model_calls': 0},
    'run_budget': {'wall_seconds': 180, 'native_FA_calls': 1, 'batch_size_for_native_LSE': 2,
                   'diagnostic_finite_calls': 3, 'diagnostic_phase_launches_on_success': 9,
                   'extra_score_MMA_tiles_per_Phase1_tile': 2, 'extra_warmup': 0,
                   'model_calls': 0, 'DT_calls': 0, 'scorer_calls': 0, 'FT_calls': 0,
                   'backward_calls': 0, 'new_samples': 0},
    'native_LSE_contract': 'Exact same public default FA route as current runner: return_attn_probs=True with dropout=0. Installed source uses return_softmax=flag and dropout>0. Require unused is None or numel()==0; no probability matrix or private buffer reads.',
    'ledger': ['D_saved-D_replay', 'D_replay-T_BF16', 'T_BF16-T_FP32', 'T_FP32-R0'],
    'identity': 'Sum four terms=D_saved-R0. D is dq*actual_deltaQ+dk*actual_deltaK with actual contiguous GQA fold; T is row sum W*(scale*(Q_CK_C^T-Q_AK_A^T)) under pinned original extension MMA. R0 is frozen native hybrid contrast at V0.',
    'precision_and_identifiability': 'T_FP32 uses current extension score MMA and finite weight, not a newly observed native internal matrix. D_replay-T_BF16 includes frozen QK endpoint interaction plus midpoint/MMA/output rounding. T_FP32-R0 includes finite softmax conditional transfer, BF16 upstream seed, B1/B2 endpoint drift and native output precision. Explicit replay/cast terms must be assessed before structural ranking.',
    'allEOS_guard': 'Use original B1 step20 deleting all eligible positions, preserve B1/B2 drift, do not require zero. Saved/actual qk prediction and original route residual must match frozen hybrid evidence within1e-7.',
    'acceptance': 'All hashes and actual shapes/dtypes/masks hold; coefficients are finite and independent of which actual A is used (same frozen coefficient input). Record all dq/dk/dv and B2 output replay drift. Four-term and coordinate-group algebra closes within1e-7. Closure alone does not prove the diagnostic tile formula.',
    'stop': 'One compile attempt; first compilation/runtime/nonfinite/source/layout/invariant/closure failure ends that stage, preserve failure evidence. No automatic retry/backend/precision change, full-model run, additional point, warmup or candidate sweep.',
    'next_action_if_QK_dominates': 'Only then one fixed QK endpoint-alignment candidate deltaQ*K1^T+Q0*deltaK^T (instead of midpoint), same3 phases and GEMM count. Endpoint identity holds on the full EOS/input chord. This diagnostic does not prove its sign or MAS improvement; retain early risk and require one paired official-metric test, no endpoint/weight sweep.',
    'next_action_if_softmax_dominates': 'Only then one original-input-Jacobian plus minimum-norm conserving correction candidate. Per valid row s=z1-z0,sc=s-mean(s),g=u*V0^T,m1=J(p1)g,a=(g*deltaP-m1*s)/||sc||^2,m=m1+a*sc. Real-arithmetic secant/zero-row-sum and minimal change to m1 are provable; arbitrary-direction/MAS advantage is not. Same tile operands/GEMMs, different scalar reductions; numerical degenerate rows need explicit stable treatment before implementation.',
    'next_action_if_numerics_or_opposite_signed_terms_dominate': 'Do not assign a structural winner or automatically expand the study. Report numerical/cancellation limitation and stop for root decision.',
    'group_contract': 'T/R0 group output-query coordinates, Q/K predictions group operand coordinates; equal token indices are different roles. No group is an independent original-input causal effect.',
    'artifact_contract': 'Source private1.379GB capture stays in source directory. New dq/dk/dv/tau/center/LSE/row outputs stay in diagnostic_coefficients_private.pt, excluded from review zip. Public review contains source, receipts and signed coordinate ledgers only.',
    'files_sha256': {name: sha(value) for name, value in files.items()}}
files['protocol.json'] = json.dumps(protocol, indent=2).encode()
paths = {name: A / filename for name, filename in {
    'protocol': 'dt_fa19_qk_softmax_diagnostic_protocol_20260908.json',
    'build': 'launch_dt_fa19_qk_softmax_build_20260908.json',
    'run': 'launch_dt_fa19_qk_softmax_run_20260908.json'}.items()}
assert not any(path.exists() for path in paths.values()), 'Refuse to overwrite a frozen protocol or payload.'
blob = base64.b64encode(zlib.compress(json.dumps({name: base64.b64encode(value).decode()
                                               for name, value in files.items()}).encode())).decode()
loader = ('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path(' + repr(remote) + ');d.mkdir(exist_ok=False);'
          'files=json.loads(zlib.decompress(base64.b64decode(' + repr(blob) + ')));'
          '[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
          'f=(d/"build_driver.log").open("w");j=subprocess.Popen([' + repr(python) + ',"-B",str(d/"build.py")],'
          'stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
          '(d/"build.pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d),"stage":"build_only"}))')
run = ('import pathlib,subprocess,json;d=pathlib.Path(' + repr(remote) + ');'
       'assert (d/"build_results.json").is_file() and not (d/"results.json").exists();'
       'b=json.loads((d/"build_results.json").read_bytes());assert b["status"]=="finite_extension_compiled_not_executed";'
       'f=(d/"driver.log").open("w");j=subprocess.Popen([' + repr(python) + ',"-B",str(d/"study.py")],'
       'stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
       '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d),"stage":"run"}))')
paths['protocol'].write_bytes(files['protocol.json'])
paths['build'].write_text(json.dumps({'cmd': python + ' -c ' + shlex.quote(loader), 'timeout': 10}))
paths['run'].write_text(json.dumps({'cmd': python + ' -c ' + shlex.quote(run), 'timeout': 10}))
print(json.dumps({'protocol_sha256': sha(files['protocol.json']), 'extension_sha256': sha(files[extension_name]),
                  'paths': {name: str(path) for name, path in paths.items()}, 'remote_directory': remote,
                  'build_budget': protocol['build_budget'], 'run_budget': protocol['run_budget']}))
