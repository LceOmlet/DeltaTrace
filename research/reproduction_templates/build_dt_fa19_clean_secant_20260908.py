"""Freeze one clean-secant candidate build/run; never execute remotely."""
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
strict = A / 'snapshot${ARTIFACT_ROOT}/codex_dt_fa19_qk_softmax_diagnostic_20260908_v1'
sr = json.loads((strict / 'results.json').read_bytes())
sp = json.loads((strict / 'protocol.json').read_bytes())
receipt = json.loads((strict / 'terminal_receipt.json').read_bytes())
assert sr['status'] == 'three_FA19_conditional_score_contractions_complete'
assert sha((strict / 'results.json').read_bytes()) == receipt['files']['results.json']['sha256']
assert all(sr['points'][step]['coefficient_replay_drift'][name]['bitwise_equal']
           for step in ('1', '10', '20') for name in ('dq', 'dk', 'dv'))
extension_name = 'vendor_fa_finite_p1_clean_secant_20260908.cu'
library_name = 'libdeltatrace_fa19_clean_secant_20260908.so'
files = {'study.py': (R / 'research/reproduction_templates/dt_fa19_clean_secant_20260908.py').read_bytes(),
         'build.py': (R / 'research/reproduction_templates/dt_fa19_clean_secant_build_20260908.py').read_bytes(),
         extension_name: (R / 'research/prototypes' / extension_name).read_bytes(),
         'vendor_fa_clean_secant_20260908.py': (R / 'research/runtime/vendor_fa_clean_secant_20260908.py').read_bytes(),
         'vendor_fa_finite_bf16_d256.py': (R / 'research/runtime/vendor_fa_finite_bf16_d256.py').read_bytes()}
assert sha(files['vendor_fa_finite_bf16_d256.py']) == sp['files_sha256']['vendor_fa_finite_bf16_d256.py']
for name, value in files.items():
    if name.endswith('.py'):
        ast.parse(value, filename=name)
tree = ast.parse(files['vendor_fa_clean_secant_20260908.py'])
row_fields = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == 'ROW_FIELDS' for t in node.targets)))
assert len(row_fields) == 15
remote = '${ARTIFACT_ROOT}/codex_dt_fa19_clean_secant_20260908_v1'
python = '${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
strict_remote = '/tmp/' + strict.name
protected = list(sp['protected_sources'])
for name in ('results.json', 'protocol.json', 'build_results.json'):
    protected.append({'path': strict_remote + '/' + name, 'sha256': sha((strict / name).read_bytes())})
artifact = sr['artifacts']['diagnostic_coefficients_private.pt']
protocol = {key: sp[key] for key in ('source_results_path', 'hybrid_results_path', 'hybrid_vectors_path',
    'private_artifact_path', 'private_artifact_sha256', 'private_artifact_bytes', 'input_sha256',
    'vendor_source_parent', 'vendor_parent_sha256', 'vendor_git_commit', 'vendor_source_version',
    'installed_model_FA_version', 'same_version_source_claim', 'production_extension_path',
    'production_extension_sha256', 'production_library_path', 'production_library_sha256', 'traits')}
protocol.update({
    'purpose': 'Exactly one saved FA19 clean-secant candidate call. No new native FA, model, DT, scorer, backward, MAS/RISE or candidate sweep.',
    'fixed_steps': ['1', '10', '20'], 'CPU_extra_endpoint_guard': 'B2', 'case': 'morehopqa_1', 'layer': 19,
    'strict_results_path': strict_remote + '/results.json',
    'saved_diagnostic_artifact_path': strict_remote + '/diagnostic_coefficients_private.pt',
    'saved_diagnostic_artifact_sha256': artifact['sha256'], 'saved_diagnostic_artifact_bytes': artifact['bytes'],
    'LSE_provenance': 'Reuse strict diagnostic private public_B2_LSE from its actual default FA call. All saved strict dq/dk/dv exactly match original coefficients. No new attention call or reconstructed LSE.',
    'protected_sources': protected, 'scale': sp['native_FA_kwargs']['softmax_scale'],
    'diagnostic_extension_name': extension_name, 'diagnostic_library_name': library_name,
    'row_fields': row_fields,
    'ABI': '13 pointers: q0,k0,q1,k1,v0,u,lse0,lse1,rows,dq,dk,dv,valid_lengths;4int B/H/Hkv/T,float scale,stream. rows=[15,B,H,T]FP32. Separate clean_secant symbol and library; production unchanged.',
    'source_provenance': 'Derived directly from original BF16D256 finite extension with pinned vendor Flash_fwd_kernel_traits/flash::copy/flash::gemm/flash::gemm_rs/row reductions/conversion. Only own finite route rule and row statistics change. No vendor/native FA/header/library patch or shadow forward.',
    'finite_rule': 'For each causal row reconstruct original FP32 p0,p1 from saved endpoint scores/LSE, explicitly normalize phat0=p0/sum(p0),phat1=p1/sum(p1). g=u_BF16*V0^T. s=scale*(Q1K1^T-Q0K0^T),sc=s-mean(s). m1=phat1*(g-phat1.g);b=g.(phat1-phat0);a=(b-m1.s)/||sc||^2;m=m1+a*sc. QK midpoint and raw actual_p1 dv branch remain original.',
    'minimum_norm_proof': 'In exact real arithmetic 1.m1=0. Any feasible update h=m-m1 satisfies1.h=0 and sc.h=b-s.m1. Cauchy-Schwarz gives||h||^2>=(b-s.m1)^2/||sc||^2; equality uniquely at h=((b-s.m1)/||sc||^2)sc forsc!=0. Thus m is the unique Euclidean closest vector to m1 in the zero-row-sum and secant affine constraints. This does not imply PSD/symmetry, arbitrary conditional accuracy, Shapley signs or MAS improvement.',
    'local_reason_and_limit': 'For small same-backend deviations around input, m1 is the first-order response. On directions orthogonal to sc the correction has no effect; along the complete endpoint chord any conserving rule predicts the same target. General directions, B1/B2 drift, QK midpoint errors and whole-network propagation can defeat an advantage.',
    'normalization_scope': 'Route phat0/phat1 is explicitly normalized; exact-real-arithmetic equals native softmax. raw p0/p1 row sums and normalized-minus-raw gDeltaP are emitted. Value dv uses original unnormalized reconstructed actual_p1 generation unchanged and must match saved coefficients. Resulting finite-precision endpoint residual is reported, never assigned to final scores.',
    'stable_variance': 'FP32 s and g are anchored to first causal key. Each score tile computes mean of anchored s and a second centered-square sum, followed by Welford merging across tiles. No E[s^2]-E[s]^2 subtraction, FP64 GPU, epsilon, clipping or denominator scan. The numerator uses centered g/s probability moments in FP32; remaining near-degenerate rounding is reported via variance/numerator/alpha, correction norm and actual row conservation.',
    'degenerate_rule': 'For n=1 route weight is exactly0. If M2=0, a=0; nonzero b is an incompatible numerical constraint residual, explicitly recorded and not called conserved. Finite positive M2 is used without an epsilon. Nonfinite outputs stop with raw candidate tensors preserved; finite but excessive opposite-sign cancellation is not admission evidence.',
    'build_budget': {'compiler_attempts': 1, 'compiler_seconds': 150, 'wall_seconds': 180,
                     'tile_candidates': 1, 'GPU_kernel_calls': 0, 'model_calls': 0},
    'run_budget': {'wall_seconds': 180, 'candidate_finite_calls': 1,
                   'candidate_phase_launches_on_success': 3, 'native_FA_calls': 0,
                   'model_calls': 0, 'DT_calls': 0, 'scorer_calls': 0, 'FT_calls': 0,
                   'backward_calls': 0, 'extra_warmup': 0, 'new_samples': 0},
    'cost_changes': {'GEMM_count': 'Same original3phase endpoint score, g, and multiplier GEMMs; no extra score tiles.',
        'shared_tile_bytes': 32768, 'row_state_count_FP32': 15, 'original_row_state_count_FP32': 2,
        'row_state_bytes_at_fixed_shape': 15 * 16 * 351 * 4,
        'extra_row_state_bytes_at_fixed_shape': 13 * 16 * 351 * 4,
        'Phase0': 'First tile2 anchor row reductions; each tile2 additional vendor quad reductions for tile mean/M2 plus Welford scalar merge;6 final probability/moment row reductions instead of2. Extra live fragments/registers and occupancy effects require measurement.',
        'Phase1': 'Two extra endpoint-credit positive/negative row reductions for diagnosis. Candidate weights retain original BF16 conversion and GEMM.',
        'Phase2': 'Different scalar route weights; same QK midpoint and original raw-P1 value GEMM. No speed equivalence is claimed.',
        'timing_scope': 'One synchronized wrapper call includes preparation/checks; no repeated timing/speed benchmark. CPU ledgers and private I/O separately contribute wall time.'},
    'acceptance': 'Source/dtype/layout/hash invariants; finite values; positive row sums/nonnegative variance; n1 zero route; M2zero alpha0; dv equals old saved coefficients; old CPU ledgers match frozen evidence; coordinate-group sums close1e-7. Numerical endpoint discrepancies remain visible. These checks establish executable observation, not effectiveness or approval for production.',
    'stop': 'One build and one candidate call only. First nonfinite/source/layout/dv/invariant failure ends study, preserving evidence. No fallback, precision change, epsilon tuning, additional sample, method sweep, model call or automatic promotion.',
    'decision_after': 'Assess early/mid signed conditional error, B2/allEOS controls, row normalization residual and cancellation together. Only a justified local improvement could motivate one separate paired full-method official-metric study; this protocol authorizes no such run.',
    'artifact_scope': 'Original1.379GB and previous26.2MB private artifacts stay in their directories. New candidate coefficients/row state stay in private pt excluded from review zip. Public signed groups label coordinate roles, not original-input independent causal effects.',
    'files_sha256': {name: sha(value) for name, value in files.items()}})
files['protocol.json'] = json.dumps(protocol, indent=2).encode()
paths = {'protocol': A / 'dt_fa19_clean_secant_protocol_20260908.json',
         'build': A / 'launch_dt_fa19_clean_secant_build_20260908.json',
         'run': A / 'launch_dt_fa19_clean_secant_run_20260908.json'}
assert not any(path.exists() for path in paths.values()), 'Refuse to overwrite frozen payloads.'
blob = base64.b64encode(zlib.compress(json.dumps({name: base64.b64encode(value).decode()
                                               for name, value in files.items()}).encode())).decode()
loader = ('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path(' + repr(remote) + ');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode(' + repr(blob) + ')));'
    '[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"build_driver.log").open("w");j=subprocess.Popen([' + repr(python) + ',"-B",str(d/"build.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"build.pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d),"stage":"build_only"}))')
run = ('import pathlib,subprocess,json;d=pathlib.Path(' + repr(remote) + ');'
    'assert (d/"build_results.json").is_file() and not (d/"results.json").exists();'
    'assert json.loads((d/"build_results.json").read_bytes())["status"]=="finite_extension_compiled_not_executed";'
    'f=(d/"driver.log").open("w");j=subprocess.Popen([' + repr(python) + ',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d),"stage":"run"}))')
paths['protocol'].write_bytes(files['protocol.json'])
paths['build'].write_text(json.dumps({'cmd': python + ' -c ' + shlex.quote(loader), 'timeout': 10}))
paths['run'].write_text(json.dumps({'cmd': python + ' -c ' + shlex.quote(run), 'timeout': 10}))
print(json.dumps({'protocol_sha256': sha(files['protocol.json']), 'extension_sha256': sha(files[extension_name]),
    'paths': {name: str(path) for name, path in paths.items()}, 'remote_directory': remote,
    'build_budget': protocol['build_budget'], 'run_budget': protocol['run_budget']}))
