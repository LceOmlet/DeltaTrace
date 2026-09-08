"""Freeze one supported-secant candidate build/run; never execute remotely."""
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
extension_name = 'vendor_fa_finite_p1_supported_secant_20260908.cu'
library_name = 'libdeltatrace_fa19_supported_secant_20260908.so'
files = {'study.py': (R / 'research/reproduction_templates/dt_fa19_supported_secant_20260908.py').read_bytes(),
         'build.py': (R / 'research/reproduction_templates/dt_fa19_supported_secant_build_20260908.py').read_bytes(),
         extension_name: (R / 'research/prototypes' / extension_name).read_bytes(),
         'vendor_fa_supported_secant_20260908.py': (R / 'research/runtime/vendor_fa_supported_secant_20260908.py').read_bytes(),
         'vendor_fa_finite_bf16_d256.py': (R / 'research/runtime/vendor_fa_finite_bf16_d256.py').read_bytes()}
assert sha(files['vendor_fa_finite_bf16_d256.py']) == sp['files_sha256']['vendor_fa_finite_bf16_d256.py']
for name, value in files.items():
    if name.endswith('.py'):
        ast.parse(value, filename=name)
tree = ast.parse(files['vendor_fa_supported_secant_20260908.py'])
row_fields = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == 'ROW_FIELDS' for t in node.targets)))
assert len(row_fields) == 21
remote = '${ARTIFACT_ROOT}/codex_dt_fa19_supported_secant_20260908_v1'
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
    'purpose': 'Exactly one saved FA19 supported-secant candidate call. No new native FA, model, DT, scorer, backward, MAS/RISE or candidate sweep.',
    'fixed_steps': ['1', '10', '20'], 'CPU_extra_endpoint_guard': 'B2', 'case': 'morehopqa_1', 'layer': 19,
    'strict_results_path': strict_remote + '/results.json',
    'saved_diagnostic_artifact_path': strict_remote + '/diagnostic_coefficients_private.pt',
    'saved_diagnostic_artifact_sha256': artifact['sha256'], 'saved_diagnostic_artifact_bytes': artifact['bytes'],
    'LSE_provenance': 'Reuse strict diagnostic private public_B2_LSE from its actual default FA call. All saved strict dq/dk/dv exactly match original coefficients. No new attention call or reconstructed LSE.',
    'protected_sources': protected, 'scale': sp['native_FA_kwargs']['softmax_scale'],
    'diagnostic_extension_name': extension_name, 'diagnostic_library_name': library_name,
    'row_fields': row_fields,
    'ABI': '13 pointers: q0,k0,q1,k1,v0,u,lse0,lse1,rows,dq,dk,dv,valid_lengths;4int B/H/Hkv/T,float scale,stream. rows=[21,B,H,T]FP32. Separate supported_secant symbol and library; production unchanged.',
    'source_provenance': 'Direct derivative of the previously compiled isolated clean-secant source, itself derived from pinned original BF16D256 finite extension. Reuses unmodified vendor Flash_fwd_kernel_traits/flash::copy/flash::gemm/flash::gemm_rs/row reductions/conversion. Only own finite rule and row scalar statistics change. No native/vendor header/library patch or shadow attention forward.',
    'finite_rule': 'lp_i=FP32(score_i*scale-native_saved_LSE_i);r_i=exp2(lp_i*log2e);Z_i=sum(r_i);phat_i=r_i/Z_i. g=u_BF16*V0^T;s=scale*(Q1K1^T-Q0K0^T). a=phat1*(g-phat1.g);b=g.(phat1-phat0);d=phat1-phat0. For x=abs(lp1-lp0),D_raw=sum(max_probability_by_lp*(-expm1(-x))*x). kappa=(b-a.s)/D_raw when positive,otherwise0;m=a+kappa*d. QK midpoint and original raw actual_p1 dv remain unchanged.',
    'FP32_actual_definition': 'Probability sums and base/target moments use original FP32 vendor reductions. g and s moments are anchored at first valid causal key, as in prior candidate; a and d are recomputed from the same score tiles and stored reciprocal row sums. D_raw uses the original expm1f stable logmean primitive without division by a small log gap; each term is nonnegative. Dnorm=sum(d*s),sum(m),sum(d),actual sum(m*s) and positive/negative contractions are independently saved in Phase1. No Dnorm-based correction is fed back. All residual beyond the denominator formula is explicitly labeled remaining FP32 arithmetic error.',
    'ideal_minimum_metric_proof': 'For ideal normalized softmax endpoints let w=LM(p0,p1),W=sum(w),mu=w.s/W. Then w*(s-mu)=DeltaP, and D=s.DeltaP=KL(p1||p0)+KL(p0||p1). The unique minimizer of sum(h_j^2/w_j) subject1.h=0,s.h=b-a.s is h=((b-a.s)/D)*DeltaP. This follows weighted Cauchy-Schwarz on the zero-sum subspace. a=J(p1)g. This generally differs from the original LM multiplier; two effective keys leave no nontrivial alternative conserving rule.',
    'support_bound_proof': 'For F(t)=E_softmax(z0+t*s)[g],b-a.s=-integral_0^1 t*Fsecond(t)dt;|Fsecond(t)|<=osc(g)*Var_t(s),D=integral_0^1 Var_t(s)dt. Hence |kappa|<=osc(g),and |h_j|<=osc(g)|DeltaP_j|<=osc(g)(p0_j+p1_j). This is a mathematical support bound, not evidence that rare-key spread caused prior NI regression.',
    'normalization_scope': 'In ideal row normalization D_raw=Dnorm=Jeffreys, so the rule is the stated unique projection. With actual raw masses Z_i, exact-real algebra gives D_raw=Z1*KL(phat1||phat0)+Z0*KL(phat0||phat1)+(Z1-Z0)*log(Z1/Z0)>=min(Z0,Z1)*Dnorm. Therefore default-precision raw D is deliberately NOT called exact normalized Jeffreys. If other arithmetic were exact the residual is (b-a.s)*(Dnorm/D_raw-1);actual row residual and the remaining arithmetic discrepancy are both reported. The ideal |kappa| bound and mass-corrected osc(g)/min(Z0,Z1) bound are audited with no clipping.',
    'stable_denominator': 'Sum nonnegative exp(max(lp0,lp1))*(-expm1(-abs_delta_lp))*abs_delta_lp in FP32; max probability uses the already reconstructed endpoint selected by lp, not an extra exponentiation. No subtraction of two second moments/KL means, weighted Welford, new phase/score pass/GEMM, FP64 GPU, epsilon or denominator scan.',
    'degenerate_rule': 'n=1 route weight is exactly0. D_raw==0 sets kappa0; preserve numerator and actual endpoint incompatibility residual. Positive finite D_raw is used without epsilon. Nonfinite row/output values stop after private tensors are saved. An excessive signed cancellation or endpoint residual is not accepted as method evidence merely because a sum is small.',
    'local_reason_and_limit': 'The candidate keeps native-input local response while bounding correction to probability change at the two endpoints in the ideal rule. The previous Euclidean rule lacks this bound; its NI full-pilot regression does not establish rare-key spreading as the empirical cause. This unique replacement is justified by the constrained geometry, not post-hoc direction/layer/threshold selection. Conditional accuracy, whole-method RISE/MAS, speed and promotion are not guaranteed.',
    'build_budget': {'compiler_attempts': 1, 'compiler_seconds': 150, 'wall_seconds': 180,
                     'tile_candidates': 1, 'GPU_kernel_calls': 0, 'model_calls': 0},
    'run_budget': {'wall_seconds': 180, 'candidate_finite_calls': 1,
                   'candidate_phase_launches_on_success': 3, 'native_FA_calls': 0,
                   'model_calls': 0, 'DT_calls': 0, 'scorer_calls': 0, 'FT_calls': 0,
                   'backward_calls': 0, 'extra_warmup': 0, 'new_samples': 0},
    'cost_changes': {'GEMM_count': 'Same original3phase endpoint score,g,and multiplier GEMMs; no new score tile or native attention call.',
        'shared_tile_bytes': 32768,'row_state_count_FP32':21,'original_row_state_count_FP32':2,
        'row_state_bytes_at_fixed_shape':21*16*351*4,'extra_row_state_bytes_at_fixed_shape':19*16*351*4,
        'Phase0': 'First-tile2 anchor reductions;7 final sum reductions for probability/base moments and positive D_raw,2 final max reductions for osc(g). Nonnegative expm1 scalar denominator; Euclidean tile-mean/M2 Welford removed. Additional register lifetime/occupancy effects are not assumed free.',
        'Phase1': 'Seven row sum reductions for signed endpoint contractions,Dnorm,row sum,direction sum,correction signs. Both raw endpoint probabilities needed for normalized DeltaP. Original BF16 multiplier conversion and GEMM unchanged.',
        'Phase2': 'Both raw probabilities needed for DeltaP; same QK midpoint and original raw-P1 dv GEMM. No speed equivalence claim.',
        'timing_scope': 'One synchronized call includes wrapper prepare/checks; no warmup/repeat or production speed claim. CPU fixed ledgers and private I/O contribute wall time.'},
    'acceptance': 'Source/dtype/layout/hash invariants;finite values;positive row masses/nonnegative D_raw and osc(g);n1 zero route;D_raw0 implies kappa0;dv bitwise original;old CPU ledgers match frozen evidence;coordinate-group sums close1e-7. Preserve all nonzero bound/constraint/rounding discrepancies for interpretation; passing executable checks is not effectiveness.',
    'stop': 'One build and one candidate call only. First nonfinite/source/layout/dv/invariant failure ends study, preserving evidence. No fallback, precision change, epsilon tuning, additional sample, method sweep, model call or automatic promotion.',
    'decision_after': 'Assess early/mid signed conditional error, B2/allEOS controls, actual row constraint residual,Dnorm/D_raw,kappa support bounds and cancellation together. Only a justified local improvement could motivate one separate paired full-method official-metric study; this protocol authorizes no such run.',
    'previous_candidate_source_sha256':sha((R/'research/prototypes/vendor_fa_finite_p1_clean_secant_20260908.cu').read_bytes()),
    'artifact_scope': 'Original1.379GB and previous26.2MB private artifacts stay in their directories. New candidate coefficients/row state stay in private pt excluded from review zip. Public signed groups label coordinate roles, not original-input independent causal effects.',
    'files_sha256': {name: sha(value) for name, value in files.items()}})
files['protocol.json'] = json.dumps(protocol, indent=2).encode()
paths = {'protocol': A / 'dt_fa19_supported_secant_protocol_20260908.json',
         'build': A / 'launch_dt_fa19_supported_secant_build_20260908.json',
         'run': A / 'launch_dt_fa19_supported_secant_run_20260908.json'}
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
