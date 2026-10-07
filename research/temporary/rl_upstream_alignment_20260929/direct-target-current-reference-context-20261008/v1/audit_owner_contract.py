"""Read actual saved owner source and quantify measured credit approximation error.

No model execution, alternative estimator, tolerance, or production change.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
AUDIT = HERE.parents[1]


def binding(path):
    raw = path.read_bytes()
    return dict(path=path.as_posix(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


source_path = AUDIT / 'direct-target-existing-pv-rule-20261008/v1/joint-finite-sources.json'
sources = json.loads(source_path.read_bytes())
files = {item['module']: item for item in sources['files']}
runner = files['qwen35_dense_finite_runner']
assert hashlib.sha256(runner['text'].encode('utf8')).hexdigest() == runner['sha256']
tree = ast.parse(runner['text'])
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name == 'Qwen35DenseFiniteRunner')
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
contract = []
for node in methods:
    contract.append(dict(name=node.name, line=node.lineno,
                         signature=ast.unparse(node.args),
                         docstring=ast.get_docstring(node)))
effect = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_token_effect')
effect_source = ast.get_source_segment(runner['text'], effect)

context_path = REPO / 'experiments/rl/results_current_token_reference_context_20261008.json'
context = json.loads(context_path.read_bytes())
causal_path = REPO / 'experiments/rl/results_credit_causal_diagnosis_20261008.json'
causal = json.loads(causal_path.read_bytes())
assert context['case']['native_sha256'] == context['launch']['case_binding']['native_sha256']

errors = {}
text = causal['selected_actual_tokens']['TextCraft_Format']
app = causal['selected_actual_tokens']['AppWorld_newline']
for name, reward, approximate, measured in [
    ('TextCraft_Format', text['reward'], text['DT_d'], text['native_single_delete_d']),
    ('AppWorld_newline_fresh_replay', app['reward'],
     app['original_symmetric_memory']['d'], app['native_single_delete_d']),
]:
    epsilon = approximate - measured
    measured_A = reward * -math.expm1(-measured)
    approximate_A = reward * -math.expm1(-approximate)
    errors[name] = dict(
        reward=reward, joint_DT_d=approximate, native_single_deletion_d=measured,
        d_error=epsilon,
        joint_DT_implied_probability_ratio=math.exp(-approximate),
        measured_probability_ratio=math.exp(-measured),
        ratio_distortion_factor=math.exp(-epsilon),
        computed_native_A_FP64=measured_A,
        computed_joint_DT_A_FP64=approximate_A,
        coefficient_error=approximate_A-measured_A,
        error_from_identity=reward*math.exp(-measured)*-math.expm1(-epsilon),
        scope='FP64 descriptive arithmetic from stored measurements; not a new numerical reference, acceptance tolerance or production credit.')

ep = context['endpoints']
F, D, B, C = [ep[key] for key in [
    'F_factual_all_sources', 'D_factual_delete_only_candidate',
    'B_all_prior_sources_EOS', 'C_only_candidate_restored']]
algebra = dict(
    joint_factual_vs_all_EOS=F-B,
    candidate_factual_single_deletion=F-D,
    other_sources_factual_group_deletion=F-C,
    sum_of_two_factual_group_deletions=(F-D)+(F-C),
    nonadditivity=(F-D)+(F-C)-(F-B),
    scope='Two groups: this actual newline and the other actual sources. This is not a sum over all individual tokens, a population rate, or a quality metric.')

result = dict(
    status='Actual owner interface and measured approximation scope audited; no algorithm change',
    actual_runner={k:runner[k] for k in ('module','path','sha256')},
    methods=contract, final_contraction=dict(line=effect.lineno, source=effect_source),
    verified_capabilities=[
        'attribute accepts caller-supplied interleaved reference/factual pairs and a fixed target selection.',
        'The same owner can score a factual-versus-single-EOS pair; the existing real diagnostics already did so.',
        'A joint pair returns a vector from one set of finite coefficients contracted with all actual endpoint embedding displacements.',
        'The inspected actual runner class has no separate full-vector factual-single-deletion estimator interface.'],
    inference=[
        'This is not a missing reward input, missing environment simulator, or a failure of the ideal Q/V/A identity.',
        'The current joint finite vector is a joint-intervention decomposition, used by the fixed plan as the practical approximation to individual deletion d.',
        'The current actual four corners show nonadditivity; joint conservation alone therefore cannot establish factual single-deletion accuracy.',
        'The measured large error is not fixed by matching only the global sum or by hiding the exponential response with a multiplier.'],
    error_identity='If epsilon=d_est-d_native, A_est-A_native = r*exp(-d_native)*(1-exp(-epsilon)).',
    measured_coefficient_errors=errors,
    current_actual_two_group_nonadditivity=algebra,
    diagnostics_remain_separate=[
        'Official FA/FLA operator reference tests and assertions.',
        'Original author cumulative deletion, RISE and MAS on actual trajectories.',
        'Actual selected-token native deletion and background-interaction probes.',
        'Actual failed B4 and exact32768 DT memory lifetime regression.'],
    pending_user_decision='Whether to authorize research into bounded native single-deletion refinement beyond the fixed one-joint-DT estimator; no selection rule, threshold, replacement coefficient or new query has been implemented.',
    current_runtime_observation=binding(sorted(HERE.glob('observation-*.json'))[-1]),
    sources=[binding(source_path),binding(context_path),binding(causal_path),
             binding(REPO/'experiments/rl/PLAN.md')],
    operations=dict(model=0, DT=0, backward=0, optimizer=0, rollout=0,
                    formal_restart=False, checkpoint_restore=False,
                    production_modified=False))
(HERE / 'owner-contract-analysis.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(json.dumps(dict(methods=[item['name'] for item in contract],
                     actual_runner_sha256=runner['sha256'],
                     errors=errors, nonadditivity=algebra), ensure_ascii=False))
