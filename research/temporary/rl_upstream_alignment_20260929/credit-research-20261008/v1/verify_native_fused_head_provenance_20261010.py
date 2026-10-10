"""Bind the active head to its accepted patch and original saved owner receipts.

Source/receipt inspection only. This does not run a model or numerical test,
replace an owner implementation, or establish the cause of the NaN incident.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'experiments/rl/current_runtime.json').exists())
RAW = HERE / 'direct-credit-records-20261009-v1'
source_path = RAW / 'native-nonfinite-math-source-1791603249.json'
observation_path = RAW / 'native-actor-observer-observation-1791602977.json'
source = json.loads(source_path.read_bytes())
observation = json.loads(observation_path.read_bytes())
head = next(f for f in source['files'] if f['path'].endswith('experimental/torch_functional.py'))
current = next(f['text'] for f in source['additional_sources'] if f['path'] == head['path'])
accepted_commit = 'dc4e4d7349c26d08eaceaac6d00a57c0e1fbfda7'
patch_relative = 'experiments/rl/patch_actor_fused_head.py'
patch_blob = subprocess.check_output(['git', 'show', accepted_commit + ':' + patch_relative], cwd=REPO)
# Load the accepted owner's existing patch; do not reproduce its transformations.
namespace = {'__name__': 'accepted_owner_patch'}
exec(compile(patch_blob, accepted_commit + ':' + patch_relative, 'exec'), namespace)
expected = namespace['patch_precision'](head['official_source'])
assert expected == current
assert hashlib.sha256(expected.encode()).hexdigest() == head['sha256']
local_patch = (REPO / patch_relative).read_bytes()
assert ast.dump(ast.parse(local_patch), include_attributes=False) == ast.dump(ast.parse(patch_blob), include_attributes=False)

receipt_path = REPO / 'experiments/rl/results_actor_b8.json'
receipt = json.loads(receipt_path.read_bytes())
deployment_path = REPO / 'experiments/rl/results_first_formal_update.json'
deployment = json.loads(deployment_path.read_bytes())
assert deployment['deployment']['actor_head_effective_sha256']['verl/utils/experimental/torch_functional.py'] == head['sha256']
assert receipt['owner']['commit'].startswith(source['upstream_commit'])
historical_paths = [
    'research/temporary/rl_upstream_alignment_20260929/owner-entropy-20260930/final-official-head-case2.log',
    'research/temporary/rl_upstream_alignment_20260929/owner-entropy-20260930/final-fused-temperature1.json',
    'research/temporary/rl_upstream_alignment_20260929/owner-entropy-20260930/materialized-temperature09.json',
    'research/temporary/rl_upstream_alignment_20260929/owner-b8-dispatch-20260930/live-TextCraft-complete.json',
]

def artifact(path):
    blob = path.read_bytes()
    return dict(path=str(path), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())

historical = []
for relative in historical_paths:
    path = REPO / relative
    record = artifact(path)
    assert record['sha256'] == receipt['receipts'][relative]
    record['matches_recorded_sha256'] = True
    historical.append(record)
test_source = source['owner_head_test']
test_assertions = [ast.unparse(n) for n in ast.walk(ast.parse(test_source['text']))
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('assert_close', 'assertEqual')] if test_source is not None else None
report = dict(
    observed_unix=source['unix'], scope=__doc__,
    upstream_commit_from_runtime=source['upstream_commit'],
    upstream_full_commit_from_original_receipt=receipt['owner']['commit'], active_head_path=head['path'],
    active_head_resolved=head['resolved'], active_head_sha256=head['sha256'],
    pristine_head_sha256=head['official_sha256'],
    accepted_patch_commit=accepted_commit, accepted_patch_source_sha256=hashlib.sha256(patch_blob).hexdigest(),
    active_source_equals_accepted_patch_applied_to_pinned_owner=True,
    local_patch_matches_accepted_patch_AST=True,
    head_is_pristine_upstream=False,
    differences=[dict(name=f['name'], diff=f['source_diff']) for f in head['functions'] if not f['matches_pinned_owner_AST']],
    original_head_test=dict(path=test_source['path'], sha256=test_source['sha256'], assertions=test_assertions) if test_source is not None else None,
    original_saved_test_scope=receipt['scope'], original_saved_owner=receipt['owner'],
    historical_receipts=historical,
    current_runtime=dict(formal_pid=observation['formal_pid'], formal_birth=observation['formal_birth'],
        observed_unix=observation['unix'], workers=[dict(pid=w['pid'], birth=w['birth'], phase=w['phase'],
            PSS_bytes=w['PSS_bytes']) for w in observation['workers']]),
    source_artifacts=[artifact(source_path), artifact(observation_path), artifact(receipt_path), artifact(deployment_path)],
    NaN_root_cause_localized=False, repair_deployed=False,
    numerical_version_changed=False, model_calls=0, DT_calls=0, backward_calls=0,
    production_changes=0,
    limitations=[
        'This verifies source identity and the integrity of historical receipts; no numerical test was rerun.',
        'The historical owner-head tests do not prove that the current whole model cannot generate a nonfinite gradient.',
        'The incident at iteration14 lacks saved pre-update actor inputs, parameters, moments and RNG; these cannot be reconstructed from finite iteration18 inputs.',
        'No evidence attributes the NaN incident to the accepted head patch, and this audit does not exclude it as a possible cause.',
        'The frozen source and native initialization log identify this active branch; this inspection did not query live sys.modules or change worker state.',
        'The remote host has no git executable; pristine source revision was not newly checked with git. Exact source SHA and the previously pinned version are recorded.',
        'The copied pristine checkout has no tests/kernels/test_linear_cross_entropy.py file. Historical original-test receipts were checked by SHA, but the test source and its assertions were not newly inspected.',
    ],
)
out = REPO / 'experiments/rl/results_textcraft_fused_owner_provenance_20261010.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(saved=str(out), active_head_sha256=head['sha256'],
    source_matches_accepted_patch=True, historical_receipts_verified=len(historical),
    root_cause_localized=False, production_changes=0), ensure_ascii=False))
