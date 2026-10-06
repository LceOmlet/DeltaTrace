"""Compose frozen formal preparation with the existing accepted query repair.

Only a new entry copy is prepared. Neither original entry, VERL, runtime
manifest nor a training process is changed.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import prepare_textcraft_whitened_formal as prior
from stage_environment_entry import AUDIT, REPO, ROOT, SCP, SSH

OUT = ROOT + '/receipts/textcraft-official-whitening-formal-20261006-v2'
LOCAL = AUDIT / 'textcraft-degradation-20261005/official-whitening-formal-20261006/v2'
ENTRY = ROOT + '/candidates/textcraft-official-whitening-formal-20261006-v2/entry'
FORMAL_OUTPUT = ROOT + '/runs/textcraft-official-whitening-20261006-v2/textcraft-dt'
QUERY = ROOT + '/receipts/textcraft-query-clock-gradient-20261006-v2/reward_readout_clock_candidate.py'
QUERY_SHA = '94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
PRIOR_SHA = 'a2e79d15f72f3f93f04530b024ba69ae4455d98e8e65466404b1d69d3409a4ca'


def preparation_script():
    assert hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest() == PRIOR_SHA
    original = (prior.OUT, prior.FORMAL_OUTPUT)
    try:
        prior.OUT, prior.FORMAL_OUTPUT = OUT, FORMAL_OUTPUT
        script = prior.preparation_script()
    finally:
        prior.OUT, prior.FORMAL_OUTPUT = original
    result = json.loads((REPO / 'experiments/rl/results_textcraft_learning_degradation_20261006.json').read_text(encoding='utf-8'))
    semantic = result['prepared_query_clock_semantic_repair']
    assert semantic['production_source']['sha256'] == QUERY_SHA
    completed = result['native_query_clock_gradient_observation']['execution']['completed']
    references = [semantic['receipt'], semantic['independent_review'], completed]
    for item in references:
        assert hashlib.sha256((REPO / item['path']).read_bytes()).hexdigest() == item['sha256']
    evidence = dict(source_patch_commit='afe59dd08354e71052b8391e7d6ae57b98c70eb4',
        query_source=dict(path=QUERY,sha256=QUERY_SHA),existing_receipts=references,
        matched_queries=semantic['matched_native_candidate_queries'],
        existing_CPU_tests=semantic['adapter_unit_tests_passed'],
        owner_contract=dict(PLAN_lines=[52,59,162],
            native_author_sha256='04148278478913787dc2f69d364c03d66eee50b586ef62595f73ae079cd54897',
            native_author_lines=[234,246,264,275],
            producer_sha256='0ad37a17aede30089fd2ac9a609a42689e6a3a68d00b601520db0a5cff8b8e6e',
            producer_lines=[383,392]),
        scope='Existing two-clause semantic repair aligns the return forecast with execution of the already emitted response. Not a demonstrated weak-PG or learning repair.',
        additional_source_difference='Full 94a source also contains the existing optional prefix_lease_factory=None interface. Frozen TextCraft producer supplies no such argument, so the default original path is used.')
    adapter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    marker = "original=payload['original'];white=payload['white'];root=Path("
    assert script.count(marker) == 1
    overlay = r'''
import shutil
baseline_entry=Path(payload['original']['entry']);candidate_entry=Path(@ENTRY@)
query=Path(@QUERY@);assert sha(query)==@QUERY_SHA@
for name,expected in payload['original']['entry_sha256'].items():
 assert sha(baseline_entry/name)==expected,name
assert not candidate_entry.exists(), 'Preserve each separately prepared entry'
shutil.copytree(baseline_entry,candidate_entry,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(query,candidate_entry/'reward_readout.py')
source_changes={name:dict(original=expected,candidate=sha(candidate_entry/name))
 for name,expected in payload['original']['entry_sha256'].items()
 if sha(candidate_entry/name)!=expected}
assert source_changes=={'reward_readout.py':dict(
 original='228afbc7a10841d482c3d73def59dfe9ef192c057a97d76dca57369502a10137',candidate=@QUERY_SHA@)}
payload['original']['entry']=str(candidate_entry)
payload['original']['entry_sha256']['reward_readout.py']=@QUERY_SHA@
payload['semantic_repair']=json.loads(@EVIDENCE@)
payload['semantic_repair']['baseline_entry']=str(baseline_entry)
payload['semantic_repair']['entry_source_changes']=source_changes
payload['semantic_repair']['composition_source_sha256']=@ADAPTER_SHA@
'''
    overlay = (overlay.replace('@ENTRY@', repr(ENTRY)).replace('@QUERY@', repr(QUERY))
        .replace('@QUERY_SHA@', repr(QUERY_SHA)).replace('@EVIDENCE@', repr(json.dumps(evidence)))
        .replace('@ADAPTER_SHA@', repr(adapter_sha)))
    script = script.replace(marker, overlay + '\n' + marker)
    script = script.replace('import verl.utils.torch_functional as functional',
        'import verl.utils.torch_functional as functional\nimport reward_readout as readout')
    modules = "('trainer',trainer),('official_helper',functional)]"
    assert modules in script
    script = script.replace(modules, "('trainer',trainer),('official_helper',functional),('reward_readout',readout)]")
    checked = "assert inspection['imports']['trainer']['sha256']==white['trainer_sha256']"
    assert checked in script
    script = script.replace(checked, checked + "\nassert inspection['imports']['reward_readout']==dict(path=str(entry/'reward_readout.py'),sha256=" + repr(QUERY_SHA) + ')')
    record = "input_receipts=payload['input_receipts'],entry=str(entry)"
    assert record in script
    script = script.replace(record,
        "input_receipts=payload['input_receipts'],semantic_repair=payload['semantic_repair'],entry=str(entry)")
    ast.parse(script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    return script


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    script = preparation_script()
    (LOCAL / 'prepare.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL / 'prepare.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}' for name in
        ('prepared.json','native-interface-inspection.json','inspection.stdout.txt',
         'effective-config.yaml','launch-plan.json')] + [str(LOCAL)], check=True)
