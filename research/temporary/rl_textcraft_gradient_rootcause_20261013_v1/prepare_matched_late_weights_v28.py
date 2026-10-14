"""Use the same matched facts/common coefficients at the original late weights."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
previous=json.loads((OUT/'matched-action-observation-target-derivatives-v20/case-manifest.json').read_bytes())
score=json.loads((OUT/'matched-target-score-readout-v27/native-target-score-accounting.json').read_bytes())
DEST=OUT/'matched-history-late-weights-v28'
cases=previous['cases']
for c in cases:assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['SHA256']
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),cases=cases,checkpoint=80,
    checkpoint_path=str(ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints/global_step_80/actor'),
    passive_boundary_readout_enabled=True,save_native_probability_carrier=True,
    expected_trainable_SHA256_by_rank={
        '0':'8ceb690d8b62f80db7d921f2e05be155efedb2ada9c960174b56bafee7abcd68',
        '1':'5f19de20ccfc55aa9f85db5fa9bcd152bea18d7bb3dc00b6bf18445c236a9182'},
    question='Does the controlled28.27x target-gradient amplification persist at the actual late weights, or is it mostly old-policy confidence on late histories?',
    why_needed='Original checkpoint5 native score readout on the same15 pairs gives late/early logit-force norm bounds6.664-13.328, compatible with the measured10.8628x hidden adjoint. Action/get dominate the late score-force lower-bound energy; some late Action tokens have logp-13.59 at checkpoint5. This directly exposes an old-weight versus actual late-weight ambiguity in using the28.27x result as the historical mechanism. Existing same-late-input cross-checkpoint results use different, unmatched carriers, so they do not resolve this contrast.',
    previous_score_ratio_bounds=score['late_over_early_score_norm_bounds'],
    limits=[
        'Identical15 original factual pairs and common coefficients fromv20/v23; no new, shortened, regenerated or intervened input.',
        'Two existing original PG backward passes and original compute_log_prob, at checkpoint80. Optimizer/scheduler do not step; no DT, generation, validation, GRPO model or training call.',
        'This tests a specific ambiguity exposed by the actual probability magnitudes, not a new candidate credit method or validation-quality effect.',
        'Early inputs are off-policy relative to checkpoint80. A fixed-weight derivative contrast is not a full historical optimizer update or a prediction of fresh training.',
        'Parameter norms and output-force bounds have distinct scopes. The finite contrast alone cannot establish the85-iteration root cause or accept a repair.'],
    accepted_repair=None,deployed_repair=None)
DEST.mkdir(exist_ok=True)
path=DEST/'case-manifest.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
