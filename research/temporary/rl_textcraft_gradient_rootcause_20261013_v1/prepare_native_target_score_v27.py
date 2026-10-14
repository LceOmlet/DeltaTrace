"""Bind original native probability readout to the completed v20/v23 inputs."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
previous=json.loads((OUT/'matched-action-observation-target-derivatives-v20/case-manifest.json').read_bytes())
DEST=OUT/'matched-target-score-readout-v27'
cases=previous['cases']
assert [c['name'] for c in cases]==['early','late']
for c in cases:assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['SHA256']
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),cases=cases,checkpoint=5,
    score_readout_only=True,passive_boundary_readout_enabled=False,
    expected_trainable_SHA256_by_rank={
        '0':'964c46365a4045fe1f107028820cf88392d75534a7bdc3f98adb00c2df664f27',
        '1':'60404f8692dfb5ebf6835c1407a1187a0792271c86c914ab347055a87750e75c'},
    question='How much of the measured final-hidden adjoint ratio10.8628 is already present in the output-logit score force on the same15 matched histories/common coefficients?',
    why_needed='Native v23 measured a28.9587x parameter-gradient ratio,10.8628x final-hidden adjoint ratio and2.66586x remaining G/H ratio. Native lm_head.forward is bypassed by the original fused path, so logit-score magnitude is missing. Original compute_log_prob on the identical factual carriers supplies target probabilities and mathematical score-norm bounds without another backward or a shadow head.',
    limits=[
        'Same15 saved successful factual pairs, checkpoint5 and original v20 carriers; no new or altered input, generation or credit.',
        'This job calls only original initialization/checkpoint restoration and compute_log_prob; no parameter backward, optimizer, scheduler, DT, validation or GRPO call.',
        'Target probabilities bound the logit-score norm between the coefficient-weighted deficit norm and sqrt(2) times that norm; the complete vocabulary distribution is not retained.',
        'Bounds account for possible score magnitude, not independent causal shares, parameter gradients, or the full85-iteration mechanism.',
        'The readout uses original native precision. Rounding and differences from previously instrumented runs must be retained.'],
    accepted_repair=None,deployed_repair=None)
DEST.mkdir(exist_ok=True)
path=DEST/'case-manifest.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
