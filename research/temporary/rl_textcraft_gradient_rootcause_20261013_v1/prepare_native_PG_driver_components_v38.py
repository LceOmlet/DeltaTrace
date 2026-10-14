"""Prepare a bounded decomposition of the actual DT policy gradient, not DT error.

Use the already prepared original Q/V/mean coefficients and original inputs.
Native VERL supplies all losses/backward. Read its unchanged optimizer moment
to relate the current component directions to actual recent gradient history.
"""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-PG-driver-components-v38'
prepared=json.loads((OUT/'native-DT-QV-gradient-components-v31/case-manifest.json').read_bytes())
mid=json.loads((OUT/'native-midpoint-fixed-seed-v35/case-manifest.json').read_bytes())
carrier=prepared['cases'][0]
assert hashlib.sha256(Path(carrier['path']).read_bytes()).hexdigest()==carrier['SHA256']
cases=[]
for cp,source in [(40,mid),(80,prepared)]:
    spec=source['cases'][0]
    digests=spec.get('expected_trainable_SHA256_by_rank',source.get('expected_trainable_SHA256_by_rank'))
    cases.append(dict(name=f'original81_QV_weights{cp}',path=carrier['path'],SHA256=carrier['SHA256'],rows=64,
                      checkpoint=cp,checkpoint_path=spec.get('checkpoint_path',source['checkpoint_path']),
                      expected_trainable_SHA256_by_rank=digests))
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),checkpoint=40,
            checkpoint_path=mid['checkpoint_path'],cases=cases,
            diagnostic_advantage_views=prepared['diagnostic_advantage_views'],
            native_optimizer_moment_readout=True,passive_boundary_readout_enabled=False,
            original_native_whitening=prepared['actual_native_whitening'],
            prepared_coefficient_norms=prepared['native_CPU_loss_coefficient_norms'],
            question='In the saved actual DT policy gradient, does the factual sampled-return Q term itself carry the growing norm, or does the subtracted V/common-centering combination create it, and which directions align with the original recent optimizer gradient history?',
            why_needed='Completed fixed-input/fixed-seed native VJPs establish late internal sensitivity growth0.0604 at5,0.0863 at40,1.1039 at80. Native same-reward-group and target partitions already localize the large late DT signal. They do not distinguish the sampled-return term from its actual correction terms. CPU coefficient norms Q0.02600,V0.01609,center0.00512,DT0.01721 cannot answer the parameter-gradient direction/cancellation question. These existing loss components provide the direct missing driver evidence without estimating DT/world error, changing credit, studying shortening, or running a new training trial.',
            discriminating_quantity='Four original native PG component gradients and their Gram at each original weight40/80, on identical saved iteration81 inputs/frozen original Q/V/mean coefficients; read-only dot products with the actual loaded native optimizer exp_avg.',
            limits=prepared['limits']+[
                'Original iteration81 data and coefficient snapshot are fixed at both weights. At40 this is an off-policy derivative diagnostic, not a reconstructed historical iteration41 update.',
                'Native exp_avg is a past clipped-gradient EMA, not an exact parameter-update direction or proof that a current component caused the entire weight trajectory.',
                'The center component is included only for the additive loss identity. No whitening-to-shortening causal chain or credit-quality experiment is evaluated.',
                'The old v31 preparation remains unsubmitted; only its immutable CPU coefficient artifact is reused here.'],
            accepted_repair=None,deployed_repair=None)
DEST.mkdir(exist_ok=True)
path=DEST/'case-manifest.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
