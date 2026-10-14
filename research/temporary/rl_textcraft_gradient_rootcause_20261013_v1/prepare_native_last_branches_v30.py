"""Locate the evidenced last-decoder adjoint gain, without changing computation."""
import hashlib
import json
from pathlib import Path
import time

OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
previous=json.loads((OUT/'matched-history-late-weights-v28/case-manifest.json').read_bytes())
state=json.loads((OUT/'matched-history-late-weights-v28/result.json').read_bytes())
assert state['phase']=='complete_native_gradient_chain_localization_not_repair'
owner=json.loads((OUT/'native-decoder-owner-v29.json').read_bytes())
assert owner['owner_SHA256']=='59f9c339e3c01672b67ba09b0e56e2e960ba28c6c3d9975faadf00d15ee0b4c8'
DEST=OUT/'matched-last-decoder-branches-v30'
report=dict(previous)
report.update(unix=time.time(),scope=__doc__,destination=str(DEST),decoder_branch_layers=[31],
    question='Which native last decoder residual branch accounts for the measured late-history versus early-history adjoint gain at checkpoint80?',
    why_needed='Completedv28 has overlapping output-score norm bounds, final-hidden late/early ratio0.917, but parameter-gradient ratio4.34354. Decoder31 output-adjoint ratio is1.219, input-adjoint ratio4.037, so a material first amplification is inside that actual block. Native owner source shows attention and MLP residual branches; total input Tensor hooks alone cannot distinguish branch gradients. CPU checkpoint geometry also shows the dominant LoRA B gradient cannot be explained by a uniform LoRA A norm increase (max operator-norm ratio1.0167).',
    limits=previous['limits']+[
        'Only the actual final decoder attention/MLP/norm branches are added to the existing passive readout. No operator, gradient or loss is replaced.',
        'PyTorch native full backward hooks may wrap module inputs/outputs in views. Numerical differences fromv28 must be reported; no bitwise identity or new tolerance is assumed.',
        'A Tensor hook on a shared residual input gives the total adjoint, not the module branch grad_input. This job records both with distinct labels.',
        'Norm identities can describe residual cancellation but do not by themselves prove a kernel defect, the85-iteration root cause, or a repair.'])
for c in report['cases']:assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['SHA256']
DEST.mkdir(exist_ok=True)
path=DEST/'case-manifest.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
