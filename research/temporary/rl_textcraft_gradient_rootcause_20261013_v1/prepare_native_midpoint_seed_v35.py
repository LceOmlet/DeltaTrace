"""Anchor the measured internal VJP growth to an existing intermediate checkpoint.

One checkpoint40 fixed-input VJP, reusing the exact completed v33 checkpoint80
block31-output seed. No training, validation, alternate credit, or GRPO run.
"""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-midpoint-fixed-seed-v35'
seed_root=OUT/'native-fixed-seed-propagation-v33'
assert json.loads((seed_root/'result.json').read_bytes())['phase']=='complete_native_gradient_chain_localization_not_repair'
previous=json.loads((seed_root/'case-manifest.json').read_bytes())
carrier=previous['cases'][0]
shared=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
checkpoint=shared/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints/global_step_40/actor'
digests={}
for rank in (0,1):
    state=json.loads((OUT/f'historical-target-derivatives-v1/iteration41-mini0-rank{rank}-state.json').read_bytes())
    assert state['trainable_SHA256_before']==state['trainable_SHA256_after']
    assert state['scheduler_before']==state['scheduler_after']==40
    digests[str(rank)]=state['trainable_SHA256_before']
hashes={}
for rank in (0,1):
    for micro in range(8):
        path=seed_root/f'seed-late80_native-rank{rank}-micro{micro}.pt'
        hashes[f'{rank}:{micro}']=hashlib.sha256(path.read_bytes()).hexdigest()
case=dict(name='late40_seed80',path=carrier['path'],SHA256=carrier['SHA256'],rows=64,
          checkpoint=40,checkpoint_path=str(checkpoint),expected_trainable_SHA256_by_rank=digests,
          boundary_seed_source_case='late80_native',boundary_seed_source_directory=str(seed_root),
          boundary_seed_SHA256_by_rank_micro=hashes)
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),checkpoint=40,
            checkpoint_path=str(checkpoint),cases=[case],decoder_branch_layers=[29,30,31],
            passive_boundary_readout_enabled=True,native_ratio_chain_readout=True,
            question='Did the fixed-input/fixed-incoming-seed internal sensitivity increase occur mainly before checkpoint40 or during the later interval that contains the steep gradient growth?',
            why_needed='Completedv33 measures parameter VJP0.06040356 at5 versus1.10390448 at80, using identical input and the same actual BF16 block31-output seed. That18.2755x endpoint contrast establishes internal geometry/state change but not its time course. Historical native grad medians remain0.0685 around31-40 and rise to0.335/0.502 around71-85. Original GRPO and DT both have roughly8x LoRA-B norm growth from5 to40, so generic parameter accumulation alone is not specific. One existing checkpoint40 VJP is the missing temporal anchor, not a new training trial.',
            discriminating_quantity='Original checkpoint40 native VJP norm and29/30/31 branch adjoints under the exact same v33 input and unchanged checkpoint80 seed, placed between the completed5/80 measurements.',
            limits=['One finite native VJP, not a PPO training update or validation test.',
                    'No new input, advantage, seed direction, rescaling, model implementation, optimizer/scheduler advance, DT, generation or GRPO call.',
                    'The existing historical checkpoint40/trainable digest and completed native seed files define identity. Recorded low-precision variation is retained without a new tolerance.',
                    'Three fixed-carrier checkpoints locate timing; they do not prove a universal functional law or complete85-step cause.'],
            accepted_repair=None,deployed_repair=None)
DEST.mkdir(exist_ok=True)
path=DEST/'case-manifest.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
