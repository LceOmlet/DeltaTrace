"""Separate measured internal propagation change from changed output direction.

Use one saved fixed late-input carrier and its common original coefficients.
Two original checkpoint native PG passes, plus one explicitly diagnostic VJP
using checkpoint80's unchanged decoder31-output seed at checkpoint5.
No optimizer/scheduler update, alternate advantage, model/kernel implementation,
generation, DT estimation, validation, or GRPO call.
"""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-fixed-seed-propagation-v33'
prior=json.loads((OUT/'matched-action-observation-target-derivatives-v20/case-manifest.json').read_bytes())
carrier=next(c for c in prior['cases'] if c['name']=='late')
assert hashlib.sha256(Path(carrier['path']).read_bytes()).hexdigest()==carrier['SHA256']
shared=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
relative=Path('runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints')
checkpoints={5:shared/relative/'global_step_5/actor',80:ROOT/relative/'global_step_80/actor'}
digests={5:{'0':'964c46365a4045fe1f107028820cf88392d75534a7bdc3f98adb00c2df664f27',
            '1':'60404f8692dfb5ebf6835c1407a1187a0792271c86c914ab347055a87750e75c'},
         80:{'0':'8ceb690d8b62f80db7d921f2e05be155efedb2ada9c960174b56bafee7abcd68',
             '1':'5f19de20ccfc55aa9f85db5fa9bcd152bea18d7bb3dc00b6bf18445c236a9182'}}
cases=[]
for name,cp,capture,source in [('late80_native',80,True,None),('late5_native',5,False,None),
                               ('late5_seed80',5,False,'late80_native')]:
    cases.append(dict(name=name,path=carrier['path'],SHA256=carrier['SHA256'],rows=64,
                      checkpoint=cp,checkpoint_path=str(checkpoints[cp]),
                      expected_trainable_SHA256_by_rank=digests[cp],capture_boundary_seed=capture,
                      boundary_seed_source_case=source))
report=dict(unix=time.time(),scope=__doc__,destination=str(DEST),checkpoint=80,
            checkpoint_path=str(checkpoints[80]),cases=cases,decoder_branch_layers=[29,30,31],
            passive_boundary_readout_enabled=True,native_ratio_chain_readout=True,
            question='Does the observed8.09494x relative directional gain through upper decoder29-31 persist when the exact same actual decoder31-output adjoint is used at both weights?',
            why_needed='Savedv23/v28 records already hold input and loss coefficients fixed. Checkpoint80 reduces final-hidden force to0.36256x but increases native parameter G to3.33821x. Upper31/30/29 relative directional gains are2.25897/2.06272/1.73725; upper input RMS medians change only3-4%. Norm quotients cannot distinguish changed derivative geometry from changed incoming direction. The exact native seed intervention fills that specific gap, rather than testing why histories shortened or whether one update improves validation.',
            discriminating_quantity='The actual unchanged checkpoint80 block31-output BF16 adjoint, reused at checkpoint5 on the same original input, with native input/output adjoints and residual branch module grad_input/output at29/30/31; ordinary native passes retain original parameter PG norms.',
            limits=['Two original native PG passes and one diagnostic fixed-seed VJP only. The latter is not a complete policy-loss gradient above block31 and is not used for training.',
                    'Inputs, masks, common loss coefficients, LoRA8/16,32k capacity setting,B4 and native owners remain unchanged. Checkpoint loads restore original states; no optimizer or scheduler advances.',
                    'Identical output seed isolates the incoming direction. Different weights also induce different internal forward states; this does not distinguish parameter geometry from activation geometry by itself.',
                    'The entire actual native seed is reused without scaling, normalization, clipping, token selection, or approximate reconstruction.',
                    'Passive module hooks wrap native views; measured low-precision variation is retained, without invented tolerances.',
                    'The bounded15 successful facts and old checkpoints cannot alone establish the full85-step root cause or accept a repair.',
                    'No DT accuracy, credit-quality, history-shortening, rollout, validation or GRPO experiment.'],
            accepted_repair=None,deployed_repair=None)
DEST.mkdir(exist_ok=True)
path=DEST/'case-manifest.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
