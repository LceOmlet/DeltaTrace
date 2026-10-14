"""Prepare actual full-policy adjoint exchange using only native interfaces.

Two original weights and two actual full-loss decoder31 adjoints form a2x2
derivative comparison on the unchanged original81 complete global64 carrier.
No parameter intervention, credit change, training update or quality test.
"""
import copy
import hashlib
import json
import re
import time
from pathlib import Path

from verl import DataProto

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT / 'native-full-policy-direction-exchange-v86'
original = json.loads((OUT / 'native-PG-driver-components-v38/case-manifest.json').read_bytes())
assert not (DEST / 'job.json').exists() and not (DEST / 'case-manifest.json').exists()
case_by_weight = {c['checkpoint']: c for c in original['cases']}
carrier_path = Path(case_by_weight[40]['path'])
assert case_by_weight[40]['path'] == case_by_weight[80]['path']
assert hashlib.sha256(carrier_path.read_bytes()).hexdigest() == case_by_weight[40]['SHA256']
data = DataProto.load_from_disk(str(carrier_path))
assert len(data) == 64
assert (data.batch['diagnostic_full_DT'] == data.batch['advantages']).all().item()
chain_path = OUT / 'native-upper-adapter-partition-v81/iterationDT80_upper24_31_from40-rank0-chain.json'
chain = json.loads(chain_path.read_bytes())
names = [p['name'] for p in chain['native_accumulation_readout']['parameters']]
assert len(names) == 496
assert all(re.search(r'\.layers\.(\d+)\.', name) and
           int(re.search(r'\.layers\.(\d+)\.', name)[1]) <= 31 for name in names)
cases = []
for step in (40, 80):
    case = copy.deepcopy(case_by_weight[step])
    case.update(name=f'full{step}_native', capture_boundary_seed=True,
                captured_seed_receipt=str(DEST / f'captured-full{step}-seed-bindings.json'))
    cases.append(case)
for step, source_step in ((80, 40), (40, 80)):
    case = copy.deepcopy(case_by_weight[step])
    case.update(name=f'full_weights{step}_seed{source_step}',
                boundary_seed_source_case=f'full{source_step}_native',
                boundary_seed_source_directory=str(DEST),
                boundary_seed_hash_receipt=str(DEST / f'captured-full{source_step}-seed-bindings.json'))
    cases.append(case)
DEST.mkdir(exist_ok=True)
report = dict(unix=time.time(), status='prepared_only_no_GPU_submission', scope=__doc__,
    destination=str(DEST), checkpoint=40, checkpoint_path=case_by_weight[40]['checkpoint_path'], cases=cases,
    diagnostic_advantage_views={'full_DT': 'diagnostic_full_DT'},
    passive_boundary_readout_enabled=True, native_ratio_chain_readout=True,
    decoder_branch_layers=[29, 30, 31], save_native_probability_carrier=True,
    question='Does the measured internal sensitivity growth amplify the actual complete policy-loss directions, and how much does actual B4 directional accumulation change?',
    why_needed='Completedv83 exchanges upper24-31 parameters. The conditional15-target fixed-seed norms still follow the body (1.07808 versus0.08793), but actual full-PG readouts differ (0.78844 versus2.06841). Full-PG B4 direction factors are0.65390/1.43763. This directly prevents using the selected-target seed as a proxy for the complete actual PG. The whole-loss adjoints and original B4 Gram provide the missing direct chain comparison.',
    discriminating_quantity='Original full81 PG at weights40 and80, followed by unchanged whole BF16 decoder31-output adjoints exchanged between these same weights. Original native full-shard PG/VJP, hidden adjoints and B4 increment Gram; current-session natural baselines and actual old-logprob carriers are saved.',
    decision='Distinguish actual loss-direction alignment and B4 cancellation from conditional propagation on one selected-target direction before attributing the increasing full training norm or accepting a repair.',
    counterfactual_scope='Read-only derivatives of the actual original81 complete first global64 at rho1 and original40/80 weights. Not a reconstruction of the historical41 batch or a full training trajectory.',
    no_trainable_parameters_above_decoder31=dict(source=str(chain_path),
        SHA256=hashlib.sha256(chain_path.read_bytes()).hexdigest(), native_trainable_parameter_count=len(names)),
    planned_native_full_policy_derivatives=2, planned_native_fixed_seed_VJPs=2,
    model_training_updates=0, optimizer_updates=0, scheduler_updates=0,
    DT_calls=0, generation_calls=0, validation_score_calls=0, GRPO_calls=0,
    formal_training_changes=0, rootcause_repair_verified=False, accepted_repair=None, deployed_repair=None,
    limits=['Original full256-whitened advantages, source/target/observation masks, IDs, order, B4/global64/1/16 and32k/LoRA8/16 are unchanged. No credit or PPO loss implementation is copied.',
        'The exact complete BF16 seed is captured by the existing native output hook, serialized once, bound by SHA256, and passed unchanged to the existing native replacement hook. No normalization, rescaling or selected-token replacement.',
        'Natural full policy gradients and exchanged-seed VJPs are different observables. The latter are not training updates, operator norms or independent causal fractions.',
        'At weights40 the original81 advantages and inputs are fixed diagnostic data, not the historical41 rollout. This experiment alone cannot prove the entire85-iteration root cause or from-scratch stability.',
        'Native low-precision variation, actual same-point checks and Gram residuals remain recorded. No extra tolerance or all-layer-decrease requirement.'])
path = DEST / 'case-manifest.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(receipt=str(path), report=report)))
