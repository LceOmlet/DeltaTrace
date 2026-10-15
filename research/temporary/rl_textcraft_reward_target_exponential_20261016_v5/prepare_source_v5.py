"""Small v5 seams over the actual efficient v3/v4 owner snapshots.

No numerical DT or official GRPO/PPO implementation is copied or changed.
The native DT tree is inherited without the withdrawn affine seed extension.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREVIOUS = HERE.parent / 'rl_textcraft_weighted_reward_targets_20261015_v4'


def replace(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


manifest = {}
parents = {
    'textcraft_owner_rollout.py':'af2aa6c1752d02249dafad726b8feab1f7e5e1050a89b3b2c90c81792dfe5f1e',
    'reward_readout.py':'131989164f94cc7ef13814caa27deef59c0a11ecf968761265d38bea8ebac473',
    'counterfactual.py':'4efb4c0e88fff4591e117e7d6efd297acc3f9d74dfbcd7b1c15e807c67d2a723',
    'dt_training_batch.py':'61cacb7957d912040c12e7282c42d5d6fd17376e97051c78bfa5e3d103ef0b46',
}
for name in ('textcraft_owner_rollout.py', 'reward_readout.py', 'counterfactual.py', 'dt_training_batch.py'):
    source = PREVIOUS / 'candidate-source' / name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==parents[name],name
    text = source.read_text(encoding='utf-8')
    if name == 'textcraft_owner_rollout.py':
        text = replace(text,
            "        weighted = os.environ.get('DT_CREDIT_MODE') == 'weighted_reward_target_finite_v4'",
            "        reward_settlement_targets = os.environ.get('DT_CREDIT_MODE') in (\n"
            "            'weighted_reward_target_finite_v4', 'reward_target_exponential_v5')")
        text = text.replace('if weighted and settled_turn', 'if reward_settlement_targets and settled_turn')
        text = text.replace('if weighted and turn_index', 'if reward_settlement_targets and turn_index')
        text = replace(text, '            if weighted:', '            if reward_settlement_targets:')
    elif name == 'reward_readout.py':
        anchor = "        self.weighted_target_credit = os.environ.get('DT_CREDIT_MODE') == 'weighted_reward_target_finite_v4'"
        text = replace(text, anchor, anchor + "\n        self.reward_target_exponential = os.environ.get('DT_CREDIT_MODE') == 'reward_target_exponential_v5'")
        anchor = "                          output_seed='c_k / native_span_EOS_logp_difference')"
        text = replace(text, anchor, anchor + "\n        elif self.reward_target_exponential:\n"
            "            report.update(target_semantics='reward_settlement_action_spans',\n"
            "                          credit_transform='per_event_exponential_then_credit_sum',\n"
            "                          output_seed='original_unscaled_native_span_logp',\n"
            "                          coefficient_kind='stable_target_credit; native_TextCraft_uses_official_GRPO',\n"
            "                          source_credit_normalization=None)")
        text = replace(text, "        value = float(row['dt_direct_reward'])",
            "        future = row.get('dt_future_event_mask', torch.ones_like(policy)).detach().cpu()\n"
            "        if future.shape != policy.shape or future.dtype != torch.bool:\n"
            "            raise ValueError('Reward event future mask must retain original boolean suffix slots')\n"
            "        prior = prior & future\n"
            "        value = float(row['dt_direct_reward'])")
        text = replace(text, '                    policy=policy, target=target, prior=prior,',
            '                    policy=policy, target=target, prior=prior, future_event_mask=future,')
        text = replace(text, "                torch.ones((1, 1, width), dtype=torch.bool), item['policy'][None, :],",
            "                item['future_event_mask'][None, None, :], item['policy'][None, :],")
    elif name == 'counterfactual.py':
        text = text.replace('input entries are the approved v4', 'input entries are the withdrawn v4')
        text = replace(text, '    interpreted as log probability ratios. The default ratio path is unchanged.',
            '    interpreted as log probability ratios. DTPO v5 uses the original default\n'
            '    exponential path, with official group-relative credit as the supplied\n'
            '    event coefficient. No source normalization is performed.')
    elif name == 'dt_training_batch.py':
        # Canonical HEAD was verified byte-for-byte equal to this actual v4
        # snapshot before adding the event packing seam in its owning module.
        assert hashlib.sha256(source.read_bytes()).hexdigest()=='61cacb7957d912040c12e7282c42d5d6fd17376e97051c78bfa5e3d103ef0b46'
        text = (REPO/'experiments/rl/dt_training_batch.py').read_text(encoding='utf-8')
        assert hashlib.sha256(text.encode()).hexdigest()=='5a5474084dd939339830595e58366fbff881be2d11a6d236b867bc89ed6c9643','Use this version snapshot, not future canonical edits'
    ast.parse(text)
    destination = HERE / 'candidate-source' / name
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(text, encoding='utf-8', newline='\n')
    manifest[name] = dict(parent_source=str(source.relative_to(REPO)),
        parent_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        path=str(destination.relative_to(REPO)), SHA256=hashlib.sha256(destination.read_bytes()).hexdigest())

(HERE / 'prepared-source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
