"""Make minimal, default-inert extensions to the pinned credit/trainer owners."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODE = 'reward_attribution_then_group_v12'


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    entry = HERE/'candidate-entry'
    entry.mkdir(exist_ok=True)
    for name in ['reward_readout.py', 'textcraft_owner_rollout.py']:
        text = (HERE/'owner-sources'/name).read_text(encoding='utf-8')
        text = replace_once(text, "'reward_target_exponential_v11')", "'reward_target_exponential_v11', '"+MODE+"')")
        (entry/name).write_text(text, encoding='utf-8', newline='')
    text = (HERE/'owner-sources/dt_training_batch.py').read_text(encoding='utf-8')
    text = replace_once(text, "    rewards = data.batch['dt_grpo_multipliers'].double().cpu()[first]", "    import os\n    multiplier_key = ('dt_reward_multipliers' if os.environ.get('DT_CREDIT_MODE') == '"+MODE+"'\n                      else 'dt_grpo_multipliers')\n    rewards = data.batch[multiplier_key].double().cpu()[first]")
    text = replace_once(text, "'reward_target_exponential_v11')", "'reward_target_exponential_v11', '"+MODE+"')")
    text = text.replace("os.environ.get('DT_CREDIT_MODE') == 'reward_target_exponential_v11'", "os.environ.get('DT_CREDIT_MODE') in ('reward_target_exponential_v11', '"+MODE+"')")
    text = text.replace("[DT v11 empty-target GRPO]", "[DT empty-target broadcast]")
    text = replace_once(text, "    import math\n    import numpy as np\n", "    import math\n    import os\n    import numpy as np\n")
    text = replace_once(text, "            if 'credit' not in event:", "            coefficient_key = 'reward' if os.environ.get('DT_CREDIT_MODE') == '"+MODE+"' else 'credit'\n            if coefficient_key not in event:")
    text = replace_once(text, "            coefficient = float(event['credit'])", "            coefficient = float(event['reward'] if os.environ.get('DT_CREDIT_MODE') == '"+MODE+"' else event['credit'])")
    text = replace_once(text, "                           credit=float(rewards[source_index]),", "                           credit=float(rewards[source_index]), reward=float(rewards[source_index]),")
    text = replace_once(text, "        dt_grpo_multipliers=torch.tensor(coefficients, dtype=torch.float64)),", "        **{('dt_reward_multipliers' if os.environ.get('DT_CREDIT_MODE') == '"+MODE+"' else 'dt_grpo_multipliers'):\n           torch.tensor(coefficients, dtype=torch.float64)}),")
    text += (HERE/'group_reward_extension.py').read_text(encoding='utf-8')
    (entry/'dt_training_batch.py').write_text(text, encoding='utf-8', newline='')

    text = (HERE/'owner-sources/ray_trainer.py').read_text(encoding='utf-8')
    start = text.index('                                # Native GRPO owns group membership, reward reduction and normalization.')
    end = text.index('                                from dt_training_batch import compute_training_credit', start)
    original = text[start:end]
    helper = '''def prepare_dt_credit_multipliers(data, *, norm_adv_by_std_in_grpo, multi_turn):
    """Keep original GRPO preparation, except v12's raw-outcome boundary."""
    import os
    if os.environ.get("DT_CREDIT_MODE") == "reward_attribution_then_group_v12":
        from dt_training_batch import prepare_raw_outcome_credit
        return prepare_raw_outcome_credit(data)
'''
    helper += ''.join('    '+line[32:]+'\n' for line in original.splitlines())
    helper = helper.replace('batch.select(', 'data.select(').replace('batch.batch[', 'data.batch[')
    helper = helper.replace('self.config.actor_rollout_ref.rollout.multi_turn.enable', 'multi_turn')
    helper += '    return data\n\n\n'
    text = text[:start]+'''                                prepare_dt_credit_multipliers(
                                    batch, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable)
'''+text[end:]
    anchor = 'def compute_advantage(data: DataProto,'
    assert text.count(anchor) == 1
    text = text.replace(anchor, helper+anchor)
    text = replace_once(text, '        advantages = data.batch["dt_token_advantages"]\n        import os\n', '''        advantages = data.batch["dt_token_advantages"]
        import os
        if os.environ.get("DT_CREDIT_MODE") == "reward_attribution_then_group_v12":
            from dt_training_batch import apply_attributed_group_advantage
            return apply_attributed_group_advantage(data)
''')
    (HERE/'candidate-ray_trainer.py').write_text(text, encoding='utf-8', newline='')
    for p in [entry/'dt_training_batch.py', entry/'reward_readout.py', entry/'textcraft_owner_rollout.py', HERE/'candidate-ray_trainer.py']:
        compile(p.read_text(encoding='utf-8'),str(p),'exec')
    print('Prepared three credit owners and minimal native trainer seam; no deployment.')


if __name__ == '__main__':
    main()
