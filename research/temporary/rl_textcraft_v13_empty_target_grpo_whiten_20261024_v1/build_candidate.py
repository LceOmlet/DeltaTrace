"""Extend the pinned v11 credit boundary and native trainer for v13 only."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'rl_textcraft_v12_attribution_then_group_20261024_v1/owner-sources'
MODE = 'reward_target_exponential_whiten_v13'


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    entry = HERE/'candidate-entry'
    entry.mkdir(exist_ok=True)
    for name in ['dt_training_batch.py', 'reward_readout.py', 'textcraft_owner_rollout.py']:
        text = (BASE/name).read_text(encoding='utf-8')
        text = replace_once(text, "'reward_target_exponential_v11')",
                            "'reward_target_exponential_v11', '"+MODE+"')")
        if name == 'dt_training_batch.py':
            text = text.replace("os.environ.get('DT_CREDIT_MODE') == 'reward_target_exponential_v11'",
                                "os.environ.get('DT_CREDIT_MODE') in ('reward_target_exponential_v11', '"+MODE+"')")
            text = text.replace('[DT v11 empty-target GRPO]', '[DT empty-target GRPO]')
        (entry/name).write_text(text, encoding='utf-8', newline='')
    text = (BASE/'ray_trainer.py').read_text(encoding='utf-8')
    text = replace_once(text,
        '        advantages = data.batch["dt_token_advantages"]\n        import os\n',
        '''        advantages = data.batch["dt_token_advantages"]
        import os
        if os.environ.get("DT_CREDIT_MODE") == "reward_target_exponential_whiten_v13":
            # One native post-attribution batch whitening, including fallback g.
            mean = verl_F.masked_mean(advantages, response_mask)
            scale = torch.rsqrt(verl_F.masked_var(advantages, response_mask) + 1e-8)
            whitened = verl_F.masked_whiten(advantages, response_mask, shift_mean=True)
            data.batch["dt_q_estimates"] = torch.where(
                response_mask, data.batch["dt_q_estimates"] * scale, 0.0)
            data.batch["dt_v_estimates"] = torch.where(
                response_mask, (data.batch["dt_v_estimates"] + mean) * scale, 0.0)
            data.batch["dt_token_advantages"] = torch.where(response_mask, whitened, 0.0)
            data.batch["advantages"] = data.batch["dt_token_advantages"]
            data.batch["returns"] = data.batch["dt_q_estimates"]
            if not all(torch.isfinite(data.batch[k]).all() for k in
                       ("dt_token_advantages", "dt_q_estimates", "dt_v_estimates")):
                raise RuntimeError("v13 whitened credit contains non-finite values")
            data.meta_info["dt_v13_whitening"] = dict(
                raw_mean=float(mean), scale=float(scale), policy_tokens=int(response_mask.sum()),
                shift_mean=True, second_normalization=False)
            print(f'[DT v13 post-gu whiten] mean={float(mean)} scale={float(scale)} '
                  f'policy_tokens={int(response_mask.sum())} shift_mean=True '
                  f'A_min={float(whitened[response_mask].min())} '
                  f'A_max={float(whitened[response_mask].max())} second_normalization=False', flush=True)
            return data
''')
    (HERE/'candidate-ray_trainer.py').write_text(text, encoding='utf-8', newline='')
    for p in [*entry.glob('*.py'), HERE/'candidate-ray_trainer.py']:
        compile(p.read_text(encoding='utf-8'),str(p),'exec')
    print('Prepared v11 fallback plus native post-gu masked_whiten; no deployment.')


if __name__ == '__main__':
    main()
