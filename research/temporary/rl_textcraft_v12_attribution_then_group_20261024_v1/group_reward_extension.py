

def prepare_raw_outcome_credit(data):
    """Use the pinned GRPO owner's outcome reduction, without group-normalizing it."""
    # Same native outcome: environment scores after the owner's reward adjustments.
    data.batch['dt_reward_multipliers'] = data.batch['token_level_rewards'].sum(-1).detach()
    return data


@torch.no_grad()
def apply_attributed_group_advantage(data, epsilon=1e-6):
    """One retained-policy reward table, one pooled mean/std per original uid.

    DT's raw A0 is the shaped reward r*u (or r on self/target-free tokens).
    Q=Q0/s, V=(V0+mu)/s preserve A=Q-V under this group affine transform.
    Observation/padding are excluded; no second batch variance scaling follows.
    """
    mask = data.batch['response_mask'].bool()
    raw = data.batch['dt_token_advantages'].double()
    q0 = data.batch['dt_q_estimates'].double()
    v0 = data.batch['dt_v_estimates'].double()
    groups = {}
    for row, uid in enumerate(data.non_tensor_batch['uid']):
        groups.setdefault(uid, []).append(row)
    result = {name: torch.zeros_like(raw) for name in CREDIT_KEYS}
    statistics = []
    for uid, rows in groups.items():
        index = torch.as_tensor(rows, device=raw.device)
        active = mask[index]
        selected = raw[index][active]
        if selected.numel() == 0:
            continue
        # Pinned GRPO uses these PyTorch reductions; here they see the token table.
        mean = torch.mean(selected)
        std = torch.std(selected, correction=0)
        scale = std + epsilon
        result['dt_token_advantages'][index] = torch.where(active, (raw[index]-mean)/scale, 0.)
        result['dt_q_estimates'][index] = torch.where(active, q0[index]/scale, 0.)
        result['dt_v_estimates'][index] = torch.where(active, (v0[index]+mean)/scale, 0.)
        statistics.append(dict(uid=str(uid), policy_tokens=selected.numel(),
            reward_mean=float(mean), reward_std=float(std), denominator=float(scale)))
    for name, value in result.items():
        if not torch.isfinite(value).all():
            raise RuntimeError('v12 group-normalized credit is non-finite')
        data.batch[name] = value.float()
    data.batch['advantages'] = data.batch['dt_token_advantages']
    data.batch['returns'] = data.batch['dt_q_estimates']
    data.meta_info['dt_v12_group_statistics'] = statistics
    values = data.batch['advantages'][mask]
    print(f'[DT v12 attribution then group] groups={len(statistics)} '
          f'policy_tokens={values.numel()} '
          f'A_min={float(values.min()) if values.numel() else 0.} '
          f'A_max={float(values.max()) if values.numel() else 0.} '
          f'group_std_min={min((x["reward_std"] for x in statistics), default=0.)} '
          f'epsilon={epsilon} second_normalization=False', flush=True)
    return data
