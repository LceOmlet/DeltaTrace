"""Readout boundary contracts. Fake runner below is explicitly a test double."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from reward_readout import RewardAlphabet, EventRatioReadout


class Tokenizer:
    def __init__(self):
        self.queries = []

    def encode(self, text, add_special_tokens=False):
        if text in ('0', '1', '2'):
            return [int(text)]
        self.queries.append(text)
        return [8, 9]


class Runner:
    def __init__(self):
        self.model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.empty(1)))
        self.prefixes, self.query_prefixes = [], []

    def forward_prefix(self, ids, past_key_values=None):
        cache = [] if past_key_values is None else past_key_values
        cache.extend(ids[0].tolist())
        self.prefixes.append(tuple(cache))
        logits = torch.full((1, 1, 12), -torch.inf)
        logits[..., 2] = 0  # Unused by event readout: it never samples tokens.
        return SimpleNamespace(past_key_values=cache, logits=logits)

    def read_outcomes(self, ids, labels, past_key_values=None):
        self.query_prefixes.append(tuple(past_key_values))
        past_key_values.extend(ids[0].tolist())  # Test native in-place cache mutation.
        value = (sum(past_key_values) + len(past_key_values)) * .03
        return torch.tensor([[value, -.1, -value]])[:, :len(labels)].log_softmax(-1)


def row(step, reward, *, active=True):
    return dict(traj_uid='same', env_step=step, active_masks=active, rewards=reward,
                input_ids=torch.tensor([0, 4, 5, 6, 7, 0]),
                responses=torch.tensor([6, 7, 0]),
                attention_mask=torch.tensor([0, 1, 1, 1, 1, 0]),
                dt_env_outcome={'observation': 'NEVER INCLUDE FUTURE STATE', 'done': step == 1})


def test_queries_have_horizon_and_absence_without_sampled_future():
    tokenizer = Tokenizer()
    alphabet = RewardAlphabet.for_task('Sokoban')
    alphabet.query_ids(tokenizer, current_step=1, event_step=4, max_steps=15)
    text = tokenizer.queries[0]
    assert 'interaction is 2' in text and 'interaction 5' in text and '15 interactions' in text
    assert 'never occurs' in text and '10.9' in text and '-0.1' in text
    assert alphabet.observed_index(10.8999999) == 2
    with pytest.raises(ValueError, match='outside'):
        alphabet.observed_index(0.9)  # Multi-box reward is outside the accepted one-box setting.


def test_pre_post_boundaries_reuse_reads_without_sampling_or_query_leaks(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Extra token sampling is forbidden')
    monkeypatch.setattr(torch, 'multinomial', forbidden)
    runner, tokenizer = Runner(), Tokenizer()
    rows = [row(0, -.1), row(1, 10.9), row(2, 0, active=False)]
    original = deepcopy(rows)
    readout = EventRatioReadout(runner, tokenizer, task='Sokoban', max_steps=15, event_batch_size=1)
    output = readout.episode(rows)
    assert runner.query_prefixes[:4] == [(4, 5, 8), (4, 5, 8), (4, 5, 6, 8), (4, 5, 6, 8)]
    assert (4, 5, 6, 7, 8) in runner.query_prefixes
    assert all(2 not in prefix for prefix in runner.query_prefixes)
    assert (4, 5, 6, 7) in runner.prefixes  # Factual advance has no query prefix.
    assert all(9 not in prefix for prefix in runner.prefixes)
    assert all('NEVER INCLUDE FUTURE STATE' not in query for query in tokenizer.queries)
    assert readout.last_report['prefix_prefill_tokens'] == 4
    assert readout.last_report['factual_decode_tokens'] == 4
    # Two responses of two tokens: each has three boundary reads, not four.
    assert readout.last_report['boundary_readouts'] == 6
    assert len(runner.prefixes) == 9
    for before, after in zip(original, rows):
        torch.testing.assert_close(before['input_ids'], after['input_ids'])
    for values in output:
        assert values['dt_token_advantages'][-1] == 0
        torch.testing.assert_close(values['dt_q_estimates']-values['dt_v_estimates'],
                                   values['dt_token_advantages'])
    assert output[0]['dt_token_advantages'][0] != output[0]['dt_token_advantages'][1]
    assert not output[-1]['dt_token_advantages'].any()


def test_zero_observed_rewards_skip_native_work_without_inventing_signal():
    runner, tokenizer = Runner(), Tokenizer()
    readout = EventRatioReadout(runner, tokenizer, task='Webshop', max_steps=15)
    output = readout.episode([row(0, 0)])
    assert not runner.prefixes
    assert not output[0]['dt_token_advantages'].any()
    assert readout.last_report['nonzero_reward_events'] == 0


def test_readout_context_includes_query_and_does_not_truncate():
    readout = EventRatioReadout(Runner(), Tokenizer(), task='Sokoban', max_steps=15, max_length=5)
    with pytest.raises(ValueError, match='context 6 exceeds cap 5'):
        readout.episode([row(0, -.1)])


def test_reward_label_or_horizon_mismatch_is_not_silently_repaired():
    with pytest.raises(ValueError, match='future event'):
        RewardAlphabet.for_task('Sokoban').query_ids(Tokenizer(), current_step=2, event_step=1, max_steps=15)
    with pytest.raises(ValueError, match='No approved'):
        RewardAlphabet.for_task('invented')


def test_installed_hf_hybrid_cache_branch_expansion_and_isolation():
    from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionLayer
    cache = Cache(layers=[DynamicLayer(), LinearAttentionLayer()])
    keys, values = torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)
    conv, recurrent = torch.randn(1, 8, 4), torch.randn(1, 2, 4, 4)
    cache.layers[0].update(keys, values)
    cache.layers[1].update_conv_state(conv)
    cache.layers[1].update_recurrent_state(recurrent)
    branch = deepcopy(cache)
    branch.reorder_cache(torch.zeros(4, dtype=torch.long))
    for name in ['keys', 'values']:
        original, expanded = getattr(cache.layers[0], name), getattr(branch.layers[0], name)
        torch.testing.assert_close(expanded, original.expand(4, -1, -1, -1))
    for name in ['conv_states', 'recurrent_states']:
        original, expanded = getattr(cache.layers[1], name), getattr(branch.layers[1], name)
        assert expanded.shape[0] == 4
        torch.testing.assert_close(expanded[3], original[0])
        expanded.zero_()
        assert original.count_nonzero() > 0


def test_unchanged_event_distribution_has_zero_advantage():
    runner = Runner()
    runner.read_outcomes = lambda ids, labels, past_key_values: torch.zeros(1, len(labels)).log_softmax(-1)
    readout = EventRatioReadout(runner, Tokenizer(), task='Sokoban', max_steps=15)
    output = readout.episode([row(0, -.1)])
    assert readout.last_report['boundary_readouts'] == 3
    assert not output[0]['dt_token_advantages'].any()


def test_probability_before_token_is_policy_marginal_and_reward_composition_is_unbiased():
    # Enumerated distribution fixture: the identity under test is conditional
    # expectation, not a claim that this double predicts real task rewards.
    policy = torch.tensor([.25, .75], dtype=torch.float64)
    after = torch.tensor([[.8, .1, .1], [.2, .3, .5]], dtype=torch.float64)
    before = policy @ after
    rewards = torch.tensor([-.1, 0., 10.9], dtype=torch.float64)
    from counterfactual import reward_event_token_credit
    q, v = after @ rewards, before @ rewards
    for action in range(2):
        d = (after[action].log() - before.log()).reshape(3, 1, 1)
        values = reward_event_token_credit(d, rewards[:, None],
            torch.ones_like(d, dtype=torch.bool), torch.ones(3, 1, dtype=torch.bool))
        torch.testing.assert_close(after[action] @ values.q_estimates[:, 0], q[action])
        torch.testing.assert_close(after[action] @ values.v_estimates[:, 0], v)
        torch.testing.assert_close(after[action] @ values.advantages[:, 0], q[action] - v)


def test_tool_observation_starts_fresh_boundary_and_is_not_an_action():
    runner = Runner()
    rows = [row(0, -.1), row(1, 10.9)]
    # Next prompt contains the original response followed by observation 10,11.
    # The preceding response's last read must NOT replace this row's first read.
    rows[1]['input_ids'] = torch.tensor([4, 5, 6, 7, 10, 11, 6, 7, 0])
    rows[1]['attention_mask'] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0])
    readout = EventRatioReadout(runner, Tokenizer(), task='Sokoban', max_steps=15, event_batch_size=1)
    output = readout.episode(rows)
    assert runner.query_prefixes[-3:] == [
        (4, 5, 6, 7, 10, 11), (4, 5, 6, 7, 10, 11, 6), (4, 5, 6, 7, 10, 11, 6, 7)]
    assert readout.last_report['policy_tokens'] == 4
    assert readout.last_report['boundary_readouts'] == 6
    assert all(r['dt_token_advantages'].shape == (3,) for r in output)
