"""EOS event adapter contracts with an explicitly labelled owner test double."""
from copy import deepcopy
import json
import string
from types import SimpleNamespace

import pytest
import torch

from reward_readout import RewardAlphabet, EventRatioReadout, DirectActionTargetReadout
from deltatrace_rollout import DeltaTraceRolloutProducer


class Tokenizer:
    eos_token_id = 99

    def __init__(self):
        self.queries = []

    def encode(self, text, add_special_tokens=False):
        labels = string.digits + string.ascii_uppercase + string.ascii_lowercase
        if len(text) == 1 and text in labels:
            return [labels.index(text)]
        self.queries.append(text)
        return [8, 9]


class Targets:
    """Captures the exact arguments sent to the real owner selection class."""
    def __init__(self, cases, offsets, length, device, **kwargs):
        self.cases, self.offsets, self.length = cases, offsets, length
        self.outcomes = kwargs['outcome_token_ids']

    def sample_sums(self, values):
        return values  # This test double has one target per case.


class Runner:
    def __init__(self):
        self.calls, self.releases = [], 0
        self.model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.empty(1)),
                                     release_owner_params=self.release)

    def release(self):
        self.releases += 1

    def attribute(self, pair, mask, selection, **kwargs):
        self.calls.append((pair.clone(), mask.clone(), selection))
        assert mask.eq(1).all()
        assert selection.offsets == [[0]]
        assert selection.cases[0]['prompt_length'] == pair.shape[1] - 1
        changed = (pair[0] != pair[1]).nonzero().flatten()
        assert pair[0, changed].eq(99).all()
        # Deliberately nonuniform signed contributions, different per event.
        signed = torch.zeros(1, pair.shape[1], dtype=torch.float64)
        scale = .01 * (int(selection.cases[0]['target_ids'][0]) + 1)
        signed[0, changed] = scale * torch.arange(1, len(changed) + 1, dtype=torch.float64) * (-1.)**torch.arange(len(changed))
        return signed, dict(root_effect=float(signed.sum()),
                            compiled_seed_logprob_effect=float(signed.sum()),
                            target_logp0=[-2.], target_logp1=[-2. + float(signed.sum())],
                            complete_attribution_seconds_with_diagnostics=0.)

    def forward_prefix(self, *args, **kwargs):
        raise AssertionError('Per-token prefix forecasting is not the accepted path')

    read_outcomes = forward_prefix


def row(step, reward, *, active=True):
    return dict(traj_uid='same', env_step=step, active_masks=active, rewards=reward,
                input_ids=torch.tensor([0, 4, 5, 6, 7, 0]),
                responses=torch.tensor([6, 7, 0]),
                attention_mask=torch.tensor([0, 1, 1, 1, 1, 0]),
                dt_env_outcome={'observation': 'NEVER INCLUDE FUTURE STATE', 'done': step == 1})


def readout(runner=None, tokenizer=None, **kwargs):
    return EventRatioReadout(runner or Runner(), tokenizer or Tokenizer(),
                            task=kwargs.pop('task', 'Sokoban'), max_steps=15,
                            minibatch_size=kwargs.pop('minibatch_size', 1),
                            packed_answer_targets=Targets, **kwargs)


@pytest.mark.parametrize('task,values', [('SkyRL-SQL', (-1., 0., 1.)), ('TextCraft', (0., 1.))])
def test_environment_reward_alphabets_and_actual_sampling(task, values):
    tokenizer = Tokenizer()
    alphabet = RewardAlphabet.for_task(task)
    assert alphabet.values == values
    sampling = dict(temperature=.6, top_p=.95, max_tokens=3000, stop=['</sql>', '</solution>'])
    dt = readout(tokenizer=tokenizer, task=task, sampling=sampling)
    result = dt.episode([row(0, values[-1])])[0]
    assert '"temperature":0.6' in tokenizer.queries[0]
    assert 'temperature 1 with no' not in tokenizer.queries[0]
    torch.testing.assert_close(result['dt_token_advantages'], result['dt_q_estimates'] - result['dt_v_estimates'])


def test_queries_encode_reward_events_without_revealing_sampled_future():
    tokenizer = Tokenizer()
    alphabet = RewardAlphabet.for_task('Sokoban')
    alphabet.query_ids(tokenizer, current_step=1, max_steps=15)
    text = tokenizer.queries[0]
    assert 'interaction is 2' in text and '15 interactions' in text
    assert 'SUM of all official rewards' in text and 'Exclude earlier rewards' in text
    assert 'never occur' in text and '10.9' in text and '-0.1' in text
    assert 'after generation of the current response has ended, possibly at its token limit.' in text
    assert 'environment immediately processing that response as emitted' in text
    assert 'unfinished response' not in text and 'Imagine completing that response' not in text
    assert alphabet.observed_index(10.8999999) == 16
    with pytest.raises(ValueError, match='outside'):
        alphabet.observed_index(0.9)


def test_official_finite_vector_per_return_and_original_token_alignment(monkeypatch):
    monkeypatch.setattr(torch, 'multinomial', lambda *a, **kw: pytest.fail('No reference sampling'))
    runner, tokenizer = Runner(), Tokenizer()
    rows = [row(0, -.1), row(1, 10.9), row(2, 0, active=False)]
    original = deepcopy(rows)
    dt = readout(runner, tokenizer)
    output = dt.episode(rows)
    assert len(runner.calls) == 2  # One whole-return target per response.
    assert runner.releases == 2
    assert runner.calls[0][0].tolist() == [[4, 5, 99, 99, 8, 9, 17], [4, 5, 6, 7, 8, 9, 17]]
    assert runner.calls[1][0].tolist() == [[4, 5, 99, 99, 8, 9, 16], [4, 5, 6, 7, 8, 9, 16]]
    assert runner.calls[0][2].outcomes == list(range(31))
    d0, d1 = torch.tensor([.18, -.36]), torch.tensor([.17, -.34])
    expected = 10.8 * (-torch.expm1(-d0))
    torch.testing.assert_close(output[0]['dt_token_advantages'][:2], expected)
    torch.testing.assert_close(output[1]['dt_token_advantages'][:2], 10.9 * (-torch.expm1(-d1)))
    assert not output[-1]['dt_token_advantages'].any()
    for before, after, values in zip(original, rows, output):
        torch.testing.assert_close(before['input_ids'], after['input_ids'])
        assert values['dt_token_advantages'][-1] == 0
        torch.testing.assert_close(values['dt_q_estimates']-values['dt_v_estimates'], values['dt_token_advantages'])
    assert all('NEVER INCLUDE FUTURE STATE' not in q for q in tokenizer.queries)
    assert dt.last_report['per_token_probability_queries'] == 0
    replay = dt.last_report['minimum_log_ratio_batch']
    assert len(replay['samples']) == 1
    sample = replay['samples'][0]
    assert sample['selected_input_ids'] == runner.calls[0][0][1].tolist()
    assert sample['trace']['source_log_ratio_min_index'] == 1
    assert sample['trace']['reference_target_logp'] == -2.
    assert sample['trace']['factual_target_logp'] == pytest.approx(-2.18)
    assert sample['observed_return'] == pytest.approx(10.8)


def test_zero_observed_rewards_do_not_invent_a_signal():
    runner = Runner()
    dt = readout(runner, task='Webshop')
    output = dt.episode([row(0, 0)])
    assert not runner.calls
    assert not output[0]['dt_token_advantages'].any()


def test_native_nonfinite_failure_retains_replay_and_propagates(capsys):
    import json
    class FailedRunner(Runner):
        def attribute(self, *args, **kwargs):
            raise ValueError('Nonfinite DT coefficients; deferred check failed before return.')
    dt = readout(FailedRunner(), task='Webshop')
    with pytest.raises(ValueError, match='Nonfinite DT coefficients'):
        dt.episode([row(0, 10)])
    line = next(line for line in capsys.readouterr().out.splitlines()
                if line.startswith('[DT EOS failed minibatch] '))
    replay = json.loads(line.split('] ', 1)[1])
    assert replay['selected_input_ids'][0][2:4] == [6, 7]
    assert replay['reference_input_ids'][0][2:4] == [99, 99]
    assert replay['samples'][0]['observed_return'] == 10


class BatchedRunner(Runner):
    def attribute(self, pair, mask, selection, **kwargs):
        self.calls.append((pair.clone(), mask.clone(), selection))
        outputs, roots = [], []
        for index, case in enumerate(selection.cases):
            end = case['prompt_length'] + 1
            assert pair[2*index:2*index+2, end:].eq(99).all()
            original = Runner()
            selected = Targets([case], [[0]], end, pair.device, outcome_token_ids=selection.outcomes)
            signed, detail = original.attribute(pair[2*index:2*index+2, :end],
                                                mask[2*index:2*index+2, :end], selected)
            outputs.append(torch.nn.functional.pad(signed, (0, pair.shape[1]-end)))
            roots.append(detail['root_effect'])
        return torch.cat(outputs), dict(root_effect=sum(roots),
            compiled_seed_logprob_effect=sum(roots), target_logp0=[-2.]*len(roots),
            target_logp1=[-2.+v for v in roots], complete_attribution_seconds_with_diagnostics=0.)


def test_minibatch_preserves_each_episode_event_and_expm1():
    episodes = [[row(0, -.1), row(1, 10.9)], [row(0, -.1)], [row(0, -.1)]]
    # Different true prefix lengths exercise compute padding after the target.
    episodes[1][0]['input_ids'] = torch.cat((torch.tensor([4, 4, 4]), episodes[1][0]['input_ids']))
    episodes[1][0]['attention_mask'] = torch.cat((torch.ones(3, dtype=torch.long), episodes[1][0]['attention_mask']))
    expected = [readout().episode(rows) for rows in episodes]
    runner = BatchedRunner()
    dt = readout(runner, minibatch_size=4)
    actual = dt.episodes(episodes)
    assert len(runner.calls) == 1 and runner.calls[0][0].shape[0] == 8
    assert dt.last_report['finite_trace_calls'] == 1
    assert dt.last_report['event_contrasts'] == 4
    for before, after in zip(expected, actual):
        for a, b in zip(before, after):
            for name in a:
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert all(t['conservation_verified'] for t in dt.last_report['traces'])
    replay = dt.last_report['minimum_log_ratio_batch']
    assert len(replay['samples']) == 4
    for index, sample in enumerate(replay['samples']):
        trace = sample['trace']
        assert trace['reference_target_logp'] == -2.
        assert trace['factual_target_logp'] - trace['reference_target_logp'] == pytest.approx(trace['root_effect'])
        selected = torch.full((trace['compute_tokens'],), replay['eos_token_id'])
        selected[:len(sample['selected_input_ids'])] = torch.tensor(sample['selected_input_ids'])
        reference = selected.clone()
        reference[sample['source_start']:sample['source_end']] = replay['eos_token_id']
        assert torch.equal(reference, runner.calls[0][0][2*index])
        assert torch.equal(selected, runner.calls[0][0][2*index+1])


def test_rounding_audit_failure_preserves_raw_token_credit_and_failed_status():
    class RoundedRunner(Runner):
        def attribute(self, *args, **kwargs):
            signed, detail = super().attribute(*args, **kwargs)
            detail['root_effect'] += .5
            detail['compiled_seed_logprob_effect'] += .5
            return signed, detail
    expected = readout().episode([row(0, -.1)])[0]
    dt = readout(RoundedRunner())
    actual = dt.episode([row(0, -.1)])[0]
    # A numerical failure must not rescale, zero, broadcast or otherwise
    # substitute a different token signal. Nor may it be reported as passed.
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert dt.last_report['conservation_failures'] == 1
    assert not dt.last_report['traces'][0]['conservation_verified']
    assert dt.last_report['traces'][0]['conservation_tolerance'] == .02
    assert dt.last_report['traces'][0]['conservation_residual'] == pytest.approx(.5)


@pytest.mark.parametrize('failure', ['nonfinite', 'seed_mismatch'])
def test_invalid_owner_values_are_still_rejected(failure):
    class InvalidRunner(Runner):
        def attribute(self, *args, **kwargs):
            signed, detail = super().attribute(*args, **kwargs)
            if failure == 'nonfinite':
                signed[0, 2] = float('nan')
            else:
                detail['compiled_seed_logprob_effect'] += .5
            return signed, detail
    with pytest.raises((ValueError, AssertionError), match='non-finite|endpoint/seed mismatch'):
        readout(InvalidRunner()).episode([row(0, -.1)])


def test_total_context_includes_readout_and_target_no_silent_truncation():
    dt = readout(max_length=6)
    with pytest.raises(ValueError, match='context 7 exceeds cap 6'):
        dt.episode([row(0, -.1)])


def test_readout_budget_reuses_actual_query_encoding():
    tokenizer = Tokenizer()
    assert RewardAlphabet.for_task('Sokoban').readout_token_budget(tokenizer, 15) == 3
    assert len(tokenizer.queries) == 15


@pytest.mark.parametrize('horizon', [1, 5, 15])
@pytest.mark.parametrize('solves', [False, True])
def test_process_rewards_require_only_linear_attribution_requests(horizon, solves):
    rows = [row(i, 10.9 if solves and i == horizon - 1 else -.1) for i in range(horizon)]
    runner = BatchedRunner()
    dt = readout(runner, minibatch_size=4)
    output = dt.episode(rows)
    assert dt.last_report['event_contrasts'] == horizon
    assert len(runner.calls) == (horizon + 3) // 4
    assert sum(len(call[2].cases) for call in runner.calls) == horizon
    for i, result in enumerate(output):
        expected_return = (11 if solves else 0) - .1 * (horizon - i)
        assert result['dt_q_estimates'][0] == pytest.approx(expected_return)
        assert result['dt_token_advantages'][0] != result['dt_token_advantages'][1]
        assert result['dt_token_advantages'][-1] == 0


def test_longer_action_span_does_not_add_dt_requests():
    short = readout(BatchedRunner(), minibatch_size=4)
    long = readout(BatchedRunner(), minibatch_size=4)
    short.episode([row(i, -.1) for i in range(15)])
    rows = [row(i, -.1) for i in range(15)]
    for r in rows:
        r['responses'] = torch.tensor([6, 7, 6, 7, 0])
        r['input_ids'] = torch.cat((torch.tensor([4, 5]), r['responses']))
        r['attention_mask'] = torch.tensor([1, 1, 1, 1, 1, 1, 0])
    long.episode(rows)
    assert short.last_report['finite_trace_calls'] == long.last_report['finite_trace_calls'] == 4
    assert long.last_report['policy_tokens'] == 2 * short.last_report['policy_tokens']


def test_full_32768_adapter_endpoint_preserves_ids():
    # Capacity of adapter tensors only; NOT a 32k model/GPU training result.
    r = row(0, -.1)
    r['input_ids'] = torch.cat((torch.full((32762,), 5), r['responses']))
    r['attention_mask'] = torch.ones_like(r['input_ids'])
    r['attention_mask'][-1] = 0
    runner = Runner()
    dt = readout(runner)
    dt.episode([r])
    assert runner.calls[0][0].shape == (2, 32767)
    assert dt.last_report['max_length'] == 32768
    # Add one original prompt token, reaching the exact configured total cap.
    r['input_ids'] = torch.cat((torch.tensor([5]), r['input_ids']))
    r['attention_mask'] = torch.cat((torch.tensor([1]), r['attention_mask']))
    dt.episode([r])
    assert runner.calls[-1][0].shape == (2, 32768)


def test_past_tool_observation_is_preserved_and_not_a_source_action():
    runner = Runner()
    rows = [row(0, -.1), row(1, 10.9)]
    rows[1]['input_ids'] = torch.tensor([4, 5, 6, 7, 10, 11, 6, 7, 0])
    rows[1]['attention_mask'] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0])
    output = readout(runner).episode(rows)
    pair = runner.calls[-1][0]
    assert pair[:, :6].tolist() == [[4, 5, 6, 7, 10, 11]] * 2
    assert (pair[0] != pair[1]).nonzero().flatten().tolist() == [6, 7]
    assert all(values['dt_token_advantages'].shape == (3,) for values in output)


def test_actual_eos_action_is_kept_with_zero_eos_contrast():
    r = row(0, -.1)
    r['responses'][1] = r['input_ids'][-2] = 99
    result = readout().episode([r])[0]
    assert result['dt_token_advantages'][1] == 0
    assert result['dt_q_estimates'][1] != 0  # Action not masked as padding.


def test_owner_parameters_released_on_failure():
    runner = Runner()
    def fail(*a, **kw):
        raise RuntimeError('owner failure')
    runner.attribute = fail
    with pytest.raises(RuntimeError, match='owner failure'):
        readout(runner).episode([row(0, -.1)])
    assert runner.releases == 1


def test_minimum_replay_is_logged_before_a_later_batch_failure(capsys):
    class FailsOnSecondBatch(Runner):
        def attribute(self, *args, **kwargs):
            if self.calls:
                raise RuntimeError('later owner failure')
            return super().attribute(*args, **kwargs)
    runner = FailsOnSecondBatch()
    with pytest.raises(RuntimeError, match='later owner failure'):
        readout(runner).episode([row(0, -.1), row(1, 10.9)])
    lines = capsys.readouterr().out.splitlines()
    records = [json.loads(line.removeprefix('[DT EOS minimum] '))
               for line in lines if line.startswith('[DT EOS minimum] ')]
    assert len(records) == 1
    sample = records[0]['samples'][0]
    assert sample['selected_input_ids'] == runner.calls[0][0][1].tolist()
    assert sample['trace']['owner_batch_index'] == 0
    assert sample['source_signed'] == pytest.approx([.18, -.36])
    assert any('batch=1/2 d_min=' in line and 'd_max=' in line for line in lines)
    assert runner.releases == 2


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('native_fp16', [False, True])
def test_actor_attention_and_training_mode_restored_after_finite_trace(fails, native_fp16):
    class Model:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation='sdpa')
            self.training = True
        def set_attn_implementation(self, name):
            self.config._attn_implementation = name
        def train(self, mode):
            self.training = mode
        def eval(self):
            self.training = False
        def modules(self):
            return iter([self])
    actor = Model()
    calls = []
    g = torch.ones(2, dtype=torch.float32)
    state = object()
    def original_fla(q, k, v, *, g, beta, **kwargs):
        calls.append((q.dtype, k.dtype, v.dtype, beta.dtype, g, kwargs['initial_state']))
        return v, kwargs['initial_state']
    actor.chunk_gated_delta_rule = original_fla
    producer = object.__new__(DeltaTraceRolloutProducer)
    producer.actor = actor
    producer.readout_options = {'task': 'Sokoban'}
    producer.native_fla_fp16 = native_fp16
    producer.runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(language_model=actor)))
    def episode(rows):
        assert not actor.training and actor.config._attn_implementation == 'flash_attention_2'
        operand = torch.ones(2, dtype=torch.bfloat16)
        output, returned_state = actor.chunk_gated_delta_rule(
            operand, operand, operand, g=g, beta=operand, initial_state=state)
        expected_dtype = torch.float16 if native_fp16 else torch.bfloat16
        assert calls[-1][:4] == (expected_dtype,) * 4
        assert calls[-1][4] is g and calls[-1][5] is state
        assert output.dtype == operand.dtype and returned_state is state
        if fails:
            raise RuntimeError('trace error')
        return []
    producer.readout = SimpleNamespace(episodes=lambda episodes, **kwargs: [episode(episodes[0])], last_report={})
    if fails:
        with pytest.raises(RuntimeError, match='trace error'):
            producer.attribute_episode([], 123)
    else:
        producer.attribute_episode([], 123)
    assert actor.training and actor.config._attn_implementation == 'sdpa'
    assert actor.chunk_gated_delta_rule is original_fla
    assert len(calls) == 1  # The representation boundary must not repeat the operator.


def test_reward_label_or_horizon_mismatch_is_not_silently_repaired():
    with pytest.raises(ValueError, match='current step'):
        RewardAlphabet.for_task('Sokoban').query_ids(Tokenizer(), current_step=15, max_steps=15)
    with pytest.raises(ValueError, match='No approved'):
        RewardAlphabet.for_task('invented')


class NoTraceRunner:
    """No simulated DT values: the analytic branches must not call a model."""
    model = SimpleNamespace(execution_device=torch.device('cpu'),
                            lm_head=SimpleNamespace(weight=torch.empty(0)))

    def attribute(self, *args, **kwargs):
        raise AssertionError('An analytically zero/target-self branch must not call DT')


def direct_row(*, reward=.5, target=(True, False, True)):
    # Small CPU transport inputs, not task/model/DT numerical evidence.
    return dict(input_ids=torch.tensor([0, 10, 11, 12, 13, 14, 0]),
                attention_mask=torch.tensor([0, 1, 1, 1, 1, 1, 0]),
                responses=torch.tensor([12, 13, 14, 0]),
                policy_mask=torch.tensor([True, False, True, False]),
                target_mask=torch.tensor([*target, False]), dt_direct_reward=reward,
                traj_uid='cpu-interface', dt_direct_response_count=2)


@pytest.mark.parametrize('reward,target,expected', [
    (.5, (True, False, True), [.5, 0., .5, 0.]),
    (.5, (False, False, False), [0., 0., 0., 0.]),
    (0., (False, False, True), [0., 0., 0., 0.]),
])
def test_direct_analytic_cases_use_real_composition_without_dt(reward, target, expected):
    dt = DirectActionTargetReadout(NoTraceRunner(), Tokenizer(), task='TextCraft',
                                  packed_answer_targets=None)
    result = dt.trajectories([direct_row(reward=reward, target=target)])[0]
    assert result['dt_token_advantages'].tolist() == expected
    torch.testing.assert_close(result['dt_token_advantages'],
                               result['dt_q_estimates'] - result['dt_v_estimates'])
    assert dt.last_report['finite_trace_calls'] == 0
    assert dt.last_report['query_tokens'] == dt.last_report['synthetic_labels'] == 0
    assert dt.last_report['original_response_rows'] == [2]
    if not any(target):
        assert result['dt_q_estimates'].tolist() == [.5, 0., .5, 0.]
        assert torch.equal(result['dt_q_estimates'], result['dt_v_estimates'])


def test_direct_joint_selection_preserves_gaps_and_original_suffix_ids():
    from pathlib import Path
    import ast
    import copy
    # Execute the unchanged owner's selection class only. This checks its CPU
    # packing ABI, not its head, finite propagation, or official tolerances.
    path = Path(__file__).resolve().parents[2]/'deltatrace/clean/qwen35/qwen35_answer_finite.py'
    source = ast.parse(path.read_bytes())
    definition = next(n for n in source.body if isinstance(n, ast.ClassDef)
                      and n.name == 'PackedAnswerTargets')
    namespace = dict(torch=torch, copy=copy)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                 str(path), 'exec'), namespace)
    item = DirectActionTargetReadout._prepare_row(direct_row(), 0)
    selection = namespace['PackedAnswerTargets']([item['case']], [item['target_offsets']],
                                                item['selected'].numel(), 'cpu')
    assert item['selected'].tolist() == [10, 11, 12, 13, 14]
    assert item['target_offsets'] == [0, 2]
    assert selection.labels.tolist() == [12, 14]  # The intervening O is not a target.
    assert selection.positions.tolist() == [1, 3]
    assert selection.outcome_token_ids is None


def test_direct_observation_cannot_be_marked_as_an_executed_target():
    value = direct_row(target=(True, True, True))
    with pytest.raises(ValueError, match='Executed targets must be policy actions'):
        DirectActionTargetReadout._prepare_row(value, 0)


def test_direct_reference_omits_causal_zero_tail_and_preserves_full_scatter(monkeypatch):
    # A transport-only stub records the endpoint IDs. Its marker values are
    # deliberately not model estimates or a numerical attribution reference.
    import reward_readout
    seen = []

    def trace_transport(runner, reference, factual, cases, offsets, **kwargs):
        seen.append((reference.clone(), factual.clone(), offsets))
        assert factual.tolist() == [[10, 11, 12, 13, 14, 15]]
        assert reference.tolist() == [[10, 11, 99, 13, 14, 15]]
        assert offsets == [[1, 3]]
        marker = torch.ones_like(factual, dtype=torch.float64)
        marker[:, 2] = 0.  # Only the prior source is retained from this stub.
        return marker, None, dict(root_effect=0., policy_credit_signed_sum=0.,
                                  conservation_residual=0.)

    monkeypatch.setattr(reward_readout, 'trace_token_attribution', trace_transport)
    value = dict(input_ids=torch.tensor([0, 10, 11, 12, 13, 14, 15, 16, 0]),
                 attention_mask=torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 0]),
                 responses=torch.tensor([12, 13, 14, 15, 16, 0]),
                 policy_mask=torch.tensor([True, True, False, True, True, False]),
                 target_mask=torch.tensor([False, True, False, True, False, False]),
                 dt_direct_reward=.5)
    dt = DirectActionTargetReadout(NoTraceRunner(), Tokenizer(), task='TextCraft',
                                  packed_answer_targets=None)
    result = dt.trajectories([value])[0]
    assert len(seen) == 1
    assert result['dt_token_advantages'].tolist() == [0., .5, 0., .5, 0., 0.]
    assert result['dt_q_estimates'][4].item() == result['dt_v_estimates'][4].item() == .5
    assert dt.last_report['actual_context_lengths'] == [7]
    assert dt.last_report['causal_context_lengths'] == [6]
    assert dt.last_report['causal_suffix_tokens_omitted'] == 1


def test_direct_terminal_observation_tail_does_not_expand_required_target_prefix():
    # CPU packing/analytic-interface regression only. These IDs do not stand
    # in for the unavailable failing rollout or a real DT numerical test.
    row = dict(input_ids=torch.arange(32835), attention_mask=torch.ones(32835),
               responses=torch.arange(1, 32835),
               policy_mask=torch.zeros(32834, dtype=torch.bool),
               target_mask=torch.zeros(32834, dtype=torch.bool),
               dt_direct_reward=.75, traj_uid='cpu-terminal-tail')
    row['policy_mask'][32765] = row['target_mask'][32765] = True
    original = row['input_ids'].clone()
    dt = DirectActionTargetReadout(NoTraceRunner(), Tokenizer(), task='AppWorld',
                                  packed_answer_targets=None)
    result = dt.trajectories([row])[0]
    assert torch.equal(row['input_ids'], original)
    assert result['dt_token_advantages'].shape == row['responses'].shape
    assert result['dt_token_advantages'][32765].item() == .75
    assert torch.count_nonzero(result['dt_token_advantages']) == 1
    assert dt.last_report['actual_context_lengths'] == [32835]
    assert dt.last_report['causal_context_lengths'] == [32767]
    assert dt.last_report['finite_trace_calls'] == 0


def test_direct_required_target_prefix_still_rejects_over_cap():
    row = dict(input_ids=torch.arange(32769), attention_mask=torch.ones(32769),
               responses=torch.arange(1, 32769),
               policy_mask=torch.ones(32768, dtype=torch.bool),
               target_mask=torch.zeros(32768, dtype=torch.bool),
               dt_direct_reward=.75, traj_uid='cpu-required-overflow')
    row['target_mask'][-1] = True
    dt = DirectActionTargetReadout(NoTraceRunner(), Tokenizer(), task='AppWorld',
                                  packed_answer_targets=None)
    with pytest.raises(ValueError, match='context 32769 exceeds cap 32768'):
        dt.trajectories([row])
