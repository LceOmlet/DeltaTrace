"""LOOP native reward data enters the existing DT Q/V/A composition unchanged.

The DT runner here is the existing explicitly labelled unit-test double.
These tests do not assert real-model attribution accuracy or training readiness.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from reward_readout import RewardAlphabet, EventRatioReadout
from test_reward_readout import Runner, Tokenizer, Targets, row


def test_native_environment_returns_are_not_rescored_or_scaled():
    receipt = Path(os.environ.get('DT_LOOP_ENVIRONMENT_RECEIPT',
        str(Path(__file__).parents[2] / 'research/temporary/rl_upstream_alignment_20260929/loop-project-environment-comparison.json')))
    for case in json.loads(receipt.read_text())['cases']:
        native_return = case['loop_return']
        tokenizer, runner = Tokenizer(), Runner()
        readout = EventRatioReadout(runner, tokenizer, task='AppWorld', max_steps=40,
            packed_answer_targets=Targets, minibatch_size=1,
            appworld_num_tests=case['loop_evaluation']['num_tests'])
        # Prefix identities are a small test fixture; only return/denominator
        # come from the saved real native environment run.
        output = readout.episode([row(0, 0.), row(1, native_return)])
        assert len(runner.calls) == 2
        for credit in output:
            torch.testing.assert_close(credit['dt_q_estimates'], torch.tensor([native_return, native_return, 0.]))
            torch.testing.assert_close(credit['dt_token_advantages'],
                                       credit['dt_q_estimates'] - credit['dt_v_estimates'])
        assert readout.alphabet.values == (0., .5, 1.)
        assert all('Successful completion gives 10' not in q for q in tokenizer.queries)
        assert all('fraction of official tests' in q for q in tokenizer.queries)
        assert all('including when the interaction or context budget' in q for q in tokenizer.queries)
        assert all('NEVER INCLUDE FUTURE STATE' not in q for q in tokenizer.queries)


def test_missing_native_metadata_cannot_fall_back_to_legacy_reward():
    with pytest.raises(ValueError, match='native eval_result.num_tests'):
        RewardAlphabet.for_task('AppWorld')
    with pytest.raises(ValueError, match='no VERL invalid-action penalty'):
        RewardAlphabet.for_task('AppWorld', invalid_action_penalty_coef=.1, appworld_num_tests=2)


def test_partial_success_and_zero_reward_keep_the_existing_composition():
    runner = Runner()
    readout = EventRatioReadout(runner, Tokenizer(), task='AppWorld', max_steps=40,
        packed_answer_targets=Targets, minibatch_size=1, appworld_num_tests=3)
    output = readout.episode([row(0, 1 / 3)])
    torch.testing.assert_close(output[0]['dt_q_estimates'], torch.tensor([1 / 3, 1 / 3, 0.]))
    assert len(runner.calls) == 1
    with pytest.raises(ValueError, match='outside'):
        readout.alphabet.observed_index(10.)
    zero = readout.episode([row(0, 0.)])
    assert len(runner.calls) == 1
    assert zero[0]['dt_token_advantages'].eq(0).all()
