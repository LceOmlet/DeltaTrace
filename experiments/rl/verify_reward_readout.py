"""Official EOS DT on real positive task fixtures, with single-EOS spot checks.

Uses scripted/oracle actions explicitly; not policy training or success rates.
No fabricated DT vectors. Run only on an idle GPU in the recorded environment.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import torch


class RecordFirstTrace:
    """Retain one real owner artifact for a native-forward numerical audit."""
    def __init__(self, runner):
        self.runner, self.first = runner, None
        self.model = runner.model

    def attribute(self, *args, **kwargs):
        signed, detail = self.runner.attribute(*args, **kwargs)
        if self.first is None:
            self.first = (args[0].detach().clone(), args[2], signed.clone())
        return signed, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['Sokoban', 'Webshop', 'AppWorld', 'all'], default='all')
    parser.add_argument('--rollout-fixtures', type=Path,
                        help='Reuse explicitly scripted official task artifacts exported on another host')
    args = parser.parse_args()
    root = Path(os.environ['DT_ROOT'])
    env = json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_text())['qwen35']
    sys.path[:0] = [str(root), env['official_root'], str(root / 'clean/qwen35'), env['ft_extension_root']]
    os.environ.update(env.get('runtime_environment', {}))
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from profiles.official import make_qwen35_runner
    from finite_fla_gpu import make_compiled_finite_pullback, verify_native_sources
    from qwen35_answer_finite import PackedAnswerTargets
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from reward_readout import EventRatioReadout
    if args.rollout_fixtures:
        fixtures = json.loads(args.rollout_fixtures.read_text())
        tasks = [(name, None) for name in fixtures['tasks']]
    else:
        from verify_task_reward_events import sokoban_events, task_rows
        from verify_positive_task_rewards import webshop_success, appworld_success
        tasks = [('Sokoban', sokoban_events), ('Webshop', webshop_success), ('AppWorld', appworld_success)]

    verify_native_sources(env['native_stage_source_sha256'])
    tokenizer = AutoTokenizer.from_pretrained(env['checkpoint'], local_files_only=True)
    print('loading existing Qwen checkpoint; no installs/downloads', flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        env['checkpoint'], dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': 'cuda:0'}, local_files_only=True,
    ).eval()
    execution = {}
    if env.get('dt_dynamic_shapes', False):
        execution = dict(dynamic_shapes=True, compiler_options=env.get('dt_compiler_options', {}))
    owner = make_qwen35_runner(
        model, VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256']),
        make_compiled_finite_pullback(reuse_scalar_products=False, **execution), **execution,
    )
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope='official task reward artifacts and native finite DT; not actor training',
                  model=env['checkpoint'], context_cap=32768,
                  action_source='disclosed scripted/oracle positive fixtures; not model success rates',
                  ratio_source='official EOS finite attribution', tasks={}, training_verified=False)
    for task, fn in tasks:
        if args.task not in ('all', task):
            continue
        if args.rollout_fixtures:
            artifact = fixtures['tasks'][task]
            rows, lengths = artifact['rows'], artifact['lengths']
            for row in rows:
                for key in ['input_ids', 'responses', 'attention_mask']:
                    row[key] = torch.tensor(row[key], dtype=torch.long)
        else:
            transitions = fn()
            rows, lengths = task_rows(task, transitions, tokenizer)
        recorded = RecordFirstTrace(owner)
        readout = EventRatioReadout(recorded, tokenizer, task=task, max_steps=15,
                                   packed_answer_targets=PackedAnswerTargets, max_length=32768)
        values = readout.episode(rows)
        reports = []
        for row, credit in zip(rows, values):
            mask = row['attention_mask'][-row['responses'].numel():].bool()
            torch.testing.assert_close(credit['dt_token_advantages'],
                                       credit['dt_q_estimates']-credit['dt_v_estimates'], atol=2e-5, rtol=1e-5)
            assert not credit['dt_token_advantages'][~mask].any()
            reports.append(dict(step=int(row['env_step']), reward=float(row['rewards']),
                                advantages=credit['dt_token_advantages'][mask].tolist(),
                                q=credit['dt_q_estimates'][mask].tolist(),
                                v=credit['dt_v_estimates'][mask].tolist()))
        pair, selection, signed = recorded.first
        labels = selection.outcome_token_ids
        target_id = int(selection.labels[0])
        target_index = int((labels == target_id).nonzero()[0])
        factual = pair[1:2, :-1]
        with torch.no_grad():
            factual_lp = owner.read_outcomes(factual, labels)[0, target_index]
            audit = []
            # Numerical audit only: never used as training credit or fallback.
            for position in (pair[0] != pair[1]).nonzero().flatten().tolist()[:3]:
                deleted = factual.clone()
                deleted[0, position] = tokenizer.eos_token_id
                deleted_lp = owner.read_outcomes(deleted, labels)[0, target_index]
                direct = float(factual_lp-deleted_lp)
                estimate = float(signed[0, position])
                audit.append(dict(position=position, direct_single_eos_log_ratio=direct,
                                  dt_signed_estimate=estimate, difference=estimate-direct))
        result['tasks'][task] = dict(readout=readout.last_report, row_lengths=lengths,
                                     credit_rows=reports, native_single_eos_spot_check=audit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(task, json.dumps(readout.last_report), flush=True)
    result['status'] = 'completed_fixture_checks_not_training'
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
