"""Default-inert native multi-turn data seam; no actor/loss/optimizer changes."""
from pathlib import Path
from patch_verl_environment_entry import replace_once


def patch_collector(source):
    old = '        gen_batch_output: DataProto = self.gather_rollout_data(\n'
    new = ('        gather = self.gather_rollout_data\n'
           '        if hasattr(envs, "gather_rollout_data"):\n'
           '            from functools import partial\n'
           '            gather = partial(envs.gather_rollout_data, self)\n'
           '        gen_batch_output: DataProto = gather(\n')
    return replace_once(source, old, new)


def patch_trainer(source):
    source = replace_once(source,
        '    attention_mask = data.batch["attention_mask"]\n    return attention_mask[:, -response_length:]\n',
        '    attention_mask = data.batch.get("loss_mask", data.batch["attention_mask"])\n'
        '    return attention_mask[:, -response_length:]\n')
    source = replace_once(source,
        '        if config.actor_rollout_ref.rollout.multi_turn.enable:\n',
        '        if config.actor_rollout_ref.rollout.multi_turn.enable and not config.env.get("factory"):\n')
    old = ('                            dt_values = compute_training_credit(\n'
           '                                batch, self.actor_rollout_wg,\n'
           '                                eos_token_id=int(self.tokenizer.eos_token_id),\n'
           '                                pad_token_id=int(self.tokenizer.pad_token_id),\n'
           '                            )\n')
    new = ('                            if "dt_response_slices" in batch.non_tensor_batch:\n'
           '                                from owner_trajectory_batch import trajectory_credit\n'
           '                                dt_values = trajectory_credit(\n'
           '                                    batch, self.envs.credit_responses, self.actor_rollout_wg,\n'
           '                                    eos_token_id=int(self.tokenizer.eos_token_id),\n'
           '                                    pad_token_id=int(self.tokenizer.pad_token_id))\n'
           '                            else:\n' + ''.join('    '+line+'\n' for line in old.splitlines()))
    return replace_once(source, old, new)


def apply(root):
    for name, patch in [
        ('agent_system/multi_turn_rollout/rollout_loop.py', patch_collector),
        ('verl/trainer/ppo/ray_trainer.py', patch_trainer),
    ]:
        path = Path(root) / name
        source = patch(path.read_text())
        compile(source, str(path), 'exec')
        path.write_text(source, newline='\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    apply(parser.parse_args().root)
