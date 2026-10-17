"""Prepare a separate, unaccepted DP-local TextCraft transport candidate."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = HERE / 'original'
source = HERE / 'source'


def pinned(name, expected):
    path = original / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    return path.read_text()


owner = pinned('textcraft_owner_rollout.py',
    'f52a408e17191bbceb6aaf3709ec4ee2d222403dedcd357f8f42d64c53b93325')
signature = '    def collect_native_trajectories(self, collector, gen_batch, actor_rollout_wg, is_train):'
assert owner.count(signature) == 1
wrapper = '''    def collect_native_trajectories(self, collector, gen_batch, actor_rollout_wg, is_train):
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        prepared = gen_batch.select(meta_info_keys=list(gen_batch.meta_info))
        prepared.meta_info['native_textcraft_configuration'] = OmegaConf.to_container(self.config, resolve=True)
        prepared.meta_info['native_textcraft_is_train'] = bool(is_train)
        prepared, padding = pad_dataproto_to_divisor(prepared, actor_rollout_wg.world_size)
        output = actor_rollout_wg.generate_sequences(prepared)
        # The original trajectory_credit API reads this argument before its
        # direct-target branch. That branch owns v6 and needs only the exact
        # artifacts already returned in output; per-turn carriers stay local.
        self.credit_responses = None
        if 'dt_response_slices' in output.non_tensor_batch:
            # These legacy source indices describe the concatenated carriers,
            # not credit values. Restore their original global row namespace.
            source_base, arrays = 0, []
            for part in output.chunk(actor_rollout_wg.world_size):
                mappings = np.empty(len(part), dtype=object)
                mappings[:] = [[(source + source_base, start, length)
                    for source, start, length in row]
                    for row in part.non_tensor_batch['dt_response_slices']]
                arrays.append(mappings)
                source_base += int(part.batch['task_rounds'].sum().item())
            output.non_tensor_batch['dt_response_slices'] = np.concatenate(arrays)
        native = self.native['eval' if self.evaluation else 'train']
        rollout = native['rollout'] if self.evaluation else native['actor_rollout_ref']['rollout']
        return unpad_dataproto(output, padding * int(rollout['n']))

    def _collect_native_trajectories(self, collector, gen_batch, actor_rollout_wg, is_train):'''
candidate_owner = owner.replace(signature, wrapper)
source.joinpath('textcraft_owner_rollout.py').write_text(candidate_owner, encoding='utf-8')
source.joinpath('owner_environment_configs.json').write_bytes(
    original.joinpath('owner_environment_configs.json').read_bytes())

worker = pinned('fsdp_workers.py',
    '198f509c9a0af8a03d54f22e66f5f0d5593ab4c9eb7b27777c201ef323d2fecb')
anchor = '    def generate_sequences(self, prompts: DataProto):\n'
assert worker.count(anchor) == 2  # Actor owner followed by the async subclass.
addition = '''        if 'native_textcraft_configuration' in prompts.meta_info:
            from textcraft_native_dp_owner import collect_native_textcraft
            return collect_native_textcraft(self, prompts)

'''
candidate_worker = worker.replace(anchor, anchor + addition, 1)
source.joinpath('fsdp_workers.py').write_text(candidate_worker, encoding='utf-8')


def methods(text, class_name):
    cls = next(n for n in ast.parse(text).body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


a, b = methods(owner, 'TextCraftOwner'), methods(candidate_owner, 'TextCraftOwner')
old = a.pop('collect_native_trajectories')
new = b.pop('_collect_native_trajectories')
new.name = old.name
assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
b.pop('collect_native_trajectories')
assert {k: ast.dump(v) for k, v in a.items()} == {k: ast.dump(v) for k, v in b.items()}
a, b = methods(worker, 'ActorRolloutRefWorker'), methods(candidate_worker, 'ActorRolloutRefWorker')
assert a.keys() == b.keys()
assert isinstance(b['generate_sequences'].body[0], ast.If)
b['generate_sequences'].body.pop(0)
assert {k: ast.dump(v) for k, v in a.items()} == {k: ast.dump(v) for k, v in b.items()}
for path in source.glob('*.py'):
    compile(path.read_bytes(), str(path), 'exec')
record = dict(status='prepared_only_unaccepted_no_runtime_or_training_changes',
    original_owner_body_AST_unchanged=True, original_worker_default_branch_AST_unchanged=True,
    only_worker_extension='Default-inert metadata boundary in existing generate_sequences; no new dispatcher or generation loop',
    original_AgentGym_loop_or_reward_or_DT_or_actor_or_core_copied_or_changed=False,
    source_SHA256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in source.iterdir() if p.is_file()})
HERE.joinpath('receipts/source-preparation.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
