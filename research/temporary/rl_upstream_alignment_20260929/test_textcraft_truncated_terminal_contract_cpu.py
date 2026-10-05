"""CPU source-contract reproduction; this is not a Torch/DT numerical test.

Execute the pinned AgentGym truncation method and the exact current TextCraft
source-row assembly AST. Execute only the original reward-selection expression
from episode_returns; no replacement tensor library or credit algorithm is used.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / 'research/temporary/rl_upstream_alignment_20260929'
OWNER = AUDIT / ('recipe-sources/AgentGym-RL-'
                 '82402a99c62a293735a3f412fb8ac9a600673bc0/AgentGym-RL/'
                 'verl/workers/rollout/schemas.py')
TEXTCRAFT = ROOT / 'experiments/rl/textcraft_owner_rollout.py'
BASELINE = AUDIT / 'textcraft-late-sampling-audit-20261005/baseline'
BEFORE_TEXTCRAFT = BASELINE / 'textcraft_owner_rollout.py'
COUNTERFACTUAL = BASELINE / 'counterfactual.py'


def tree(path):
    return ast.parse(path.read_text(encoding='utf-8'))


def actual_truncate():
    owner = tree(OWNER)
    cls = next(node for node in owner.body
               if isinstance(node, ast.ClassDef) and node.name == 'RolloutHandler')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'truncate_output_ids')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(OWNER), 'exec'), namespace)
    return namespace['truncate_output_ids']


def actual_source_rows(handler, turns, path=BEFORE_TEXTCRAFT):
    method = next(node for node in ast.walk(tree(path))
                  if isinstance(node, ast.FunctionDef)
                  and node.name == 'collect_native_trajectories')
    start = next(index for index, node in enumerate(method.body)
                 if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Tuple)
                 and [getattr(target, 'id', None) for target in node.targets[0].elts]
                 == ['sources', 'maps'])
    namespace = dict(handlers=[handler], records=[turns], ids=['actual-episode'], scores=[1.0])
    block = copy.deepcopy(method.body[start:start + 2])
    exec(compile(ast.fix_missing_locations(ast.Module(body=block, type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace['sources'], namespace['maps']


def actual_selected_reward_values(rows):
    method = next(node for node in tree(COUNTERFACTUAL).body
                  if isinstance(node, ast.FunctionDef) and node.name == 'episode_returns')
    assignment = next(node for node in method.body
                      if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == 'rewards')
    # The torch.tensor input expression is plain Python. This executes that
    # exact expression, without reimplementing tensor/cumsum/Q/V behaviour.
    expression = ast.Expression(body=copy.deepcopy(assignment.value.args[0]))
    return eval(compile(ast.fix_missing_locations(expression), str(COUNTERFACTUAL), 'eval'),
                {'rows': rows})


def native_list_artifacts(turn_count):
    prompt_ids = [7] * 512
    input_ids = prompt_ids.copy()
    turns = []
    for step in range(turn_count):
        # Fits the author's 512-token per-turn generation budget. Twenty
        # complete 511-token actions leave 20 retained tokens of action 21;
        # action 22 is wholly outside the official 10240 training response.
        response = [100 + step] * 511
        turns.append(dict(start=len(input_ids), native_prompt_ids=input_ids.copy(),
                          native_response_ids=response))
        input_ids.extend(response)
    # Only list artifacts are needed by the original truncation method.
    handler = SimpleNamespace(input_ids=input_ids, prompt_ids=prompt_ids,
        attention_mask=[1] * len(input_ids), position_ids=list(range(len(input_ids))),
        loss_mask=[0] * len(prompt_ids) + [1] * (len(input_ids) - len(prompt_ids)),
        prompt_attention_mask=[1] * len(prompt_ids),
        prompt_position_ids=list(range(len(prompt_ids))),
        prompt_loss_mask=[0] * len(prompt_ids),
        max_response_len=10240, max_model_len=10752)
    actual_truncate()(handler)
    return handler, turns


class TextCraftTruncatedTerminalContract(unittest.TestCase):
    def test_before_source_is_actual_downloaded_deployment(self):
        source = json.loads((BASELINE / 'source.json').read_text(encoding='utf-8'))
        for path in (BEFORE_TEXTCRAFT, COUNTERFACTUAL):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             source['files'][path.name]['sha256'])

    def test_actual_terminal_beyond_train_cap_is_marked_inactive(self):
        handler, turns = native_list_artifacts(22)
        rows, maps = actual_source_rows(handler, turns)
        self.assertEqual(len(handler.response_ids), 10240)
        self.assertEqual([row['rewards'] for row in rows], [0.0] * 21 + [1.0])
        self.assertEqual([row['active_masks'] for row in rows], [True] * 21 + [False])
        self.assertEqual(maps, [[(i, i * 511, 511) for i in range(20)] + [(20, 10220, 20)]])
        self.assertEqual(actual_selected_reward_values(rows), [0.0] * 22)
        # Consequently every original reverse cumulative sum input is zero.
        # No Torch/DT operation is claimed to have been executed here.

    def test_no_truncation_keeps_real_terminal_reward(self):
        handler, turns = native_list_artifacts(2)
        rows, maps = actual_source_rows(handler, turns)
        self.assertEqual([row['active_masks'] for row in rows], [True, True])
        self.assertEqual(actual_selected_reward_values(rows), [0.0, 1.0])
        self.assertEqual(maps, [[(0, 0, 511), (1, 511, 511)]])

    def test_event_activity_and_existing_native_slices_are_independent(self):
        handler, turns = native_list_artifacts(22)
        rows, maps = actual_source_rows(handler, turns)
        factual_rows, after_maps = actual_source_rows(handler, turns, TEXTCRAFT)
        self.assertEqual(actual_selected_reward_values(factual_rows), [0.0] * 21 + [1.0])
        eligible = {source for slices in maps for source, _, length in slices if length > 0}
        self.assertEqual(eligible, set(range(21)))
        self.assertNotIn(21, eligible)
        self.assertEqual(len(rows[20]['response_ids']), 511)
        self.assertEqual(maps[0][20][2], 20)
        self.assertEqual(factual_rows[20]['response_ids'], rows[20]['response_ids'])
        self.assertEqual([row['rewards'] for row in factual_rows],
                         [row['rewards'] for row in rows])
        self.assertEqual(after_maps, maps)

    def test_unexecuted_padding_contract_still_excludes_its_reward(self):
        rows = [dict(rewards=1.0, active_masks=True),
                dict(rewards=123.0, active_masks=False)]
        self.assertEqual(actual_selected_reward_values(rows), [1.0, 0.0])


if __name__ == '__main__':
    print(json.dumps(dict(scope='CPU original AST reward/mapping contract only; no Torch/DT numeric test',
        sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (OWNER, BEFORE_TEXTCRAFT, TEXTCRAFT, COUNTERFACTUAL)}), ensure_ascii=False))
    unittest.main()
