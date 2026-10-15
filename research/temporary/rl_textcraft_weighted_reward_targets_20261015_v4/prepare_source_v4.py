"""Apply the approved v4 seams to canonical owners and the deployed snapshot."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def replace(text, old, new):
    if text.count(old) != 1:
        raise AssertionError((old[:100], text.count(old)))
    return text.replace(old, new, 1)


def answer(text):
    text = replace(text, 'import copy\n', 'import copy\nimport math\n')
    text = replace(text, '*,outcome_token_ids=None):',
                   '*,outcome_token_ids=None,target_span_credits=None):')
    anchor = '    def pack_hidden(self,paired_hidden):'
    extension = '''        self.credit_groups=None
        self.credit_seed_weights=None
        if target_span_credits is not None:
            if len(target_span_credits)!=self.batch:
                raise ValueError('One credit-span partition per target case is required.')
            groups=[];credits=[];group_samples=[]
            for b,(offsets,spans) in enumerate(zip(self.offsets,target_span_credits)):
                owners={}
                for span_offsets,credit in spans:
                    span_offsets=list(span_offsets);credit=float(credit)
                    if not math.isfinite(credit) or not span_offsets:
                        raise ValueError('Credit spans must be nonempty with finite credit.')
                    group=len(credits);credits.append(credit);group_samples.append(b)
                    for offset in span_offsets:
                        if offset not in offsets or offset in owners:
                            raise ValueError('Credit spans must partition the selected target offsets.')
                        owners[offset]=group
                if set(owners)!=set(offsets):
                    raise ValueError('Credit spans must cover every selected target offset.')
                groups.extend(owners[offset] for offset in offsets)
            self.credit_groups=torch.tensor(groups,dtype=torch.long)
            self.credit_budgets=torch.tensor(credits,dtype=torch.float64)
            self.credit_group_samples=group_samples

    def prepare_credit_seed(self,logp0,logp1):
        """Frozen affine span readout; no epsilon or attribution correction."""
        if self.credit_groups is None:return
        delta=logp1.detach().double().cpu()-logp0.detach().double().cpu()
        denominator=torch.zeros_like(self.credit_budgets).index_add_(0,self.credit_groups,delta)
        active=self.credit_budgets!=0
        if not bool(torch.isfinite(denominator).all()) or bool((active & (denominator==0)).any()):
            raise FloatingPointError('Nonzero target credit has a zero/nonfinite EOS score difference.')
        slopes=torch.zeros_like(denominator)
        slopes[active]=self.credit_budgets[active]/denominator[active]
        self.credit_seed_weights=slopes[self.credit_groups].to(device=self.labels.device,dtype=torch.float32)
        if not bool(torch.isfinite(self.credit_seed_weights).all()):
            raise FloatingPointError('Target-credit finite seed is not representable.')
        self.credit_span_deltas=denominator.tolist()

    def weight_target_values(self,values):
        if self.credit_seed_weights is None:return values
        return values*self.credit_seed_weights.to(device=values.device,dtype=values.dtype)

'''
    text = replace(text, anchor, extension + anchor)
    text = replace(text, 'def _answer_seed_rule(z0,z1,target,weight):',
                   'def _answer_seed_rule(z0,z1,target,weight,multipliers=None):')
    text = replace(text, '    hidden=_linear_transpose(seed,weight)',
                   '    if multipliers is not None:seed=seed*multipliers[:,None]\n    hidden=_linear_transpose(seed,weight)')
    text = replace(text, 'result=self.seed(z0,z1,target,weight)',
                   'result=self.seed(z0,z1,target,weight,multipliers=selection.credit_seed_weights)')
    if '_TEXT_TARGET_ROW_LIMIT' in text:
        text = replace(text, 'target[start:start+_TEXT_TARGET_ROW_LIMIT],weight)',
                       'target[start:start+_TEXT_TARGET_ROW_LIMIT],weight,\n                                 multipliers=None if selection.credit_seed_weights is None else selection.credit_seed_weights[start:start+_TEXT_TARGET_ROW_LIMIT])')
    return text


def runner(text):
    anchor = "        with torch.no_grad():mnorm,seed=timed('finite_seed'"
    text = replace(text, anchor,
                   '        selection.prepare_credit_seed(root_lp0,root_lp1)\n' + anchor)
    text = replace(text, 'effect_G=float((root_lp1.double()-root_lp0.double()).sum())',
                   'effect_G=float(selection.weight_target_values(root_lp1.double()-root_lp0.double()).sum())')
    text = replace(text, 'compiled_seed_G=float((lp1.double()-lp0.double()).sum())',
                   'compiled_seed_G=float(selection.weight_target_values(lp1.double()-lp0.double()).sum())')
    anchor = "        if self.defer_diagnostics:\n            flags=torch.stack(validity)"
    text = replace(text, anchor, """        if selection.credit_seed_weights is not None:
            info.update(target_seed_weights=selection.credit_seed_weights.detach().cpu().tolist(),
                        target_span_deltas=selection.credit_span_deltas,
                        target_span_credits=selection.credit_budgets.tolist(),
                        target_span_samples=selection.credit_group_samples)
""" + anchor)
    return text


def trace(text):
    text = replace(text, '    outcome_token_ids: list[int] | None = None,',
                   '    outcome_token_ids: list[int] | None = None,\n    target_span_credits: Any = None,')
    anchor = '    selection = packed_answer_targets('
    text = replace(text, anchor,
                   "    if target_span_credits is not None:options['target_span_credits']=target_span_credits\n"+anchor)
    anchor = '        roots = selection.sample_sums(delta)'
    text = replace(text, anchor, """        if 'target_seed_weights' in detail:
            delta=delta*torch.as_tensor(detail['target_seed_weights'],device=delta.device,dtype=delta.dtype)
""" + anchor)
    return text


def counterfactual(text):
    text = replace(text, '    self_target_mask: torch.Tensor | None = None,',
                   '    self_target_mask: torch.Tensor | None = None,\n    linear_attribution: bool = False,')
    text = replace(text, '    event_advantages = weighted_reward * (-torch.expm1(-d))\n    event_baselines = weighted_reward * torch.exp(-d)',
                   '''    if linear_attribution:
        # v4 receives a normalized finite allocation, not a log probability ratio.
        event_advantages = weighted_reward * d
        event_baselines = weighted_reward - event_advantages
    else:
        event_advantages = weighted_reward * (-torch.expm1(-d))
        event_baselines = weighted_reward * torch.exp(-d)''')
    return text


def readout(text):
    text = replace(text, 'import math\n', 'import math\nimport os\n')
    anchor = '        self.task, self.max_length = task, max_length'
    text = replace(text, anchor, anchor+"\n        self.weighted_target_credit = os.environ.get('DT_CREDIT_MODE') == 'weighted_reward_target_finite_v4'")
    text = replace(text, "                    ratios=torch.zeros(width, dtype=torch.float32),",
                   "                    span_credits=row.get('dt_target_span_credits'),\n                    ratios=torch.zeros(width, dtype=torch.float32),")
    anchor = "                [item['target_offsets'] for item in batch],\n                packed_answer_targets=self.packed_answer_targets,"
    text = replace(text, anchor, """                [item['target_offsets'] for item in batch],
                packed_answer_targets=self.packed_answer_targets,
                **({'target_span_credits':[item['span_credits'] for item in batch]}
                   if self.weighted_target_credit else {}),""")
    anchor = "                self_target_mask=item['target'][None, None, :],"
    text = replace(text, anchor, anchor+'\n                linear_attribution=self.weighted_target_credit,')
    anchor = "        report['raw_advantage_groups'] = groups"
    text = replace(text, anchor, """        if self.weighted_target_credit:
            report.update(target_semantics='reward_settlement_action_spans',
                          credit_transform='linear_finite_allocation',
                          output_seed='c_k / native_span_EOS_logp_difference')
""" + anchor)
    return text


def batch(text):
    anchor = "    requests.meta_info.update(eos_token_id=int(eos_token_id), pad_token_id=int(pad_token_id),\n                              dt_target_semantics='native_joint_action_target')"
    text = replace(text, anchor, """    import os
    weighted = os.environ.get('DT_CREDIT_MODE') == 'weighted_reward_target_finite_v4'
    if weighted:
        spans = np.empty(len(selected), dtype=object)
        for i,item in enumerate(selected):
            spans[i] = [(span['target_offsets'],span['credit_fraction'])
                        for span in item['reward_target_spans'] if span['target_offsets']]
        requests.non_tensor_batch['dt_target_span_credits'] = spans
"""+anchor)
    anchor = '        self_target_mask=target[:, None, :])'
    text = replace(text, anchor, '        self_target_mask=target[:, None, :], linear_attribution=weighted)')
    return text


def rollout(text):
    anchor = '        direct_artifacts = []'
    text = replace(text, anchor, "        weighted = os.environ.get('DT_CREDIT_MODE') == 'weighted_reward_target_finite_v4'\n"+anchor)
    text = replace(text, '        for h, turns in zip(handlers, records):',
                   '        for trajectory_index, (h, turns) in enumerate(zip(handlers, records)):')
    anchor = '            for turn in turns:\n                metadata = turn.get(\'executed_payload_metadata\')'
    text = replace(text, anchor, """            # AgentGym emits one terminal score and its actual task_rounds.
            # Zero scores are settlements too; no text or nonzero-reward heuristic.
            settled_turn = int(output.batch['task_rounds'][trajectory_index]) - 1
            if weighted and settled_turn != len(turns)-1:
                raise ValueError('Native reward clock differs from captured action identities')
            for turn_index, turn in enumerate(turns):
                if weighted and turn_index != settled_turn:
                    continue
                metadata = turn.get('executed_payload_metadata')""")
    anchor = "            artifact['payload_metadata'] = parser_metadata"
    text = replace(text, anchor, anchor+"""
            if weighted:
                artifact['reward_target_spans'] = [dict(
                    event_index=0, action_turn_index=settled_turn,
                    reward=float(scores[trajectory_index]), credit_fraction=1.,
                    target_offsets=[i for i,selected in enumerate(artifact['target_mask']) if selected],
                    reward_owner='AgentGym terminal scores/task_rounds')]
""")
    return text


changes = {'qwen35_answer_finite.py':answer, 'qwen35_dense_finite_runner.py':runner,
           'deltatrace_credit.py':trace, 'counterfactual.py':counterfactual,
           'reward_readout.py':readout, 'dt_training_batch.py':batch,
           'textcraft_owner_rollout.py':rollout}
manifest={}
for name,patch in changes.items():
    canonical=REPO/('deltatrace/clean/qwen35' if name.startswith('qwen35_') else 'experiments/rl')/name
    for source,destination in ((HERE/'baseline'/name,HERE/'candidate-source'/name),(canonical,canonical)):
        original=source.read_text(encoding='utf-8')
        after=patch(original)
        ast.parse(after)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(after,encoding='utf-8',newline='\n')
        manifest[str(destination.relative_to(REPO))]=hashlib.sha256(destination.read_bytes()).hexdigest()
(HERE/'prepared-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
