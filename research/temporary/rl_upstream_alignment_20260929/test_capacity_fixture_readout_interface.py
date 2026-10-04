"""CPU contracts for the original capacity fixture's readout option seam.

Source tests require no training runtime. Optional Torch tests call the exact
helper and current EventRatioReadout._prepare_episode, without a model/runner
forward. The tokenizer is an explicit test double, not a Qwen token-count or
capacity-performance claim.
"""
import ast
import copy
import hashlib
import os
from pathlib import Path

import pytest


AUDIT=Path(__file__).resolve().parent
REPO=AUDIT.parents[2]
SOURCE=Path(os.environ.get('DT_CAPACITY_FIXTURE_SOURCE',
    REPO/'experiments/rl/verify_dt_context_capacity.py'))
ORIGINAL_AST_SHA='874880fc2b3a0f1bd41c60ecdd3f4030739cca77b78116807cfdf3dd03c37271'


def helper_node():
    return next(node for node in ast.parse(SOURCE.read_bytes()).body
                if isinstance(node,ast.FunctionDef) and node.name=='capacity_fixture')


def legacy_node():
    original=copy.deepcopy(helper_node())
    original.args.kwonlyargs=[]
    original.args.kw_defaults=[]
    query=next(node for node in ast.walk(original) if isinstance(node,ast.Call)
               and ast.unparse(node.func)=='alphabet.query_ids')
    query.keywords=[keyword for keyword in query.keywords if keyword.arg!='sampling']
    next(keyword for keyword in query.keywords if keyword.arg=='max_steps').value=ast.Constant(value=15)
    assert hashlib.sha256(ast.dump(original).encode()).hexdigest()==ORIGINAL_AST_SHA
    return original


def test_only_existing_query_owner_options_changed():
    legacy_node()  # Every other original AST statement and default stays identical.
    node=helper_node()
    assert [arg.arg for arg in node.args.kwonlyargs]==['max_steps','sampling']
    assert [ast.literal_eval(value) for value in node.args.kw_defaults]==[15,None]
    query=next(item for item in ast.walk(node) if isinstance(item,ast.Call)
               and ast.unparse(item.func)=='alphabet.query_ids')
    assert {item.arg:ast.unparse(item.value) for item in query.keywords}=={
        'current_step':'step','max_steps':'max_steps','sampling':'sampling'}


def actual_helper(torch,*,legacy=False):
    namespace=dict(torch=torch,hashlib=hashlib)
    node=legacy_node() if legacy else helper_node()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 str(SOURCE),'exec'),namespace)
    return namespace['capacity_fixture']


class TokenizerDouble:
    eos_token_id=99

    def encode(self,text,add_special_tokens=False):
        # Exact text identity yields exact test-token identity; not Qwen BPE.
        return list(text.encode('utf8'))


def original_row(torch,*,step):
    return dict(traj_uid='fixture',env_step=step,active_masks=True,rewards=1.,
        responses=torch.tensor([71,72,0,0]),
        input_ids=torch.tensor([0,31,32,71,72,0,0]),
        attention_mask=torch.tensor([0,1,1,1,1,0,0]))


def test_default_helper_preserves_original_tokens_and_details(monkeypatch):
    torch=pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(REPO/'experiments/rl'))
    from reward_readout import RewardAlphabet
    original=original_row(torch,step=0)
    tokenizer=TokenizerDouble()
    alphabet=RewardAlphabet.for_task('AppWorld',appworld_num_tests=2)
    before=actual_helper(torch,legacy=True)(original,tokenizer,alphabet,response_tokens=512)
    after=actual_helper(torch)(original,tokenizer,alphabet,response_tokens=512)
    for key in ('responses','input_ids','attention_mask'):
        assert torch.equal(before[0][key],after[0][key])
    assert torch.equal(before[1],after[1])
    assert before[2:]==after[2:]


@pytest.mark.parametrize('max_steps,step,sampling',[
    (15,0,None),(40,25,None),(40,25,dict(temperature=1.,max_tokens=1500)),
])
def test_helper_factual_matches_current_readout_prepare_exactly(monkeypatch,max_steps,step,sampling):
    torch=pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(REPO/'experiments/rl'))
    from reward_readout import EventRatioReadout
    tokenizer=TokenizerDouble()
    readout=EventRatioReadout(object(),tokenizer,task='AppWorld',max_steps=max_steps,
        packed_answer_targets=object(),appworld_num_tests=2,sampling=sampling)
    row,factual,detail,_=actual_helper(torch)(original_row(torch,step=step),tokenizer,
        readout.alphabet,response_tokens=512,max_steps=readout.max_steps,sampling=readout.sampling)
    report=dict(nonzero_reward_events=0,policy_tokens=0,actual_row_lengths=[])
    _,requests=readout._prepare_episode([row],report,[1.])
    assert len(requests)==1
    request=requests[0]
    prepared=torch.cat([request[name] for name in ('prompt','actions','query','target')])
    assert torch.equal(prepared,factual)
    assert request['context_tokens']==detail['dt_input_tokens']==32768
    assert request['query_tokens']==detail['query_tokens']
    assert request['source_step']==step and request['actions'].numel()==512
