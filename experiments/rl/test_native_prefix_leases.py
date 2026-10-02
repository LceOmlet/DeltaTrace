"""Exact-artifact and unchanged credit-boundary contracts, not DT numerics."""
import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
OWNER_PATH = Path(os.environ.get('DT_PREFIX_ARTIFACT_SOURCE',
    ROOT/'deltatrace/clean/qwen35/qwen35_native_prefix_artifacts.py'))


def test_moved_artifact_body_is_the_recorded_incremental_owner():
    old = ast.parse(subprocess.check_output(['git','show',
        '6ef58eb:research/temporary/rl_upstream_alignment_20260929/native_prefix_artifacts_candidate.py'],
        cwd=ROOT,text=True,encoding='utf8'))
    new = ast.parse(OWNER_PATH.read_text(encoding='utf8'))
    new.body.pop()  # Only the new exact data lease was added after the move.
    old.body.pop(0);new.body.pop(0)  # Module documentation.
    assert ast.dump(old) == ast.dump(new)


def test_lease_preserves_identity_and_rejects_a_different_factual_prefix(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(OWNER_PATH.parent))
    import qwen35_native_prefix_artifacts as owner
    source = SimpleNamespace(config=object(),input_ids=torch.arange(3*128).reshape(3,128))
    seen=[]
    monkeypatch.setattr(owner,'compose_native_prefix_cache',lambda *a,**kw: seen.append((a,kw)))
    lease=owner.NativePrefixLease([(source,2),(source,0)],64)
    ids=source.input_ids[[2,0],:64].clone()
    lease(ids)
    assert len(seen)==1 and seen[0][0] == (source.config,lease.sources,64)
    wrong=ids.clone();wrong[0,0] += 1
    with pytest.raises(ValueError,match='exact factual token IDs'):lease(wrong)
    assert len(seen)==1  # Mismatch is rejected before the owning Cache call.


def test_explicit_factory_preserves_all_owner_endpoints_and_token_qva():
    torch = pytest.importorskip('torch')
    from test_reward_readout import BatchedRunner,readout,row
    inputs=[row(i,-.1) for i in range(5)]
    original=BatchedRunner();expected=readout(original,minibatch_size=4).episode(copy.deepcopy(inputs))
    leases=[object(),object()];observed=[]
    def factory(runner,requests,**kwargs):
        observed.append(([(r['traj_uid'],r['source_step'],r['context_tokens']) for r in requests],kwargs))
        return leases,{'measurement':'explicit test-double factory, no model/cache computation'}
    actual_runner=BatchedRunner();calls=actual_runner.attribute
    def attribute(*args,**kwargs):
        index=len(actual_runner.calls)
        assert kwargs.pop('prefix_cache_provider') is leases[index]
        return calls(*args,**kwargs)
    actual_runner.attribute=attribute
    dt=readout(actual_runner,minibatch_size=4,prefix_lease_factory=factory)
    result=dt.episode(inputs)
    assert len(observed)==1 and observed[0][1]==dict(minibatch_size=4,eos_token_id=99)
    for left,right in zip(original.calls,actual_runner.calls):
        assert torch.equal(left[0],right[0]) and torch.equal(left[1],right[1])
    for left,right in zip(expected,result):
        for key in left:assert torch.equal(left[key],right[key])
    assert dt.last_report['event_contrasts']==5


@pytest.mark.parametrize('other_rank_capture_rows',[None,8])
def test_factory_uses_native_padding_and_only_factual_history(monkeypatch,other_rank_capture_rows):
    torch = pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(OWNER_PATH.parent))
    import qwen35_native_prefix_artifacts as owner
    from native_prefix_leases import prepare_native_prefix_leases
    calls=[]
    def capture(model,ids,lengths,**kwargs):
        calls.append((ids.clone(),lengths,kwargs))
        return SimpleNamespace(input_ids=ids.cpu(),config=kwargs['config'])
    monkeypatch.setattr(owner.NativePrefixArtifacts,'capture',capture)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    group=object()
    mesh=SimpleNamespace(ndim=1,size=lambda:2,get_group=lambda:group)
    weight=SimpleNamespace(device_mesh=mesh) if other_rank_capture_rows else torch.empty(1)
    if other_rank_capture_rows:
        def all_reduce(value,*,op,group):
            assert op is torch.distributed.ReduceOp.MAX
            value.fill_(other_rank_capture_rows)
        monkeypatch.setattr(torch.distributed,'all_reduce',all_reduce)
    runner=SimpleNamespace(model=SimpleNamespace(lm_head=SimpleNamespace(weight=weight),
        execution_device=torch.device('cpu'),_conditional=SimpleNamespace(config=object()),
        synchronize_prefix_start=lambda n:n))
    requests=[]
    for width in (128,192,256):
        for uid in ('first','second'):
            prompt=torch.arange(width)+(1000 if uid=='second' else 0)
            requests.append(dict(traj_uid=uid,prompt=prompt,start=width,
                actions=torch.tensor([-11,-12]),query=torch.tensor([-13]),target=torch.tensor([-14])))
    leases,report=prepare_native_prefix_leases(runner,requests,minibatch_size=4,eos_token_id=99)
    assert [lease.prefix_length for lease in leases]==[128,256]
    expected_calls=2 if other_rank_capture_rows else 1
    assert len(calls)==report['capture_rounds']==expected_calls
    assert report['local_unique_histories']==2
    for ids,lengths,kwargs in calls:
        assert ids.shape==(4,256) and lengths==[128,256]
        assert not torch.isin(ids,torch.tensor([-11,-12,-13,-14])).any()
        assert kwargs['config'] is runner.model._conditional.config
    assert all(len(lease.sources)==count for lease,count in zip(leases,[4,2]))


def test_factory_rejects_incompatible_history_before_capturing(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(OWNER_PATH.parent))
    import qwen35_native_prefix_artifacts as owner
    from native_prefix_leases import prepare_native_prefix_leases
    monkeypatch.setattr(owner.NativePrefixArtifacts,'capture',lambda *a,**k: pytest.fail('No capture on mismatched IDs'))
    runner=SimpleNamespace(model=SimpleNamespace(synchronize_prefix_start=lambda n:n))
    requests=[dict(traj_uid='same',start=128,prompt=torch.arange(128)),
              dict(traj_uid='same',start=192,prompt=torch.arange(192)+1)]
    with pytest.raises(ValueError,match='do not share'):
        prepare_native_prefix_leases(runner,requests,minibatch_size=4,eos_token_id=99)
