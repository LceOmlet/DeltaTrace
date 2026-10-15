"""Real owner DP transport of distinct credited target events.

The worker uses sentinel log effects only to test packaging/composition. The
separate native GPU check uses the actual DT owner at these same interfaces.
"""
from copy import deepcopy

import pytest
import torch

from counterfactual import reward_event_token_credit
from dt_training_batch import CREDIT_KEYS,compute_direct_target_credit
from test_direct_target_transport import artifact,native_training_data
from verl import DataProto
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.base.decorator import dispatch_dp_compute_data_proto,collect_dp_compute_data_proto


class EventTransportWorker(WorkerGroup):
    def __init__(self):
        self._workers=[None,None];self.calls=[];self.returned={}
    def compute_dt_token_advantages(self,data):
        args,kwargs=dispatch_dp_compute_data_proto(self,data);assert not kwargs
        chunks=args[0];self.calls.append(chunks)
        outputs=[]
        for chunk in chunks:
            assert len(chunk)%4==0
            # A distinct per-event DT response, rather than a joint sum.
            position=torch.arange(chunk.batch['responses'].shape[-1],dtype=torch.float32)
            effect=position[None,:]*.125+chunk.batch['target_mask'].float().sum(-1)[:,None]*.25
            effect=effect+chunk.batch['dt_direct_reward'].float()[:,None]*.5
            values=reward_event_token_credit(effect[:,None,:],chunk.batch['dt_direct_reward'].float()[:,None],
                chunk.batch['dt_future_event_mask'][:,None,:],chunk.batch['policy_mask'],
                self_target_mask=chunk.batch['target_mask'][:,None,:])
            result=DataProto.from_dict(tensors=dict(dt_token_advantages=values.advantages,
                dt_q_estimates=values.q_estimates,dt_v_estimates=values.v_estimates))
            outputs.append(result)
            for row,uid in enumerate(chunk.non_tensor_batch['traj_uid']):
                self.returned[str(uid)]={key:result.batch[key][row].clone() for key in CREDIT_KEYS}
        return collect_dp_compute_data_proto(self,outputs)


def events(item,credits=(-.1,1.)):
    width=len(item['response_ids'])
    item['dt_reward_target_events']=[
        dict(event_id='first',target_offsets=[1],credit=credits[0],
             future_policy_mask=[i<=1 for i in range(width)]),
        dict(event_id='second',target_offsets=[4],credit=credits[1],
             future_policy_mask=[i<=4 for i in range(width)])]
    return item


@pytest.mark.parametrize('credits',[(-.1,1.),(-1.,1.)])
def test_multitarget_DP_preserves_signed_components_and_retained_slots(credits):
    items=[events(artifact(i),credits) for i in range(5)]
    # Duplicate an original trajectory with different retained response slots.
    repeated=deepcopy(items[0]);repeated['retained_response_positions']=[3,4,5,-1]
    data=native_training_data(items+[repeated],identities=[f'uid-{i}' for i in range(5)]+['uid-0'],
                              multipliers=[sum(credits)]*6)
    worker=EventTransportWorker()
    output=compute_direct_target_credit(data,worker,eos_token_id=99,pad_token_id=0)
    assert len(worker.calls)==1
    assert [len(chunk) for chunk in worker.calls[0]]==[8,8]  # 10 events, original padding to B16
    assert len(worker.returned)==10
    for row,item in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
        uid=str(data.non_tensor_batch['traj_uid'][row])
        per_event=[worker.returned[str((uid,'str',event['event_id']))] for event in item['dt_reward_target_events']]
        for key in CREDIT_KEYS:
            expected=per_event[0][key]+per_event[1][key]
            for slot,position in enumerate(item['retained_response_positions']):
                assert output.batch[key][row,slot]==(expected[position] if position>=0 else 0)
    if sum(credits)==0:
        assert bool(output.batch['dt_token_advantages'].any())  # Not skipped by rollout g=0.


def test_earlier_target_is_source_for_later_event_and_future_masks_are_retained():
    item=events(artifact(0))
    data=native_training_data([item],multipliers=[.9]);worker=EventTransportWorker()
    compute_direct_target_credit(data,worker,eos_token_id=99,pad_token_id=0)
    rows={str(uid): (chunk,i) for chunk in worker.calls[0]
          for i,uid in enumerate(chunk.non_tensor_batch['traj_uid'])}
    chunk,row=rows[str(('uid-0','str','second'))]
    assert chunk.batch['policy_mask'][row,1] and not chunk.batch['target_mask'][row,1]
    assert chunk.batch['dt_future_event_mask'][row].tolist()==[True,True,True,True,True,False]
    assert not chunk.batch['policy_mask'][row,2]  # Observation remains excluded.


def test_multiple_payload_pieces_of_one_event_are_not_credited_twice():
    item=artifact(0)
    item['dt_reward_target_events']=[dict(event_id=0,target_offsets=[1,4],credit=.75,
                                         future_policy_mask=[True]*6)]
    worker=EventTransportWorker()
    result=compute_direct_target_credit(native_training_data([item]),worker,eos_token_id=99,pad_token_id=0)
    assert len(worker.returned)==1
    assert result.batch['dt_q_estimates'][0,0]==.75
    assert result.batch['dt_token_advantages'][0,1]==.75


def test_mixed_event_counts_and_zero_component_keep_native_equal_rank_calls():
    items=[events(artifact(0)),artifact(1),events(artifact(2))]
    items[2]['dt_reward_target_events'].append(dict(event_id='zero',target_offsets=[1,4],credit=0.,
                                                   future_policy_mask=[True]*6))
    worker=EventTransportWorker()
    result=compute_direct_target_credit(native_training_data(items),worker,eos_token_id=99,pad_token_id=0)
    assert len(worker.calls)==1 and [len(chunk) for chunk in worker.calls[0]]==[4,4]
    assert len(worker.returned)==5
    assert not any("'zero'" in uid for uid in worker.returned)
    assert all(bool(torch.isfinite(result.batch[key]).all()) for key in CREDIT_KEYS)


@pytest.mark.parametrize('defect',['duplicate_event','missing_credit','observation_target','wrong_future_slots'])
def test_invalid_event_contract_is_rejected_before_collective(defect):
    item=events(artifact(0));worker=EventTransportWorker()
    if defect=='duplicate_event':item['dt_reward_target_events'][1]['event_id']='first'
    elif defect=='missing_credit':del item['dt_reward_target_events'][1]['credit']
    elif defect=='observation_target':item['dt_reward_target_events'][1]['target_offsets']=[2]
    else:item['dt_reward_target_events'][1]['future_policy_mask']=[True]
    with pytest.raises(ValueError):
        compute_direct_target_credit(native_training_data([item]),worker,eos_token_id=99,pad_token_id=0)
    assert not worker.calls
