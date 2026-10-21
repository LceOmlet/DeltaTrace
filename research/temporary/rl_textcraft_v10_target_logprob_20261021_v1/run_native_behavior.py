"""Reuse the v5 native B4/DP2 harness; observe the existing endpoint scores and Q/V/A."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-v10-target-logprob-20261021-v1'
owner=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source/run_native_behavior_v5.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='ae4ba91b1978d572c17a5ab4589c0d2804be1eefb8d59d1de308f17d22815541'
code=owner.read_text()


def once(before,after):
    global code
    assert code.count(before)==1,before
    code=code.replace(before,after)


once('Native B4/rank credit transport and exact-32768 capacity, zero optimizer.',
     'Native DP2 B4 target endpoint integration only; no capacity experiment or optimizer update.')
once("AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'",
     "AUDIT=ROOT/'receipts/textcraft-v10-target-logprob-20261021-v1'")
once('base.AUDIT=AUDIT',"base.AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'")
once("AUDIT/'prepared-launch.json'","AUDIT/'dt-prepared.json'")
once("_temp_dir='/tmp/ray-tc-exponential-v5'","_temp_dir='/tmp/ray-textcraft-v10-native-20261021'")
once('        import reward_readout', '''        import reward_readout
        expected_readout=ROOT/'candidates/textcraft-v10-target-logprob-20261021-v1/entry/reward_readout.py'
        assert Path(reward_readout.__file__).resolve()==expected_readout
        assert hashlib.sha256(expected_readout.read_bytes()).hexdigest()==json.loads(
            (AUDIT/'dt-prepared.json').read_bytes())['source_bindings'][str(expected_readout)]''')
once("            native_attribute=runner.attribute;native_credit=reward_readout.reward_event_token_credit",
     "            native_attribute=runner.attribute;native_credit=reward_readout.reward_event_token_credit\n"
     "            endpoint_calls=[]")
once("                signed,detail=native_attribute(pair,mask,selection,*args,**kwargs)",
     "                signed,detail=native_attribute(pair,mask,selection,*args,**kwargs)\n"
     "                endpoint_calls.append(dict(pair=pair.detach().cpu(),signed=signed.detach().cpu(),\n"
     "                    samples=selection.samples.detach().cpu(),positions=selection.positions.detach().cpu(),\n"
     "                    labels=selection.labels.detach().cpu(),\n"
     "                    logp0=torch.as_tensor(detail['target_logp0']).detach().cpu(),\n"
     "                    logp1=torch.as_tensor(detail['target_logp1']).detach().cpu()))\n"
     "                torch.save(endpoint_calls[-1],AUDIT/f'{stage}-rank{self.rank}-endpoint-call{len(endpoint_calls)}.pt')")
once("                own=kwargs['self_target_mask'];active=future&policy[:,None,:]&(b[:,:,None]!=0)",
     "                own=kwargs['self_target_mask'];assert own is None\n"
     "                active=future&policy[:,None,:]&(b[:,:,None]!=0)")
once('                estimated=active&~own','                estimated=active')
once('                expected=torch.where(own,weighted,expected).sum(1)','                expected=expected.sum(1)')
start=code.index('            for i,row in enumerate(to_list_of_dict(data)):')
end=code.index("            state.update(status='completed'",start)
code=code[:start]+'''
            torch.save(dict(original_request=data,returned_native_DataProto=output),
                       AUDIT/f'{stage}-rank{self.rank}-native-output-before-oracle.pt')
            readout=producer.direct_readout
            assert readout.target_token_logprob
            prepared=[readout._prepare_row(row,i) for i,row in enumerate(to_list_of_dict(data))]
            requests=sorted([item for item in prepared if item['reward']!=0
                and item['target_offsets'] and bool(item['prior'].any())],key=lambda item:item['selected'].numel())
            offset=0;target_checks=0
            for call in endpoint_calls:
                count=call['pair'].shape[0]//2
                batch=requests[offset:offset+count]
                for sample,item in enumerate(batch):
                    positions=call['positions'][call['samples']==sample]+1
                    labels=call['labels'][call['samples']==sample]
                    assert torch.equal(item['selected'][positions],labels)
                    assert torch.equal(call['pair'][2*sample+1,:item['selected'].numel()],item['selected'])
                    d=torch.zeros(item['width'],dtype=torch.float32)
                    prior=item['prior'][item['suffix_positions']]
                    values=call['signed'][sample,item['prompt_length']:item['selected'].numel()]
                    d[item['suffix_positions'][prior]]=values[prior].to(dtype=d.dtype)
                    # Independent scatter uses the native causal predictor positions,
                    # rather than repeating the readout helper's target-mask mapping.
                    suffix_positions=item['suffix_positions'][positions-item['prompt_length']]
                    d[suffix_positions]=(call['logp1']-call['logp0'])[call['samples']==sample].to(dtype=d.dtype)
                    expected=native_credit(d[None,None,:],torch.tensor([[item['reward']]]),
                        item['future_event_mask'][None,None,:],item['policy'][None,:])
                    for key,value in [('dt_q_estimates',expected.q_estimates),
                                      ('dt_v_estimates',expected.v_estimates),
                                      ('dt_token_advantages',expected.advantages)]:
                        assert torch.equal(output.batch[key][item['index']].cpu(),value[0]),key
                    target_checks+=len(positions)
                offset+=count
            assert offset==len(requests) and target_checks>0
            torch.save(dict(native_endpoint_calls=endpoint_calls,
                original_request=data,returned_native_DataProto=output,
                target_tokens_verified=target_checks),AUDIT/f'{stage}-rank{self.rank}-native-tensors.pt')
            state['target_endpoint_integration']=dict(target_tokens_verified=target_checks,
                exact_native_QVA_expression=True,independent_native_predictor_position_scatter=True,
                self_target_constant_override=False,extra_forward_for_target_credit=0)
'''+code[end:]
once("            assert state['report']['credit_transform']=='per_event_exponential_then_credit_sum'",
     "            assert state['report']['credit_transform']=='per_event_exponential_then_credit_sum'\n"
     "            assert state['report']['target_self_constant_override'] is False")
start=code.index('        reused=AUDIT/')
end=code.index("        write(AUDIT/'native-behavior-status.json',state)",
               code.index("        state.update(status='completed'",start))
code=code[:start]+'''        state['status']='native-B4-target-endpoint';write(AUDIT/'native-behavior-status.json',state)
        group.verify_credit(small)
        state.update(status='completed',finished=time.time(),
                     validation_stages=['native-DP2-B4-target-endpoint-integration'],
                     DT_kernel_or_model_changes=0,additional_forward_for_target_credit=0)
'''+code[end:]
(AUDIT/'source/composed-native-behavior-owner.py').write_text(code)
exec(compile(code,str(owner),'exec'),globals())
