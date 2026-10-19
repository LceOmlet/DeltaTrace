"""One original-capacity call inside the next existing native DT RPC.

Use the original prepared-batch owner, then return the original training batch.
Ranks0/1 consume B4/exact32768, others the original small B4, preserving the
same FSDP collective sequence without eight simultaneous 32k host root caches.
This is per-rank capacity validation, not an eight-rank all-32k memory claim.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1','root-pageable-native-capacity-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
observation = '''    import json,time,torch
    producer=getattr(actor,'_deltatrace_producer',None)
    runner=getattr(producer,'runner',None)
    result['root_pageable_capacity_installed']=False
    fixture_path=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4/root-pageable-capacity-inputs.pt')
    fixture_receipt=json.loads(fixture_path.with_suffix('.json').read_bytes())
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest()==fixture_receipt['inputs_SHA256']
    if producer is not None and runner is not None and not getattr(producer,'_root_pageable_capacity_v1',False):
        assert (runner.pin_root_host,runner.pin_replay_host)==(True,True)
        native=producer.attribute_prepared_batch
        def verify_then_native(data):
            producer.attribute_prepared_batch=native
            flags=(runner.pin_root_host,runner.pin_replay_host)
            cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
            output_path=fixture_path.with_name('root-pageable-capacity-rank'+str(actor.rank)+'.json')
            state=dict(status='started_capacity_validation',rank=actor.rank,pid=os.getpid(),
                started=time.time(),fixture=str(fixture_path),
                fixture_kind='capacity' if actor.rank in (0,1) else 'small',
                additional_validation_prepared_batch_calls=1,additional_optimizer_calls=0,
                added_production_calls=0,candidate_deployed=False,
                scope='B4/rank exact32768 on ranks0/1; other ranks participate using original small fixture. Not all-eight-rank 32k capacity or production speed.')
            output_path.write_text(json.dumps(state,indent=2)+chr(10))
            try:
                fixture=torch.load(fixture_path,map_location='cpu',weights_only=False)[state['fixture_kind']]
                state['input_shape']=list(fixture.batch['input_ids'].shape)
                assert len(fixture)==4
                if actor.rank in (0,1):assert state['input_shape']==[4,32768]
                runner.pin_root_host=False;runner.pin_replay_host=True
                start=time.perf_counter()
                candidate=native(fixture)
                state['candidate_seconds']=time.perf_counter()-start
                state['native_report']=producer.direct_readout.last_report
                if actor.rank in (0,1):
                    assert state['native_report']['actual_context_lengths']==[32768]*4
                    assert state['native_report']['minibatch_size']==4
                values={key:candidate.batch[key] for key in ('dt_q_estimates','dt_v_estimates','dt_token_advantages')}
                state['all_QVA_finite']={key:bool(torch.isfinite(value).all()) for key,value in values.items()}
                assert all(state['all_QVA_finite'].values())
                state['numerical_scope']='Finiteness and native interface/capacity only; no new official numerical tolerance claim'
                state['RNG_unchanged']=bool(torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()))
                state['status']='completed_original_owner_capacity'
                del candidate,values,fixture
            except Exception as error:
                state.update(status='candidate_capacity_failed',error=type(error).__name__+': '+str(error))
            finally:
                runner.pin_root_host,runner.pin_replay_host=flags
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                state.update(finished=time.time(),original_options_restored=(runner.pin_root_host,runner.pin_replay_host)==flags,
                             original_method_restored=producer.attribute_prepared_batch==native)
                output_path.write_text(json.dumps(state,indent=2,allow_nan=False)+chr(10))
            return native(data)
        producer.attribute_prepared_batch=verify_then_native
        producer._root_pageable_capacity_v1=True
        result['root_pageable_capacity_installed']=True
    result['root_pageable_capacity_scope']='One next existing DT RPC; two ranks B4/32768, other six original small B4; no new model process or optimizer call'
    return result'''
code = code.replace(anchor, "assert code.count('    return result')==1\ncode=code.replace('    return result',"+repr(observation)+")\n"+anchor)
exec(compile(code,str(owner),'exec'),globals())
