"""Apply one verified existing DT storage option at an idle worker RPC.

This does not change FA/FLA/VERL arithmetic or invoke a model. The original
read-only metadata owner supplies process/import identities. Its final receipt
is explicitly relabelled as a runtime resource override, not read-only work.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
capacity_path=AUDIT/'root-pageable-native-capacity-results.json'
pair_path=AUDIT/'existing-DT-root-pageable-pair-v2-results.json'
capacity=json.loads(capacity_path.read_bytes())
pair=json.loads(pair_path.read_bytes())
assert capacity['status']=='completed' and sorted(capacity['completed_ranks'])==list(range(8))
assert pair['successful_same_input_pair_count']==8
for row in pair['same_input_pairs']:
    assert row['pair']['raw_DT_vector_equal'] and row['pair']['rng_unchanged']
    assert row['pair']['original_options_restored']
for row in capacity['rows']:
    value=row['value']
    assert value['status']=='completed_original_owner_capacity'
    assert value['original_options_restored'] and value['original_method_restored']
    assert value['RNG_unchanged'] and all(value['all_QVA_finite'].values())
    if value['rank'] in (0,1):
        assert value['native_report']['actual_context_lengths']==[32768]*4
        assert value['native_report']['minibatch_size']==4

owner=Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code=owner.read_text().replace('v9-negative2-eight-GPU-initial-v1','root-pageable-resource-accepted-v1')
anchor="exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor)==1
override='''    import time,inspect
    producer=getattr(actor,'_deltatrace_producer',None)
    assert producer is not None
    runner=producer.runner
    source=Path(inspect.getfile(type(runner)))
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
    before=(runner.pin_root_host,runner.pin_replay_host)
    assert before==(True,True), before
    assert producer.attribute_prepared_batch.__name__=='attribute_prepared_batch'
    runner.pin_root_host=False
    result['root_pageable_resource_override']=dict(
        status='applied_existing_root_storage_option',applied_unix=time.time(),
        before=dict(pin_root_host=before[0],pin_replay_host=before[1]),
        after=dict(pin_root_host=runner.pin_root_host,pin_replay_host=runner.pin_replay_host),
        native_runner=str(source),native_runner_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        actor_scheduler_epoch=actor.actor_lr_scheduler.last_epoch,
        arithmetic_changed=False,model_calls=0,DT_calls=0,optimizer_updates=0,
        capacity_scope='B4/exact32768 on two ranks with all eight participating; not eight simultaneous max-length capacity',
        persistence='Current loaded runner only; original environment/source receipt remains unchanged',
        verification_receipts=['root-pageable-native-capacity-results.json','existing-DT-root-pageable-pair-v2-results.json'])
    return result'''
code=code.replace(anchor,
    "assert code.count('    return result')==1\ncode=code.replace('    return result',"+repr(override)+")\n"+
    "code=code.replace('completed_read_only_worker_binding_check','completed_runtime_root_storage_override')\n"+
    "code=code.replace('Unchanged original metadata RPC; absent lazy modules remain unverified','Original metadata RPC with accepted root-only storage override; arithmetic unchanged; current loaded runners only')\n"+
    anchor)
exec(compile(code,str(owner),'exec'),globals())
