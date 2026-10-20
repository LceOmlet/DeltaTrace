"""Extend the unchanged all-rank metadata RPC with existing readout flags."""
from pathlib import Path
import hashlib

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/inspect_native_workers.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '7f3c940abc65da9317a9e5f74182f5080cae066a116ddd6beaebc7476b36bb1d'
code = owner.read_text().replace('textcraft-eight-gpu-sequential-20261017-v1',
    'textcraft-v5-credit-integration-20261020-v1')
anchor = "    result['DP_world_size'] = actor.world_size"
assert code.count(anchor) == 1
extension = '''
    producer=getattr(actor,'_deltatrace_producer',None)
    readout=getattr(producer,'direct_readout',None)
    result['actual_v5_readout_flags']=(None if readout is None else {
        key:getattr(readout,key,None) for key in
        ('reward_target_exponential','clip_exponential_credit','symmetric_probability_ratio','weighted_target_credit')})
    report=getattr(readout,'last_report',{}) if readout is not None else {}
    result['actual_v5_credit_report']={key:report.get(key) for key in
        ('credit_transform','source_credit_normalization','event_u_clip','coefficient_kind','finite_trace_calls')}
    result['actor_optimizer_step_values']=sorted({float(s['step'].item())
        for s in actor.actor_optimizer.state.values() if 'step' in s})
'''
code = code.replace(anchor, anchor+extension)
exec(compile(code, str(owner), 'exec'), globals())
