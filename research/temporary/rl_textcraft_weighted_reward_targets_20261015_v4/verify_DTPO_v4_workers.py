"""Reuse original initial-state RPCs and the math-preserving v4 recorder."""
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4'
source=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/verify_DT_group_workers_v40.py'
code=source.read_text()
old="job = json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())"
assert old in code
code=code.replace(old,"job = json.loads((ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/DTPO-v4-job.json').read_bytes())")
code=code.replace('initial-state-and-existing-passive-records-v40.json','initial-state-and-existing-passive-records-v4.json')
code=code.replace("passive_path=OUT/'install_passive_records_native_graph.py'",
    "passive_path=ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source/install_passive_records_v4.py'")
extra='''
def read_v4_environment(worker):
    import os
    return dict(pid=os.getpid(),credit_mode=os.environ.get('DT_CREDIT_MODE'),
                DT_ROOT=os.environ.get('DT_ROOT'),VERL_ROOT=os.environ.get('VERL_ROOT'))

'''
code=code.replace('def main():',extra+'def main():',1)
old="        report.update(complete=True,completed_unix=time.time(),status='actual_initial_owner_and_existing_passive_recorder_verified')"
assert old in code
code=code.replace(old,"""        report['actual_v4_environment']=ray.get([a.execute_with_func_generator.remote(func=read_v4_environment) for a in actors],timeout=60)
        for value in report['actual_v4_environment']:
            assert value['credit_mode']=='weighted_reward_target_finite_v4'
            assert value['DT_ROOT']==launch['effective_environment']['DT_ROOT']
            assert value['VERL_ROOT']==launch['effective_environment']['VERL_ROOT']
"""+old)
exec(compile(code,str(source),'exec'),globals())
