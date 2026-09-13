"""Start the fixed cost experiment after the existing quality/export job completes."""
import json,pathlib,re,subprocess,sys,time,traceback
ROOT=pathlib.Path('/mnt/geogpt-doc-new/deepresearch/lzq/deltatrace_qwen35_20260912')
OWN=pathlib.Path(__file__).resolve().parent
status=OWN/'cost_controller_status.json'
state=dict(status='waiting_for_quality_and_export',started_unix=time.time())
def save():
    temp=status.with_suffix('.partial');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(status)
save()
try:
    while True:
        quality=json.loads((ROOT/'paper_comparison_status.json').read_bytes())
        if quality['status']=='failed':raise RuntimeError('Upstream quality/export failed: '+quality.get('error',''))
        if quality['status']=='complete':break
        time.sleep(30)
    recovery=json.loads((ROOT/'paper_recovery_dynamic/results.json').read_bytes())
    export=json.loads((ROOT/'export_dynamic_with_ifr/verification.json').read_bytes())
    assert recovery['status']=='complete' and len(recovery['cases'])==448
    assert export['status']=='complete' and export['cases']==1243
    while True:
        gpu=subprocess.run(['/usr/bin/mx-smi'],capture_output=True,text=True,check=True)
        pids=sorted(set(map(int,re.findall(r'^\|\s+0\s+(\d+)\s+\S+',gpu.stdout,re.MULTILINE))))
        if not pids:break
        state.update(status='waiting_for_idle_gpu',other_gpu_pids=pids);save();time.sleep(30)
    # The worker also checks exclusive GPU ownership immediately after loading
    # and around every example. No other process is stopped by this controller.
    state.update(status='cost_running',cost_started_unix=time.time());save()
    command=[str(ROOT/'env/bin/python'),'-u',str(OWN/'benchmark_complete_calls.py'),
        '--run-root',str(ROOT),'--protocol',str(OWN/'protocol.json'),'--output',str(OWN/'cost_v1')]
    state['command']=command;save()
    with (OWN/'cost.log').open('xb') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0, f'Cost worker exited {result.returncode}; inspect cost.log and results.json'
    cost=json.loads((OWN/'cost_v1/results.json').read_bytes())
    assert cost['status']=='complete' and sum(len(x['calls']) for x in cost['examples'])==192
    state.update(status='complete',finished_unix=time.time());save()
except Exception:
    state.update(status='failed',error=traceback.format_exc());save();raise
