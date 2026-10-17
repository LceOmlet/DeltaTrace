"""Adapt campaign identity/empty queues around the unchanged original renderer."""
import hashlib
from pathlib import Path

source = Path(__file__).resolve().parent.parent / 'rl_textcraft_gradient_rootcause_20261013_v1/plot_current_approved_DTPO_v63.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'f6ed1ff351b9f8ea7980577d4897acd59e953bc105defe47fe4cb3e700d46244'
code = source.read_text()
start = code.index("remote = r'''")
end = code.index("'''.replace('ROOT_VALUE', repr(entry.ROOT))", start) + len("'''.replace('ROOT_VALUE', repr(entry.ROOT))")
remote = r'''
import hashlib,json,runpy
from pathlib import Path
ROOT=Path(ROOT_VALUE)
AUDIT=ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1'
state_path=AUDIT/'campaign-state.json'
state=json.loads(state_path.read_bytes())
prepared=json.loads((AUDIT/'campaign-prepared.json').read_bytes())
parser_path=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(parser_path.read_bytes()).hexdigest()=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner=runpy.run_path(str(parser_path),run_name='native_eight_GPU_comparison_parser')
snapshot=owner['collect'](str(ROOT),[j['output'] for j in state['jobs']])
actual={j['method']:j for j in state['jobs']}
for job in snapshot['jobs']:
    selected=actual[job['method']]
    assert (job['pid'],job['observed_process_created_unix'])==(selected['pid'],selected['birth'])
    assert selected['devices']==list(range(8)) and selected['budget']['total_iterations']==200
    job['campaign_status']=selected['status']
for pending in prepared['jobs']:
    if pending['method'] not in actual:
        snapshot['jobs'].append(dict(task='TextCraft',environment_task='TextCraft',method=pending['method'],
            budget=200,pid=None,pid_alive=False,observed_process_created_unix=None,
            devices=list(range(8)),gpu=0,run_dir=pending['output'],checkpoint_step=None,
            exit_code=None,metrics=[],phases=[],records=[],errors=[],campaign_status='queued_not_started'))
snapshot['jobs'].sort(key=lambda j:prepared['order'].index(j['method']))
snapshot['DT_version']=prepared['version']
snapshot['DT_display_label']='DTPO v6（8卡）'
snapshot['source_label']='TextCraft; fresh native 8-GPU sequential 200-iteration comparison'
snapshot['campaign_source']=dict(path=str(state_path),SHA256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
    status=state['status'],order=state['order'])
snapshot['original_parser_source']=dict(path=str(parser_path),SHA256=hashlib.sha256(parser_path.read_bytes()).hexdigest())
snapshot['gradient_norm_scope']='Original first-returned rank preclip shard norm; no missing global norm estimated'
snapshot['collection_model_calls']=0
print(json.dumps(snapshot))
'''
code = code[:start] + 'remote=' + repr(remote) + ".replace('ROOT_VALUE', repr(entry.ROOT))" + code[end:]
code = code.replace("assert last['dt']>0,'Current DTPO has no completed native iteration yet; source snapshot retained.'",
                    '# Pending combinations and fresh empty histories retain missing metrics.')
code = code.replace('/330', '/200').replace("label='原生 GRPO'", "label='原生 GRPO（8卡）'").replace("label='原生 PPO'", "label='原生 PPO（8卡）'")
anchor = "        if ax.lines:ax.legend(frameon=False,fontsize=8)\n        else:ax.text(.5,.5,'原日志无此指标',ha='center',va='center',transform=ax.transAxes)"
assert code.count(anchor) == 1
code = code.replace(anchor, """        has_data=bool(ax.lines)
        for job in selected:
            if not job['metrics']:
                style=dict(styles[job['method']])
                style['label']+='（待运行）' if job.get('campaign_status')=='queued_not_started' else '（无完成迭代）'
                ax.plot([],[],**style)
        if ax.lines:ax.legend(frameon=False,fontsize=8)
        if not has_data:
            ax.text(.5,.5,'共同窗口尚为空；缺失不填零' if limit==0 else '原日志尚无此指标',ha='center',va='center',transform=ax.transAxes)""")
exec(compile(code, str(source), 'exec'), globals())
