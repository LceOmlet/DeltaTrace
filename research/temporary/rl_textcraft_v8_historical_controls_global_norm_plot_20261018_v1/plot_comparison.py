"""Reuse the accepted historical-control plot; put norm estimates on the same panel.

Native metric values are retained. Only a separately named plot metric is
converted, with its factor, assumption and original value exported to CSV.
"""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
owner=HERE.parent/'rl_textcraft_equal_workload_comparison_20261018_v1/plot_comparison.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='78d3d75a28f37ad178a8309fafb3115d801b33dc929549c136f2c12eed69c146'
source=owner.read_text(encoding='utf-8')
source=source.replace('textcraft-eight-gpu-sequential-20261018-v5',
    'textcraft-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8')
source=source.replace('textcraft-dtpo-v6-8gpu-200-20261018-v5',
    'textcraft-dtpo-symmetric-probability-ratio-postgroup-variance-v8-8gpu-200-20261018-v8')
source=source.replace('textcraft-equal-workload-current-DTPO-historical-controls-20261018-v1',
    'textcraft-v8-historical-PPO-GRPO-global-norm-20261018-v1')
source=source.replace('DTPO v6（8卡）','DTPO v8（8卡）')

projection='''
import math
conversion=[]
for job in snapshot['jobs']:
    world_size=len(job['devices'])
    is_current=job['method']=='dt'
    assert world_size==(8 if is_current else 2)
    factor=1.0 if is_current else math.sqrt(world_size)
    scope=('measured_global_preclip_native_DTensor_full_tensor' if is_current
           else 'estimated_global_preclip_equal_shard_gradient_energy')
    job['gradient_display_rule']=dict(scope=scope,factor=factor,world_size=world_size,
        original_field='actor/grad_norm',plot_field='plot/grad_norm_global',
        assumption=None if is_current else 'Equal gradient squared norm in each shard at each native optimizer update')
    for metric in job['metrics']:
        raw=metric['values'].get('actor/grad_norm')
        if raw is not None and math.isfinite(raw):
            metric['values']['plot/grad_norm_global']=raw*factor
    conversion.append(dict(algorithm=job['method'],run=job['run_dir'],**job['gradient_display_rule']))
snapshot['gradient_norm_conversion']=dict(status='explicit_estimates_for_historical_controls_only',
    native_values_overwritten=False,per_update_energy_balance_assumption=True,rules=conversion,
    historical_global_norms_measured=False,training_changes=0)
snapshot['comparison_scope']='Current v8 and independent historical PPO/GRPO; no run concatenation. Native eight-GPU controls remain separate queued experiments.'

def _norm_csv_fields(job,metric,key):
    rule=job['gradient_display_rule']
    if key=='plot/grad_norm_global':
        return [rule['scope'],rule['factor'],metric['values']['actor/grad_norm']]
    if key in ('actor/grad_norm','actor/grad_norm_local'):
        scope=('measured_global_preclip_native_DTensor_full_tensor'
            if key=='actor/grad_norm' and job['method']=='dt' else 'first_returned_rank_preclip_local_shard')
        return [scope,1.0,metric['values'][key]]
    return ['', '', '']
'''
anchor="snapshot['renderer_source'] = dict(path=str(renderer), SHA256=renderer_sha)\n"
assert source.count(anchor)==1
source=source.replace(anchor,anchor+projection)

presentation='''
code=code.replace("('actor/grad_norm','首 rank 分片梯度范数（原生日志值）'",
    "('plot/grad_norm_global','全局梯度范数（历史对照为估算）'")
code=code.replace("('plot/grad_norm_global','全局梯度范数（历史对照为估算）','每迭代原日志值')",
    "('plot/grad_norm_global','全局梯度范数（历史对照为估算）','裁剪前全局范数')")
plot_line="                ax.plot(xs,ys,linewidth=1.7,markersize=4,**styles[job['method']])"
assert code.count(plot_line)==1
code=code.replace(plot_line,"""                plot_style=dict(styles[job['method']])
                if key=='plot/grad_norm_global':
                    plot_style['label']+='（全局实测）' if job['method']=='dt' else '（×√2估算）'
                ax.plot(xs,ys,linewidth=1.7,markersize=4,**plot_style)""")
code=code.replace('每轮均256条轨迹、global minibatch 64、1 epoch、4次actor更新；历史对照独立保留。分片梯度不作跨卡数全局比较。',
    '梯度：v8为全局实测；旧PPO/GRPO为首rank分片×√2估算（分片梯度能量均衡假设）。其余指标沿用原日志。')
code=code.replace("['algorithm','iteration','metric','value','original_sources']",
    "['algorithm','iteration','metric','value','original_sources','norm_scope','conversion_factor','original_value']")
code=code.replace("json.dumps(m['sources'],ensure_ascii=False)])",
    "json.dumps(m['sources'],ensure_ascii=False),*_norm_csv_fields(job,m,key)])")
code=code.replace('report=dict(output=str(output),files=',
    "report=dict(gradient_norm_conversion=snapshot['gradient_norm_conversion'],output=str(output),files=")
'''
anchor="exec(compile(code, str(renderer), 'exec'), globals())"
assert source.count(anchor)==1
source=source.replace(anchor,presentation+'\n'+anchor)
exec(compile(source,str(owner),'exec'),globals())
