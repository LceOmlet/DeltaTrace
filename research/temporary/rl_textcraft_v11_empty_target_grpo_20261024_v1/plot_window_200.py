
import ast,csv,datetime as dt,hashlib,json,math,shutil
from pathlib import Path
base=Path('research/temporary/rl_textcraft_v11_empty_target_grpo_20261024_v1')
saved=Path((base/'latest-output.txt').read_text(encoding='utf-8-sig').strip())
snapshot_path=saved/'source-snapshot.json'
snapshot=json.loads(snapshot_path.read_text(encoding='utf-8-sig'))
renderer=Path('research/temporary/rl_textcraft_gradient_rootcause_20261013_v1/plot_current_approved_DTPO_v63.py')
owner=Path('research/temporary/rl_textcraft_equal_workload_comparison_20261018_v1/plot_comparison.py')
adapter=Path('research/temporary/rl_textcraft_v8_historical_controls_global_norm_plot_20261018_v1/plot_comparison.py')
zero_owner=Path('research/temporary/rl_textcraft_v5_eight_gpu_20261019_v1/plot_hourly.py')
expected={
 renderer:'f6ed1ff351b9f8ea7980577d4897acd59e953bc105defe47fe4cb3e700d46244',
 owner:'78d3d75a28f37ad178a8309fafb3115d801b33dc929549c136f2c12eed69c146',
 adapter:'0a1d44d6b28a496cc90c903445af3a98d7239bf8a59b31d9f1ffa0dd82c34a20',
 zero_owner:'0b6da9c6e4e59ea8b05d7fce7eecc8d92a75e3d478ce2dff87bddef25aa730fb'
}
for p,want in expected.items():
 assert hashlib.sha256(p.read_bytes()).hexdigest()==want,str(p)
output=saved/'requested-window-200'
output.mkdir(exist_ok=True)
original=renderer.read_text(encoding='utf-8')
owner_text=owner.read_text(encoding='utf-8')
start=owner_text.index("code = original[original.index('import matplotlib'):]")
end=owner_text.index('code = code.replace("csv_path=output/',start)
ns={'original':original}
exec(compile(owner_text[start:end],str(owner),'exec'),ns)
tree=ast.parse(adapter.read_text(encoding='utf-8'))
presentation=next(ast.literal_eval(n.value) for n in tree.body
 if isinstance(n,ast.Assign) and len(n.targets)==1
 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='presentation')
exec(compile(presentation.replace('v8为全局实测','本次v11为全局实测'),
 str(adapter),'exec'),ns)
# Reuse the existing zero-iteration presentation guard from the same plot owner.
zero_tree=ast.parse(zero_owner.read_text(encoding='utf-8'))
zero_presentation=next(ast.literal_eval(n.value) for n in zero_tree.body
 if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name)
 and n.target.id=='presentation')
exec(compile(zero_presentation,str(zero_owner),'exec'),ns)
render_code=ns['code']
render_code=render_code[:render_code.index('\ndraw(jobs,common,')]
anchor='        ax.margins(x=.04,y=.12)'
assert render_code.count(anchor)==1
render_code=render_code.replace(anchor,anchor+'\n        ax.set_xlim(0,200)')
stamp=dt.datetime.fromtimestamp(snapshot['collected_at'],
 dt.timezone(dt.timedelta(hours=8)))
shutil.copyfile(snapshot_path,output/'source-snapshot.json')
shutil.copyfile(saved/'workload-audit.json',output/'workload-audit.json')
scope={'snapshot':snapshot,'stamp':stamp,'output':output,
 'snapshot_path':output/'source-snapshot.json','math':math,'json':json}
exec(compile(render_code,str(renderer),'exec'),scope)
scope['draw'](scope['jobs'],200,
 'TextCraft · v11 / PPO / GRPO：前200轮（已有数据）',
 'algorithm-comparison-first-200')
with (saved/'metrics-with-sources.csv').open(encoding='utf-8-sig',newline='') as src,\
 (output/'metrics-with-sources.csv').open('w',encoding='utf-8-sig',newline='') as dest:
 reader=csv.DictReader(src)
 writer=csv.DictWriter(dest,fieldnames=reader.fieldnames)
 writer.writeheader()
 for row in reader:
  if int(row['iteration'])<=200:writer.writerow(row)
counts={j['method']:max((m['step'] for m in j['metrics']
 if m['step']<=200),default=0) for j in snapshot['jobs']}
receipt={
 'request':'三种算法绘制截取到200 it的图像',
 'source_snapshot':{'path':str(snapshot_path),
  'SHA256':hashlib.sha256(snapshot_path.read_bytes()).hexdigest()},
 'collected_at_Beijing':stamp.isoformat(),
 'iteration_range':[0,200],
 'metric_iteration_filter':'step<=200',
 'actual_data_endpoints':counts,
 'renderer_sources':{str(p):want for p,want in expected.items()},
 'presentation_change':'Original draw function, limit=200, axes.set_xlim(0,200); existing norm projection retained from source snapshot',
 'data_interpolation':False,'smoothing':False,'missing_values_filled':False,
 'new_model_DT_optimizer_environment_calls':0,
 'new_collection_calls':0,
 'files':{p.name:{'path':str(p),
  'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [
   output/'algorithm-comparison-first-200.png',
   output/'algorithm-comparison-first-200.svg',
   output/'metrics-with-sources.csv']}
}
(output/'plot-receipt.json').write_text(
 json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
j=next(x for x in snapshot['jobs'] if x['method']=='dt')
m=max(j['metrics'],key=lambda x:x['step']) if j['metrics'] else None
keys=['episode/reward/mean','actor/entropy_loss','actor/grad_norm','actor/kl_loss','response_length/mean','episode/tool_call_count/mean']
b=snapshot['active_execution_binding']
print(json.dumps({'window_200_output':str(output),
 'actual_data_endpoints':counts,'collected_at_Beijing':stamp.isoformat(),
 'job':{k:j.get(k) for k in ['run_dir','budget','pid','pid_alive','observed_process_created_unix','phase_label','exit_code','checkpoint_step']},
 'latest_metric':{'step':m['step'],'values':{k:m['values'][k] for k in keys},'sources':m['sources']} if m else None,
 'binding':{k:b.get(k) for k in ['run','source_manifest_SHA256','execution_source_commit','method_source_commit','credit_mode','native_SSM_cache_dtype','controller','driver','live_workers','completed_runtime_overrides']},
 'observer':snapshot.get('existing_observer_state')},ensure_ascii=False))

