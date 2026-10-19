"""Compose the original native collector, archived controls and approved renderer."""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
owner=HERE.parent/'rl_textcraft_equal_workload_comparison_20261018_v1/plot_comparison.py'
adapter=HERE.parent/'rl_textcraft_v8_historical_controls_global_norm_plot_20261018_v1/plot_comparison.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='78d3d75a28f37ad178a8309fafb3115d801b33dc929549c136f2c12eed69c146'
assert hashlib.sha256(adapter.read_bytes()).hexdigest()=='0a1d44d6b28a496cc90c903445af3a98d7239bf8a59b31d9f1ffa0dd82c34a20'
source=owner.read_text(encoding='utf-8')
source=source.replace('textcraft-eight-gpu-sequential-20261018-v5','textcraft-v5-eight-gpu-20261019-v1')
source=source.replace('textcraft-dtpo-v6-8gpu-200-20261018-v5','textcraft-dtpo-v5-exponential-8gpu-200-20261019-v1')
source=source.replace('textcraft-equal-workload-current-DTPO-historical-controls-20261018-v1',
    'textcraft-v5-eight-gpu-historical-PPO-GRPO-global-norm-20261019-v1')
source=source.replace('DTPO v6（8卡）','DTPO v5（新8卡）')

# Both controls are stopped. Use their already preserved original-owner
# artifacts, including original paths/line numbers and collection timestamp.
start=source.index("job_path=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json'")
end=source.index('common_keys=',start)
source=source[:start]+source[end:]
source=source.replace("snapshot['jobs'].append(historical_grpo)",
    "snapshot['jobs'].extend([historical_ppo,historical_grpo])")
anchor="snapshot['renderer_source'] = dict(path=str(renderer), SHA256=renderer_sha)\n"
assert source.count(anchor)==1
tree=ast.parse(adapter.read_text(encoding='utf-8'))
strings={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
    and node.targets[0].id in ('projection','presentation')}
projection=strings['projection'].replace('Current v8','Current eight-GPU v5')
projection += "\nsnapshot['historical_control_source_snapshot']=dict(path=str(historical_path),SHA256=hashlib.sha256(historical_path.read_bytes()).hexdigest(),original_collection_time=historical.get('collected_at'),frozen_control_histories=True)\n"
source=source.replace(anchor,anchor+projection)
presentation=strings['presentation'].replace('v8为全局实测','新八卡v5为全局实测')
presentation += '''
code=code.replace("assert last['dt']>0,'Current DTPO has no completed native iteration yet; source snapshot retained.'", "assert any(last.values()), 'No completed native metrics'")
code=code.replace("primary_figure=str(output/'algorithm-comparison-common-window.png')", "primary_figure=str(output/('algorithm-comparison-common-window.png' if common else 'algorithm-comparison-full.png'))")
'''
anchor="exec(compile(code, str(renderer), 'exec'), globals())"
assert source.count(anchor)==1
source=source.replace(anchor,presentation+'\n'+anchor)
exec(compile(source,str(owner),'exec'),globals())
