"""Localize the saved source-token contribution along unchanged DT transport.

The layer differences are a localization tool, not a root-cause verdict.
No single-deletion oracle or global-conservation acceptance is introduced.
"""
import json,math
from pathlib import Path
import torch

DEST=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-neg11873-rootcause-20261015-v1')
torch.set_num_threads(2)
for rank in [0,1]:
    path=DEST/f'native-transport-v1-rank{rank}.pt'
    data=torch.load(path,map_location='cpu',weights_only=True)
    metadata=data['metadata'];payload=data['payload'];focus=data['focus_row']
    assert metadata['diagnostic_error'] is None and len(metadata['records'])==len(payload)
    records=[]
    for site,p in zip(metadata['records'],payload):
        index=p['focus_local_index'];values=p['token_effects'];m=p['focus_coefficient'].double()
        delta=p['focus_endpoint1'].double()-p['focus_endpoint0'].double()
        effect=float(values[focus,index]);coefficient_norm=float(m.norm());displacement_norm=float(delta.norm())
        product=coefficient_norm*displacement_norm
        records.append(dict(call_index=site['call_index'],phase=site['phase'],layer=site['current_layer_index'],
            block_type=site['block_type'],focus_effect=effect,
            coefficient_norm=coefficient_norm,displacement_norm=displacement_norm,
            cosine=effect/product if product else None,
            focus_cpu_dot=float((m*delta).sum()),
            largest_token_abs_effect_in_same_row=float(values[focus].abs().max()),
            native_coordinate=dict(local_index=index,original_position=p['focus_original_position'],
                                   row_starts=p['row_starts'],shared_prefix=p['native_shared_prefix']),
            nearby_effects=values[focus,max(0,index-5):index+6].tolist()))
    layers=[]
    for layer in reversed(range(32)):
        subset=[x for x in records if x['layer']==layer]
        assert len(subset)==3,(layer,[x['phase'] for x in subset])
        root,replay,incoming=subset
        assert 'root' in root['phase'] and 'replay' in replay['phase'] and 'input' in incoming['phase']
        layers.append(dict(layer=layer,block_type=root['block_type'],
            output_root=root['focus_effect'],output_replay=replay['focus_effect'],input=incoming['focus_effect'],
            weighted_replay_minus_root=replay['focus_effect']-root['focus_effect'],
            input_minus_output_root=incoming['focus_effect']-root['focus_effect'],
            input_minus_output_replay=incoming['focus_effect']-replay['focus_effect']))
    result=dict(rank=rank,scope=__doc__,checkpoint=25,incident_exact_replay=False,
        focus_uid=data['focus_uid'],records=records,layers_in_reverse_order=layers,
        largest_absolute_layer_transfers=sorted(layers,key=lambda x:abs(x['input_minus_output_root']),reverse=True)[:8],
        attribution_final=records[-1]['focus_effect'],new_model_calls=0,new_tolerance=None)
    (DEST/f'native-transport-v1-rank{rank}-CPU-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','layers_in_reverse_order']},ensure_ascii=False))
