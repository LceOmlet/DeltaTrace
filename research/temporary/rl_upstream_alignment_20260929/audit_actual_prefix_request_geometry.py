"""Read original saved request geometry with CPU Torch only; no model calls.

Uses the existing SSH/environment constants and frozen literal-request receipt.
It only counts saved ID tensors and observes the original B4 padding/prefix
boundaries. This is workload accounting, not a fixture or capacity gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import ENTRY, REPO, SSH


REMOTE_CPU_STATISTICS = r'''
import hashlib,json,os,pathlib,resource,statistics
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MACA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(4)
spec=json.loads(@SPEC@)
def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def summary(values):
    ordered=sorted(values)
    return dict(count=len(ordered),min=min(ordered),median=statistics.median(ordered),
                mean=statistics.mean(ordered),max=max(ordered),sum=sum(ordered))
def configuration_mentions(value,path=''):
    found=[]
    if isinstance(value,dict):
        for key,item in value.items():
            child=(path+'.'+key).lstrip('.')
            if key.rsplit('.',1)[-1] in ('max_response_length','max_tokens','max_model_len','max_length'):
                found.append(dict(path=child,value=item))
            found.extend(configuration_mentions(item,child))
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):found.extend(configuration_mentions(item,f'{path}[{i}]'))
    elif isinstance(value,str) and ('max_response_length=' in value or 'max_tokens=' in value):
        found.append(dict(path=path,value=value))
    return found
result=dict(scope='CPU saved-ID workload accounting; no CUDA API, model, forward, capacity gate or fixture construction',
            literal_source_receipt=spec['source_receipt'],cpu_gpu_visibility=dict(
                CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
                MACA_VISIBLE_DEVICES=os.environ['MACA_VISIBLE_DEVICES']),ranks=[])
for name,expected in sorted(spec['files'].items()):
    path=pathlib.Path(name)
    actual=file_sha(path)
    if actual!=expected:raise ValueError(f'Literal request source changed: {path}: {actual}')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    requests=sorted(payload['requests'],key=lambda request:request['context_tokens'])
    rows=[]
    for index,request in enumerate(requests):
        lengths={key:int(request[key].numel()) for key in ('prompt','actions','query','target')}
        rows.append(dict(sorted_row=index,row_index=request['row_index'],
            traj_uid=request['traj_uid'],source_step=request['source_step'],
            prompt_tokens=lengths['prompt'],action_tokens=lengths['actions'],
            factual_tokens=lengths['prompt']+lengths['actions'],query_tokens=lengths['query'],
            target_tokens=lengths['target'],context_tokens=request['context_tokens'],
            source_start=request['start'],source_end=request['end'],
            sum_saved_components=sum(lengths.values()),
            saved_tensor_devices={key:str(request[key].device) for key in lengths}))
    batches=[]
    for offset in range(0,len(requests),4):
        batch=requests[offset:offset+4]
        # Original native_prefix_leases.py prefix_lengths expression. This is
        # a scalar observation only, not cache preparation or runner execution.
        boundary=min(request['start'] for request in batch)//64*64
        padded=max(request['context_tokens'] for request in batch)
        batches.append(dict(batch_index=offset//4,sorted_rows=list(range(offset,offset+len(batch))),
            rows=len(batch),prompt_min=min(request['prompt'].numel() for request in batch),
            prompt_max=max(request['prompt'].numel() for request in batch),
            action_min=min(request['actions'].numel() for request in batch),
            action_max=max(request['actions'].numel() for request in batch),
            context_min=min(request['context_tokens'] for request in batch),context_max=padded,
            local_common_prefix_boundary=boundary,local_suffix_tokens=padded-boundary,
            original_padded_paired_ID_shape=[2*len(batch),padded],
            local_suffix_paired_ID_shape=[2*len(batch),padded-boundary]))
    options=path.parent/'native-launch-options.json'
    metadata=dict(path=str(path),sha256=actual,bytes=path.stat().st_size,
        payload_keys=list(payload),request_keys=list(requests[0]) if requests else [],
        request_count=len(requests),unique_trajectories=len({r['traj_uid'] for r in requests}),
        distributions={key:summary([row[key] for row in rows]) for key in (
            'prompt_tokens','action_tokens','factual_tokens','query_tokens','target_tokens','context_tokens')},
        batches=batches,rows=rows,
        original_option_file=dict(path=str(options),exists=options.exists()),
        payload_configuration_mentions=configuration_mentions({k:v for k,v in payload.items()
            if k in ('readout_options','configuration','config','sampling')}))
    if options.exists():
        metadata['original_option_file'].update(sha256=file_sha(options),
            length_mentions=configuration_mentions(json.loads(options.read_bytes())))
    result['ranks'].append(metadata)
    del payload,requests,rows,batches
# The frozen v2 owner used a full-value MIN across sharding ranks. Later
# local-prefix owner changes synchronize only branch participation. Report
# both saved-bank geometric boundaries without pretending to run a collective.
if len(result['ranks'])==2:
    left,right=result['ranks']
    for lb,rb in zip(left['batches'],right['batches']):
        common=min(lb['local_common_prefix_boundary'],rb['local_common_prefix_boundary'])
        for batch in (lb,rb):
            batch['frozen_v2_cross_rank_MIN_prefix']=common
            batch['frozen_v2_cross_rank_MIN_suffix_tokens']=batch['context_max']-common
for rank in result['ranks']:
    rank['selected_rows_40_43']=[row for row in rank['rows'] if 40<=row['sorted_row']<44]
    rank['selected_B4_40_43']=next((batch for batch in rank['batches'] if batch['batch_index']==10),None)
    rank['largest_context_batch']=max(rank['batches'],key=lambda batch:batch['context_max'])
    rank['largest_local_suffix_batch']=max(rank['batches'],key=lambda batch:batch['local_suffix_tokens'])
    rank['largest_action_row']=max(rank['rows'],key=lambda row:row['action_tokens'])
result['CPU_process_max_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-receipt',type=Path,default=REPO/'experiments/rl/results_native_prefix_reuse_workload_20261004.json')
    parser.add_argument('--output',type=Path,default=REPO/'experiments/rl/results_actual_prefix_request_geometry_20261004.json')
    args=parser.parse_args()
    receipt=json.loads(args.source_receipt.read_bytes())
    files=receipt['literal_request_files']
    spec=dict(files=files,source_receipt=dict(path=str(args.source_receipt),
        sha256=hashlib.sha256(args.source_receipt.read_bytes()).hexdigest()),
        existing_configuration=receipt.get('configuration'))
    code=REMOTE_CPU_STATISTICS.replace('@SPEC@',repr(json.dumps(spec)))
    script=f"source {ENTRY}/metax-entry.env.sh\nexport CUDA_VISIBLE_DEVICES=''\nexport MACA_VISIBLE_DEVICES=''\n\"$VENV_PYTHON\" - <<'PY'\n{code}\nPY\n"
    process=subprocess.run(SSH+['bash','-s'],input=script.encode('utf8'),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
    if process.returncode:
        print(process.stderr.decode('utf8','replace'))
        raise SystemExit(process.returncode)
    result=json.loads(process.stdout)
    result['existing_receipt_configuration']=receipt.get('configuration')
    result['audit_script_source']=dict(path=str(Path(__file__)),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(args.output),ranks=[{
        'path':rank['path'],'request_count':rank['request_count'],
        'distributions':rank['distributions'],'selected_B4_40_43':rank['selected_B4_40_43'],
        'largest_context_batch':rank['largest_context_batch'],
        'largest_local_suffix_batch':rank['largest_local_suffix_batch'],
        'original_option_file':rank['original_option_file'],
    } for rank in result['ranks']],CPU_process_max_RSS_KiB=result['CPU_process_max_RSS_KiB']),indent=2))


if __name__=='__main__':main()
