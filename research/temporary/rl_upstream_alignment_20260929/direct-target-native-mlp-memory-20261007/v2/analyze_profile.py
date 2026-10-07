"""Source-bound CPU analysis of actual native DT memory phases and values."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE/'actual-results/results'
GIB = 1024**3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def storage_bytes(items):
    return sum(value['storage_bytes'] for value in {item['ptr']:item for item in items}.values())


def cache_bytes(event):
    return storage_bytes([item for layer in event['owners'].get('replay_cache',[]) for item in layer['tensors']])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['3','5','6'],default='3')
    args=parser.parse_args()
    global ROOT
    destination=HERE if args.version=='3' else HERE/('candidate-analysis-v'+args.version)
    destination.mkdir(exist_ok=True)
    if args.version!='3':ROOT=HERE/('actual-results-v'+args.version)/'results'
    phases = [[json.loads(line) for line in (ROOT/f'rank{rank}-phases.jsonl').read_bytes().splitlines()]
              for rank in (0,1)]
    physical = []
    for line in (ROOT/'physical-mx-smi.jsonl').read_bytes().splitlines():
        event=json.loads(line)
        gpu=None
        for row in event['stdout'].splitlines():
            board=re.match(r'^\|\s*(\d+)\s+MetaX\s',row)
            if board:
                gpu=int(board[1])
            memory=re.search(r'(\d+)/(\d+) MiB',row)
            if memory and gpu in (4,5):
                physical.append(dict(unix=event['unix'],gpu=gpu,used_mib=int(memory[1]),total_mib=int(memory[2])))
    ranks=[]
    rows=[]
    for rank, events in enumerate(phases):
        original=torch.load(ROOT/f'rank{rank}-original_gpu_captures.pt',map_location='cpu',weights_only=False)
        offload=torch.load(ROOT/f'rank{rank}-existing_offload_mixer.pt',map_location='cpu',weights_only=False)
        failed=torch.load(ROOT/f'rank{rank}-failed-actual-complete.pt',map_location='cpu',weights_only=False)
        signed_equal=torch.equal(original['signed'],offload['signed'])
        qva_equal={name:all(torch.equal(before[name],after[name])
                    for before,after in zip(original['values'],offload['values']))
                   for name in original['values'][0]}
        assert signed_equal and all(qva_equal.values())
        modes=[]
        for mode in ('original_gpu_captures','existing_offload_mixer','failed_actual_B4_existing_offload'):
            group=[event for event in events if event.get('mode')==mode]
            start=next(event for event in group if event['phase']=='DT_begin')
            end=next(event for event in group if event['phase']=='DT_complete')
            samples=[event for event in physical if event['gpu']==rank+4 and start['unix']<=event['unix']<=end['unix']]
            entry=dict(mode=mode,seconds=end['seconds'],
                sampled_physical_peak_mib=max(item['used_mib'] for item in samples),
                pss_max_bytes=max(item['pss_bytes'] for item in group),
                recorded_allocated_peak_bytes=end.get('peak_allocated'),
                note='Physical sampled peak includes warm allocator pool; only first pair has reset peak stats')
            for event in group:
                if 'layer' not in event:continue
                row=dict(rank=rank,mode=mode,layer=event['layer'],phase=event['phase'],unix=event['unix'],
                    allocated_bytes=event['allocated'],reserved_bytes=event['reserved'],
                    runtime_device_free_bytes=event['device_free'],pss_bytes=event['pss_bytes'],
                    replay_cache_gpu_storage_bytes=cache_bytes(event),
                    mixer_capture_gpu_storage_bytes=storage_bytes(event['owners'].get('mc',[])))
                rows.append(row)
            first=next(event for event in group if event['phase']=='before_native_layer')
            last=next(event for event in reversed(group) if event['phase']=='after_finite_layer')
            entry.update(cache_first_bytes=cache_bytes(first),cache_last_bytes=cache_bytes(last),
                         cache_growth_bytes=cache_bytes(last)-cache_bytes(first))
            modes.append(entry)
        previous_full=None
        if args.version!='3':
            previous=torch.load(HERE/f'actual-results/results/rank{rank}-failed-actual-complete.pt',map_location='cpu',weights_only=False)
            previous_full=dict(signed_exact_equal=torch.equal(previous['signed'],failed['signed']),
                signed_maxabs=float((previous['signed']-failed['signed']).abs().max()),
                QVA_exact_equal={name:all(torch.equal(a[name],b[name]) for a,b in zip(previous['values'],failed['values'])) for name in previous['values'][0]},
                comparison_scope='Same original failed B4; former actor-only existing-offload result versus candidate with official async-vLLM sleep lifecycle')
        single_source = None
        if args.version == '6':
            single=torch.load(ROOT/f'rank{rank}-single-source-eos.pt',map_location='cpu',weights_only=False)
            single_source=dict(candidate_signed=float(single['signed'][0,7260]),
                signed_sum=float(single['signed'].sum()),
                other_positions_nonzero=int(torch.count_nonzero(single['signed']))-int(single['signed'][0,7260]!=0),
                details=single['detail'],
                scope='Same producer, prefix cache, native and finite owners; only this source token replaced by EOS. Diagnostic, not a changed training estimator.')
        ranks.append(dict(rank=rank,signed_exact_equal=signed_equal,QVA_exact_equal=qva_equal,modes=modes,
            single_source_eos_diagnostic=single_source,
            full_failed_B4_previous_comparison=previous_full,
            failed_batch_all_finite=all(bool(torch.isfinite(v).all()) for item in failed['values'] for v in item.values()),
            failed_batch_trace_calls=failed['report']['finite_trace_calls'],
            failed_batch_actual_context_lengths=failed['report']['actual_context_lengths'],
            failed_batch_causal_context_lengths=failed['report']['causal_context_lengths'],
            App_extreme_current_d=float(original['signed'][0,7260]),
            App_extreme_saved_original_d=-3.1117619098301255))
    header=list(rows[0])
    with (destination/'memory-phases.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=header);writer.writeheader();writer.writerows(rows)
    result=dict(scope=('Actual saved B4 and original failed final B4. Existing mixer offload plus consumed-layer cache lifetime patch; original native/finite/PPO/reward/LoRA/B4 math unchanged.' if args.version!='3' else 'Actual saved B4 and original failed final B4. Only existing mixer offload changed; original native/finite/PPO/reward/LoRA/B4 owners unchanged.'),
        ranks=ranks,source_transport=dict(path=str(ROOT.parent/'transport.json'),sha256=sha(ROOT.parent/'transport.json')),
        physical_scope=('Original async actor and AsyncLLMServerManager wake/sleep, co-resident official vLLM, no formal restart. Not full32k capacity or restored training-health proof.' if args.version!='3' else 'Actor-only original FSDP initialization; no co-resident vLLM and no formal restart. Not full32k capacity or restored training-health proof.'),
        timing_scope='Original mode ran first, offload second after warmup; no speedup ratio claim.',
        tolerances='Signed and QVA exactly equal on real comparison B4. No new FA/FLA tolerance or numerical correction.',
        numerical_scope=('The long actual B4 is compared with the completed original offload-only v3 vector. Exact equality or observed differences are reported, without a new tolerance.' if args.version!='3' else 'The long failed B4 has a completed finite vector, but no successful original offload-disabled full vector exists.'),
        cache_scope=('Candidate replaces a consumed original DynamicLayer with a new original empty DynamicLayer only after its finite consumer; no other layer state or math changed.' if args.version!='3' else 'Storage metadata confirms already consumed DynamicLayer keys/values remain GPU-resident until the attribute call ends; do not infer cross-batch leak.'))
    (destination/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    colors={'original_gpu_captures':'#c65a36','existing_offload_mixer':'#267f9a','failed_actual_B4_existing_offload':'#537d35'}
    labels={'original_gpu_captures':'Original B4, captures on GPU','existing_offload_mixer':'Same B4, existing offload','failed_actual_B4_existing_offload':'Original failed B4, existing offload'}
    for rank in (0,1):
        for mode,color in colors.items():
            selected=[row for row in rows if row['rank']==rank and row['mode']==mode and row['phase']=='before_native_mlp']
            axes[rank,0].plot([31-row['layer'] for row in selected],[row['allocated_bytes']/GIB for row in selected],label=labels[mode],color=color)
        axes[rank,0].set(title=f'Rank {rank}: live allocation before native MLP',xlabel='Reverse decoder replay index',ylabel='Torch live GiB (not physical VRAM)')
        for mode,color in colors.items():
            selected=[row for row in rows if row['rank']==rank and row['mode']==mode and row['phase']=='after_finite_layer']
            axes[rank,1].plot([31-row['layer'] for row in selected],[row['replay_cache_gpu_storage_bytes']/GIB for row in selected],label=labels[mode],color=color)
        axes[rank,1].set(title=f'Rank {rank}: retained native replay cache',xlabel='Completed reverse decoder index',ylabel='Deduplicated GPU storage GiB')
        for ax in axes[rank]:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(destination/'memory-phases.png',dpi=150)
    if args.version != '3':
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
        for rank,color in ((0,'#267f9a'),(1,'#c65a36')):
            events=[e for e in phases[rank] if e.get('mode')=='failed_actual_B4_existing_offload']
            start=next(e['unix'] for e in events if e['phase']=='DT_begin')
            end=next(e['unix'] for e in events if e['phase']=='DT_complete')
            samples=[e for e in physical if e['gpu']==rank+4 and start<=e['unix']<=end]
            axes[0].plot([e['unix']-start for e in samples],
                [e['used_mib']/1024 for e in samples],color=color,label=f'GPU {rank+4}, current patch')
            before=[json.loads(line) for line in (HERE/f'actual-results/results/rank{rank}-phases.jsonl').read_bytes().splitlines()]
            old=[e for e in before if e.get('mode')=='failed_actual_B4_existing_offload' and e['phase']=='after_finite_layer']
            new=[e for e in events if e['phase']=='after_finite_layer']
            axes[1].plot([31-e['layer'] for e in old],[cache_bytes(e)/GIB for e in old],
                color=color,linestyle='--',label=f'Rank {rank}, previous retained cache')
            axes[1].plot([31-e['layer'] for e in new],[cache_bytes(e)/GIB for e in new],
                color=color,label=f'Rank {rank}, consumed cache released')
        axes[0].axhline(64,color='#444444',linestyle=':',label='Device capacity 64 GiB')
        axes[0].set(title='Actual failed B8 replay, official async vLLM asleep',
            xlabel='Seconds since DT began',ylabel='Physical mx-smi GiB',ylim=(0,67))
        axes[1].set(title='Same actual B4 rows: retained K/V storage',
            xlabel='Completed reverse decoder index',ylabel='Deduplicated GPU cache GiB')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(destination/'physical-and-cache.png',dpi=150)
    assert not torch.cuda.is_initialized()
    print(json.dumps(result))


if __name__=='__main__':main()
