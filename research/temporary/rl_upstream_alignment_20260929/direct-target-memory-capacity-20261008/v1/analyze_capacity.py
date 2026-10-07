"""Describe actual memory lifetimes and unchanged repeated QVA, CPU only."""
import csv
import hashlib
import json
from pathlib import Path
import re

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE/'actual-results/results'
GIB = 1024**3


def cache_bytes(event, kind=None):
    layers = event.get('replay_cache',event.get('owners',{}).get('replay_cache',[]))
    if kind is not None:
        layers = [layer for layer in layers if layer['type']==kind]
    storages = {tensor['ptr']:tensor['storage_bytes'] for layer in layers for tensor in layer['tensors']}
    return sum(storages.values())


def main():
    phases = [[json.loads(line) for line in (ROOT/f'rank{rank}-phases.jsonl').read_bytes().splitlines()] for rank in (0,1)]
    physical = []
    for line in (ROOT/'physical-mx-smi.jsonl').read_bytes().splitlines():
        entry = json.loads(line)
        gpu = None
        for row in entry['stdout'].splitlines():
            board = re.match(r'^\|\s*(\d+)\s+MetaX\s',row)
            if board:
                gpu = int(board[1])
            memory = re.search(r'(\d+)/(\d+) MiB',row)
            if memory and gpu in (4,5):
                physical.append(dict(unix=entry['unix'],gpu=gpu,used_mib=int(memory[1]),total_mib=int(memory[2])))
    modes = ('exact32768_first','exact32768_repeat','original_failed_B4')
    ranks, csvrows = [], []
    for rank, events in enumerate(phases):
        summaries = []
        for mode in modes:
            group = [e for e in events if e['mode']==mode]
            begin = next(e for e in group if e['phase']=='DT_begin')
            end = next(e for e in group if e['phase']=='DT_complete')
            samples = [e for e in physical if e['gpu']==rank+4 and begin['unix']<=e['unix']<=end['unix']]
            first = next(e for e in group if e['phase']=='before_native_layer')
            final = next(e for e in reversed(group) if e['phase']=='after_finite_layer')
            value = torch.load(ROOT/f'rank{rank}-{mode}.pt',map_location='cpu',weights_only=False)
            summaries.append(dict(mode=mode,seconds=end['seconds'],
                sampled_physical_peak_mib=max(e['used_mib'] for e in samples),
                pss_peak_bytes=max(e['pss_bytes'] for e in group),
                begin_allocated_bytes=begin['allocated'],end_allocated_bytes=end['allocated'],
                allocated_end_minus_begin_bytes=end['allocated']-begin['allocated'],
                begin_reserved_bytes=begin['reserved'],end_reserved_bytes=end['reserved'],
                final_replay_cache_gpu_storage_bytes=cache_bytes(final),first_replay_cache_gpu_storage_bytes=cache_bytes(first),
                final_replay_cache_by_native_type={kind:cache_bytes(final,kind) for kind in {layer['type'] for layer in final['replay_cache']}},
                all_QVA_finite=all(bool(torch.isfinite(t).all()) for row in value['values'] for t in row.values()),
                DT_calls=value['report']['finite_trace_calls'],causal_lengths=value['report']['causal_context_lengths']))
            for e in group:
                if 'layer' in e:
                    csvrows.append(dict(rank=rank,mode=mode,layer=e['layer'],phase=e['phase'],unix=e['unix'],
                        allocated_bytes=e['allocated'],reserved_bytes=e['reserved'],pss_bytes=e['pss_bytes'],replay_cache_storage_bytes=cache_bytes(e)))
        comparisons = [e for e in events if e['phase'] in ('repeat_comparison','real_regression_comparison')]
        ranks.append(dict(rank=rank,modes=summaries,comparisons=comparisons))
    result = dict(status='bounded_exact32768_DT_capacity_and_actual_failed_B4_regression_complete_not_formally_deployed',
        ranks=ranks,launch=json.loads((HERE/'actual-results/launch.json').read_bytes()),
        source_transport=dict(path=str(HERE/'actual-results/transport.json'),sha256=hashlib.sha256((HERE/'actual-results/transport.json').read_bytes()).hexdigest()),
        scope='Two consecutive exact32768 storage-stress calls per rank, original per-GPU B4; original failed actual B4 afterwards. Existing mixer offload and consumed-cache lifetime repair; original VERL async actor and original asleep vLLM present.',
        limitations=['Observation-extended exact32768 input is capacity-only, not an official task trajectory or credit-accuracy test.',
            'DT only: no rollout, PPO update, checkpoint restore or formal restart. Rank8/alpha16/B4 unchanged.',
            'Two repeats bound this storage regression; not an indefinite absence-of-leak guarantee.',
            'Torch allocator counters include MetaX virtual/sleep pools; physical VRAM comes only from mx-smi.',
            'Unchanged QVA compared with original Torch dtype assertions and exact equality; no new FA/FLA finite-attribution tolerance or credit correction.'])
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'memory-phases.csv').open('w',newline='',encoding='utf8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    colors = ('#247d94','#ca673b')
    start = min(e['unix'] for events in phases for e in events if e['phase']=='DT_begin')
    for rank,events in enumerate(phases):
        samples = [e for e in physical if e['gpu']==rank+4 and e['unix']>=start]
        axes[0].plot([(e['unix']-start)/60 for e in samples],[e['used_mib']/1024 for e in samples],color=colors[rank],label=f'Physical GPU{rank+4}')
        before = [e for e in events if e['phase']=='before_native_mlp']
        axes[1].plot([(e['unix']-start)/60 for e in before],[e['allocated']/GIB for e in before],color=colors[rank],label=f'Rank{rank} live allocator before MLP')
    for e in phases[0]:
        if e['phase']=='DT_begin':
            for ax in axes:
                ax.axvline((e['unix']-start)/60,color='#777777',linestyle=':',linewidth=.8)
            axes[0].text((e['unix']-start)/60+.05,4,e['mode'],rotation=90,va='bottom',fontsize=8)
    axes[0].axhline(64,color='#444444',linestyle='--',label='64 GiB device capacity')
    axes[0].set(ylabel='Physical mx-smi GiB',ylim=(0,67),title='Exact 32768 DT repeated, then original failed B4; official vLLM asleep')
    axes[1].set(ylabel='Torch live GiB (virtual pools included)',xlabel='Minutes since first DT began',title='Phase storage observations; allocator counters are not physical VRAM')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(HERE/'capacity-memory.png',dpi=150)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(ranks=ranks,analysis=str(HERE/'analysis.json'),cuda_initialized=False)))


if __name__ == '__main__':
    main()
