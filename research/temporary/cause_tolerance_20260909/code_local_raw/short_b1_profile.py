"""Profile the existing short-B1 DT harness without changing its calculations.

Two ordinary warm calls, one instrumented call, and one separate output audit
for the 128-token builder case. No other attribution method is invoked.
"""
from pathlib import Path
import hashlib


def main():
    original=Path('/tmp/codex_short_b1_efficiency_20260910_v1/benchmark_v2.py')
    data=original.read_bytes()
    assert hashlib.sha256(data).hexdigest()=='eb5bd45e572971f42e2d64fbc86f6691c444bd3d4df0bdee2e1acc727412ba75'
    source=data.decode().replace('\r\n','\n')
    replacements=[
      ("n_runs=6 if args.method in ['deltatrace_retained','ifr_multi_hop','ifr_multi_hop_both'] else 3",'n_runs=3'),
      ("phase='original3' if rep<3 else 'warm3'","phase='warm' if rep<2 else 'profile'"),
      ('status,wall,alloc,reserved,by_device=bench.measure(runner,[0],catch_oom=True)',
       '''if rep<2:
                        status,wall,alloc,reserved,by_device=bench.measure(runner,[0],catch_oom=True)
                    else:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                            status,wall,alloc,reserved,by_device=bench.measure(runner,[0],catch_oom=True)
                        report['profile_operators']=[{'key':e.key,'count':e.count,'input_shapes':e.input_shapes,
                            'self_cpu_us':e.self_cpu_time_total,'total_cpu_us':e.cpu_time_total,
                            'self_device_us':e.self_device_time_total,'total_device_us':e.device_time_total}
                            for e in prof.key_averages(group_by_input_shape=True)]
                        report['profile_scope']='Diagnostic timing only; does not establish acceleration.' '''.rstrip()),
      ("vectors[str(input_len)]=vec", "vectors[str(input_len)]=vec\n                if args.family=='qwen35':case['existing_controller_diagnostics']=result['details']"),
      ("'sample_batch':1,'generation_calls':0,'cases':[],'rows':[],'initialization':[],",
       "'sample_batch':1,'generation_calls':0,'cases':[],'rows':[],'initialization':[],\n      'purpose':'Profile current retained DT short B1, no speed candidate or baseline-method expansion',\n      'frozen_parent_driver_sha256':'eb5bd45e572971f42e2d64fbc86f6691c444bd3d4df0bdee2e1acc727412ba75',")]
    for before,after in replacements:
        assert source.count(before)==1,before
        source=source.replace(before,after)
    namespace={'__name__':'__main__','__file__':str(Path(__file__).resolve())}
    exec(compile(source,str(Path(__file__).resolve()),'exec'),namespace)


if __name__=='__main__':main()
