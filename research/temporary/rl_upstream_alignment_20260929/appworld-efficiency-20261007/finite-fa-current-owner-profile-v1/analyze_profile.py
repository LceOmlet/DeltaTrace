"""Separate actual kernel events; never sum nested profiler CPU/device totals."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def artifact(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    report = dict(status='unchanged_deployed_FA_component_profiles_completed',
        scope='One instrumented call per rank, both serially on free GPU2, using saved actual B4 decoder3 inputs. Not full DT, formal wall-time savings, or a 32k operator profile.',
        execution=artifact(HERE/'execution.json'), ranks=[],
        production_changes=0, checkpoint_operations=0, model_loads=0,
        numerical_test_claim=False,
        interpretation=['Phase1 dQ is the largest measured finite-FA component.',
            'Phase0 and Phase1 repeat score calculations; the prior fused candidate was slower and remains rejected.',
            'Trace resource fields are metadata. Register count alone does not prove dynamic spills or causality.',
            'No model/checkpoint/production profiler attachment and no changed formula or tolerance.'])
    for rank in (0, 1):
        root = HERE/f'rank{rank}'
        value = json.loads((root/'result.json').read_bytes())
        trace = json.loads((root/'trace.json').read_bytes())
        kernels = [event for event in trace['traceEvents'] if event.get('cat') == 'kernel']
        phases = []
        for event in kernels:
            match = re.search(r'deltatrace_fa_finite_p1_kernel<(\d),', event['name'])
            if match:
                phases.append(dict(phase=int(match[1]), original_name=event['name'],
                    device_microseconds=event['dur'], original_resource_metadata=event['args']))
        assert sorted(row['phase'] for row in phases) == [0, 1, 2]
        total = sum(event['dur'] for event in kernels)
        for row in phases:
            row['fraction_of_summed_kernel_device_duration'] = row['device_microseconds']/total
        report['ranks'].append(dict(rank=rank, result=artifact(root/'result.json'), trace=artifact(root/'trace.json'),
            phases=phases, kernel_count=len(kernels), summed_kernel_device_microseconds=total,
            other_kernel_device_microseconds=total-sum(row['device_microseconds'] for row in phases),
            other_kernel_names=[event['name'] for event in kernels if 'deltatrace_fa_finite_p1_kernel<' not in event['name']],
            dtype_geometry=dict(operands=value['operands'], lengths=value['lengths'],
                query_starts=value['query_starts'], coefficient_starts=value['coefficient_starts']),
            torch_peak_allocated_bytes=value['torch_peak_allocated_bytes'],
            torch_peak_reserved_bytes=value['torch_peak_reserved_bytes']))
    output = HERE/'profile-analysis.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),sha256=artifact(output)['sha256'],
        ranks=[dict(rank=row['rank'], total_gpu_ms=row['summed_kernel_device_microseconds']/1000,
            phases=[dict(phase=p['phase'],ms=p['device_microseconds']/1000,fraction=p['fraction_of_summed_kernel_device_duration']) for p in row['phases']]) for row in report['ranks']])))


if __name__ == '__main__':
    main()
