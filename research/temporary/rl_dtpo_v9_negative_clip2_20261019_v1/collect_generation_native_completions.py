"""Reduce native completion records obtained by the original job-log reader."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rl_dtpo_v9_clip_u_20261019_v1'))
from transport import run_python


def main():
    metadata_reader = HERE / 'collect_existing_DT_timings.py'
    assert hashlib.sha256(metadata_reader.read_bytes()).hexdigest() == 'b841476e54813430a7991dae0db0125873cb61dd620179c6b52e5a088d2ce228'
    call, = [n for n in ast.walk(ast.parse(metadata_reader.read_text()))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run_python']
    prefix = ast.literal_eval(call.args[0]).split('\ndef summarize(records):', 1)[0]
    prefix = prefix.replace('native_DT_existing_timing', 'native_generation_completions')
    prefix += '\nnative_records=rows\nnative_errors=errors\n'
    report = json.loads(run_python(prefix + r'''
import collections,hashlib,json,runpy,time
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
reader=R/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(reader.read_bytes()).hexdigest()=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner=runpy.run_path(str(reader),run_name='original_completion_record_reader')
state=json.loads((R/'receipts/textcraft-eight-gpu-v9-negative-clip2-20261019-v4/campaign-state.json').read_bytes())
job,=owner['collect'](str(R),[state['jobs'][0]['output']])['jobs']
rows=[];errors=list(native_errors);superseded_rows=[]
for record in native_records:
    value=record['value']
    events=value['native_completion_events'];n=value['requests']
    identities={e['request_id'] for e in events}
    valid=len(events)==len(identities)==n and n>0 and not value.get('native_completion_observation_errors')
    reduction=dict(completion_coverage_valid=valid)
    if valid:
        start=value['started'];end=value['finished'];span=end-start
        times=sorted(e['finished'] for e in events)
        valid=all(start<=t<=end for t in times)
        reduction['timestamps_inside_original_call']=valid
        if valid:
            waits=[end-t for t in times]
            active=n;last=start;bins=collections.Counter()
            for t in times+[end]:
                label='0' if active==0 else '1-4' if active<=4 else '5-8' if active<=8 else '9-16' if active<=16 else '17-32' if active<=32 else '>32'
                bins[label]+=t-last
                active=max(0,active-1);last=t
            reduction.update(mean_request_wait_until_batch_return_seconds=sum(waits)/n,
                max_request_wait_seconds=max(waits),
                request_lifetime_fraction=sum(t-start for t in times)/(n*span),
                wall_time_by_unfinished_request_count_seconds=dict(bins))
    decoded=dict(path=record['path'],line=record['line'],value=value,reduction=reduction)
    if 'native_completion_observation_errors' in value:rows.append(decoded)
    else:superseded_rows.append(decoded)
keys={(r['value']['rank'],r['value']['call']) for r in rows}
complete=keys=={(r,c) for r in range(8) for c in (1,2)} and len(rows)==16 and not errors
complete=complete and all(r['reduction']['completion_coverage_valid'] and r['reduction'].get('timestamps_inside_original_call') for r in rows)
result=dict(status='completed_native_completion_observation' if complete else 'partial_native_completion_observation',
 collected=time.time(),reader=str(reader),reader_SHA256=hashlib.sha256(reader.read_bytes()).hexdigest(),
 native_metadata_reader_SHA256='b841476e54813430a7991dae0db0125873cb61dd620179c6b52e5a088d2ce228',
 rows=rows,superseded_rows=superseded_rows,errors=errors,additional_generation_calls=0,model_calls=0,optimizer_calls=0,
 interpretation='Completion times are host observations of original vLLM RequestOutput.finished. Request wait until batch return measures the owner barrier; it is neither GPU idle time nor a predicted speedup. Native step methods restore after each call; the generation wrapper restores after call2.',
 latest_complete_training_iteration=owner['completed'](job),phase=job['phase_label'])
if rows or superseded_rows or errors:
    target=A/'native-generation-completions.json';target.write_text(json.dumps(result,indent=2)+'\n');result['remote_receipt']=str(target)
print(json.dumps(result))
''', timeout=45))
    if report['rows'] or report['superseded_rows'] or report['errors']:
        target = HERE.parents[2] / 'experiments/rl/results_dtpo_v9_negative2_generation_completions_20261019_v4.json'
        target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], calls=len(report['rows']), errors=report['errors'],
        completed=report['latest_complete_training_iteration'], phase=report['phase'],
        summaries=[dict(rank=r['value']['rank'],call=r['value']['call'],requests=r['value']['requests'],
                        duration=r['value']['native_generate_seconds'],**r['reduction']) for r in report['rows']])))
    print(json.dumps(dict(superseded_summaries=[dict(rank=r['value']['rank'],call=r['value']['call'],
        requests=r['value']['requests'],duration=r['value']['native_generate_seconds'],**r['reduction'])
        for r in report['superseded_rows']])))


if __name__ == '__main__':
    main()
