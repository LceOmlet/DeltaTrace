"""Join completed native request observations to the original dmon reducer."""
import contextlib
import hashlib
import io
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'


def main():
    observations = json.loads((AUDIT / 'native-generation-completions.json').read_bytes())
    assert observations['status'] == 'completed_native_completion_observation'
    calls = [r['value'] for r in observations['rows']]
    assert len(calls) == 16
    lower = min(r['started'] for r in calls)
    upper = max(r['finished'] for r in calls)
    source = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/summarize_native_utilization_v2.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'e6f3b17198f34ca0b5946637dcbd3ee4920bb08b9783bbc374ffe85bb14199e3'
    destination = AUDIT / 'native-generation-completion-utilization'
    destination.mkdir(exist_ok=False)
    code = source.read_text()
    code = code.replace("AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'", 'AUDIT = Path(' + repr(str(destination)) + ')')
    code = code.replace("PHYSICAL = AUDIT / 'native-mx-smi-dmon-timestamped.jsonl'",
                        "PHYSICAL = ROOT / 'receipts/six-run-native-utilization-20261017-v1/native-mx-smi-dmon-timestamped.jsonl'")
    code = code.replace('-8gpu-200-20261017-v1', '-8gpu-200-20261019-v4')
    code = code.replace('lower, upper = None, None', f'lower, upper = {lower!r}, {upper!r}')
    anchor = '        entry = groups.setdefault(label, {})'
    assert code.count(anchor) == 1
    code = code.replace(anchor, '''        if len(label)==4 and label[-1]=='ray::WorkerDict.actor_rollout_generate_sequences':
            current=[r for r in observed_calls if r['started']<=stamp<r['finished']]
            if len(current)==8 and {r['rank'] for r in current}==set(range(8)) and len({r['call'] for r in current})==1:
                unfinished=sum(r['requests']-sum(e['finished']<=stamp for e in r['native_completion_events']) for r in current)
                fraction=unfinished/sum(r['requests'] for r in current)
                bucket='0-25%' if fraction<=.25 else '25-50%' if fraction<=.5 else '50-75%' if fraction<=.75 else '75-100%'
                label=label+('unfinished_native_requests_'+bucket,)
            else:
                label=label+('outside_common_observed_call',)
''' + anchor)
    namespace = {'__name__': 'original_completion_utilization_reducer', '__file__': str(source), 'observed_calls': calls}
    exec(compile(code, str(source), 'exec'), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        namespace['main']()
    physical = json.loads((destination / 'native-utilization-phase-summary.json').read_bytes())
    groups = []
    for phase in physical['phases']:
        devices = list(phase['devices'].values())
        n = sum(d['gpu_utilization_percent']['samples'] for d in devices)
        groups.append(dict(identity=phase['identity'], card_samples=n,
            mean_GPU_Util_percent=sum(d['gpu_utilization_percent']['mean'] * d['gpu_utilization_percent']['samples'] for d in devices) / n))
    report = dict(status='completed_read_only_completion_utilization_join', source_observations=str(AUDIT / 'native-generation-completions.json'),
        physical_report=str(destination / 'native-utilization-phase-summary.json'),
        original_reducer_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(), window=physical['window'], groups=groups,
        interpretation='All eight original phase brackets and observed call identities must agree. Buckets use the total unfinished native request fraction across all eight ranks, so no unverified rank-to-physical-board mapping is assumed. This short observational window mixes prefill/decode and cannot establish a causal speedup.',
        model_calls=0,training_changes=0)
    target=destination / 'completion-utilization-receipt.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(receipt=str(target),groups=groups)))


if __name__ == '__main__':
    main()
