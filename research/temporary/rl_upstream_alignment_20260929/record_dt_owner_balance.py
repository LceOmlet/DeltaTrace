"""Record the candidate separately from the unchanged live runtime snapshot."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[3]
audit=Path(__file__).resolve().parent/'dt-owner-balance-20261001'
suite=ET.parse(audit/'cpu-tests.xml').getroot().find('testsuite')
names=('experiments/rl/dt_training_batch.py','experiments/rl/test_distributed_credit.py')
result=dict(
    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    deployment='CPU-tested candidate only; existing formal driver functions and frozen entries are unchanged',
    owner='VERL-agent 20bd331bdbc9026a5668e11362178e10ab7400c8',
    owner_function='verl.utils.seqlen_balancing.get_seqlen_balanced_partitions',
    source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
    tests=dict(passed=int(suite.attrib['tests']),failures=int(suite.attrib['failures']),
        errors=int(suite.attrib['errors']),skipped=int(suite.attrib['skipped']),
        scope='Actual VERL partition/reorder/DP dispatch/pad/collect with synthetic credit ratios; exact identity and complete-return transport, not model numerics or speed',
        receipt=str((audit/'cpu-tests.xml').relative_to(root)).replace('\\','/')),
    workload=json.loads((audit/'completed-partition-comparison.json').read_text()),
    unchanged=['DT Q/V/A formula','DT numerical core','FA/FLA tests and tolerances',
        'VERL PPO and optimizer','official trajectory/global-minibatch budgets',
        'per-GPU actor/DT microbatch4','LoRA rank8 alpha16'],
    limitations=['Recorded length-work reduction is not measured wall-clock speedup.',
        'The 12-second Python stack sample shows CUDA synchronization, not a breakdown of GPU kernels versus collective wait.',
        'No current formal training process was paused, restarted or hotpatched to apply this candidate.'])
(root/'experiments/rl/results_dt_owner_balance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(tests=result['tests'],source_sha256=result['source_sha256']),indent=2))
