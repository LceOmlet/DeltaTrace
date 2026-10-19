"""Route the accepted native CPU checkpoint reader to an exact v9 campaign job.

The existing reader owns Torch mmap, scheduler/optimizer/RNG/data and native
AppWorld sampler inspection. This adapter changes task identity and rank count.
It never creates/restores a training checkpoint or invokes a model.
"""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT / 'receipts/appworld-formal-training-20261017-v1/source/verify_completed_checkpoint.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'd5a6f3ee52e822356e5cb4d3df73215f3818ede9e033eda9e4b53719636ae865'
code = owner.read_text()
code = code.replace("AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'",
    "AUDIT = ROOT / 'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'")
code = code.replace("method = sys.argv[1]\nassert method in ('dt', 'ppo')",
    "task,method=sys.argv[1:3]\nassert task in ('TextCraft','AppWorld')\nassert method in ('dt','ppo','grpo')")
start = code.index('job, = [j for j in ')
end = code.index('source = json.loads', start)
code = code[:start] + '''segment=ROOT/'receipts'/(task.lower()+'-eight-gpu-v9-clip-u-20261019-v2')
prepared=json.loads((segment/'campaign-prepared.json').read_bytes())
expected,=[j for j in prepared['jobs'] if j['method']==method]
state_path=segment/'campaign-state.json'
state=json.loads(state_path.read_bytes()) if state_path.exists() else {}
selected=[j for j in state.get('jobs',[]) if j['method']==method and j['run']==expected['run']]
if not selected or not selected[0].get('pid'):
    print(json.dumps(dict(status='pending_requested_campaign_job_not_started',task=task,method=method,run=expected['run'])))
    raise SystemExit(0)
job,=selected
formal,=[j for j in json.loads((ROOT/'formal-training.json').read_bytes())['jobs']
    if j.get('run')==expected['run'] and j.get('pid')==job['pid'] and j.get('birth')==job['birth']]
assert formal['source_receipt']==job['source_receipt']
world_size=len(job['devices'])
assert world_size==8
''' + code[end:]
anchor = "__file__, method, '--native'"
assert code.count(anchor) == 1
code = code.replace(anchor, "__file__, task, method, '--native'")
code = code.replace('for rank in (0, 1):', 'for rank in range(world_size):')
code = code.replace("f'{kind}_world_size_2_rank_{rank}.pt'", "f'{kind}_world_size_{world_size}_rank_{rank}.pt'")
start = code.index("assert source.get('sampler_checkpoint_artifact')")
end = code.index('assert not torch.cuda.is_initialized()', start)
sampler = code[start:end]
# Read the actual fixed trainer artifact. Metadata labels are not an additional
# checkpoint requirement; a conflicting explicit label still needs inspection.
sampler = sampler.replace("assert source.get('sampler_checkpoint_artifact') == 'loop_sampler.pt', 'Actual LOOP sampler state was not included by this execution'",
    "assert source.get('sampler_checkpoint_artifact','loop_sampler.pt')=='loop_sampler.pt'\nassert 'loop_sampler.pt' in (Path(source['verl_root'])/'verl/trainer/ppo/ray_trainer.py').read_text()")
sampler = sampler.replace("checkpoint / source['sampler_checkpoint_artifact']", "checkpoint / 'loop_sampler.pt'")
sampler = sampler.replace('assert len(sampler_states) == 2', 'assert len(sampler_states) == world_size')
code = code[:start] + "sampler_metadata=[];sampler_path=None\nif task=='AppWorld':\n" + ''.join(
    '    ' + line if line.strip() else line for line in sampler.splitlines(True)) + code[end:]
code = code.replace('sampler_state_path=str(sampler_path)', "sampler_state_path=str(sampler_path) if sampler_path is not None else None")
code = code.replace('hashlib.sha256(sampler_path.read_bytes()).hexdigest()',
    'hashlib.sha256(sampler_path.read_bytes()).hexdigest() if sampler_path is not None else None')
code = code.replace("method=method, version=job['version'], step=step,", "task=task, method=method, version=job['version'], step=step,")
code = code.replace("scope='Completed native marker and readable saved model/optimizer/scheduler/training and rollout RNG/data plus actual LOOP sampler RNG/order/prefetch state. CPU validation instances only; no training restore or trajectory equivalence claim'",
    "scope='Eight native ranks: readable model/optimizer/scheduler/training and rollout RNG/data; AppWorld also has its native LOOP sampler RNG/order/prefetch state. CPU reads only; no training restore or trajectory equivalence claim'")
EXECUTED_SOURCE_SHA256 = hashlib.sha256(code.encode()).hexdigest()
exec(compile(code, str(owner), 'exec'), globals())
