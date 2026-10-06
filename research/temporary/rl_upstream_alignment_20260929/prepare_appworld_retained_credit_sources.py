"""Reuse the accepted retained-action seam on the frozen AppWorld entry.

CPU preparation only. No running entry, model, service or job is modified.
The shared two-file fix is taken from its recorded Git commit; no credit,
environment or training implementation is copied into this preparer.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import ENTRY, REPO, ROOT, SCP, SSH


FIX = 'e42f5968f8cf62d447f6f30262c9672b9c9ddc52'
FILES = {
    'owner_trajectory_batch.py': (
        'trajectory_credit',
        '2d3a93f03ba13b49056209c5be5488da6bea0fb9ba1f58d8549b358194b6aca3',
        '95042a7a610e4c7781982873b5730d4bb744145dbcabe61961d55621e83d8939'),
    'dt_training_batch.py': (
        'compute_training_credit',
        'da9b8a01c3bb9ba00fe3388fd95f93961d33c44bdf0eccf32e7b5846e2ffcae2',
        'cdefd4b7ec764dccf1b7ab3d5783102264c969f0d343fb949782dfda58f92223'),
}
LOCAL = Path(__file__).parent / 'appworld-efficiency-20261006' / 'retained-action-v1'
REMOTE = ROOT + '/receipts/appworld-efficiency-20261006/retained-action-v1'
BASE = ROOT + '/candidates/appworld-retained-credit-sources-20261006-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-metrics', action='store_true',
                        help='Prepare a separate v2 entry with CPU source-work counters.')
    args = parser.parse_args()
    local, receipt_root, candidate_root = LOCAL, REMOTE, BASE
    files = dict(FILES)
    if args.work_metrics:
        local = LOCAL.with_name('retained-action-v2')
        receipt_root = REMOTE.replace('retained-action-v1', 'retained-action-v2')
        candidate_root = BASE.replace('20261006-v1', '20261006-v2')
        function, before, _ = files['dt_training_batch.py']
        metric_blob = (REPO / 'experiments/rl/dt_training_batch.py').read_bytes()
        baseline = subprocess.check_output(
            ['git', 'show', f'{FIX}:experiments/rl/dt_training_batch.py'], cwd=REPO)

        class RemoveWorkCounters(ast.NodeTransformer):
            def visit_Assign(self, node):
                if any(isinstance(t, ast.Name) and t.id in ('nonzero_before', 'retained_sources')
                       for t in node.targets):
                    return None
                return self.generic_visit(node)

            def visit_Expr(self, node):
                if (isinstance(node.value, ast.Call) and
                    isinstance(node.value.func, ast.Name) and node.value.func.id == 'print' and
                    node.value.args and isinstance(node.value.args[0], ast.JoinedStr) and
                    '[DT source workload]' in ast.unparse(node.value.args[0])):
                    return None
                return self.generic_visit(node)

        assert ast.dump(RemoveWorkCounters().visit(ast.parse(metric_blob))) == ast.dump(ast.parse(baseline)), \
            'Work counters must not change the accepted credit computation'
        files['dt_training_batch.py'] = (function, before, hashlib.sha256(metric_blob).hexdigest())
    # Keep v1's immutable receipt and source; v2 composes that same accepted
    # two-function seam with counters only, not another dispatch algorithm.
    local.mkdir(parents=True, exist_ok=True)
    for name, (_, _, digest) in files.items():
        blob = (metric_blob if args.work_metrics and name == 'dt_training_batch.py' else
                subprocess.check_output(['git', 'show', f'{FIX}:experiments/rl/{name}'], cwd=REPO))
        assert hashlib.sha256(blob).hexdigest() == digest
        (local / name).write_bytes(blob)
    test = 'test_distributed_credit.py'
    (local / test).write_bytes(subprocess.check_output(
        ['git', 'show', f'HEAD:experiments/rl/{test}'], cwd=REPO))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    code = r'''
import ast,hashlib,json,os,psutil,shutil,subprocess,time
from pathlib import Path
root=Path(@ROOT@);receipt=Path(@REMOTE@);base=Path(@BASE@)
old=root/'candidates/appworld-official-whitening-20261006-v1/entry'
run=root/'runs/appworld-official-whitening-20261006-v2/appworld-dt'
driver=psutil.Process(3592468);assert driver.create_time()==1791291997.14
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=json.loads((run/'source.json').read_bytes())
assert sha(run/'source.json')=='2827c5ae8785c498785b639573769d7db0208d3e3540087a4a0816b6968a34ac'
assert not base.exists(), 'Inspect existing candidate; do not replace a frozen entry'
assert not (receipt/'prepared.json').exists()
for name,digest in source['entry_sha256'].items():assert sha(old/name)==digest,name
entry=base/'entry'
shutil.copytree(old,entry,ignore=shutil.ignore_patterns('__pycache__'))
changes={}
for name,(function,before,after) in @FILES@.items():
 assert sha(old/name)==before and sha(receipt/name)==after
 original=ast.parse((old/name).read_text());candidate=ast.parse((receipt/name).read_text())
 original.body=[node for node in original.body if not isinstance(node,ast.FunctionDef) or node.name!=function]
 candidate.body=[node for node in candidate.body if not isinstance(node,ast.FunctionDef) or node.name!=function]
 assert ast.dump(original)==ast.dump(candidate), 'Change outside accepted seam: '+name
 shutil.copy2(receipt/name,entry/name)
 changes[name]={'before_sha256':before,'after_sha256':after,'function':function,'source_commit':@FIX@}
unchanged={}
for name,digest in source['entry_sha256'].items():
 if name not in changes:
  assert sha(entry/name)==digest,name
  unchanged[name]=digest
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='-1',MACA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 VERL_ROOT=source['verl_root'],DT_ENTRY_ROOT=str(entry))
env['PYTHONPATH']=':'.join([str(entry),source['verl_root'],env.get('PYTHONPATH','')])
nodes=['test_native_two_rank_dispatch_keeps_complete_future_returns',
 'test_zero_batch_does_not_enter_fsdp_collectives',
 'test_training_slices_do_not_drop_later_executed_reward',
 'test_empty_training_slice_selection_avoids_all_dt_calls',
 'test_trajectory_uses_native_slice_sources_after_complete_returns',
 'test_owner_balance_restores_credit_after_padding_and_duplicate_rows',
 'test_native_denominators_stay_collectively_aligned_without_changing_credit']
xml=receipt/'cpu-tests.xml';started=time.time()
r=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib',
 *[str(receipt/'test_distributed_credit.py')+'::'+n for n in nodes],
 '--junitxml='+str(xml)],cwd=entry,env=env,capture_output=True)
(receipt/'cpu-tests.stdout.txt').write_bytes(r.stdout+r.stderr)
record={'status':'prepared_only_CPU_interface_tests_passed' if r.returncode==0 else 'prepared_only_CPU_tests_failed',
 'observed_unix':time.time(),'wall_seconds':time.time()-started,'preparation_repository_commit':@REVISION@,
 'source_fix_commit':@FIX@,'work_counters':@WORK_COUNTERS@,
 'work_counters_source_commit':@REVISION@ if @WORK_COUNTERS@ else None,
 'prior_entry':str(old),'entry':str(entry),'verl_root':source['verl_root'],
 'dt_root':source['dt_root'],'driver_pid':3592468,'driver_birth':driver.create_time(),
 'source_receipt':str(run/'source.json'),'source_receipt_sha256':sha(run/'source.json'),
 'changes':changes,'unchanged_entry_sha256':unchanged,
 'cpu_tests':{'returncode':r.returncode,'xml':str(xml),'sha256':sha(xml) if xml.exists() else None,
 'scope':'Actual VERL dispatch/pad/collect and Q/V transport with fixed ratio test doubles; no DT numerical or performance claim'},
 'scope':'Only select existing retained source identities after complete returns. No reward, native trajectory, sampling, PPO, whitening, LoRA, batch or tolerance change.',
 'deployment':False,'GPU_calls':0,'training_calls':0}
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
r.check_returncode()
'''
    for key, value in {'ROOT': ROOT, 'REMOTE': receipt_root, 'BASE': candidate_root, 'FILES': files,
                       'FIX': FIX, 'REVISION': revision, 'WORK_COUNTERS': args.work_metrics}.items():
        code = code.replace('@' + key + '@', repr(value))
    ast.parse(code)
    mkdir = subprocess.run(SSH + ['mkdir', '-p', receipt_root], capture_output=True)
    mkdir.check_returncode()
    subprocess.run(SCP + [str(local / name) for name in [*files, test]] +
                   [f'{SSH[-1]}:{receipt_root}/'], check=True)
    script = f'source {ENTRY}/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n{code}\nPY\n'
    (local / 'prepare.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (local / 'prepare.stdout.txt').write_bytes(result.stdout + result.stderr)
    for name in ('prepared.json', 'cpu-tests.xml', 'cpu-tests.stdout.txt'):
        subprocess.run(SCP + [f'{SSH[-1]}:{receipt_root}/{name}', str(local / name)], check=True)
    result.check_returncode()
    print(json.dumps({'local': str(local), 'remote': receipt_root, 'candidate': candidate_root}))


if __name__ == '__main__':
    main()
