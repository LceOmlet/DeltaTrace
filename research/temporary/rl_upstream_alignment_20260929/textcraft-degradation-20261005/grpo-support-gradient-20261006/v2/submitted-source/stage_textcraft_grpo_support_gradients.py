"""Compose the existing guarded stager; prepare does CPU inspection only."""
import argparse
import hashlib
import json
import subprocess

import stage_textcraft_native_readout as prior
from stage_environment_entry import AUDIT, ROOT, SSH, SCP

OUT = ROOT + '/receipts/textcraft-grpo-support-gradient-20261006-v2'
LOCAL = AUDIT / 'textcraft-degradation-20261005/grpo-support-gradient-20261006/v2'
ADAM = ROOT + '/receipts/textcraft-native-adam-20261006-v1'
NAME = 'verify_textcraft_grpo_support_gradients.py'
DEPS = ('observe_textcraft_grpo_support_gradients.py',)
PINNED = {
    ADAM + '/verify_textcraft_native_adam.py': 'cdb45238843b685234f21b882f5635b22a1bdba072948354a92ab98360268901',
    ADAM + '/observe_native_adam_update.py': '74b66bf8bb703c66ae17e2cbb8506a18fe7685b3b073fd94b8ca07a6e43be36c',
    ADAM + '/effective-config.yaml': '9f663a7ddad06203edaaa59ec0237c1bdf64ac1ced88e9ce00d598f9a556b4dd',
    prior.BASE + '/observe_native_optimizer_minibatch.py': '022466b2bac94617b8e627ea027fb3afdf4490dc4bc2bcaa11944ab444e36214',
    prior.BASE + '/observe_native_actor_loss_gradients.py': '94219328ba644a5c0118f4a3bd2561b16f969643f2cd2915047202a7ff085047',
    prior.BASE + '/observe_textcraft_native_batches.py': '429ca8254e54ed3ec4882027e72d74a59aba0d9069a92086943a5817102daa4d',
    ROOT + '/receipts/textcraft-native-readout-20261006-v2/verify_textcraft_native_readout.py': '1fd8d8981bf82f0992db78cce43251cd2040afb4a3f039d702eb528b8d8f69cb',
}


def prepared_script(mode, hashes):
    script = prior.SCRIPT
    old = "cases=base/'readout-first-response-cases.json'\nassert cases.is_file(), 'CPU mapping must prepare and verify the exact original prefixes first'\ndata=json.loads(cases.read_bytes()); assert [len(rows) for rows in data['rank_cases']]==[32,32]\nassert len({row['traj_uid'] for rows in data['rank_cases'] for row in rows})==64"
    new = "cases=base/'native-optimizer-minibatch.pkl'\nboundary=json.loads((base/'native-minibatch-update-boundary.json').read_bytes())\nassert hashlib.sha256(cases.read_bytes()).hexdigest()==boundary['snapshot']['sha256']"
    assert old in script
    script = script.replace(old, new)
    script = script.replace("PYTHONPATH=str(out)+':'+tail", "PYTHONPATH=str(out)+':'+str(base)+':__ADAM__:'+str(base.parent/'textcraft-native-readout-20261006-v2')+':'+tail")
    script = script.replace("ast.parse(source.read_bytes())", "ast.parse(source.read_bytes())\nextra_hashes=__EXTRA_HASHES__\npinned=__PINNED__\nfor filename, expected in extra_hashes.items():\n p=out/filename; assert hashlib.sha256(p.read_bytes()).hexdigest()==expected; ast.parse(p.read_bytes())\nfor filename, expected in pinned.items():\n p=pathlib.Path(filename); assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,(filename,'Pinned owner/config mismatch')")
    script = script.replace("launch=json.loads((base/'launch.json').read_bytes())", "env['DT_TEXTCRAFT_ADAM_RECIPE_ROOT']='__ADAM__'\nlaunch=json.loads((base/'launch.json').read_bytes())")
    script = script.replace("diagnostic_source=dict(path=str(source),sha256='__SHA__'),", "diagnostic_source=dict(path=str(source),sha256='__SHA__'),\n diagnostic_dependencies={n:dict(path=str(out/n),sha256=h) for n,h in extra_hashes.items()},\n reused_diagnostic_sources=pinned,")
    script = script.replace("'sources','diagnostic_source','input','config_source','checkpoint'", "'sources','diagnostic_source','diagnostic_dependencies','reused_diagnostic_sources','input','config_source','checkpoint'")
    script = script.replace("role='Isolated 64 real first-response prefixes; original native outcome reader; no rollout/DT/backward/update'", "role='Prepared isolated same saved native64/checkpoint25; two support groups each two original backward passes with optimizer/scheduler no-op; no rollout or DT'")
    script = script.replace("native_batch_per_call=4,finite_trace_calls=0", "native_batch_per_call=4,planned_backward_passes_per_rank=4,finite_trace_calls=0")
    script = script.replace("status='native_reader_submitted'", "status='native_support_gradient_submitted'")
    marker = "if mode=='prepare':\n with (out/'native-owner-inspection.stdout.txt').open('wb') as log:"
    callsite = """if mode=='prepare':
 code=\"import hashlib,importlib,json,pathlib,os,psutil,torch; names=('observe_native_optimizer_minibatch','observe_native_actor_loss_gradients','observe_textcraft_native_batches'); modules={n:importlib.import_module(n) for n in names}; sources={n:dict(path=str(pathlib.Path(m.__file__).resolve()),sha256=hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest()) for n,m in modules.items()}; observer=modules['observe_native_optimizer_minibatch']; assert observer.observe_native_optimizer_minibatch.__globals__ is vars(observer); value=dict(sources=sources,pid=os.getpid(),pid_birth=psutil.Process().create_time(),rss_bytes=psutil.Process().memory_info().rss,cuda_initialized=torch.cuda.is_initialized(),distributed_initialized=torch.distributed.is_initialized(),scope='Fresh Python subprocess using worker PYTHONPATH only; no driver import/sys.path edits, model or loss calls.'); pathlib.Path('__OUT__/worker-callsite-import-inspection.json').write_text(json.dumps(value,indent=2)+chr(10))\"
 with (out/'worker-callsite-import.stdout.txt').open('wb') as log:
  callsite=subprocess.run([env['VENV_PYTHON'],'-c',code],cwd=out,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
 assert callsite.returncode==0, 'Recorded helper callsite import failure before GPU submission'
 imports=json.loads((out/'worker-callsite-import-inspection.json').read_bytes())
 for item in imports['sources'].values():
  assert item['sha256']==pinned[item['path']], 'Helper callsite actual path/hash mismatch'
 assert not imports['cuda_initialized'] and not imports['distributed_initialized']
 receipt['worker_callsite_import_inspection']=dict(path=str(out/'worker-callsite-import-inspection.json'),sha256=hashlib.sha256((out/'worker-callsite-import-inspection.json').read_bytes()).hexdigest())
 with (out/'native-owner-inspection.stdout.txt').open('wb') as log:"""
    assert marker in script
    script = script.replace(marker, callsite)
    return (script.replace('__OUT__', OUT).replace('__BASE__', prior.BASE).replace('__MODE__', mode)
        .replace('__NAME__', NAME).replace('__SHA__', hashes[NAME]).replace('__ADAM__', ADAM)
        .replace('__EXTRA_HASHES__', repr({name: hashes[name] for name in DEPS})).replace('__PINNED__', repr(PINNED)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    hashes = {name: hashlib.sha256((AUDIT / name).read_bytes()).hexdigest() for name in (NAME, *DEPS)}
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(SCP + [str(AUDIT / name) for name in hashes] + [f'{SSH[-1]}:{OUT}/'], check=True)
    script = prepared_script(args.mode, hashes)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / f'{args.mode}.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    if args.mode == 'prepare':
        subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}' for name in
            ('prepared.json', 'native-owner-inspection.json', 'native-owner-inspection.stdout.txt',
             'worker-callsite-import-inspection.json', 'worker-callsite-import.stdout.txt')] + [str(LOCAL)], check=True)
