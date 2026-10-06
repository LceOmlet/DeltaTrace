"""Compose the existing guarded stager for an isolated official-whitening probe.

Root prepares the frozen VERL copy and its one trainer seam beforehand. This
entry reuses a recorded recovered environment, existing assets and caches; it
does not patch, install, recreate, or infer a live parent's environment.
"""
import argparse
import ast
import hashlib
import subprocess

import stage_textcraft_grpo_support_gradients as prior
from stage_environment_entry import AUDIT, ROOT, SSH, SCP

OUT = ROOT + '/receipts/textcraft-official-whitening-20261006-v1'
LOCAL = AUDIT / 'textcraft-degradation-20261005/official-whitening-20261006/v1'
NAME = 'verify_textcraft_official_whitening.py'
LABEL_ROOT = ROOT + '/receipts/textcraft-label-gradient-20261006-v2'
VERL = ROOT + '/candidates/textcraft-official-whitening-20261006-v1/verl'
TRAINER_SHA = '7366557b482e604d66f47bdc4841ea147ffaac7c80538fa000544bb4e92eb619'
ENVIRONMENT_FILE = OUT + '/recovered-runtime-environment.json'
ENVIRONMENT_SHA = '9c7a56dd8a0e49558b7378be0e8ea3784fce49a75524738819b4e5d4b3c3cea7'
PINNED = dict(prior.PINNED)
PINNED[LABEL_ROOT + '/observe_textcraft_label_gradients.py'] = (
    '9ed52c51f5eac34a2b12d5425e386f4b3e5b5945f2c6b61e01b25aba837eddb7')


def prepared_script(mode, hashes, *, verl_root, trainer_sha, environment_file, environment_sha):
    # Compose the existing ownership/PID/resource/staging implementation. Only
    # this diagnostic's sources and a recorded environment provider are named.
    old = {name: getattr(prior, name) for name in ('OUT', 'LOCAL', 'NAME', 'DEPS', 'PINNED')}
    try:
        prior.OUT, prior.LOCAL, prior.NAME, prior.DEPS, prior.PINNED = (
            OUT, LOCAL, NAME, (), {**PINNED, OUT + '/' + NAME: hashes[NAME]})
        script = prior.prepared_script(mode, hashes)
    finally:
        for name, value in old.items():
            setattr(prior, name, value)
    parent = "parent=psutil.Process(prior['reused_environment_pid'])\nassert abs(parent.create_time()-prior['reused_environment_pid_birth'])<.05\nenv=parent.environ(); env.pop('RAY_ADDRESS',None)"
    assert parent in script
    recovered = """environment_file=pathlib.Path(__ENVFILE__)
assert hashlib.sha256(environment_file.read_bytes()).hexdigest()==__ENVSHA__
recovered=json.loads(environment_file.read_bytes())
assert all(isinstance(k,str) and isinstance(v,str) for k,v in recovered['environment'].items())
for filename,expected in recovered['sources'].items():
 assert hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest()==expected,(filename,'Recovered environment source changed')
import os
env=os.environ.copy(); env.update(recovered['environment'])
for key in ('RAY_ADDRESS','MACA_VISIBLE_DEVICES','RAY_TMP','RAY_TMPDIR'):
 env.pop(key,None)
frozen_verl_root=roots['verl']; overlay_verl_root=pathlib.Path(__VERLROOT__)
overlay_trainer=overlay_verl_root/'verl/trainer/ppo/ray_trainer.py'
assert hashlib.sha256(overlay_trainer.read_bytes()).hexdigest()==__TRAINERSHA__
overlay_sources={str(overlay_trainer):__TRAINERSHA__}
for filename,expected in checks.items():
 p=pathlib.Path(filename)
 if p.is_relative_to(frozen_verl_root):
  relative=p.relative_to(frozen_verl_root)
  if relative==pathlib.Path('verl/trainer/ppo/ray_trainer.py'):
   continue
  actual=overlay_verl_root/relative
  assert hashlib.sha256(actual.read_bytes()).hexdigest()==expected,(str(actual),'Frozen VERL non-trainer source changed')
  overlay_sources[str(actual)]=expected
roots['verl']=overlay_verl_root"""
    script = script.replace(parent, recovered)
    # The prior environment may contain an original frozen VERL path rather
    # than the AppWorld alias handled by the existing stager.
    marker = "env.update(PYTHONPATH=str(out)+':'+str(base)+':" + prior.ADAM
    assert marker in script
    script = script.replace(marker,
        "tail=tail.replace(str(frozen_verl_root),str(overlay_verl_root))\n" + marker)
    path_marker = "PYTHONPATH=str(out)+':'+str(base)+':" + prior.ADAM + ":'"
    assert path_marker in script
    script = script.replace(path_marker,
        "PYTHONPATH=str(out)+':'+str(overlay_verl_root)+':__LABELROOT__:'+str(base)+':" + prior.ADAM + ":'")
    env_marker = "env['DT_TEXTCRAFT_ADAM_RECIPE_ROOT']='" + prior.ADAM + "'"
    assert env_marker in script
    script = script.replace(env_marker, env_marker +
        "\nenv['DT_TEXTCRAFT_WHITENING_ROOT']=str(out)" +
        "\nenv['DT_TEXTCRAFT_WHITENING_TRAINER_SHA']=__TRAINERSHA__")
    provider_marker = "devices=[4,5],reused_environment_pid=parent.pid,\n reused_environment_pid_birth=parent.create_time(),observed_unix=time.time(),optimizer_steps=0,"
    assert provider_marker in script
    script = script.replace(provider_marker,
        "devices=[4,5],recovered_runtime_environment=dict(path=str(environment_file),sha256=__ENVSHA__,scope=recovered['scope']),\n overlay_verl_sources=overlay_sources,observed_unix=time.time(),optimizer_steps=0,")
    equality = "assert inspected['worker_constructor']=={key:prior_imports['worker_constructor'][key] for key in ['path','sha256']}, 'Native owner path/hash mismatch'"
    assert equality in script
    script = script.replace(equality,
        "assert inspected['worker_constructor']['sha256']==prior_imports['worker_constructor']['sha256'], 'Copied native worker bytes changed'\n receipt['original_frozen_worker_constructor']=prior_imports['worker_constructor']")
    role = "role='Prepared isolated same saved native64/checkpoint25; two support groups each two original backward passes with optimizer/scheduler no-op; no rollout or DT'"
    assert role in script
    script = script.replace(role,
        "role='Prepared saved native64/checkpoint25 raw/officially whitened full-batch advantages, original PG/H/KL and cross-PG Gram; no rollout, DT or real optimizer/scheduler step'")
    script = script.replace('planned_backward_passes_per_rank=4,finite_trace_calls=0',
        'planned_backward_passes_per_rank=6,finite_trace_calls=0')
    script = script.replace("status='native_support_gradient_submitted'",
        "status='native_official_whitening_submitted'")
    names = "names=('observe_native_optimizer_minibatch','observe_native_actor_loss_gradients','observe_textcraft_native_batches')"
    assert names in script
    script = script.replace(names,
        "names=('observe_native_optimizer_minibatch','observe_native_actor_loss_gradients','observe_textcraft_native_batches','observe_textcraft_label_gradients','verify_textcraft_native_adam','verify_textcraft_official_whitening')")
    keys = "'sources','diagnostic_source','diagnostic_dependencies','reused_diagnostic_sources','input','config_source','checkpoint'"
    assert keys in script
    script = script.replace(keys, keys + ",'recovered_runtime_environment','overlay_verl_sources'")
    script = (script.replace('__ENVFILE__', repr(environment_file))
        .replace('__ENVSHA__', repr(environment_sha)).replace('__VERLROOT__', repr(verl_root))
        .replace('__TRAINERSHA__', repr(trainer_sha)).replace('__LABELROOT__', LABEL_ROOT))
    ast.parse(script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    return script


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    parser.add_argument('--verl-root', default=VERL)
    parser.add_argument('--trainer-sha', default=TRAINER_SHA)
    parser.add_argument('--environment-file', default=ENVIRONMENT_FILE)
    parser.add_argument('--environment-sha', default=ENVIRONMENT_SHA)
    args = parser.parse_args()
    hashes = {NAME: hashlib.sha256((AUDIT / NAME).read_bytes()).hexdigest()}
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(SCP + [str(AUDIT / NAME), f'{SSH[-1]}:{OUT}/{NAME}'], check=True)
    script = prepared_script(args.mode, hashes, verl_root=args.verl_root,
        trainer_sha=args.trainer_sha, environment_file=args.environment_file,
        environment_sha=args.environment_sha)
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
             'worker-callsite-import-inspection.json', 'worker-callsite-import.stdout.txt')]
            + [str(LOCAL)], check=True)
