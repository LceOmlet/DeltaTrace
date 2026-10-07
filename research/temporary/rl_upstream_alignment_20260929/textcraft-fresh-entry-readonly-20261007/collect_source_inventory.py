"""Read source and existing receipts only; never import training/model owners."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
spec = importlib.util.spec_from_file_location("existing_ssh_configuration", AUDIT / "stage_environment_entry.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VENV = "/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python"

SOURCE_READER = r'''
import ast, difflib, hashlib, json, pathlib, shutil, subprocess, time
P=pathlib.Path
ROOT=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
TENTRY=ROOT/'candidates/textcraft-official-whitening-formal-20261006-v2/entry'
TSOURCE=ROOT/'runs/textcraft-official-whitening-20261006-v2/textcraft-dt/source.json'
ASOURCE=ROOT/'runs/appworld-fresh-row-prefix-20261007-v1/appworld-dt/source.json'
ts=json.loads(TSOURCE.read_bytes()); ap=json.loads(ASOURCE.read_bytes())
def h(p): return hashlib.sha256(P(p).read_bytes()).hexdigest()
def file(p, recorded=None):
 p=P(p); z={'path':str(p),'exists':p.is_file()}
 if z['exists']: z.update(sha256=h(p),bytes=p.stat().st_size)
 if recorded is not None: z.update(recorded_sha256=recorded,matches_recorded=z.get('sha256')==recorded)
 return z
def extract(p, terms, radius=2):
 lines=P(p).read_text().splitlines(); positions=set()
 for i,line in enumerate(lines):
  if any(term in line for term in terms): positions.update(range(max(0,i-radius),min(len(lines),i+radius+1)))
 return [{'line':i+1,'text':lines[i]} for i in sorted(positions)]
def changed_functions(p,q):
 def funcs(path):
  return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(P(path).read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 x,y=funcs(p),funcs(q)
 return {'changed':[k for k in x.keys()&y.keys() if x[k]!=y[k]],'textcraft_only':sorted(x.keys()-y.keys()),'appworld_only':sorted(y.keys()-x.keys())}
result={'scope':'Only stdlib source/receipt reads. No Torch, owner import, GPU/model/DT/RPC/test/checkpoint access. No files modified remotely.', 'observed_unix':time.time(), 'textcraft_source':file(TSOURCE,'175565cfe8c83ae146f7f06dfd93f62361705e95a0ac0853106deac10b182708'), 'appworld_source':file(ASOURCE,'c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789')}
keys=['upstream_commit','lora_rank','lora_alpha','white_patch_commit','semantic_repair','sampling','actor_fix_commit','dt_dispatch_commit']
result['textcraft_frozen_provenance']={k:ts.get(k) for k in keys}
selected=('launch_textcraft_native.py','textcraft_owner_rollout.py','textcraft_environment_entry.py','owner_trajectory_batch.py','dt_training_batch.py','deltatrace_rollout.py','reward_readout.py','owner_runtime_options.py','owner_environment_configs.json','textcraft_qwen_template.json')
result['textcraft_actual_entry']={n:file(TENTRY/n,ts['entry_sha256'].get(n)) for n in selected}
vnames=('verl/trainer/ppo/ray_trainer.py','verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py','verl/trainer/ppo/core_algos.py','verl/utils/torch_functional.py','verl/workers/sharding_manager/fsdp_vllm.py','verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py')
result['verl_owners']={n:{'textcraft':file(P(ts['verl_root'])/n,ts['verl_sha256'].get(n)),'appworld':file(P(ap['verl_root'])/n,ap['verl_sha256'].get(n))} for n in vnames}
result['verl_changed_functions']={n:changed_functions(P(ts['verl_root'])/n,P(ap['verl_root'])/n) for n in vnames if (P(ts['verl_root'])/n).is_file() and (P(ap['verl_root'])/n).is_file()}
native=P(ts['agentgym_root'])/'AgentGym-RL'
nnames=('examples/train/AgentGym-RL/textcraft_train.sh','examples/eval/textcraft_eval.sh','verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py','verl/utils/agent_dataset/rl_dataset.py')
result['native_author_files']={n:file(native/n) for n in nnames}
if shutil.which('git'):
 cp=subprocess.run(['git','-C',str(native),'rev-parse','HEAD'],capture_output=True,text=True)
 result['native_git_head']={'returncode':cp.returncode,'stdout':cp.stdout.strip(),'stderr':cp.stderr.strip()}
else:
 result['native_git_head']={'unavailable':'git executable absent; source identity supplied by preserved author receipt and actual source SHA'}
result['native_rollout_lines']=extract(native/nnames[2],['handler','truncate','max_response','rounds','scores','agentgym'],1)
result['textcraft_rollout_owner_lines']=extract(TENTRY/'textcraft_owner_rollout.py',['owner_module','vllm_rollout.py','rl_dataset.py','RolloutHandler','super().add_assistant_message','active_masks=True','length > 0','scores[index]','module.vLLMRollout','EngineTransport','generate_sequences'],2)
result['textcraft_fresh_launcher_lines']=extract(TENTRY/'launch_textcraft_native.py',['resume_mode','resume_from','--resume-from','data.max_prompt_length','data.max_response_length','ppo_epochs','total_epochs','owner_command'],2)
result['textcraft_whitening_lines']=extract(P(ts['verl_root'])/'verl/trainer/ppo/ray_trainer.py',['masked_whiten','deltatrace_advantages','_dt_raw_advantages','advantages_raw','whiten'],3)
result['appworld_whitening_lines']=extract(P(ap['verl_root'])/'verl/trainer/ppo/ray_trainer.py',['masked_whiten','deltatrace_advantages','_dt_raw_advantages','advantages_raw','whiten'],3)
result['textcraft_native_config']=json.loads((TENTRY/'owner_environment_configs.json').read_bytes())['TextCraft']
result['textcraft_startup_selected']={k:v for k,v in ts['startup_options'].items() if any(t in k for t in ('micro','lora','max_','train_batch','ppo_mini','ppo_epochs','total_epochs','env.max_steps','resume','kl','entropy'))}
result['textcraft_producer_lines']=extract(TENTRY/'deltatrace_rollout.py',['prefix_lease_factory','prepare_native_prefix_leases','Qwen35','runner','readout','lease'],1)
ae=P(ap.get('entry',str(ASOURCE.parent)))
if not (ae/'deltatrace_rollout.py').is_file():
 ae=ROOT/'candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/entry'
result['appworld_producer_lines']=extract(ae/'deltatrace_rollout.py',['prefix_lease_factory','prepare_native_prefix_leases','individual_prefixes','boundary_row_storage'],2)
result['readout_same']={'textcraft':file(TENTRY/'reward_readout.py'),'appworld':file(ae/'reward_readout.py')}
dn=('clean/qwen35/qwen35_dense_finite_runner.py','clean/qwen35/qwen35_answer_finite.py','clean/qwen35/qwen35_native_prefix_artifacts.py','clean/qwen35/vendor_fa_finite_bf16_d256.py','clean/qwen35/finite_fla_gpu.py','clean/qwen35/qwen35_gdn_finite.py')
result['dt_selected']={n:{'textcraft':file(P(ts['dt_root'])/n,ts['dt_source_sha256'].get(n)),'appworld':file(P(ap['dt_root'])/n,ap['dt_source_sha256'].get(n))} for n in dn}
result['dt_prefix_leases']={'textcraft':file(TENTRY/'native_prefix_leases.py',ts['entry_sha256'].get('native_prefix_leases.py')),'appworld':file(ae/'native_prefix_leases.py',ap['entry_sha256'].get('native_prefix_leases.py'))}
for n in ('verl/trainer/ppo/ray_trainer.py','verl/workers/actor/dp_actor.py','verl/workers/sharding_manager/fsdp_vllm.py','verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'):
 old=P(ts['verl_root'])/n;new=P(ap['verl_root'])/n
 result.setdefault('owner_source_diffs',{})[n]=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=str(old),tofile=str(new),n=3))
hfroot=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/lib')
result['current_installed_hf_source']=[file(p) for p in hfroot.glob('python*/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py')]
result['historical_hf_provenance_fields']={k:v for k,v in ts.items() if 'hf' in k.lower() or 'transform' in k.lower()}
print(json.dumps(result,ensure_ascii=False,indent=2))
'''

if __name__ == "__main__":
    command = module.SSH + [VENV, "-"]
    completed = subprocess.run(command, input=SOURCE_READER.encode(), capture_output=True)
    for name in ("source-reader.stdout.json", "source-reader.stderr.txt", "collector-receipt.json"):
        old = HERE / name
        if old.exists():
            sha = hashlib.sha256(old.read_bytes()).hexdigest()[:12]
            preserved = HERE / (name + ".before-" + sha)
            if not preserved.exists():
                preserved.write_bytes(old.read_bytes())
    (HERE / "source-reader.stdout.json").write_bytes(completed.stdout)
    (HERE / "source-reader.stderr.txt").write_bytes(completed.stderr)
    receipt = {"command": command, "returncode": completed.returncode,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reader_sha256": hashlib.sha256(SOURCE_READER.encode()).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "scope": "Read-only source and historical receipt inventory. No tests or owner imports."}
    (HERE / "collector-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if completed.returncode:
        print(completed.stderr.decode(errors="replace"))
        raise SystemExit(completed.returncode)
