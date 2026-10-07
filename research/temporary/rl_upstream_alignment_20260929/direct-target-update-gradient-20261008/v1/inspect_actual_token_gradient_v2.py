"""One real first TextCraft optimizer minibatch, no optimizer/scheduler update.

The unchanged existing native observer calls original VERL update_policy twice.
Its legacy diagnostic_grpo_advantages/grpo_pg names carry ONLY the isolated
actual Format-token DT coefficient in this diagnostic, never a GRPO method.
All other original coefficients/masks/IDs and native loss/accumulation persist.
"""
import hashlib,inspect,json,os,sys,time
from pathlib import Path
import torch,ray
from omegaconf import OmegaConf
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from inspect_extreme_endpoint import check_imports,actor_initialization_steps
import observe_native_optimizer_minibatch as observer

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/direct-target-update-gradient-20261008-v1'
SOURCE=ROOT/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
CAPTURE=ROOT/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
ACTOR='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
UID='6e76f70e-bdeb-4726-8b35-d9e21eff2f68'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(f):
 p=inspect.getsourcefile(inspect.unwrap(f));return dict(path=p,sha256=sha(p))
def phase(name,**kw):
 r=dict(phase=name,unix=time.time(),pid=os.getpid(),**kw);(OUT/'phase.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)

@ray.remote
class GradientWorker(ActorRolloutRefWorker):
 @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
 def observe_actual(self,data):
  from observe_native_optimizer_minibatch import _temporary_attribute
  assert self.actor.config.ppo_mini_batch_size==32 and self.actor.config.ppo_micro_batch_size_per_gpu==4
  assert identity(type(self.actor))['sha256']==ACTOR
  before={}
  for name,p in self.actor_module_fsdp.named_parameters():
   if p.requires_grad:
    t=p.detach();t=t.to_local() if hasattr(t,'to_local') else t;before[name]=t.cpu().clone()
  assert before and not any(torch.count_nonzero(v).item() for k,v in before.items() if '.lora_B.' in k)
  scheduler_calls=[]
  def observe_policy(*,data):return observer.observe_native_optimizer_minibatch(self.actor,data,output_path=OUT/f'rank{self.rank}-raw-observer.json',rank=self.rank)
  def no_scheduler(*args,**kwargs):scheduler_calls.append(True)
  with _temporary_attribute(observer,'ACTOR_SHA256',ACTOR),_temporary_attribute(observer,'LABELS',('dt_pg','grpo_pg')),_temporary_attribute(self.actor,'update_policy',observe_policy),_temporary_attribute(self.actor_lr_scheduler,'step',no_scheduler):
   output=super().update_actor(data)
  comparisons={}
  for name,p in self.actor_module_fsdp.named_parameters():
   if name in before:
    t=p.detach();t=t.to_local() if hasattr(t,'to_local') else t;comparisons[name]=torch.equal(before[name],t.cpu())
  assert all(comparisons.values())
  result=json.loads((OUT/f'rank{self.rank}-raw-observer.json').read_bytes())
  result.update(scope=__doc__,actual_label_meaning={'dt_pg':'Full original saved DT actor coefficient vector','grpo_pg':'Isolated existing Format-token DT coefficient; legacy observer alias only, no GRPO computation'},parameters_exact_unchanged=comparisons,scheduler_step_executed=False,scheduler_noop_calls=len(scheduler_calls),first_held_source_sha256=sha(SOURCE),gradient_adapter=identity(self.observe_actual))
  (OUT/f'rank{self.rank}-gradients.json').write_text(json.dumps(result,indent=2)+'\n');return output

def main():
 torch.set_num_threads(4)
 source=json.loads(SOURCE.read_bytes());assert sha(SOURCE)=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52';check_imports(source)
 assert sha(observer.__file__)=='022466b2bac94617b8e627ea027fb3afdf4490dc4bc2bcaa11944ab444e36214'
 from verl.workers.actor.dp_actor import DataParallelPPOActor
 from verl.trainer.ppo.core_algos import compute_policy_loss
 assert identity(DataParallelPPOActor)['sha256']==ACTOR and identity(compute_policy_loss)['sha256']==observer.CORE_SHA256
 paths=[CAPTURE/f'rank{i}-pre-update.pt' for i in (0,1)];expected=['1563ce74f298769893b360398fd1bacf16c376439b1d462e56ef9dc6f905be59','a1970da0bbf462b203314cbf29cc8ecd8c81e526fe1b6ee944978a76bbadd34e']
 assert [sha(p) for p in paths]==expected
 saved=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
 tensors={k:torch.cat([s['tensors'][k][:32] for s in saved]) for k in saved[0]['tensors']}
 tensors['position_ids']=compute_position_id_with_mask(tensors['attention_mask'])
 # Verify the position owner against a real full-history native capture, not a fixture.
 native=torch.load(CAPTURE/'rank1-readout-native-batch-21.pt',map_location='cpu',weights_only=False)
 native_checks=[]
 for row in native['rows']:
  x=row['row'];candidate=compute_position_id_with_mask(x['attention_mask'][None])[0]
  keep=x['attention_mask'].bool();native_checks.append(torch.equal(candidate[keep],x['position_ids'][keep]))
 assert all(native_checks)
 uid=[str(u) for s in saved for u in s['non_tensors']['traj_uid'][:32]]
 isolated=torch.zeros_like(tensors['advantages']);hits=[]
 for globalrow,u in enumerate(uid):
  if u!=UID:continue
  rank,row=divmod(globalrow,32);a=saved[rank]['non_tensors']['dt_direct_target_artifact'][row]
  positions=torch.tensor(a['retained_response_positions']);slot=int((positions==351).nonzero().item())
  assert int(tensors['responses'][globalrow,slot])==14606 and tensors['response_mask'][globalrow,slot]
  isolated[globalrow,slot]=tensors['advantages'][globalrow,slot];hits.append(dict(global_row=globalrow,rank=rank,local_row=row,slot=slot,token_id=14606,coefficient=float(isolated[globalrow,slot])))
 assert len(hits)==1
 tensors['diagnostic_grpo_advantages']=isolated
 cfg=OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
 for k,v in source['startup_options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
 cfg.actor_rollout_ref.actor.optim.total_training_steps=actor_initialization_steps(cfg.trainer.total_training_steps,330)
 data=DataProto.from_dict(tensors=tensors,non_tensors={'traj_uid':uid},meta_info={'temperature':cfg.actor_rollout_ref.rollout.temperature,'multi_turn':True,'global_token_num':tensors['attention_mask'].sum(-1).tolist()})
 OUT.mkdir(exist_ok=True);(OUT/'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
 inspection=dict(scope=__doc__,source_sha256=sha(SOURCE),inputs=[dict(path=str(p),sha256=sha(p)) for p in paths],selected_rows='First32 original rows of each held rank; actual first global64 optimizer minibatch',position_owner=identity(compute_position_id_with_mask),native_position_checks=native_checks,actor=identity(DataParallelPPOActor),core=identity(compute_policy_loss),observer=identity(observer.observe_native_optimizer_minibatch),isolated_positions=hits,rows=len(data),valid_tokens=int(tensors['response_mask'].sum()),cuda_initialized=torch.cuda.is_initialized(),gradient_input_fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in tensors.items()},checkpoint_restore=0,DT=0,rollout=0,optimizer_steps=0)
 inspection['import_scope']='Original owner imports may initialize CUDA; no model/forward/backward/update is created during --inspect-only.'
 (OUT/'input-inspection.json').write_text(json.dumps(inspection,indent=2)+'\n')
 if '--inspect-only' in sys.argv:print(json.dumps(inspection));return
 ray.init(num_cpus=8,include_dashboard=False)
 try:
  phase('native_actor_init_begin');group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),RayClassWithInitArgs(GradientWorker,cfg.actor_rollout_ref,'actor'));group.init_model();phase('native_actor_init_complete')
  phase('original_old_log_prob_begin');data.union(group.compute_log_prob(data.select(deepcopy=True)));phase('original_old_log_prob_complete')
  phase('original_ref_log_prob_begin');data.union(group.compute_ref_log_prob(data.select(deepcopy=True)));phase('original_ref_log_prob_complete')
  phase('original_complete_minibatch_gradients_begin');output=group.observe_actual(data);phase('original_complete_minibatch_gradients_complete')
  (OUT/'completed.json').write_text(json.dumps(dict(completed_unix=time.time(),source_sha256=sha(SOURCE),scope=__doc__,rows=64,native_backward_passes_per_rank=2,optimizer_steps=0,scheduler_steps=0,checkpoint_restore=0,DT=0,rollout=0),indent=2)+'\n')
 finally:ray.shutdown()

if __name__=='__main__':main()
