"""Original20-step RISE/MAS on one frozen NI0 input, with actual native scoring.

FT0-3 saved official scores, frozen DT, paired symmetric control and MLP candidate.
The only evaluator adaptation is the explicitly declared fixed-input formatter.
Original metric and original scorer functions execute unchanged. No FT rerun.
"""
import os,sys,json,time,hashlib,traceback,signal,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'])
sys.dont_write_bytecode=True
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'running','protocol':p,'model_loads':0,'model_forwards_started':0,'model_forwards_completed':0,
 'FT_runs':0,'attribution_calls':0,'generation_calls':0,'curves':{},'current_method':None,'calls':[]}
started=time.perf_counter();handles=[];trace=None
def save():
 f=A/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(A/'results.json')
def sources():
 out={}
 for name,want in p['files_sha256'].items():
  raw=(A/name).read_bytes();assert sha(raw)==want;out[name]=want
 for name,want in p['official_source_blob_sha1'].items():
  raw=(Path(p['official_root'])/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name;out[name]=sha(raw)
 for name,want in p['runtime_source_sha256'].items():
  raw=(Path(p['isolated_site'])/name).read_bytes();assert sha(raw)==want,name;out[name]=want
 return out
def timeout(signum,frame):raise TimeoutError('Frozen NI0 original-metric budget expired.')
try:
 signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
 import numpy as np
 import torch,transformers,flash_attn
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 from transformers.models.qwen3_5 import modeling_qwen3_5 as native
 from flashtrace.improved import faithfulness_test_skip_tokens
 from llm_attr_eval import LLMAttributionEvaluator
 from fixed_input_metric_view import FixedInputMetricView
 assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
 for path,want in p['data_sha256'].items():assert sha(Path(path).read_bytes())==want,path
 cp=Path(p['checkpoint'])
 for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
 stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
 r['weight_stats_before']=stats();assert r['weight_stats_before']==p['expected_weight_stats']
 record=json.loads(Path(p['cache_path']).read_text().splitlines()[0]);contract=json.loads(Path(p['input_contract_path']).read_bytes())
 tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
 prompt_ids=tokenizer(record['prompt'],add_special_tokens=False,return_tensors='pt').input_ids
 target_ids=tokenizer(record['target']+tokenizer.eos_token,add_special_tokens=False,return_tensors='pt').input_ids
 clean=torch.cat((prompt_ids,target_ids),1);assert clean.shape==(1,588)
 assert sha(clean.numpy().tobytes())==p['input_sha256'];keep=contract['current_keep'];keep_set=set(keep)
 assert len(keep)==310 and prompt_ids.shape==(1,340) and target_ids.shape==(1,248)
 ft=np.load(p['FT_vectors_path'])['scores'];base=np.load(p['DT_frozen_vectors_path'])['signed']
 paired=np.load(p['DT_paired_vectors_path'])
 scores={**{f'FT{i}':ft[i] for i in range(4)},'DT_frozen':base[:340],'DT_sym_control':paired['symmetric'][:340],'DT_MLP_content1':paired['content1'][:340]}
 assert list(scores)==p['method_order'] and all(v.shape==(340,) and np.isfinite(v).all() for v in scores.values())
 r['scores_sha256']={name:sha(np.asarray(v,dtype=np.float32).tobytes()) for name,v in scores.items()}
 r['gpu_free_total_before_load']=list(torch.cuda.mem_get_info());assert r['gpu_free_total_before_load'][0]>=32*1024**3
 torch.manual_seed(73);r['status']='loading_actual_model';save();tick=time.perf_counter()
 model,info=Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,attn_implementation='flash_attention_2',
  device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True)
 model.eval().requires_grad_(False);torch.cuda.synchronize();r['calls'].append({'kind':'model_load','seconds':time.perf_counter()-tick});r['model_loads']=1;assert not any(info.values())
 evaluator=LLMAttributionEvaluator(model,tokenizer);view=FixedInputMetricView(evaluator,record['prompt'])
 assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
 stock=tokenizer(evaluator.format_prompt(record['prompt']),add_special_tokens=False,return_tensors='pt').input_ids
 r['input_adapter']={'actual_shape':[1,588],'actual_input_sha256':p['input_sha256'],'stock_evaluator_prompt_length':stock.shape[1],
  'stock_evaluator_total_length':stock.shape[1]+248,'stock_evaluator_input_sha256':sha(torch.cat((stock,target_ids),1).numpy().tobytes()),
  'declaration':'Only the evaluator formatting view pins the same raw588 input used by FT and DT attribution; original metric and model-scoring methods are unchanged. Not stock run_exp input reproduction.'}
 r['versions']={'torch':torch.__version__,'transformers':transformers.__version__,'FA':flash_attn.__version__}
 def prehook(module,args,kwargs):
  x=kwargs['input_ids'].detach().cpu();mask=kwargs['attention_mask'];assert x.shape==clean.shape and bool(mask.eq(1).all())
  changed=(x[0]!=clean[0]).nonzero().flatten().tolist();assert set(changed)<=keep_set
  assert all(int(x[0,j])==tokenizer.eos_token_id for j in changed)
  if not r['curves'][r['current_method']]['input_receipts']:assert not changed
  r['model_forwards_started']+=1
  r['curves'][r['current_method']]['input_receipts'].append({'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed})
 def posthook(module,args,output):r['model_forwards_completed']+=1
 handles=[model.register_forward_pre_hook(prehook,with_kwargs=True),model.register_forward_hook(posthook)]
 def observe(frame,event,arg):
  if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
   v=frame.f_locals;row=r['curves'][r['current_method']]
   for key in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']:
    row[key]=np.asarray(v[key]).copy().tolist()
   row['sorted_keep']=[int(x) for x in v['sorted_keep']];row['attr_sum']=float(v['attr_sum']);row['return_metrics']=[float(x) for x in arg]
 for name in p['method_order']:
  r['current_method']=name;r['curves'][name]={'input_receipts':[]};r['status']='scoring_'+name;save()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
  sys.setprofile(observe)
  try:
   with torch.no_grad():result=faithfulness_test_skip_tokens(view,torch.as_tensor(scores[name],dtype=torch.float32)[None],record['prompt'],record['target'],
    keep_prompt_token_indices=keep,user_prompt_indices=list(range(340)),k=20)
  finally:sys.setprofile(None)
  torch.cuda.synchronize();row=r['curves'][name];row['seconds']=time.perf_counter()-tick
  row['peak_allocated']=torch.cuda.max_memory_allocated();row['peak_reserved']=torch.cuda.max_memory_reserved()
  assert row['return_metrics']==[float(x) for x in result] and len(row['input_receipts'])==21
  assert row['input_receipts'][-1]['deleted_positions']==keep
  assert all(np.isfinite(row[key]).all() for key in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores','return_metrics'])
  row['density_increases']=int(np.sum(np.diff(row['density'])>0));save()
 for h in handles:h.remove()
 handles=[];r['sources_after']=sources();r['weight_stats_after']=stats();assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
 assert r['model_forwards_started']==r['model_forwards_completed']==147
 r['current_method']=None;r['status']='original_RISE_MAS_on_fixed_NI0_input_completed'
except Exception:
 sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
 signal.alarm(0)
 for h in handles:h.remove()
 r['job_seconds']=time.perf_counter()-started;save()
 with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in list(p['files_sha256'])+['protocol.json','results.json']:z.write(A/name,name)
 print(json.dumps({'status':r['status'],'metrics':{k:v.get('return_metrics') for k,v in r['curves'].items()},'model_forwards':r['model_forwards_completed'],'seconds':r['job_seconds'],'error':r.get('error')}),flush=True)
