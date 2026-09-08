"""One paired DT MLP ablation on authenticated actual NI0 endpoint artifacts.

One original resident model, 32 real decoder replays, 64 finite decoder calls.
Both DT rules share the identical native captures and full-response finite seed.
No new root/FT/deletion/generation run, no scoring or target tuning.
"""
import os,sys,json,time,hashlib,traceback,signal,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
 PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
sys.dont_write_bytecode=True
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'running','protocol':p,'model_loads':0,'root_forwards':0,'decoder_replays':0,
 'finite_decoder_calls':{'symmetric':0,'content1':0},'FA_auxiliary_calls':0,'calls':[],'layers':{},
 'original_needle_calls':0,'FT_runs':0,'generation_calls':0,'deletion_queries':0,'artifacts':{}}
started=time.perf_counter()
def save():
 f=A/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(A/'results.json')
def timed(kind,fn):
 torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
 r['calls'].append({'kind':kind,'seconds':time.perf_counter()-start,'allocated_after':torch.cuda.memory_allocated()});save()
 return value
def source_receipt():
 result={}
 for name,want in p['files_sha256'].items():
  actual=sha((A/name).read_bytes());assert actual==want,name;result[name]=actual
 for name,want in p['runtime_source_sha256'].items():
  actual=sha((Path(p['isolated_site'])/name).read_bytes());assert actual==want,name;result[name]=actual
 for name,want in p['official_package_blob_sha1'].items():
  raw=(Path(p['official_root'])/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
  result[name]=sha(raw)
 return result
def copies(value,device):
 if isinstance(value,torch.Tensor):return value.detach().to(device,copy=True)
 if isinstance(value,tuple):return tuple(copies(v,device) for v in value)
 if isinstance(value,list):return [copies(v,device) for v in value]
 if isinstance(value,dict):return {k:copies(v,device) for k,v in value.items()}
 assert value is None or isinstance(value,(str,int,float,bool));return value
def effect(m,x):return float((m.double()*(x[1::2].double()-x[0::2].double())).sum())
def delta(a,b):
 a=a.float();b=b.float();return {'relative_L2':float((a-b).norm()/a.norm().clamp_min(1e-30)),'max_abs':float((a-b).abs().max())}
def timeout(signum,frame):raise TimeoutError('Frozen paired MLP ablation budget expired.')
try:
 signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds'])
 r['sources_before']=source_receipt()
 import numpy as np
 import torch,transformers,flash_attn
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 from transformers.models.qwen3_5 import modeling_qwen3_5 as native
 from transformers.integrations.flash_attention import flash_attention_forward
 from flash_attn import flash_attn_func,flash_attn_varlen_func
 import flash_attn.flash_attn_interface as fa
 from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens
 from qwen35_answer_finite import PackedAnswerTargets,FiniteAnswerOps
 from qwen35_decoder_finite import NativeDecoderCapture,FiniteBoundaryOps,attention_finite_pullback,decoder_finite_pullback
 from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback
 from native_dense_attention_capture import NativeDenseAttentionCapture
 from mlp_content1_finite import Content1MLPBoundaryOps,swiglu_content1_finite_rule
 from finite_fla_gpu import verify_native_sources,make_compiled_finite_pullback
 from vendor_fa_finite_bf16_d256 import RightPaddedLengths,VendorFAFiniteP1BF16D256
 from fla.ops.gated_delta_rule import chunk_gated_delta_rule
 import causal_conv1d
 assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
 assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
 verify_native_sources(p['native_stage_source_sha256'])
 for name in p['files_sha256']:
  if name[:-3] in sys.modules:assert Path(sys.modules[name[:-3]].__file__).resolve()==(A/name).resolve(),name
 cp=Path(p['checkpoint'])
 for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
 stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
 r['weight_stats_before']=stats();assert r['weight_stats_before']==p['expected_weight_stats']
 old=Path(p['endpoint_directory'])
 for name,want in p['endpoint_artifacts_sha256'].items():assert sha((old/name).read_bytes())==want,name
 old_result=json.loads((old/'results.json').read_bytes());assert old_result['status']=='DT_content_P1_same_official_NI0_input_completed'
 saved=torch.load(old/'actual_native_root.pt',map_location='cpu',weights_only=True)
 z=torch.load(old/'actual_packed_target_logits.pt',map_location='cpu',weights_only=True)
 root=saved['states'];kwargs=saved['layer_kwargs'];pairs=saved['input_ids'];assert pairs.shape==(2,588)
 assert sha(pairs[1].numpy().tobytes())==old_result['input']['clean_sha256']
 assert sha(pairs[0].numpy().tobytes())==old_result['input']['baseline_sha256']
 contract_raw=Path(p['input_contract_path']).read_bytes();assert sha(contract_raw)==p['input_contract_sha256']
 contract=json.loads(contract_raw);keep=contract['current_keep'];assert len(keep)==310
 assert (pairs[1]!=pairs[0]).nonzero().flatten().tolist()==keep
 tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
 assert pairs[0,keep].eq(tokenizer.eos_token_id).all()
 target=pairs[1,340:];selection=PackedAnswerTargets([{'target_ids':target,'prompt_length':340}],[list(range(248))],588,'cuda')
 r['input']=old_result['input'];r['root_effect']=old_result['root_effect']
 r['gpu_free_total_before_load']=list(torch.cuda.mem_get_info());assert r['gpu_free_total_before_load'][0]>=32*1024**3
 torch.manual_seed(73);r['status']='loading_actual_model';save()
 model,info=timed('model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
  attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
 model.eval().requires_grad_(False);r['model_loads']=1;assert not any(info.values()),info
 layers=model.model.language_model.layers;norm=model.model.language_model.norm;assert len(layers)==32
 for layer in layers:
  if layer.block_type=='linear_attention':
   assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
   assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
 r['versions']={'torch':torch.__version__,'transformers':transformers.__version__,'FA':flash_attn.__version__}
 boundaries={'symmetric':FiniteBoundaryOps(True),'content1':Content1MLPBoundaryOps(True)}
 answer=FiniteAnswerOps(True);torch.cuda.reset_peak_memory_stats();paired_start=time.perf_counter()
 with torch.no_grad():
  mnorm,seed=timed('shared_existing_finite_seed',lambda:answer(z.to('cuda'),model.lm_head,selection))
  x=root['final_norm_input'].to('cuda')
  m0=timed('shared_existing_final_norm',lambda:boundaries['symmetric'].norm_residual(x[0::2],x[1::2],norm.weight,mnorm,torch.zeros_like(mnorm),norm.eps))
 r['seed_effect']=effect(m0,x);del mnorm,seed,x,z
 current={'symmetric':m0,'content1':m0.clone()};del m0
 layout=RightPaddedLengths([588],588,'cuda');finite_fa=VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256'])
 finite_fla=make_compiled_finite_pullback(reuse_scalar_products=False)
 for i in reversed(range(32)):
  layer=layers[i];is_fa=layer.block_type=='full_attention';x=root[str(i)].to('cuda');kw=copies(kwargs[str(i)],'cuda')
  dc=NativeDecoderCapture(layer,destination='cuda')
  mc=(NativeDenseAttentionCapture(layer.self_attn,flash_attention_forward,flash_attn_varlen_func,flash_attn_func,destination='cuda') if is_fa else NativeGDNCapture(layer.linear_attn,device='cuda'))
  r['decoder_replays']+=1
  def replay():
   with torch.no_grad(),dc,mc:return layer(x,**kw)
  y=timed('actual_decoder'+str(i)+'_replay',replay)
  assert dc.calls=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
  assert mc.calls==({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if is_fa else {'module':1,'conv':1,'FLA':1,'stage':1})
  d,c,e=dc.values,mc.values,getattr(mc,'endpoints',{});scale=getattr(mc,'scale',0.0625)
  expected=root['final_norm_input' if i==31 else str(i+1)].to('cuda')
  row={'block_type':layer.block_type,'decoder_calls':dc.calls,'mixer_calls':mc.calls,'replay_vs_saved_root':delta(expected,y),'methods':{}}
  r['layers'][str(i)]=row
  for method in current:row['methods'][method]={'root_output_effect':effect(current[method],expected),'replay_output_effect':effect(current[method],y)}
  del expected,y,x
  lse=None
  if is_fa:
   args={k:v for k,v in mc.dense_arguments.items() if k!='return_attn_probs'};assert args['dropout_p']==0 and args['causal']
   r['FA_auxiliary_calls']+=1
   with torch.no_grad():aux,lse,unused=timed('public_FA'+str(i)+'_LSE',lambda:flash_attn_func(c['dense_q'],c['dense_k'],c['dense_v'],return_attn_probs=True,**args))
   assert unused is None or unused.numel()==0
   row['auxiliary_vs_actual_core']=delta(c['attention_output'],aux);del aux,unused
  del dc,mc
  if i==31:
   # Actual captured small coordinates, evaluated against independent autograd
   # at equal endpoints. This is a local rule check, not a benchmark sample.
   g=d['gate_output'][1,:8,:64].detach().float().requires_grad_(True);u=d['up_output'][1,:8,:64].detach().float().requires_grad_(True)
   with torch.enable_grad():
    a=torch.nn.functional.silu(g);reference=torch.autograd.grad((u*a).sum(),(u,g))
   actual=swiglu_content1_finite_rule(g.detach(),g.detach(),u.detach(),u.detach(),a.detach(),a.detach(),torch.ones_like(u))
   r['local_equal_endpoint_autograd']={'up':delta(reference[0],actual[0]),'gate':delta(reference[1],actual[1])}
   assert all(v['relative_L2']<1e-6 for v in r['local_equal_endpoint_autograd'].values())
   # The independent product identity uses genuine saved SiLU endpoints.
   g0,g1=[d['gate_output'][j,:8,:64].double() for j in (0,1)];u0,u1=[d['up_output'][j,:8,:64].double() for j in (0,1)];a0,a1=[d['silu_output'][j,:8,:64].double() for j in (0,1)]
   error=(u1*a1-u0*a0)-(a1*(u1-u0)+u0*(a1-a0))
   r['local_product_identity_max_abs']=float(error.abs().max());assert r['local_product_identity_max_abs']<1e-9
   del g,u,a,reference,actual,g0,g1,u0,u1,a0,a1,error
  for method in ['symmetric','content1']:
   bd=boundaries[method]
   def mixer(upstream):
    if is_fa:
     cos,sin=kw['position_embeddings'];return attention_finite_pullback(layer.self_attn,c,lse,cos,sin,upstream,finite_fa,layout,bd,False)
    return gdn_finite_pullback(layer.linear_attn,c,e,upstream,scale,finite_fla,False)
   r['finite_decoder_calls'][method]+=1
   with torch.no_grad():new,_=timed(method+'_finite_decoder'+str(i),lambda:decoder_finite_pullback(layer,d,current[method],mixer,bd,False))
   assert torch.isfinite(new).all();row['methods'][method]['input_effect']=effect(new,d['input_norm_input']);current[method]=new;del new
  del d,c,e,lse,kw;save()
 embed=root['0'].to('cuda');vectors={};r['methods']={}
 old_signed=np.load(old/'signed_result.npz')['signed']
 for method,m in current.items():
  signed=(m.double()*(embed[1::2].double()-embed[0::2].double())).sum(-1)[0]
  assert torch.isfinite(signed).all();assert all(float(signed[j])==0 for j in set(range(588))-set(keep))
  vectors[method]=signed.cpu().numpy()
  # Persist every completed vector before metric/comparison assertions.
  with (A/'signed_vectors.partial').open('wb') as handle:np.savez_compressed(handle,**vectors)
  (A/'signed_vectors.partial').replace(A/'signed_vectors.npz')
  r['artifacts']['signed_vectors.npz']={'sha256':sha((A/'signed_vectors.npz').read_bytes()),'bytes':(A/'signed_vectors.npz').stat().st_size};save()
  score=signed[:340].float().cpu();r['original_needle_calls']+=1
  rec=evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=keep,gold_prompt_token_indices=contract['current_gold'],top_fraction=0.1)
  selected=[keep[j] for j in torch.topk(score[keep].clamp(min=0),31).indices.tolist()];hits=sorted(set(selected)&set(contract['current_gold']))
  assert rec==len(hits)/40
  r['methods'][method]={'signed_sum':float(signed.sum()),'relative_residual':(r['root_effect']-float(signed.sum()))/r['root_effect'],
   'needle':rec,'selected':selected,'hits':hits,'positive_count':int((score[keep]>0).sum()),'negative_count':int((score[keep]<0).sum())}
 r['symmetric_vs_saved_score_max_abs']=float(np.max(np.abs(vectors['symmetric']-old_signed)))
 r['control_comparison']={'score_relative_L2':float(np.linalg.norm(vectors['symmetric']-old_signed)/max(np.linalg.norm(old_signed),1e-30)),
  'needle_unchanged':r['methods']['symmetric']['needle']==old_result['needle']['recovery'],
  'top31_set_unchanged':set(r['methods']['symmetric']['selected'])==set(old_result['needle']['selected']),
  'eligible_sign_changes':int(np.sum(np.sign(vectors['symmetric'][keep])!=np.sign(old_signed[keep]))),
  'interpretation':'Report native default-precision drift; same-run rules share captures and seed. No retroactive pass of v1 and no universal1e-6 threshold.'}
 torch.cuda.synchronize();r['paired_propagation_seconds_with_diagnostics']=time.perf_counter()-paired_start
 r['peak_allocated']=torch.cuda.max_memory_allocated();r['peak_reserved']=torch.cuda.max_memory_reserved()
 np.savez_compressed(A/'signed_vectors.npz',**vectors)
 r['artifacts']['signed_vectors.npz']={'sha256':sha((A/'signed_vectors.npz').read_bytes()),'bytes':(A/'signed_vectors.npz').stat().st_size}
 r['sources_after']=source_receipt();r['weight_stats_after']=stats();assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
 assert r['decoder_replays']==32 and r['finite_decoder_calls']=={'symmetric':32,'content1':32} and r['FA_auxiliary_calls']==8
 r['status']='paired_DT_MLP_content1_NI0_completed'
except Exception:
 sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
 signal.alarm(0);r['job_seconds']=time.perf_counter()-started;save()
 with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in list(p['files_sha256'])+['protocol.json','results.json','signed_vectors.npz']:
   if (A/name).exists():z.write(A/name,name)
 print(json.dumps({'status':r['status'],'methods':r.get('methods'),'error':r.get('error'),'job_seconds':r['job_seconds']}),flush=True)
