"""Compare each changed DT boundary on the same actual operands; no native patch."""
import torch
from qwen3_input_cast_boundaries import attention_layout as new_layout,norm_residual_two as new_two,norm as new_norm
from qwen3_projection_cast_boundaries import swiglu_half as new_swiglu,seed_half as new_seed,native_precast_half_linear as new_mm,qk_norm_half as new_qk
from compiled_secant_boundaries import attention_layout as old_layout,norm_residual_two as old_two
from compiled_finite_rules import rmsnorm_secant_pullback as old_norm
from compiled_swiglu_secant import swiglu_secant_multipliers as old_swiglu
from compiled_logprob_seed import compiled_seed as old_seed
from native_half_linear import native_half_linear as old_mm
records=[]
def compare(name,a,b):
 if isinstance(a,(tuple,list)):
  for i,(x,y) in enumerate(zip(a,b)):compare(name+'.'+str(i),x,y)
  return
 row={'name':name,'shape':list(a.shape),'new_dtype':str(a.dtype),'old_dtype':str(b.dtype),'exact':bool(torch.equal(a,b))}
 if a.dtype!=torch.bool:row['max_abs']=float((a.float()-b.float()).abs().max())
 records.append(row)
def seed_half(*args):
 a=new_seed(*args);b=old_seed(*args);compare('seed',a,(b[0].half(),b[1]));return a
def swiglu_half(*args):
 a=new_swiglu(*args);b=old_swiglu(*args);compare('swiglu',a,tuple(x.half() for x in b));return a
def native_precast_half_linear(m,w):
 a=new_mm(m,w);b=old_mm(m.float(),w);compare('precast_MM',a,b);return a
def native_half_linear_checked(m,w):return old_mm(m,w)
def qk_norm_half(*args):
 a=new_qk(*args);b=old_norm(*args).half();compare('qk_norm',a,b);return a
def norm(x0,x1,w,m,eps):
 a=new_norm(x0,x1,w,m,eps);b=old_norm(x0.float(),x1.float(),w.float(),m,eps);compare('final_norm',a,b);return a
def norm_residual_two(x0,x1,w,a,b,res,eps):
 value=new_two(x0,x1,w,a,b,res,eps);reference=old_two(x0.float(),x1.float(),w.float(),a,b,res,eps);compare('norm_two',value,reference);return value
def attention_layout(q,k,v,*args):
 a=new_layout(q,k,v,*args);b=old_layout(q.float(),k.float(),v.float(),*args);compare('attention_layout',a,b);return a
class PairedFA:
 def __init__(self,new,old):self.new=new;self.old=old
 def __call__(self,values,scale,activity=None):
  a=self.new(values,scale,activity);expanded=dict(values)
  for key in ['k0','k1','v0']:expanded[key]=values[key].repeat_interleave(values['q0'].shape[1]//values[key].shape[1],dim=1)
  b=self.old(expanded,scale)
  for key in a:compare('FA.'+key,a[key],b[key])
  return a
