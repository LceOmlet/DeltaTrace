"""Remove materialized GQA inputs using the pinned FA query-to-KV head mapping."""
import ast, hashlib, json
from pathlib import Path
A=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
source=(A/'vendor_fa_finite_p1.cu').read_text()
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
source=once(source,'int batch,heads,length;', 'int batch,heads,kv_heads,length;')
source=once(source,'const int64_t head_offset=int64_t(bh)*length*D;', '''const int64_t head_offset=int64_t(bh)*length*D;
    // Same GQA mapping as pinned FA flash_fwd_kernel.h: bidh / h_h_k_ratio.
    const int kv_bh=(bh/p.heads)*p.kv_heads+(bh%p.heads)/(p.heads/p.kv_heads);
    const int64_t kv_offset=int64_t(kv_bh)*length*D;
    const int64_t pair_left_offset=Phase==2?kv_offset:head_offset;
    const int64_t pair_right_offset=Phase==2?head_offset:kv_offset;''')
source=once(source,'left+head_offset+int64_t(row0)*D','left+pair_left_offset+int64_t(row0)*D')
source=once(source,'right+head_offset+int64_t(col0)*D','right+pair_right_offset+int64_t(col0)*D')
source=once(source,'values+head_offset+int64_t(col0)*D','values+(Phase==1?kv_offset:head_offset)+int64_t(col0)*D')
source=once(source,'extern "C" int deltatrace_fa_finite_p1(', 'extern "C" int deltatrace_fa_finite_p1_gqa(')
source=once(source,'int batch,int heads,int length,','int batch,int heads,int kv_heads,int length,')
source=once(source,'if(batch<1||heads<1||length<1)return -1;',
    'if(batch<1||heads<1||kv_heads<1||heads%kv_heads!=0||length<1)return -1;')
source=once(source,'(E*)dq,(E*)dk,(E*)dv,batch,heads,length,scale',
    '(E*)dq,(E*)dk,(E*)dv,batch,heads,kv_heads,length,scale')
source=source.replace(' * No integral quadrature,', ' * Compact GQA inputs; query-head outputs retain the existing FP32 grouped reduction.\n * No integral quadrature,')
(A/'vendor_fa_finite_p1_gqa.cu').write_text(source)
runtime=(A/'vendor_fa_finite_runtime.py').read_text()
runtime=runtime.replace('class VendorFAFiniteP1:', 'class VendorFAFiniteP1CompactGQA:')
runtime=runtime.replace('self.library.deltatrace_fa_finite_p1', 'self.library.deltatrace_fa_finite_p1_gqa')
runtime=runtime.replace('[ctypes.c_int]*3','[ctypes.c_int]*4')
runtime=runtime.replace('# All expanded operands are [B,H,N,128]; H corresponds to query heads.',
    '# Q/U are [B,Hq,N,128]; K/V are compact [B,Hkv,N,128].\n        # Query-head outputs keep the existing FP32 GQA sum outside this operator.')
runtime=once(runtime,"        for name in names:\n", "        kv_heads=operands['k0'].shape[1]\n        assert kv_heads>0 and heads%kv_heads==0\n        for name in names:\n")
runtime=once(runtime,'assert value.shape==reference.shape and value.device==reference.device',
    "expected=(batch,kv_heads,length,dim) if name in ('k0','k1','v0') else reference.shape\n            assert value.shape==expected and value.device==reference.device")
runtime=runtime.replace('],batch,heads,length,scale,', '],batch,heads,kv_heads,length,scale,')
runtime=runtime.replace("            activity['calls_attempted']", "            activity['query_heads']=heads\n            activity['kv_heads']=kv_heads\n            activity['GQA_input_expansion']=False\n            activity['calls_attempted']",1)
ast.parse(runtime)
(A/'vendor_fa_finite_gqa_runtime.py').write_text(runtime)
for old,new in [('qwen_signed_secant_vendor_fa.py','qwen_signed_secant_vendor_fa_gqa.py'),
                ('qwen_signed_secant_batched_vendor_fa.py','qwen_signed_secant_batched_vendor_fa_gqa.py')]:
    s=(A/old).read_text()
    s=once(s,'kr=k.repeat_interleave(groups,dim=1);vr=v.repeat_interleave(groups,dim=1)',
             '# Compact K/V share FA head mapping; no materialized GQA inputs.\n        kr=k;vr=v')
    ast.parse(s);(A/new).write_text(s)
for old,new,oldmod,newmod in [
    ('qwen_signed_secant_paired_vendor_fa.py','qwen_signed_secant_paired_vendor_fa_gqa.py','qwen_signed_secant_vendor_fa','qwen_signed_secant_vendor_fa_gqa'),
    ('qwen_signed_secant_batched_vendor_fa_public.py','qwen_signed_secant_batched_vendor_fa_gqa_public.py','qwen_signed_secant_batched_vendor_fa','qwen_signed_secant_batched_vendor_fa_gqa')]:
    s=(A/old).read_text();s=once(s,'from '+oldmod+' import','from '+newmod+' import')
    ast.parse(s);(A/new).write_text(s)
out={'status':'source_generated_not_executed','native_FA_and_finite_arithmetic_changed':False,
     'change':'Compact GQA input address mapping only; existing query-head outputs and FP32 GQA sum retained.',
     'upstream_mapping':'third_party/metax_fa_2_5_3/csrc/flash_attn/src/flash_fwd_kernel.h:122,125',
     'GQA_group_reduction_fused':False,'three_pass_finite_attention_changed':False,
     'sources':{name:sha(A/name) for name in ['vendor_fa_finite_p1_gqa.cu','vendor_fa_finite_gqa_runtime.py',
        'qwen_signed_secant_vendor_fa_gqa.py','qwen_signed_secant_batched_vendor_fa_gqa.py',
        'qwen_signed_secant_paired_vendor_fa_gqa.py','qwen_signed_secant_batched_vendor_fa_gqa_public.py']}}
(A/'fa_compact_gqa_source_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
