"""Patch the pinned finite owner in an isolated, default-inert research copy.

Only new conditional finite-row semantics are implemented. Existing native
FA traits, copies, MMA, coefficient storage and default exports are reused.
This does not patch a production path or implement a model attention kernel.
"""
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
COLLECTION=HERE.parent
AUDIT=COLLECTION.parents[1]


def replace_once(text,old,new):
    assert text.count(old)==1,(old[:100],text.count(old))
    return text.replace(old,new,1)


def main():
    owner=read_path=AUDIT/'appworld-efficiency-20261007/individual-prefix-owner-candidate-v1/candidate/vendor_fa_finite_p1_bf16_d256.cu'
    raw=owner.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='9ebcef18cec94f45a875694acac0fed4b46064f13d3d0af1a4581c931df54c8b'
    original=raw.decode()
    source=original
    source=replace_once(source,'    int query_length;\n};','''    int query_length;
    // Only the explicit research export initializes/uses these fields.
    const float *conditional_own_q0k0,*conditional_own_q1k0,*conditional_own_uv0;
    float *conditional_factual_lse,*conditional_excluded_lse,*conditional_top_key;
    float *conditional_factual_u,*conditional_excluded_u,*conditional_lse1;
    float *conditional_own_query_weight;
};
#include "conditional_attention_rows.cuh"''')
    source=replace_once(source,'template<int Phase,typename Traits=FiniteTraits>',
                       'template<int Phase,typename Traits=FiniteTraits,bool Conditional=false>')
    source=replace_once(source,'    if(row0>=valid || row0+M<=start) {\n        if constexpr(Phase==0) {\n',
'''    if(row0>=valid || row0+M<=start) {
        if constexpr(Phase==-1) {
            for(int j=tid;j<M;j+=blockDim.x) if(row0+j<query_end) {
                const int64_t pos=int64_t(bh)*query_length+row0+j-query_start;
                p.conditional_factual_lse[pos]=-INFINITY;
                p.conditional_excluded_lse[pos]=-INFINITY;
                p.conditional_top_key[pos]=-1.f;
                p.conditional_factual_u[pos]=p.conditional_excluded_u[pos]=0.f;
                p.conditional_lse1[pos]=-INFINITY;
            }
        } else if constexpr(Phase==0) {
''')
    source=replace_once(source,'    clear(denominator);clear(numerator);',
'''    clear(denominator);clear(numerator);
    ConditionalTopRow conditional_rows[decltype(size(rowcoords))::value];''')
    # Phase -1 uses the original non-retained two-operand loader, like phase0.
    source=source.replace('if constexpr(Phase!=0)', 'if constexpr(Phase>0)')
    source=source.replace('if constexpr(Phase==0) flash::', 'if constexpr(Phase<=0) flash::')
    source=replace_once(source,'''        if constexpr(Phase==2) {
            pair(k0,q0,a0,0);pair(k1,q1,a1,1);pair(v0,u,at,2);
        } else {
            pair(q0,k0,a0,0);pair(q1,k1,a1,1);pair(u,v0,at,2);
        }
''','''        if constexpr(Phase==-1) {
            // In conditional mode the explicit export's value operand is V_F.
            pair(q1,k1,a1,1);pair(u,v0,at,2);
            auto scores=make_tensor(a1.data(),flash::convert_layout_acc_rowcol(a1.layout()));
            auto uv=make_tensor(at.data(),flash::convert_layout_acc_rowcol(at.layout()));
            auto positions=make_tensor(coords.data(),flash::convert_layout_acc_rowcol(coords.layout()));
            #pragma unroll
            for(int r=0;r<size<0>(scores);++r) {
                #pragma unroll
                for(int c=0;c<size<1>(scores);++c) {
                    const int query=row0+get<0>(positions(r,c));
                    const int key=col0+get<1>(positions(r,c));
                    if(query<valid && key<valid && key<=query)
                        conditional_rows[r].add(scores(r,c)*p.scale,key,uv(r,c));
                }
            }
            continue;
        } else if constexpr(Phase==2) {
            if constexpr(Conditional) pair(k0,q1,a0,0);
            else pair(k0,q0,a0,0);
            pair(k1,q1,a1,1);pair(v0,u,at,2);
        } else {
            if constexpr(Conditional) pair(q0,k1,a0,0);
            else pair(q0,k0,a0,0);
            pair(q1,k1,a1,1);pair(u,v0,at,2);
        }
''')
    source=replace_once(source,'                const float lp0=a0(i)*p.scale-p.lse0[pos];',
'''                if constexpr(Conditional && Phase==2) {
                    const float factual_score=a1(i)*p.scale;
                    const float reference_score=a0(i)*p.scale;
                    float exclusive_lse,exclusive_u;
                    conditional_exclusive_row(p,pos,key,factual_score,at(i),exclusive_lse,exclusive_u);
                    if(exclusive_lse==-INFINITY) {
                        // The sole causal key has no key-normalization effect.
                        a0(i)=0.f;a1(i)=1.f;
                    } else {
                        a0(i)=conditional_sigmoid_secant(factual_score-exclusive_lse,
                            reference_score-exclusive_lse)*(at(i)-exclusive_u);
                        a1(i)=conditional_sigmoid(reference_score-exclusive_lse);
                    }
                    continue;
                }
                if constexpr(Conditional && Phase!=2) {
                    if(key==query) {
                        a0(i)=p.conditional_own_q0k0[pos]/p.scale;
                        a1(i)=p.conditional_own_q1k0[pos]/p.scale;
                        at(i)=p.conditional_own_uv0[pos];
                    }
                }
                const float lp0=a0(i)*p.scale-p.lse0[pos];''')
    source=replace_once(source,'                const float lp1=a1(i)*p.scale-p.lse1[pos];',
'''                const float lp1=a1(i)*p.scale-(Conditional?p.conditional_lse1[pos]:p.lse1[pos]);''')
    source=replace_once(source,'                if constexpr(Phase>0) a0(i)=lm*(at(i)-p.center[pos]);',
'''                if constexpr(Phase>0) {
                    a0(i)=lm*(at(i)-p.center[pos]);
                    if constexpr(Conditional && Phase==1) {
                        if(key==query) p.conditional_own_query_weight[pos]=float(E(a0(i)));
                    }
                }''')
    # An invalid KV cell must not retain stale score/value accumulator entries.
    source=replace_once(source,'            } else if constexpr(Phase>0) a0(i)=0.f;',
'''            } else if constexpr(Phase>0) a0(i)=0.f;''')
    source=replace_once(source,'            if constexpr(Phase==1) multiply(a0,nullptr,accQ,true);\n            else {multiply(a0,nullptr,accQ,true);multiply(a1,u,accV,false);}',
'''            if constexpr(Conditional) {
                if constexpr(Phase==1) multiply(a0,k1,accQ,false);
                else {multiply(a0,q1,accQ,false);multiply(a1,u,accV,false);}
            } else {
                if constexpr(Phase==1) multiply(a0,nullptr,accQ,true);
                else {multiply(a0,nullptr,accQ,true);multiply(a1,u,accV,false);}
            }''')
    source=replace_once(source,'    if constexpr(Phase==0) {\n        flash::quadreduce_sum',
'''    if constexpr(Phase==-1) {
        #pragma unroll
        for(int i=0;i<size(denominator);++i) {
            float logz,excluded,arg,mean,excluded_mean;
            conditional_rows[i].finish(logz,excluded,arg,mean,excluded_mean);
            const int r=row0+get<0>(rowcoords(i));
            if(get<1>(rowcoords(0))==0 && r<query_end) {
                const int64_t pos=int64_t(bh)*query_length+r-query_start;
                if(r<valid) {
                    p.conditional_factual_lse[pos]=logz;
                    p.conditional_excluded_lse[pos]=excluded;
                    p.conditional_top_key[pos]=arg;
                    p.conditional_factual_u[pos]=mean;
                    p.conditional_excluded_u[pos]=excluded_mean;
                    float own_excluded,unused;
                    conditional_exclusive_row(p,pos,r,
                        0.f,0.f,own_excluded,unused);
                    // Compute the own-key exclusive partition from the score
                    // accumulator below, not an uninitialized/stale diagonal.
                    p.conditional_lse1[pos]=own_excluded;
                } else {
                    p.conditional_factual_lse[pos]=p.conditional_excluded_lse[pos]=-INFINITY;
                    p.conditional_top_key[pos]=-1.f;
                    p.conditional_factual_u[pos]=p.conditional_excluded_u[pos]=0.f;
                    p.conditional_lse1[pos]=-INFINITY;
                }
            }
        }
    } else if constexpr(Phase==0) {
        flash::quadreduce_sum''')
    # The diagonal factual score must be retained per row for the exclusive
    # own-key partition. Reuse the conditional query-weight buffer before its
    # later phase1 consumer writes the query multiplier into the same slot.
    source=replace_once(source,'''                    if(query<valid && key<valid && key<=query)
                        conditional_rows[r].add(scores(r,c)*p.scale,key,uv(r,c));''',
'''                    if(query<valid && key<valid && key<=query) {
                        conditional_rows[r].add(scores(r,c)*p.scale,key,uv(r,c));
                        if(key==query) p.conditional_own_query_weight[
                            int64_t(bh)*query_length+query-query_start]=scores(r,c)*p.scale;
                    }''')
    # All threads finish the same tile stream before the writer reads its
    # diagonal. Shared row writes use an existing explicit global row slot.
    source=replace_once(source,'''                    conditional_exclusive_row(p,pos,r,
                        0.f,0.f,own_excluded,unused);
                    // Compute the own-key exclusive partition from the score
                    // accumulator below, not an uninitialized/stale diagonal.
                    p.conditional_lse1[pos]=own_excluded;''',
'''                    if(float(r)==arg) own_excluded=excluded;
                    else own_excluded=logz+log1pf(-expf(p.conditional_own_query_weight[pos]-logz));
                    p.conditional_lse1[pos]=conditional_logaddexp(own_excluded,p.conditional_own_q1k0[pos]);''')
    source=replace_once(source,'    if constexpr(Phase==-1) {\n        #pragma unroll',
                       '    if constexpr(Phase==-1) {\n        __syncthreads();\n        #pragma unroll')
    source=replace_once(source,'                out[pos]=E(r<valid && r>=start?accQ(i)*p.scale:0.f);',
'''                float result=accQ(i);
                if constexpr(Conditional && Phase==1) {
                    const int64_t own=kv_offset+int64_t(r)*D+d;
                    if(r<valid && r>=start) result+=p.conditional_own_query_weight[
                        int64_t(bh)*query_length+r-query_start]*(float(k0[own])-float(k1[own]));
                }
                out[pos]=E(r<valid && r>=start?result*p.scale:0.f);''')
    source+='''
// Research-only conditional-row ABI. Original exports above remain unchanged.
// v_factual is the factual V endpoint, unlike the original P1 export's V0.
extern "C" int deltatrace_fa_conditional_bf16_d256_row_cached_suffix(
    const void *q0,const void *k0,const void *q1,const void *k1,const void *v_factual,
    const void *u,const void *lse0,const void *unused_lse1,
    void *tau,void *center,void *dq,void *dk,void *dv,const void *valid_lengths,
    const void *coefficient_starts,const void *query_starts,
    const void *own_q0k0,const void *own_q1k0,const void *own_uv0,
    void *factual_lse,void *excluded_lse,void *top_key,void *factual_u,void *excluded_u,
    void *conditional_lse1,void *own_query_weight,
    int batch,int heads,int kv_heads,int length,int query_length,int query_start,
    float scale,void *stream_ptr) {
    FiniteParams p{q0,k0,q1,k1,v_factual,u,(const float*)lse0,(const float*)unused_lse1,
        (float*)tau,(float*)center,dq,dk,dv,(const int*)valid_lengths,
        (const int*)coefficient_starts,batch,heads,kv_heads,length,query_start,scale,
        (const int*)query_starts,query_length,
        (const float*)own_q0k0,(const float*)own_q1k0,(const float*)own_uv0,
        (float*)factual_lse,(float*)excluded_lse,(float*)top_key,
        (float*)factual_u,(float*)excluded_u,(float*)conditional_lse1,(float*)own_query_weight};
    if(batch<1||heads<1||kv_heads<1||heads%kv_heads||length<1||query_length<1)return -1;
    using E=mctlass::bfloat16_t;
    dim3 grid((query_length+FiniteTraits::kBlockM-1)/FiniteTraits::kBlockM,batch*heads);
    constexpr int shared=(size(typename FiniteTraits::SmemLayoutQ{})+size(typename FiniteTraits::SmemLayoutKV{}))*sizeof(E);
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    deltatrace_fa_finite_p1_kernel<-1,FiniteTraits,true><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    deltatrace_fa_finite_p1_kernel<0,FiniteTraits,true><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    deltatrace_fa_finite_p1_kernel<1,FiniteTraits,true><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    deltatrace_fa_finite_p1_kernel<2,FiniteTraits,true><<<grid,FiniteTraits::kNThreads,shared,stream>>>(p);
    return int(cudaGetLastError());
}
'''
    owners=json.loads((COLLECTION/'attention-owner-sources.json').read_bytes())
    wrapper=next(f for f in owners['tasks']['textcraft']['files'] if f['module']=='vendor_fa_finite_bf16_d256')
    assert hashlib.sha256(wrapper['text'].encode()).hexdigest()==wrapper['sha256']
    python=wrapper['text']
    python=replace_once(python,'def __call__(self, operands, scale, layout, activity=None):',
                        'def __call__(self, operands, scale, layout, activity=None, *, conditional=False):')
    python=replace_once(python,"            value = operands[name]\n            expected =",
"            value = operands['v1' if conditional and name=='v0' else name]\n            expected =")
    python=replace_once(python,"            value = operands[name]\n            assert value.shape == (batch, heads, query_length)",
"            value = torch.empty((batch,heads,query_length),device=ref.device,dtype=torch.float32) if conditional and name=='lse1' else operands[name]\n            assert value.shape == (batch, heads, query_length)")
    python=replace_once(python,'        if activity is not None:\n            activity.update',
'''        conditional_rows={}
        if conditional:
            # Explicit research ABI only; no model or production default uses it.
            operation=self.library.deltatrace_fa_conditional_bf16_d256_row_cached_suffix
            operation.argtypes=[ctypes.c_void_p]*26+[ctypes.c_int]*6+[ctypes.c_float,ctypes.c_void_p]
            operation.restype=ctypes.c_int
            buffers=values+[tau,center,dq,dk,dv,layout._tensor,layout._starts,layout._query_starts]
            for name in ('own_q0k0','own_q1k0','own_uv0'):
                value=operands[name]
                assert value.shape==(batch,heads,query_length) and value.device==ref.device and value.dtype==torch.float32
                buffers.append(value.detach().contiguous())
            for name in ('factual_lse','excluded_lse','top_key','factual_u','excluded_u','conditional_lse1','own_query_weight'):
                value=torch.empty_like(tau)
                buffers.append(value);conditional_rows[name]=value
            dimensions=[batch,heads,kv_heads,length,query_length,layout.query_start]
        if activity is not None:
            activity.update''')
    python=replace_once(python,'                                  \'bytes\': v.numel() * v.element_size()} for v in buffers])',
                       '                                  \'bytes\': v.numel() * v.element_size()} for v in buffers if v is not None])')
    python=replace_once(python,"            activity['calls_attempted'] =",
'''            if conditional:
                activity.update(conditional_research_only=True,kernel_launches_per_call=4,
                    value_operand='factual V1',endpoint_mean_buffers=0,
                    finite_rule='complete local conditional Q/K/V including own-key interaction')
            activity['calls_attempted'] =''')
    python=replace_once(python,'ctypes.c_void_p(v.data_ptr()) for v in buffers',
                       'ctypes.c_void_p(v.data_ptr() if v is not None else 0) for v in buffers')
    python=replace_once(python,"return {'dq': dq, 'dk': dk, 'dv': dv, 'tau': tau, 'center': center}",
"return {'dq': dq, 'dk': dk, 'dv': dv, 'tau': tau, 'center': center, **conditional_rows}")
    out=HERE/'prepared';out.mkdir(exist_ok=True)
    generated={'vendor_fa_finite_p1_bf16_d256.cu':source,'vendor_fa_finite_bf16_d256.py':python}
    for name,text in generated.items():
        (out/name).write_bytes(text.encode())
    patch=''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),
            fromfile='a/vendor_fa_finite_p1_bf16_d256.cu',tofile='b/vendor_fa_finite_p1_bf16_d256.cu'))
    patch+=''.join(difflib.unified_diff(wrapper['text'].splitlines(True),python.splitlines(True),
            fromfile='a/vendor_fa_finite_bf16_d256.py',tofile='b/vendor_fa_finite_bf16_d256.py'))
    (HERE/'owner.patch').write_bytes(patch.encode())
    metadata=dict(status='Prepared-only research owner extension; no numerical or collection result',
        original_cuda=dict(path=str(owner),sha256=hashlib.sha256(raw).hexdigest()),
        original_wrapper={k:v for k,v in wrapper.items() if k!='text'},
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
        generated_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in generated.items()},
        header_sha256=hashlib.sha256((HERE/'conditional_attention_rows.cuh').read_bytes()).hexdigest(),
        default_mode='Original exports and Python conditional=False; no new kernel/row buffer on default call',
        limitations=['Primitive only: runner public strict-past FA and gate seam not yet connected',
            'No claim of whole-model improvement, conservation, capacity, official tolerance or runtime speed'],
        production_modified=False,model_calls=0,GPU_calls=0)
    (HERE/'prepared-owner.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(metadata,ensure_ascii=False))


if __name__=='__main__':main()
