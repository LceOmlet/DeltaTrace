"""Complete conditional attention algebra, existing-owner work and storage audit.

Research derivation on original receipts only; no candidate code, model call,
GPU allocation, new numerical tolerance or training change is performed here.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3].parent


def read(path):
    return json.loads(Path(path).read_bytes())


def reference(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def count_geometry(lengths,starts,cuts,heads,dim):
    assert len(lengths)==len(starts)==len(cuts)
    rows=[]
    for length,start,cut in zip(lengths,starts,cuts):
        assert 0<=start<=cut<length
        # Logical causal pairs, not a native FA tile/latency model.
        query=sum(i+1 for i in range(cut,length))
        kv=(length-cut)*(length-cut+1)//2
        public=sum(i+1 for i in range(start,length))
        strict=sum(i for i in range(cut,length))
        old=7*query+5*kv+4*public
        proposed=9*query+5*kv+2*strict
        # The replacement paired public call must be included. The new rule
        # The selected equivalent local KV ordering uses factual V for the
        # key coefficient and p_CF for the value coefficient: one UV, not two.
        # The conditional native Q call requests only the same coefficient
        # suffix as the original finite owner; all K/V history is retained.
        assert proposed-old==4*query-4*public-2*(length-cut)
        assert proposed<=old
        native_reuse=7*query+5*kv
        query_tiles=kv_tiles=0
        for row0 in range(start,length,64):
            if row0+64<=cut:
                continue
            query_tiles+=(min(length,row0+64)+63)//64
            kv_tiles+=(length-row0+63)//64
        rows.append(dict(length=length,query_start=start,coefficient_start=cut,
            logical_query_pairs=query,logical_KV_pairs=kv,public_pairs=public,
            strict_past_public_pairs=strict,
            original_contraction_pair_units=old,conditional_contraction_pair_units=proposed,
            native_LSE_reuse_baseline_contraction_pair_units=native_reuse,
            original_finite_query_tiles=query_tiles,original_finite_KV_tiles=kv_tiles))
    old=sum(r['original_contraction_pair_units'] for r in rows)
    proposed=sum(r['conditional_contraction_pair_units'] for r in rows)
    reused=sum(r['native_LSE_reuse_baseline_contraction_pair_units'] for r in rows)
    return dict(rows=rows,original_contraction_pair_units=old,
        conditional_contraction_pair_units=proposed,
        ratio=proposed/old,
        native_LSE_reuse_baseline_contraction_pair_units=reused,
        conditional_over_native_LSE_reuse_ratio=proposed/reused,
        original_logical_contraction_FLOPs=old*heads*2*dim,
        conditional_logical_contraction_FLOPs=proposed*heads*2*dim)


def main():
    started=time.perf_counter()
    owners=read(HERE/'conditional-owner-lifetime.json')
    previous=read(HERE/'conditional-attention-algebra.json')
    cfg=read(HERE/'gdn-owner-readonly.json')['model_config']['text_config']
    compile_path=HERE.parents[1]/'appworld-efficiency-20261007/individual-prefix-owner-candidate-v1/compiled-owner.json'
    cuda_path=compile_path.parent/'candidate/vendor_fa_finite_p1_bf16_d256.cu'
    compiled=read(compile_path)
    assert reference(cuda_path)['sha256']==compiled['source_sha256'][cuda_path.name]
    old_owners=read(HERE/'attention-owner-sources.json')
    for task,data in old_owners['tasks'].items():
        assert compiled['library_sha256']==data['finite_library']['sha256']
    native_interface=old_owners['native_FA_public_interface']
    interface_tree=ast.parse(native_interface['text'])
    native_lse_source={}
    for class_name in ('FlashAttnFunc','FlashAttnVarlenFunc'):
        cls=next(n for n in interface_tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
        forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        body=ast.unparse(forward)
        assert 'softmax_lse' in body and 'ctx.save_for_backward' in body
        native_lse_source[class_name]=body

    # x,y are the factual/reference unnormalized probability masses for one
    # key. z is the exact sum for all other keys, t their U-dot-output mean.
    x,y,z=sp.symbols('x y z',positive=True)
    b,c,t=sp.symbols('b c t')
    pf=x/(z+x);pc=y/(z+y)
    factual=(z*t+x*b)/(z+x)
    deleted=(z*t+y*c)/(z+y)
    ordered=pf*(b-c)+(pf-pc)*(c-t)
    remainder=sp.cancel(factual-deleted-ordered)
    assert remainder==0
    # This algebraically equal row-effect ordering needs only U dot V_F in
    # the KV tile, already needed for factual/exclusive normalization. It
    # avoids the extra full UV contraction and retained V0 operand. This
    # chooses a single prototype before any candidate data, not a PV sweep.
    selected=pc*(b-c)+(pf-pc)*(b-t)
    assert sp.cancel(factual-deleted-selected)==0
    # Sigmoid secant: the stable numerator needs neither 1-exp(large) nor
    # subtraction of two nearly equal saturated probabilities.
    u,v=sp.symbols('u v',positive=True)
    stable_numerator=u/(1+u)/(1+v)*(1-v/u)
    assert sp.cancel(stable_numerator-(u/(1+u)-v/(1+v)))==0
    maximum=sp.Rational(1,4)-u/(1+u)**2
    assert sp.cancel(maximum-(u-1)**2/(4*(u+1)**2))==0
    # Same hidden row changes Q and K/V; the ordered intermediate handles
    # their diagonal interaction instead of adding two factual derivatives.
    f,m,n,s1,s0=sp.symbols('F M CF gate1 gate0')
    coupled=s1*((f-m)+(m-n))+n*(s1-s0)
    assert sp.expand(coupled-(s1*f-s0*n))==0

    owner_bindings={}
    for task,data in owners['tasks'].items():
        file=data['files'][0]
        tree=ast.parse(file['text'])
        public=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                    and n.name=='_public_varlen_attention_lse')
        node_calls=[n for n in ast.walk(public) if isinstance(n,ast.Call)
                    and isinstance(n.func,ast.Name) and n.func.id=='flash_attn_varlen_func']
        assert len(node_calls)==1
        assert '_upad_input' in ast.unparse(public)
        assert 'pad_input' in ast.unparse(public)
        owner_bindings[task]=dict(runner={k:v for k,v in file.items() if k!='text'},
            existing_public_LSE_callback=ast.unparse(public),
            resource_settings=data['resource_settings'],
            original_phase='Offload-disabled calls paired public_FA_LSE before finite MLP; offload-enabled already defers that owner call into mixer after MLP consumption.',
            proposed_seam='An explicit research mode would replace that paired LSE replay with one B4 strict-past Q0/factual-KV public call in the original mixer. Q requests only the existing coefficient suffix; all factual K/V history remains. It must not retain the original paired replay and append the new call.',
            release='The conditional rule does not read the captured global attention_output. Release or omit its restoration at the original consumer, preserving any requested observer trace; do not assume pop frees allocator/physical storage without measuring aliases and phase.',
            mask_owner='Use installed HF/FA unpad_input, indices/cu_seqlens and pad_input. Q keeps the original valid suffix starting at the existing coefficient cut, K/V retains complete history except its last valid key. Bottom-right alignment then excludes the own key for each requested query. This omits unused coefficient rows, not state context. No copied mask/parser or token identity reconstruction.')

    geometry={}
    refs=[]
    for task in ('textcraft','appworld'):
        batches=[]
        for rank in (0,1):
            path=HERE/f'layer-{task}-observations/rank{rank}.json'
            refs.append(reference(path))
            observation=read(path)
            for batch in observation['batches']:
                detail=batch['DT_detail']
                cost=count_geometry(detail['native_row_context_lengths'],
                    detail['native_row_prefix_lengths'],detail['fa_coefficient_starts'],
                    cfg['num_attention_heads'],cfg['head_dim'])
                batches.append(dict(rank=rank,batch=batch['index'],uids=batch['uids'],**cost))
        old=sum(b['original_contraction_pair_units'] for b in batches)
        proposed=sum(b['conditional_contraction_pair_units'] for b in batches)
        reused=sum(b['native_LSE_reuse_baseline_contraction_pair_units'] for b in batches)
        geometry[task]=dict(batches=batches,
            original_contraction_pair_units=old,conditional_contraction_pair_units=proposed,
            workload_ratio=proposed/old,
            native_LSE_reuse_baseline_contraction_pair_units=reused,
            conditional_over_native_LSE_reuse_ratio=proposed/reused,
            scope='Sum of arithmetic work for these original B4 calls, not quality weighting, wall-clock speed or a model-throughput projection.')
    envelope=count_geometry([32768]*4,[0]*4,[0]*4,cfg['num_attention_heads'],cfg['head_dim'])
    q=4*32768*cfg['num_attention_heads']*cfg['head_dim']*2
    kv=4*32768*cfg['num_key_value_heads']*cfg['head_dim']*2
    row=4*32768*cfg['num_attention_heads']*4
    # Explicit array accounting only; native views, temporary workspace,
    # padding, actual suffix extents and lifetimes need runtime measurement.
    storage=dict(batch=4,length=32768,query_heads=cfg['num_attention_heads'],
        kv_heads=cfg['num_key_value_heads'],head_dim=cfg['head_dim'],
        BF16_one_query_or_output=q,BF16_one_compact_K_or_V=kv,FP32_one_row_scalar=row,
        original_paired_public_arrays_upper_bytes=2*(q+kv+kv+q+q),
        replacement_B4_public_arrays_upper_bytes=q+kv+kv+q+q,
        six_FP32_row_stats_bytes=6*row,
        own_QK_QK_UV_FP32_diagonal_inputs_bytes=3*row,
        query_own_weight_FP32_bytes=row,
        original_LSE_tau_center_FP32_bytes=4*row,
        proposed_all_finite_FP32_row_buffers_bytes=13*row,
        proposed_net_finite_FP32_row_buffer_increase_bytes=9*row,
        factual_V_copy_replaces_original_reference_V_copy=True,
        original_global_attention_capture_pair_bytes=2*q,
        estimate_scope='Upper array sizes before alias/reuse analysis, not measured net allocation or predicted physical peak. Packed Q/K/V, native output and padded output may be views or overlap; account actual storage once.',
        lifecycle='A single current B4/32k attention output is 1GiB, not a bank per layer. Reuse one layer at a time; exclude the unused global attention capture from restore and consume the strict-past output at the gate. No 32-layer output bank.',
        tile_storage='Keep the original FA D256 tile/shared-memory budget. Store the query own-key weight in one explicit row-scalar buffer instead of silently enlarging the full shared tile, overwriting an input or relabeling the original tau buffer.',
        capacity_reference=reference(ROOT/'experiments/rl/results_memory_capacity_20261008.json'),
        capacity_scope='The linked 62.626953125GiB high-water belongs to the consumed-cache-release candidate, not the currently held runner and not this proposed math. It cannot prove this candidate fits.')
    result=dict(scope=__doc__,source=reference(Path(__file__)),
        inputs=[reference(HERE/'conditional-owner-lifetime.json'),
            reference(HERE/'conditional-owner-lifetime-command.sh'),
            reference(HERE/'conditional-attention-algebra.json'),
            reference(HERE/'attention-owner-sources.json'),reference(compile_path),reference(cuda_path),
            *refs],
        algebra=dict(conditional_KV_remainder=str(remainder),
            previous_equivalent_row_effect='Delta O=p_F*Delta V+(p_F-p_CF)*(V_CF-O_other)',
            selected_exact_row_effect='Delta O=p_CF*Delta V+(p_F-p_CF)*(V_F-O_other)',
            choice='Both identities have the same exact complete local row effect. Choose the latter before candidate execution because the KV owner needs only one U/V contraction and one retained value operand. This is not a statement that their coefficient vectors or end-to-end source estimates are identical; do not run a PV-order parameter sweep.',
            value_coefficient='p_CF*upstream, for this key-only counterfactual with factual query and all other keys/values factual.',
            key_coefficient='[sigmoid(s_F-logZ_other)-sigmoid(s_CF-logZ_other)]/(s_F-s_CF) times upstream dot(V_F-O_other), then scale*Q_F.',
            stable_secant='sigmoid(max(z_F,z_CF))*sigmoid(-min(z_F,z_CF))*(-expm1(-abs(delta_s)))/abs(delta_s); equal endpoints use sigmoid(z)*sigmoid(-z).',
            secant_bound='Between 0 and 1/4 by the sigmoid derivative identity. This is a property of the unmodified formula, not a clipping bound.',
            dominant_key='Track one top-key index and an independently accumulated partition/U-dot-output excluding that key. Non-top keys have p<=1/2, so exclusive mass uses stable log1p and a denominator >=1/2. No subtracting a rounded p=1 to reconstruct tiny mass.',
            streaming='Within each query tile retain the top key and online weighted sum of all other keys. When a larger key appears, move the former top into the excluded accumulator. The source dimension is not materialized.',
            own_interaction='F=O(Q1,K1,V1); M_i=O(Q1,K_i0,V_i0,others factual); CF_i=O(Q_i0,K_i0,V_i0,others factual). Compute (F-M_i)+(M_i-CF_i), and use CF_i in the original content1 gate identity.',
            query='Use both query endpoints on the same conditional KV background; retain the original logarithmic-mean center machinery with factual other keys and own reference key/value. No global midpoint of K or Q in this conditional rule.',
            single_public_call='Q0 strict-past FA supplies conditional endpoint0 partition/output on the original requested coefficient suffix. Factual row statistics supply endpoint1 own-key-excluded partition. Add each own reference-key mass by logaddexp. Thus the previously proposed second strict-past Q1 FA call is unnecessary.',
            composition='The new local finite-row coefficients have the original tensor ABI and can compose with the unchanged input norm/projection, residual, MLP and remaining decoders. This defines a research approximation to source deletion; its whole-model quality is measured, not inferred from local exactness or one sample.',
            conservation='Do not claim these conditional coefficients are the original joint content1 pullback. Their local row effects need not sum to the global paired endpoint difference. Preserve that diagnostic as a distinct quantity; do not add a rescale or label it passed.'),
        owner_bindings=owner_bindings,
        native_LSE_reuse_seam=dict(
            pinned_interface_path=native_interface['path'],sha256=native_interface['sha256'],
            native_forward_source=native_lse_source,
            evidence='The pinned native autograd forward unconditionally obtains softmax_lse and saves it locally even when public return_attn_probs=False. Current passive capture does not retain this owner-emitted field, so the controller reruns public FA for LSE.',
            smallest_action='A separate default-inert passive-capture extension can retain the original native forward LSE and original HF query indices, then use official pad_input. No FA/model callable, parser, mask or numerical formula needs replacement.',
            status='Source-supported reuse opportunity; no runtime tensor capture or equivalence result is claimed here. It must not be called verified/deployed or silently assumed for the current job.',
            comparison='Show both current-owner work and the potential baseline with native LSE reuse. Do not market removal of the baseline duplicate as a free accuracy improvement or compare only against avoidable baseline work.'),
        work=dict(original='7 query-domain contractions + 5 KV-domain contractions + two public FA endpoints (4 public-domain contractions).',
            proposed='9 query-domain contractions + 5 KV-domain contractions + one public strict-past FA endpoint (2 public-domain contractions).',
            original_custom_phases=3,proposed_custom_phases=4,
            public_FA_calls_per_full_attention_layer=dict(original=1,proposed=1),
            logical_contraction_ratio_bound=1,
            bound_scope='Only dimension-D matrix contractions at the same logical lengths/cuts, including the existing LSE replay. This is not a bound on scalar operations, total FLOPs, SFU/reduction cost, traffic, kernel time or complete DT wall time. No runtime speed claim.',
            extra_scalar_work='Stable exclusive normalization, argmax/online row reductions, logistic secants, diagonal Q/K scores and own-output combine must also be counted/profiled.',
            unchanged_calls='One complete DT/B4 per trajectory group, no per-source model forward, no extra full-model pass, no target/reward expansion.',
            frozen_collection=geometry,full32k_logical_envelope=envelope),
        storage=storage,
        numerical_scope='No new kernel is implemented or tested. If implemented, retain the original FA reference/dtype/assertions for its ordinary-gradient limit; separately report the new finite-rule algebra/approximation. Neither test substitutes for the other or for cumulative deletion/RISE/MAS.',
        bounded_next_action='A single default-inert finite-attention owner prototype can now be specified completely from these identities. Compare it on the frozen collection; do not add GDN/head corrections or expand samples to rescue a failed result. No production candidate is accepted by this derivation.',
        operations=dict(model=0,DT=0,GPU=0,update=0,production_modified=False),
        formal_textcraft=owners['formal_textcraft'],
        local_python=sys.executable,elapsed_seconds=time.perf_counter()-started)
    (HERE/'conditional-attention-feasibility.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(algebra=result['algebra']['conditional_KV_remainder'],
        collection_work_ratios={t:v['workload_ratio'] for t,v in geometry.items()},
        full32k_contraction_ratio=envelope['ratio'],storage=storage,
        operations=result['operations'],elapsed_seconds=result['elapsed_seconds']),ensure_ascii=False))


if __name__=='__main__':
    main()
