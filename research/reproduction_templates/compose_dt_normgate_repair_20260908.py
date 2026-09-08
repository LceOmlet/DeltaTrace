from pathlib import Path
import ast
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';T=R/'research/reproduction_templates'
base=(T/'dt_layer0_conditional_20260908.py').read_text()
base=base.replace("['q','k','v','beta','raw_g','o']","['q','k','v','beta','raw_g','o','g','h','v_new']")
base=base.replace('value=features(dc.values,mc.values,mc.endpoints)','value=features(dc.values,mc.values,mc.endpoints);self.last_features=value')
classes='''
class PairedFLA:
    def __init__(self):
        self.regular=make_compiled_finite_pullback(reuse_scalar_products=False)
        self.capture=make_capture_backend();self.calls=[];self.saved={};self.last_scale=None;self.last_e=None
    def __call__(self,e,do,scale,kind='current'):
        capture=(len(self.calls)==23 or kind=='candidate')
        self.last_scale=float(scale);self.last_e=id(e)
        item={'kind':kind,'captured_layer0':capture,'scale':float(scale),'do_shape':list(do.shape),'status':'entered'}
        self.calls.append(item)
        if capture:
            coeff,diag=self.capture(e,do,scale);self.saved[kind]=cpu(diag)
            item['packed_shape']=list(diag['L'].shape);item['packed_CPU_bytes']=tensor_bytes(self.saved[kind])
        else:coeff=self.regular(e,do,scale)
        item['status']='returned';return coeff

class RepairObserver(Layer0Observer):
    def __init__(self,layer,fla,boundaries,info):
        super().__init__();self.layer=layer;self.fla=fla;self.boundaries=boundaries;self.info=info
        self.other=Layer0Observer();self.control_points={}
    def decoder(self,index,d,c,e,upstream,new,terms):
        super().decoder(index,d,c,e,upstream,new,terms)
        assert self.fla.last_e==id(e) and len(self.fla.calls)==24
        scale=self.fla.last_scale
        candidate,candidate_terms,self.metadata=compute_layer0_symmetric_normgate(self.layer,d,c,e,upstream,new,terms,
            lambda ep,mo,sc:self.fla(ep,mo,sc,kind='candidate'),self.boundaries,scale)
        # Reuse the same existing CPU coefficient mapper; no candidate model forward.
        self.other.decoder(index,d,c,e,upstream,candidate,candidate_terms)
        self.other.paired=self.paired
        self.candidate=(candidate.double()*(d['input_norm_input'][1::2].double()-d['input_norm_input'][0::2].double())).sum(-1)[0].cpu()
        self.scale=scale
        row['paired_repair']=self.metadata
        assert len(self.fla.calls)==25
    def consume(self,dc,mc):
        super().consume(dc,mc);value=self.last_features
        self.control_points[str(self.current)]=value['e']
        if self.current==0:self.other.clean=self.clean
        else:self.other.points[str(self.current)]=self.other.decompose(difference(self.clean,value))

'''
base=base.replace('class ScoringLayer0Capture:',classes+'class ScoringLayer0Capture:')
base=base.replace('from flashtrace.improved import keep_token_indices','from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens,faithfulness_test_skip_tokens')
base=base.replace('from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256',
 'from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256\n    from fixed_input_metric_view import FixedInputMetricView\n    from finite_fla_coefficient_capture_20260908 import make_capture_backend\n    from layer0_normgate_paired_repair_20260908 import compute_layer0_symmetric_normgate')
base=base.replace("assert p['case_indices']==[['niah_mq_q2',1],['morehopqa',1]]","assert p['case_indices']==[['niah_mq_q2',1]]")
base=base.replace("    runner=Qwen35DenseFiniteRunner(model,","    paired_fla=PairedFLA()\n    runner=Qwen35DenseFiniteRunner(model,")
base=base.replace('make_compiled_finite_pullback(reuse_scalar_products=False))\n    evaluator','paired_fla)\n    evaluator')
cut=base.index('    for key,rec,ids,base,target,info in cases:')
prefix=base[:cut]
loop='''    quality_raw=Path(p['quality_parent_path']).read_bytes();assert sha(quality_raw)==p['quality_parent_sha256']
    quality=json.loads(quality_raw)
    for key,rec,ids,base,target,info in cases:
        old_info=quality['cases'][key]['input']
        for name,value in info.items():assert old_info[name]==value,(key,name)
        assert sha(base.numpy().tobytes())==old_info['baseline_sha256']
        gold=old_info['gold'];row={'input':info,'gold':gold,'scoring_points':{},'curves':{}};r['cases'][key]=row
        row['prior_fixed_FT_metrics']={k:v for k,v in quality['cases'][key]['curves'].items() if k.startswith('FT')}
        observer=RepairObserver(layer0,paired_fla,runner.boundaries,info)
        pairs=torch.stack((base,ids)).to('cuda');mask=torch.ones_like(pairs)
        selection=PackedAnswerTargets([{'target_ids':target,'prompt_length':info['prompt_length']}],
            [list(range(len(target)))],len(ids),'cuda')
        r['status']='same_actual_forward_paired_normgate_repair';save()
        signed,details=runner.attribute(pairs,mask,selection,select_output_rows=True,observer=observer);r['DT_calls']+=1
        signed=signed[0];candidate=observer.candidate
        for name,value in [('current',signed),('candidate',candidate)]:
            vectors[key+'_'+name+'_full']=value.numpy();vectors[key+'_'+name+'_evaluated']=value[:info['prompt_length']].float().numpy()
        np.savez_compressed(A/'vectors.npz',**vectors)
        row['DT_with_paired_diagnostics']=details;row['finite_FLA_calls']=paired_fla.calls
        reference=torch.from_numpy(parent_vectors[key+'_DT_full'])
        row['historical_vector_relative_L2_report_only']=float((signed-reference).norm()/reference.norm())
        row['historical_guard_policy']='Prior failed 2% experiment retained. This new repair uses shared actual endpoints/upstream, not old-vector admission. Historical drift is reported and not called method effect.'
        row['B2_current']=observer.B2;row['B2_candidate']=observer.other.B2
        assert bool(torch.isfinite(candidate).all()) and bool(torch.isfinite(signed).all())
        assert torch.equal(observer.boundary_coeff['0'],observer.coeff['input'])
        assert torch.equal(observer.boundary_coeff['1'],observer.coeff['upstream'])
        del pairs,mask,selection,reference
        capture=ScoringLayer0Capture(layer0,observer)
        def before_frozen(_module,args,kw):
            r['scoring_forwards']+=1;x=kw['input_ids'].detach().cpu();step=observer.current
            frozen=prior['cases'][key]['curve']['input_receipts'][step]
            assert x.shape==(1,len(ids)) and bool(kw['attention_mask'].eq(1).all())
            assert sha(x.numpy().tobytes())==frozen['input_sha256']
            assert (x[0]!=ids).nonzero().flatten().tolist()==frozen['deleted_positions']
            row['scoring_points'][str(step)]={'input_receipt':frozen}
        handles=[model.register_forward_pre_hook(before_frozen,with_kwargs=True)]
        r['status']='four_frozen_original_scores_and_real_control_states';save()
        try:
            for step in p['capture_steps']:
                observer.current=step;frozen=prior['cases'][key]['curve']['input_receipts'][step]
                prompt=ids[:info['prompt_length']].clone();prompt[frozen['deleted_positions']]=tokenizer.eos_token_id
                with torch.no_grad():lp=timed('original_frozen_scorer_'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(
                    prompt[None].to('cuda'),target[None].to('cuda')))
                point=row['scoring_points'][str(step)];point['original_native_score']=float(lp.sum().cpu())
                point['prior_original_native_score']=prior['cases'][key]['curve']['scores'][step]
                point['capture']=observer.captures[str(step)]
                if step:
                    point['current_layer0_decomposition']=observer.points[str(step)]
                    point['candidate_layer0_decomposition']=observer.other.points[str(step)]
                    actual=row['scoring_points']['0']['original_native_score']-point['original_native_score']
                    deleted=frozen['deleted_positions'];point['complete_input']={'actual_model_effect':actual}
                    for name in ['current','candidate']:
                        score=torch.from_numpy(vectors[key+'_'+name+'_evaluated'])
                        prediction=float(score[deleted].double().sum())
                        point['complete_input'][name]={'prediction':prediction,'prediction_minus_actual':prediction-actual}
                save();del lp
        finally:
            capture.close();capture=None
            for h in handles:h.remove()
            handles=[]
        assert r['scoring_forwards']==4
        artifact=A/(key+'_control_content_inputs.pt')
        torch.save({'paired':observer.paired['e'],'points':observer.control_points,
            'current':paired_fla.saved['current'],'candidate':paired_fla.saved['candidate'],'scale':observer.scale},artifact)
        row['control_content_artifact']={'file':artifact.name,'sha256':sha(artifact.read_bytes()),'bytes':artifact.stat().st_size}
        coeff_path=A/(key+'_paired_coefficients.pt');torch.save({'current':observer.coeff,'candidate':observer.other.coeff},coeff_path)
        row['coefficient_artifact']={'file':coeff_path.name,'sha256':sha(coeff_path.read_bytes()),'bytes':coeff_path.stat().st_size}
        save();del observer;gc.collect()
        view=FixedInputMetricView(evaluator,rec['prompt'])
        assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
        for method in ['current','candidate']:
            score=torch.from_numpy(vectors[key+'_'+method+'_evaluated'])
            curve={'input_receipts':[]};row['curves'][method]=curve
            curve['needle']=float(evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=info['keep'],
                gold_prompt_token_indices=gold,top_fraction=0.1)) if gold else None
            def before_metric(_module,args,kw):
                x=kw['input_ids'].detach().cpu();assert x.shape==(1,len(ids)) and bool(kw['attention_mask'].eq(1).all())
                changed=(x[0]!=ids).nonzero().flatten().tolist();assert set(changed)<=set(info['keep'])
                assert all(int(x[0,j])==tokenizer.eos_token_id for j in changed)
                if not curve['input_receipts']:assert not changed
                curve['input_receipts'].append({'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed})
                r['scoring_forwards']+=1
            def observe_metric(frame,event,arg):
                if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                    loc=frame.f_locals
                    for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']:
                        curve[name]=np.asarray(loc[name]).copy().tolist()
                    curve['sorted_keep']=[int(x) for x in loc['sorted_keep']];curve['attr_sum']=float(loc['attr_sum'])
                    curve['return_metrics']=[float(x) for x in arg]
            handles=[model.register_forward_pre_hook(before_metric,with_kwargs=True)]
            r['status']='original_needle_RISE_MAS_'+method;save();sys.setprofile(observe_metric)
            try:
                with torch.no_grad():returned=timed('original_full_curve_'+method,lambda:faithfulness_test_skip_tokens(view,
                    score[None],rec['prompt'],rec['target'],keep_prompt_token_indices=info['keep'],
                    user_prompt_indices=list(range(info['prompt_length'])),k=20))
            finally:
                sys.setprofile(None)
                for h in handles:h.remove()
                handles=[]
            assert curve['return_metrics']==[float(x) for x in returned]
            assert len(curve['input_receipts'])==21 and curve['input_receipts'][-1]['deleted_positions']==info['keep']
            assert all(np.isfinite(curve[name]).all() for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores'])
            save()
        row['acceptance_scope']='NI1 development comparison only. Original new-order curves measured. No MH confirmation or production speed claim; no automatic promotion or multi-rule sweep.'
        del signed,candidate;gc.collect()
    assert r['DT_calls']==1 and r['scoring_forwards']==46
    assert len(paired_fla.calls)==25 and sum(x['kind']=='candidate' for x in paired_fla.calls)==1
    r['sources_after']=sources();r['weight_stats_after']=stats()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['status']='NI1_paired_normgate_repair_original_metrics_complete'
'''
footer=base[base.index('except Exception:\n    r[\'status\']=\'failed\''):]
s=prefix+loop+footer
s=s.replace('Focused passive layer0 localization on frozen original scoring inputs.','Paired layer0 norm-gate repair on one original NI1 example.')
s=s.replace('Two DT calls and eight original scorer calls; no method, FT or model replacement.\nConv preactivation and SiLU remain combined: no extra preactivation calls.',
 'One actual DT with shared endpoints/upstream and one extra symmetric layer0 GDN propagation.\nFour frozen original scores plus two unchanged original 20-step curves; no FT/model replacement.')
ast.parse(s);dst=T/'dt_normgate_repair_NI1_20260908.py';assert not dst.exists();dst.write_text(s,encoding='utf-8');print(dst)
