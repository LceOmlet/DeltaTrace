"""Descriptive biased-sample endpoint analysis; no acceptance rule or new credit."""
import csv
import hashlib
import json
import math
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
population=json.loads((HERE/'population.json').read_bytes())
owner_inspection=json.loads((HERE/'runtime-owners-and-labels.json').read_bytes())
def artifact(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
results={};table=[]
for task in ('textcraft','appworld'):
    directory=HERE/('results-'+task)
    ranks=[json.loads((directory/(task+'-rank'+str(i)+'.json')).read_bytes()) for i in (0,1)]
    assert all(r['phase']=='complete' for r in ranks)
    modes={};points=[];identity=[];endpoint_files=[]
    for mode in ('identity','most_negative','most_positive','median_negative'):
        files=[directory/(task+'-rank'+str(i)+'-'+mode+'-targets.pt') for i in (0,1)]
        tensors=[torch.load(p,map_location='cpu',weights_only=False) for p in files]
        endpoint_files.extend(artifact(p) for p in files)
        rank_equal={k:torch.equal(tensors[0][k],tensors[1][k]) for k in ('samples','predictor_positions','labels','reference_target_logp','factual_target_logp')}
        modes[mode]=dict(rank0_rank1_exact_equal=rank_equal,seconds=[r['modes'][mode]['seconds'] for r in ranks],factual_equal_to_identity=[r['modes'][mode]['factual_target_logp_equal_to_identity'] for r in ranks])
        t=tensors[0]
        for r in ranks[0]['modes'][mode]['rows']:
            if r['point'] is None:
                identity.append(r['single_delete_d']);continue
            point=r['point'];sample=t['samples'].eq(r['row']);future=sample & t['predictor_positions'].ge(point['packed_slot'])
            indices=future.nonzero().flatten();order=torch.argsort(t['factual_minus_reference'][indices].abs(),descending=True)[:5];dominant=[]
            for i in indices[order].tolist():
                factual=float(t['factual_target_logp'][i]);deleted=float(t['reference_target_logp'][i])
                dominant.append(dict(predictor_position=int(t['predictor_positions'][i]),label=int(t['labels'][i]),decoded_label=owner_inspection[task]['labels'][str(int(t['labels'][i]))],factual_logp=factual,single_eos_logp=deleted,factual_probability=math.exp(factual),single_eos_probability=math.exp(deleted),effect=factual-deleted))
            native_d=r['single_delete_d'];dt_d=point['d_FP64'];dt_a=point['expected_A_FP32'];native_a=r['native_endpoint_expected_A_FP32']
            v=dict(task=task,mode=mode,row=r['row'],traj_uid=r['traj_uid'],response_slot=point['response_slot'],packed_slot=point['packed_slot'],token_id=point['token_id'],token=point['decoded_token'],context=point['context'],reward=point['reward'],DT_d=dt_d,native_single_delete_d=native_d,difference_d=dt_d-native_d,DT_expected_A_FP32=dt_a,native_endpoint_expected_A_FP32=native_a,sign_disagreement=(dt_d<0)!=(native_d<0),DT_implied_deleted_to_factual_ratio=math.exp(-dt_d),native_deleted_to_factual_ratio=math.exp(-native_d),earlier_target_count=r['earlier_targets'],earlier_delta_maxabs=r['earlier_delta_maxabs'],future_target_count=r['future_targets'],dominant_future_targets=dominant)
            points.append(v);table.append({k:v[k] for k in ('task','mode','row','traj_uid','response_slot','packed_slot','token_id','token','reward','DT_d','native_single_delete_d','difference_d','DT_expected_A_FP32','native_endpoint_expected_A_FP32','sign_disagreement')})
    prior=population['tasks'][task]['prior_A'];policy=population['tasks'][task]['policy_A'];selected=population['tasks'][task]['selected_batch']
    results[task]=dict(source=population['tasks'][task]['source'],native_batch=selected['file'],native_shape=selected['native_shape'],counterfactual_owner=owner_inspection[task],saved_native_files=population['tasks'][task]['native_files'],saved_trajectory_rows=population['tasks'][task]['saved_trajectory_rows'],unique_uids=population['tasks'][task]['unique_traj_uids'],duplicate_rows=population['tasks'][task]['duplicate_traj_rows'],prior_stats=prior,policy_stats=policy,selected_batch_negative_square_fraction=selected['fraction_of_all_saved_negative_prior_sumsq'],selected_minima_negative_square_fraction=selected['selected_most_negative_sumsq_fraction'],identity_single_delete_d=identity,modes=modes,points=points,biased_sample_sign_disagreements=sum(v['sign_disagreement'] for v in points),biased_sample_count=len(points),factual_joint_logp_drift_from_saved_formal=ranks[0]['modes']['identity']['saved_original_factual_drift'],endpoint_files=endpoint_files,owner_metadata_note='The diagnostic used inspect.getsourcefile on a torch.no_grad-decorated function, which names torch contextlib. That field is not the counterfactual owner binding. population.json independently binds the actual module bytes; the separate runtime owner inspection unwraps the function.')
assert not torch.cuda.is_initialized()
result=dict(scope=__doc__,population=artifact(HERE/'population.json'),tasks=results,limitations=['One deliberately selected B4 per task; 12 positions per task. Counts are biased diagnostic counts, not population error rates.','Population statistics weight saved request slots. TextCraft includes six repeated trajectory rows; AppWorld excludes unfinished batches.','Raw coefficient square shares are not parameter-gradient shares and are not shares after whole-batch whitening.','Fresh original VERL actors with LoRA B=0 reproduce original base endpoints; saved formal factual drift is retained rather than hidden.','Single-token probes supplement unchanged author cumulative deletion / RISE / MAS; they do not replace those metrics.','No new acceptance threshold, PPO update, DT production change, checkpoint restore or clipping.'],cuda_initialized=torch.cuda.is_initialized())
(HERE/'sample-analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with (HERE/'sample-points.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.7))
for task,color in (('textcraft','#1764ab'),('appworld','#c05126')):
    rows=results[task]['points']
    axes[0].scatter([r['native_single_delete_d'] for r in rows],[r['DT_d'] for r in rows],label=task,color=color,s=42,alpha=.85)
    axes[1].scatter([r['native_endpoint_expected_A_FP32'] for r in rows],[r['DT_expected_A_FP32'] for r in rows],label=task,color=color,s=42,alpha=.85)
for ax,title,label in zip(axes,('Log-prob deletion effects','Reward coefficients (raw)'),('d','A')):
    ax.axhline(0,color='.6',lw=.6);ax.axvline(0,color='.6',lw=.6)
    ax.set_xlabel('Original native single deletion '+label);ax.set_ylabel('Saved joint DT '+label);ax.set_title(title);ax.grid(alpha=.15);ax.legend()
axes[0].plot([-5,24],[-5,24],'--',c='.65',lw=.8)
axes[1].set_xscale('symlog',linthresh=.05);axes[1].set_yscale('symlog',linthresh=.05);axes[1].plot([-54,1],[-54,1],'--',c='.65',lw=.8)
axes[1].set_xticks([-10,-1,-.1,0,.1,1],labels=['-10','-1','-0.1','0','0.1','1'])
axes[1].set_yticks([-100,-10,-1,-.1,0,.1,1],labels=['-100','-10','-1','-0.1','0','0.1','1'])
axes[1].annotate('AppWorld newline',xy=(.75,-53.9798355),xytext=(.01,-25),arrowprops=dict(arrowstyle='->',color='.3'),fontsize=9)
fig.suptitle('24 high-impact diagnostic positions; not an unbiased error-rate sample',fontsize=11)
fig.tight_layout();fig.savefig(HERE/'sample-endpoints.png',dpi=180);plt.close(fig)
print(json.dumps({task:dict(sign_disagreements=t['biased_sample_sign_disagreements'],count=t['biased_sample_count'],identity=t['identity_single_delete_d'],minima=[v for v in t['points'] if v['mode']=='most_negative']) for task,t in results.items()},ensure_ascii=False))
