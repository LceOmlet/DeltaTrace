"""Collect original-author paper curves, preserve raw arrays, and plot them.

This only summarizes the author's actual returns; it does not reimplement
RISE/MAS, invent a numerical tolerance, or pool the four examples into a
population-quality claim.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH,SCP,REPO
from observe_author_collection import BODY

OUT=HERE/'paper-role-author-curves-20261010-v3'
REMOTE=ROOT+'/receipts/paper-role-author-curves-20261010-v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
binding=lambda p:dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
dest=REPO/'experiments/rl/results_paper_author_curves_20261010.json'
assert not dest.exists(),'Preserve completed receipt'
for version in ('v1','v2','v3'):
    for name in ('result.json','driver.log'):
        subprocess.run(SCP+[SSH[-1]+':'+ROOT+'/receipts/paper-role-author-curves-20261010-'+version+'/'+name,
                           str(HERE/('paper-role-author-curves-20261010-'+version)/name)],check=True,timeout=35)
r=json.loads((OUT/'result.json').read_bytes())
assert r['phase']=='complete' and r['native_forward_calls']==63
assert all(v==0 for v in r['operations'].values())
body='TARGET='+repr(REMOTE)+'\nROOT='+repr(ROOT)+'\nINCLUDE_PHASES=False\n'+BODY
shell='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
p=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=35)
(OUT/'completion-command.sh').write_text(shell,encoding='utf-8');p.check_returncode()
done=json.loads(p.stdout)
(OUT/'completion-and-resource.json').write_text(json.dumps(done,indent=2)+'\n',encoding='utf-8')
assert not done['driver_alive'],'Collect completion after this diagnostic releases the GPU'
assert done['files']['result.json']['sha256']==sha(OUT/'result.json')

profiles=list(r['profiles']);cases=[]
for index,base in enumerate(r['profiles']['historical-paper-clean-v1']['cases']):
    row=dict(dataset=base['dataset'],index=base['index'],input_sha256=base['input_sha256'],
             source_count=len(base['source_positions']),metrics={},endpoint_logp_observations={})
    for profile in profiles:
        case=r['profiles'][profile]['cases'][index]
        assert case['input_sha256']==base['input_sha256']
        signed=case['views']['signed_RISE'];positive=case['views']['positive_MAS']
        assert len(signed['score_points'])==len(positive['score_points'])==21
        row['metrics'][profile]=dict(RISE=signed['author_return'][0],MAS=positive['author_return'][1],
            normalized_signed_curve_all_zero=all(x==0 for x in signed['author_arrays']['normalized_model_response']),
            signed_density_min=min(signed['author_arrays']['density']))
        row['endpoint_logp_observations'][profile]={view:dict(factual=data['score_points'][0]['logp'],
            all_sources_EOS=data['score_points'][-1]['logp']) for view,data in case['views'].items()}
    old=row['metrics']['historical-paper-clean-v1']
    row['paired_metric_changes_vs_historical_vector']={profile:{name:row['metrics'][profile][name]-old[name]
        for name in ('RISE','MAS')} for profile in profiles if profile!='historical-paper-clean-v1'}
    cases.append(row)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
names={'historical-paper-clean-v1':'Original paper attribution',
       'clean-v1':'Current clean-v1','gdn-symmetric-v1':'Current gdn-symmetric-v1'}
colors={'historical-paper-clean-v1':'#555555','clean-v1':'#2A6FBB','gdn-symmetric-v1':'#BD4B22'}
fig,axes=plt.subplots(4,2,figsize=(11,12),constrained_layout=True)
for index,row in enumerate(cases):
    for profile in profiles:
        arrays=r['profiles'][profile]['cases'][index]['views']['signed_RISE']['author_arrays']
        x=[100*len(point['changed_input_positions'][2*index])/row['source_count']
           for point in r['profiles'][profile]['forward_phases']]
        axes[index,0].plot(x,arrays['scores'],label=names[profile],color=colors[profile],linewidth=1.6)
        axes[index,1].plot(x,arrays['normalized_model_response'],label=names[profile],color=colors[profile],linewidth=1.6)
    title=f"{row['dataset']} / {row['index']} ({row['source_count']} eligible source tokens)"
    axes[index,0].set_title(title);axes[index,1].set_title(title)
    axes[index,0].set_ylabel('Original target log-probability (nats)')
    axes[index,1].set_ylabel('Author normalized response')
    for ax in axes[index]:
        ax.set_xlabel('Eligible source tokens replaced by EOS (%)')
        ax.grid(alpha=.2);ax.set_xlim(0,100)
    axes[index,1].set_ylim(-.03,1.03)
axes[0,0].legend(fontsize=8)
fig.suptitle('Four literal paper examples: unchanged author cumulative deletion metric\nSame native scorer for all three attribution vectors; no DT or policy update',fontsize=12)
plot=OUT/'paper-author-deletion-curves.png';fig.savefig(plot,dpi=170);plt.close(fig)

result=dict(status='Completed original-author metrics on four literal paper examples; diagnostic subset only',
 scope=__doc__,launch=binding(OUT/'launch.json'),raw=binding(OUT/'result.json'),
 completion=binding(OUT/'completion-and-resource.json'),plot=binding(plot),
 related_DT_receipt=binding(REPO/'experiments/rl/results_paper_role_implementation_20261010.json'),
 owners=r['owners'],cases=cases,operations=r['operations'],native_forward_calls=r['native_forward_calls'],
 elapsed_seconds=r['elapsed_seconds'],resources=r['resources'],
 rejected_diagnostic_attempts=[binding(HERE/('paper-role-author-curves-20261010-'+v)/'rejected-diagnostic.json') for v in ('v1','v2')],
 interpretation=['Original metric k=20, sorting, EOS deletion, density, normalization and returns are unchanged.',
 'RISE is the signed-view original RISE return; MAS is the positive-view original MAS return. Both are lower-is-better AUCs, not percentages.',
 'Historical paper attribution is rescored using the same current native model as current vectors. These are matched diagnostic results, not the historical published metric values.',
 'The four inputs are the original available figure examples, not a randomly sampled held-out evaluation. No population accuracy, tail recall, statistical significance or repair claim follows.',
 'Raw score reversals and the original normalization are retained separately; no added smoothing, compensation, clipping, new tolerance or estimator modification. The original metric normalization and clipping remain unchanged.',
 'No production source, formal training process, LoRA setting, optimizer, reward or policy advantage was modified.'])
dest.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(receipt=binding(dest),plot=str(plot),cases=cases,native_forward_calls=r['native_forward_calls'],resources=r['resources']),ensure_ascii=False))
