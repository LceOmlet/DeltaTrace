"""Freeze exhaustive coverage and derive scheduling-only full Recall driver."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
release=json.loads((ROOT/'experiments/official/protocol.json').read_bytes())
choice=json.loads((HERE/'scope_choice.json').read_bytes())
verified=json.loads((HERE/'target_scope/analysis.json').read_bytes())
parent=json.loads((HERE/'target_scope_v1/results.json').read_bytes())
assert verified['status']=='verified_validation' and verified['case_count']==80
assert verified['run_results_sha256']==sha(HERE/'target_scope_v1/results.json')
assert verified['run_vectors_sha256']==parent['vectors_sha256']==sha(HERE/'target_scope_v1/vectors.npz')
assert parent['choice']==choice
tasks=list(choice['targets'])
plan=dict(version='full-recall-v1',spec_sha256=sha(HERE/'FULL_RECALL.md'),
    choice_sha256=sha(HERE/'scope_choice.json'),parent_results_sha256=verified['run_results_sha256'],
    parent_vectors_sha256=parent['vectors_sha256'],parent_analysis_sha256=sha(HERE/'target_scope/analysis.json'),
    tasks={},chunks={})
for t in tasks:
    reused=sorted(x['index'] for x in parent['cases'] if x['dataset']==t)
    count=release['tasks'][t]['count'];missing=[i for i in range(count) if i not in reused]
    assert len(reused)==16 and len(set(reused))==16
    plan['tasks'][t]=dict(count=count,cache_sha256=release['tasks'][t]['cache_sha256'],
        target_mode=choice['targets'][t],reused_indices=reused,new_indices=missing,control_index=min(reused))
    for j,start in enumerate(range(0,len(missing),16)):
        new=missing[start:start+16];controls=[min(reused)] if j==0 else []
        plan['chunks'][f'{t}_{j:02d}']=dict(dataset=t,new_indices=new,control_indices=controls,indices=controls+new)
plan.update(unique_cases=sum(x['count'] for x in plan['tasks'].values()),reused_cases=80,
            new_cases=sum(len(x['new_indices']) for x in plan['tasks'].values()),overlap_controls=5)
assert (plan['unique_cases'],plan['new_cases'])==(448,368)
p=HERE/'full_recall_plan.json'
if p.exists():assert json.loads(p.read_bytes())==plan
else:p.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
source=HERE/'evaluate_target_scope.py'
assert sha(source)=='d12f8492b83c5209eed3e7cc1c05343d3817ad5446ac70f6762212ba30977912'
code=source.read_text(encoding='utf-8').replace("choices=['validation']","choices=['full']")
code=code.replace("    parser.add_argument('--choice', type=Path)","    parser.add_argument('--choice', type=Path, required=True)\n    parser.add_argument('--parent', type=Path, required=True)\n    parser.add_argument('--chunk', required=True)")
code=code.replace("    if args.stage == 'validation':", "    if args.stage == 'full':")
anchor='    caches = {}'
assert code.count(anchor)==1
code=code.replace(anchor,'''    full_plan=json.loads((PILOT/'full_recall_plan.json').read_bytes())
    assert full_plan['spec_sha256']==sha((PILOT/'FULL_RECALL.md').read_bytes())
    assert full_plan['choice_sha256']==sha(args.choice.read_bytes())
    chunk=full_plan['chunks'][args.chunk]
    assert args.datasets==[chunk['dataset']]
    assert sha((args.parent/'results.json').read_bytes())==full_plan['parent_results_sha256']
    assert sha((args.parent/'vectors.npz').read_bytes())==full_plan['parent_vectors_sha256']
    parent_report=json.loads((args.parent/'results.json').read_bytes())
    assert parent_report['status']=='complete' and parent_report['choice']==choice
    parent_rows={(x['dataset'],x['index']):x for x in parent_report['cases']}
''' + anchor)
code=code.replace("        indices = split['tasks'][dataset][args.stage]", "        indices = chunk['indices']")
code=code.replace("        assert sha(raw) == split['tasks'][dataset]['cache_sha256']", "        assert sha(raw) == full_plan['tasks'][dataset]['cache_sha256']")
code=code.replace("report.update(experiment='target-scope-v1'", "report.update(experiment='target-full-v1'")
code=code.replace("    vectors = {}", "    report.update(full_plan_sha256=sha((PILOT/'full_recall_plan.json').read_bytes()),full_spec_sha256=full_plan['spec_sha256'],chunk=args.chunk,chunk_spec=chunk,parent_results_sha256=full_plan['parent_results_sha256'],parent_vectors_sha256=full_plan['parent_vectors_sha256'])\n    parent_vectors=np.load(args.parent/'vectors.npz')\n    vectors = {}")
anchor="                    row['status'] = 'complete'"
assert code.count(anchor)==1
code=code.replace(anchor,'''                    if index in chunk['control_indices']:
                        old_row=parent_rows[dataset,index]
                        for name in ('input_ids','input_sha256','target','target_mode','target_weights','keep','gold','references'):
                            assert row[name]==old_row[name],('Overlap metadata differs',name)
                        for suffix in ('DT_target_signed_full','FT_K1_prompt','FT_K3_prompt'):
                            assert np.array_equal(vectors[key+'_'+suffix],parent_vectors[key+'_'+suffix]),('Overlap vector differs',suffix)
                        row['overlap_control_bitwise_equal']=True
''' + anchor)
code=code.replace('Small Recall pilot derived from the frozen, verified source-v2 adapter.',
                  'Exhaustive Recall shard; computation inherited from the verified target adapter.')
compile(code,'evaluate_full_recall.py','exec')
(HERE/'evaluate_full_recall.py').write_text(code,encoding='utf-8')
print(json.dumps(dict(plan_sha256=sha(p),driver_sha256=sha(HERE/'evaluate_full_recall.py'),chunks=len(plan['chunks']),new_cases=368,overlap_controls=5),indent=2))
