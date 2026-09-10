"""Reuse independent input reconstruction and add fixed scope statistics."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'analyze_answer_pilot.py').read_text(encoding='utf-8')
head=source[:source.index('    table = {(x[')]
head=head.replace('answer-pilot-v1','target-scope-v1').replace('answer_split.json','scope_split.json').replace('ANSWER_PILOT.md','TARGET_SCOPE_VALIDATION.md')
head=head.replace("    p.add_argument('--freeze-choice', action='store_true')\n",'')
head=head.replace("    assert r['driver_sha256'] == digest(HERE / ('evaluate_answer.py' if r['stage'] == 'development' else 'evaluate_answer_matched.py'))",
    "    assert r['driver_sha256'] == digest(HERE / 'evaluate_target_scope.py')\n    assert r['stage']=='validation'\n    assert r['choice']==json.loads((HERE/'scope_choice.json').read_bytes())\n    assert r['choice']['weighted_sources']==r['weighted_sources']")
start=head.index("    stage = r['stage']")
end=head.index("    assert len(r['cases'])",start)
head=head[:start]+'''    stage='validation';tasks=TASKS
    expected={(t,i,r['choice']['targets'][t]) for t in tasks for i in split['tasks'][t][stage]}
    assert len(expected)==80
''' + head[end:]
head=head.replace("            if mode == 'answer_conditioned':\n                assert all((item['start'], item['end']) == (0, len(target_ids)-2)\n                    for item in row[f'FT_K{hops}_actual_target_aggregation']), 'FT must retain all reasoning-hop support'",
    "            assert all((item['start'],item['end'])==(0,len(target_ids)-2) for item in row[f'FT_K{hops}_actual_target_aggregation'])")
tail='''    from scope_statistics import summarize_scope
    statistics=summarize_scope(rows,split)
    out=dict(status='verified_validation',case_count=80,driver_sha256=r['driver_sha256'],
        analyzer_sha256=digest(Path(__file__)),statistics_sha256=digest(HERE/'scope_statistics.py'),
        run_results_sha256=digest(a.run/'results.json'),run_vectors_sha256=r['vectors_sha256'],
        split_sha256=r['split_sha256'],choice_sha256=digest(HERE/'scope_choice.json'),
        plan_sha256=r['plan_sha256'],choice=r['choice'],
        original_inputs_gold_references_and_answer_substrings_verified=True,
        actual_DT_FT_target_weights_and_full_hop_support_verified=True,
        all_recall_curves_recomputed_from_vectors=True,
        completed_gpu_operation_seconds=sum(x['seconds'] for x in r['costs'] if x['status']=='returned'),
        costs=r['costs'],residuals=residuals,assignment_diagnostics=diagnostics,**statistics)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\\n',encoding='utf-8')
    with (a.output/'cases.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in out.items() if k in ('status','case_count','primary_comparisons','confirmed_scoped_advantages','confirmed_scoped_disadvantages','confirmed_same_view_shared_advantage')},indent=2))


if __name__=='__main__':
    main()
'''
code=head+tail
code=code.replace('Verify matched answer-target controls and select using development cases only.',
    'Verify reserved task-specific target validation without candidate selection.')
compile(code,'analyze_target_scope.py','exec')
(HERE/'analyze_target_scope.py').write_text(code,encoding='utf-8')
print('Generated fixed-scope verifier')
