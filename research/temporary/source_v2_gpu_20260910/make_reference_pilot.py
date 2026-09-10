"""Build the two-reference answer-only pilot from the verified target adapter."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
split=json.loads((HERE/'answer_split.json').read_bytes())
split['version']='reference-pilot-v1';split['plan_sha256']=sha(HERE/'REFERENCE_PILOT.md')
path=HERE/'reference_split.json'
if path.exists():assert json.loads(path.read_bytes())==split
else:path.write_text(json.dumps(split,indent=2)+'\n',encoding='utf-8')
source=HERE/'evaluate_answer_matched.py'
assert sha(source)=='775f7359a357c00e847b67364b4b6f5f9eea64a8b952d827dbcbbbd751663b8f'
code=source.read_text(encoding='utf-8')
code=code.replace('answer-pilot-v1','reference-pilot-v1').replace('answer_split.json','reference_split.json').replace('ANSWER_PILOT.md','REFERENCE_PILOT.md')
code=code.replace("    parser.add_argument('--choice', type=Path)","    parser.add_argument('--choice', type=Path)\n    parser.add_argument('--parent', type=Path, help='Development: verified answer-target pilot for repeat controls')")
code=code.replace("        assert args.choice is None", "        assert args.choice is None and args.parent is not None")
code=code.replace("        if choice:\n            assert choice['FT_initial_target_adapter_sha256'] == report['FT_initial_target_adapter']['sha256']\n",'')
code=code.replace("                modes = ['full', 'answer_conditioned', 'answer_only'] if choice is None else ['full', choice['target_mode']]", "                modes = ['answer_only']")
code=code.replace("        report['weighted_sources'] =", '''        control_vectors = control_rows = None
        if args.parent:
            control_report = json.loads((args.parent/'results.json').read_bytes())
            assert control_report['status'] == 'complete' and control_report['experiment'] == 'answer-pilot-v1'
            assert sha((args.parent/'vectors.npz').read_bytes()) == control_report['vectors_sha256']
            control_vectors = np.load(args.parent/'vectors.npz')
            control_rows = {(r['dataset'],r['index']):r for r in control_report['cases'] if r['target_mode']=='answer_only'}
            report['parent_control'] = {'results_sha256': sha((args.parent/'results.json').read_bytes()), 'vectors_sha256': control_report['vectors_sha256']}
        space_ids = tokenizer(' ', add_special_tokens=False)['input_ids']
        assert len(space_ids) == 1 and space_ids[0] != tokenizer.eos_token_id
        report['space_reference_token_id'] = space_ids[0]
        report['weighted_sources'] =''')
start=code.index('                    full_reference = reference_token_ids(')
end=code.index('                    method_identity()',start)
code=code[:start]+'''                    refs = {name: reference_token_ids(row['input_ids'], [positions[j] for j in author_keep], token)
                        for name, token in [('eos',tokenizer.eos_token_id),('space',space_ids[0])]}
                    row['references'] = {name:sha(np.asarray(ref,dtype=np.int64).tobytes()) for name,ref in refs.items()}
                    for ref, reference in refs.items():
                        signed, detail = timed(key+'_DT_'+ref, lambda reference=reference:attribute(reference,target_weights))
                        detail['sign_scope'] = 'Signed allocation relative to the explicit full author-eligible ' + ref + ' token reference; fixed generated final-answer target.'
                        vectors[key+'_DT_'+ref+'_signed_full'] = signed.numpy()
                        row['DT_'+ref+'_details'] = detail
                        score('DT_'+ref,signed.numpy()[positions])
                    mean_signed = .5*(vectors[key+'_DT_eos_signed_full']+vectors[key+'_DT_space_signed_full'])
                    vectors[key+'_DT_mean_signed_full'] = mean_signed
                    score('DT_mean',mean_signed[positions])
                    if control_rows is not None:
                        parent_row = control_rows[dataset,index]
                        assert row['input_sha256'] == parent_row['input_sha256']
                        assert np.array_equal(vectors[key+'_DT_eos_signed_full'],control_vectors[key+'_DT_target_signed_full']), 'EOS DT control did not reproduce'
                        row['DT_EOS_parent_bitwise_equal'] = True
''' + code[end:]
code=code.replace("                            score(f'FT_K{hops}', ft_score.numpy())", "                            score(f'FT_K{hops}', ft_score.numpy())\n                            if control_vectors is not None:\n                                assert np.array_equal(ft_score.numpy(),control_vectors[key+f'_FT_K{hops}_prompt']), 'FT control did not reproduce'\n                                row[f'FT_K{hops}_parent_bitwise_equal'] = True")
compile(code,'evaluate_reference.py','exec')
path=HERE/'evaluate_reference.py';path.write_text(code,encoding='utf-8')
print(json.dumps({'driver_sha256':sha(path),'split_sha256':sha(HERE/'reference_split.json'),'plan_sha256':split['plan_sha256']}))
