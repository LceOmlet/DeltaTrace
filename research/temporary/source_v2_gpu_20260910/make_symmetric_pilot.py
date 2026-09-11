"""Generate a separate layerwise symmetric finite-rule experiment."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

src = HERE / 'weighted_secant.py'
assert sha(src) == '4f072111ca55174daee41f9a8240e01dd258e34649ec65afb53bfffb2534de74'
code = src.read_text(encoding='utf-8')
code = code.replace("assert pv_rule == 'content_P1' and callable(finite_attention)",
                    "assert pv_rule in ('content_P1', 'layer_symmetric') and callable(finite_attention)")
anchor = "            assert all(torch.isfinite(value).all() for value in outputs.values())"
assert code.count(anchor) == 1
code = code.replace(anchor, '''            if pv_rule == 'layer_symmetric':
                swapped_activity = {'layer':li, 'direction':'swapped_operands_same_native_capture'}
                finite_activity.append(swapped_activity)
                reverse = finite_attention({'q0':q1,'k0':kr1,'q1':q0,'k1':kr0,
                    'v0':vr1,'u':mout,'lse0':lse1,'lse1':lse0},layer.self_attn.scaling,swapped_activity)
                assert all(torch.isfinite(value).all() for value in reverse.values())
                assert (reverse['tau']>0).all()
                for field in ('dq','dk','dv'):
                    outputs[field] = .5*(outputs[field].float()+reverse[field].float())
                del reverse
''' + anchor)
code = code.replace("'pv_rule_scope': 'Content P1 only: deltaP*V0+P1*deltaV; finite logmean softmax and symmetric QK allocation.",
                    "'pv_rule_scope': 'Explicit ' + pv_rule + ' finite PV allocation; layer_symmetric averages original and swapped finite multipliers at each layer, using identical native endpoint capture; finite logmean softmax and symmetric QK allocation.")
code = code.replace("Content-P1 finite attention uses traceable FA-framework extension", "Explicit PV finite attention uses traceable FA-framework extension")
compile(code, 'symmetric_secant.py', 'exec')
(HERE / 'symmetric_secant.py').write_text(code, encoding='utf-8')
paired = (HERE / 'weighted_paired.py').read_text(encoding='utf-8').replace('from weighted_secant import', 'from symmetric_secant import')
(HERE / 'symmetric_paired.py').write_text(paired, encoding='utf-8')

split = json.loads((HERE / 'answer_split.json').read_bytes())
split.update(version='symmetric-pilot-v1', plan_sha256=sha(HERE / 'SYMMETRIC_PILOT.md'))
path = HERE / 'symmetric_split.json'
if path.exists():
    assert json.loads(path.read_bytes()) == split
else:
    path.write_text(json.dumps(split, indent=2) + '\n', encoding='utf-8')

src = HERE / 'evaluate_reference.py'
assert sha(src) == '919b60c5dc923a8a8831476c6d94794fe75f05b59eff1e4caeb87abc22bd0fd0'
code = src.read_text(encoding='utf-8')
code = code.replace('reference-pilot-v1', 'symmetric-pilot-v1').replace('reference_split.json', 'symmetric_split.json').replace('REFERENCE_PILOT.md', 'SYMMETRIC_PILOT.md')
code = code.replace('No model, attention, FT method, or evaluator function is replaced here.',
    'The native model and FT remain unchanged; a separate finite PV rule is tested.')
code = code.replace('from weighted_paired import propagate_paired_secant as propagate_weighted',
    'from symmetric_paired import propagate_paired_secant as propagate_weighted')
code = code.replace("('weighted_secant.py', 'weighted_paired.py')", "('symmetric_secant.py', 'symmetric_paired.py')")
start = code.index("        space_ids = tokenizer(' '")
end = code.index("        report['weighted_sources']", start)
code = code[:start] + code[end:]
code = code.replace('def attribute(reference_override=None, weights=None):',
                    "def attribute(reference_override=None, weights=None, rule='content_P1'):")
code = code.replace("propagate_weighted(model, before, after, pv_rule='content_P1',", "propagate_weighted(model, before, after, pv_rule=rule,")
start = code.index('                    refs = {name: reference_token_ids(')
end = code.index('                    method_identity()', start)
code = code[:start] + '''                    reference = reference_token_ids(row['input_ids'], [positions[j] for j in author_keep], tokenizer.eos_token_id)
                    row['references'] = {'eos':sha(np.asarray(reference,dtype=np.int64).tobytes())}
                    for rule in ('content_P1','layer_symmetric'):
                        signed, detail = timed(key+'_DT_'+rule, lambda rule=rule:attribute(reference,target_weights,rule))
                        detail['sign_scope'] = 'Signed ' + rule + ' allocation, original author-eligible EOS reference, fixed final-answer target.'
                        assert len(detail['finite_attention_activity']) == (72 if rule=='layer_symmetric' else 36)
                        assert all(x['calls_enqueued']==1 for x in detail['finite_attention_activity'])
                        vectors[key+'_DT_'+rule+'_signed_full'] = signed.numpy()
                        row['DT_'+rule+'_details'] = detail
                        score('DT_'+rule,signed.numpy()[positions])
                    if control_rows is not None:
                        parent_row = control_rows[dataset,index]
                        assert row['input_sha256'] == parent_row['input_sha256']
                        assert np.array_equal(vectors[key+'_DT_content_P1_signed_full'],control_vectors[key+'_DT_target_signed_full']), 'Original DT control did not reproduce'
                        row['DT_P1_parent_bitwise_equal'] = True
''' + code[end:]
code = code.replace("        assert choice['status'] == 'frozen_for_validation'", "        assert choice['status'] == 'frozen_for_validation'\n        assert choice['rule'] == 'layer_symmetric'\n        assert choice['symmetric_sources'] == {name:sha((PILOT/name).read_bytes()) for name in ('symmetric_secant.py','symmetric_paired.py')}")
compile(code, 'evaluate_symmetric.py', 'exec')
(HERE / 'evaluate_symmetric.py').write_text(code, encoding='utf-8')
print(json.dumps({name:sha(HERE/name) for name in ('SYMMETRIC_PILOT.md','symmetric_split.json','symmetric_secant.py','symmetric_paired.py','evaluate_symmetric.py')},indent=2))
