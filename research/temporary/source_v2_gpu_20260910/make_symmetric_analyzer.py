"""Derive input/target verification from the completed reference audit."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
code=(HERE/'analyze_reference_pilot.py').read_text(encoding='utf-8')
code=code.replace('reference-pilot-v1','symmetric-pilot-v1').replace('reference_split.json','symmetric_split.json').replace('REFERENCE_PILOT.md','SYMMETRIC_PILOT.md').replace('evaluate_reference.py','evaluate_symmetric.py')
code=code.replace("REFS=['eos','space','mean']", "REFS=['content_P1','layer_symmetric']")
start=code.index("    space,=tok.encode(")
end=code.index("    stage=r['stage']",start)
code=code[:start]+code[end:]
code=code.replace("assert eos!=space and hashlib", "assert hashlib")
code=code.replace("for name,token in [('eos',eos),('space',space)]:", "for name,token in [('eos',eos)]:")
start=code.index("        mean=.5*(vectors[")
end=code.index('        if parent_rows is not None:',start)
code=code[:start]+code[end:]
code=code.replace("'DT_eos_signed_full'", "'DT_content_P1_signed_full'")
start=code.index("                if ref!='mean':")
end=code.index('            else:',start)
code=code[:start]+'''                assert ref in REFS
                detail=row[method+'_details'];l0,l1=np.asarray(detail['endpoint_target_logprobs32'],dtype=np.float64)
                delta=float(((l1-l0)*weights).sum());deltas[ref]=delta
                assert np.isclose(delta,detail['target_delta_score32_sum64'],atol=1e-8)
                assert np.isclose(signed.sum(),detail['signed_sum'],atol=1e-10)
                assert detail['pv_rule']==ref
                activity=detail['finite_attention_activity']
                assert len(activity)==(72 if ref=='layer_symmetric' else 36)
                assert all(x['calls_attempted']==x['calls_enqueued']==1 for x in activity)
                assert sorted(x['layer'] for x in activity)==sorted(list(range(36))*(2 if ref=='layer_symmetric' else 1))
                assert np.array_equal(row['DT_content_P1_details']['endpoint_target_logprobs32'],row['DT_layer_symmetric_details']['endpoint_target_logprobs32'])
                residuals.append({'dataset':t,'index':i,'rule':ref,'delta':delta,'residual':detail['unassigned_total']})
''' + code[end:]
start=code.index("        residuals.append({'dataset':t,'index':i,'reference':'mean'")
end=code.index('    table=',start)
code=code[:start]+code[end:]
code=code.replace("        chosen=ranked[0]", "        chosen='layer_symmetric'  # The only preregistered candidate; P1 is a control.")
code=code.replace("'reference':chosen", "'rule':chosen, 'symmetric_sources':r['weighted_sources']")
code=code.replace("chosen=r['choice']['reference']", "chosen=r['choice']['rule'];assert chosen=='layer_symmetric'\n        assert r['choice']['symmetric_sources']==r['weighted_sources']")
code=code.replace("Independently verify fixed answer-only EOS/space reference comparisons.",
                  "Independently verify the fixed answer-only layerwise PV comparison.")
compile(code,'analyze_symmetric_pilot.py','exec')
(HERE/'analyze_symmetric_pilot.py').write_text(code,encoding='utf-8')
print('Generated symmetric analyzer')
