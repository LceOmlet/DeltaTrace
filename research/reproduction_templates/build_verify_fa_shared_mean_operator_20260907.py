from pathlib import Path
A=Path(__file__).resolve().parent
s=(A/'verify_fa_compact_gqa_operator_20260907.py').read_text()
s=s.replace('codex_fa_compact_gqa_operator_20260907_v1','codex_fa_shared_mean_operator_20260907_v1')
s=s.replace("'operator_exact_no_end_to_end_claim'", "'shared_mean_operator_exact_no_end_to_end_claim'")
s=s.replace("'compact'", "'shared_mean'")
s=s.replace("assert new['activity']['GQA_input_expansion'] is False", "assert new['activity']['GQA_input_expansion'] is False\n    assert new['activity']['global_endpoint_mean_buffers']==0\n    assert len(old['activity']['buffer_contract'])==15 and len(new['activity']['buffer_contract'])==13")
s=s.replace('fa_compact_gqa_operator_summary_20260907.json','fa_shared_mean_operator_summary_20260907.json')
s=s.replace("'next':'At most8 whole-model old/new attributions: one NI2 and one real B4 group, each one warm+one measured per method. No new curves/FT/VJPs.'", "'next':'Assess paired local timings and proof of shared-memory midpoint reuse before any small whole-network screen. No new curves/FT/VJPs.'")
(A/'verify_fa_shared_mean_operator_20260907.py').write_text(s)
print(A/'verify_fa_shared_mean_operator_20260907.py')
