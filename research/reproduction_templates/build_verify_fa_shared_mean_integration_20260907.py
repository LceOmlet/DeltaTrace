from pathlib import Path
A=Path(__file__).resolve().parent
s=(A/'verify_fa_input_preparation_integration_20260907.py').read_text()
s=s.replace('fa_input_preparation','fa_shared_mean').replace('FA_input_preparation','FA_shared_mean')
s=s.replace('fused_prepare','shared_mean')
s=s.replace('for pos in [1,3,4,7]:assert shapes[pos][1]==8', 'for pos in [1,3,4]:assert shapes[pos][1]==8')
s=s.replace('assert shapes[12]==shapes[13]==shapes[14]==shapes[0]', '''if mode=='old':
                    assert len(shapes)==15 and shapes[7][1]==8
                    assert shapes[12]==shapes[13]==shapes[14]==shapes[0]
                else:
                    assert len(shapes)==13
                    assert shapes[10]==shapes[11]==shapes[12]==shapes[0]
                    assert item['global_endpoint_mean_buffers']==0 and item['extra_shared_tile'] is False''')
s=s.replace("if mode=='shared_mean':assert item['input_preparation']=='public_torch_compile_inductor_same_expressions'", "if mode=='shared_mean':assert item['endpoint_mean']=='FP32_add_FP16_store_in_FA_shared_tile_from_existing_loads'")
s=s.replace("out['conclusion']='Full signed vectors and endpoints remain exact; actual native FA and finite kernels unchanged. Report all measured latency; fewer wrapper kernels do not establish useful end-to-end speed. No new curves/FT comparison.'", """assert len(wrapper)==252
out['conclusion']='Shared-storage endpoint means are integrated into genuine FA-framework finite kernels and verified on original B1/B4. Same-job whole vectors and endpoints exact,0 global mean buffers,original shared-memory size,7 remaining preparation kernels/layer. Report paired whole-model cost; no FT/quality or arbitrary long-input claim.'""")
s=s.replace("assert len(wrapper)==252", """assert len(wrapper)==252
out['groups'][0]['timing_caveat']='Original NI2 first measured1.795s is anomalously high against second1.002s; the median ratio is retained but not a causal30% speedup claim. Preparation~0.014s, propagation1.555/0.757s locates the fluctuation inside propagation; cause not established.'
out['preferred_scope']='Shared-mean reuse is a verified FA engineering candidate for the tested B1/B4 scopes; stable broad performance, long input, independent quality and FT superiority remain unproved.'""")
(A/'verify_fa_shared_mean_integration_20260907.py').write_text(s)
print(A/'verify_fa_shared_mean_integration_20260907.py')
