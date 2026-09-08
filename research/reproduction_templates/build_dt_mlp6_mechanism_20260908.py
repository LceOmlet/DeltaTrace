"""Freeze one3GEMM MLP6 mechanism graph andCPU algebra; never launch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1';p=json.loads((D/'protocol.json').read_bytes())
raw=(D/'results.json').read_bytes();actual=json.loads(raw);assert actual['status']=='decoder19_6_actual_conditional_observation_complete'
assert actual['private_artifact']['sha256']=='0b33ce76ca28fd00375e931a4769fbf4868cf2ff0cddd7040785bc42a6a2672a'
files={name:(D/name).read_bytes() for name in p['files_sha256'] if name not in ['study.py','decoder19_conditional_decomposition_20260908.py']}
for name,value in files.items():assert sha(value)==p['files_sha256'][name],name
files['study.py']=(R/'research/reproduction_templates/dt_mlp6_mechanism_20260908.py').read_bytes()
files['weights_metadata.json']=(A/'snapshot${ARTIFACT_ROOT}/codex_dt_mlp6_weights_metadata_20260908.json').read_bytes()
metadata=json.loads(files['weights_metadata.json'])
assert metadata['index_sha256']=='26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5'
for name,value in files.items():
    if name.endswith('.py'):ast.parse(value,filename=name)
for shard,stat in metadata['shard_stats'].items():assert p['expected_weight_stats'][shard]==stat
for name in ['capture_steps','selected_layers','coarse_boundaries','norm_gate_rules','frozen_metrics_path','frozen_metrics_sha256',
             'production_results_path','production_results_sha256','production_vectors_path','production_vectors_sha256','diagnostic_change']:
    p.pop(name,None)
p.update(scope='One lowcostMLP6 mechanism diagnostic on actual saved MH1 layer6 operands/coefficients. No model load,score,attribution,candidate,orbenchmark.',
    layer_index=6,fixed_steps=[1,10,20],cases=['morehopqa_1'],case_indices=[['morehopqa',1]],
    actual_results_path='${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1/results.json',actual_results_sha256=sha(raw),
    actual_artifact_path='${ARTIFACT_ROOT}/codex_dt_decoder19_6_conditional_20260908_v1/MH1_decoder19_6_actual_private.pt',
    actual_artifact_sha256=actual['private_artifact']['sha256'],
    method_change='None. One diagnostic graph reuses original _linear_transpose/_mm and symmetric swiglu_finite_rule with original fullgraph=True,dynamic=False,triton.cudagraphs=False,max_autotune=False options. Added retained outputs may alter compiler scheduling; explicitly compare recomputedmnorm to originalsavedmnorm and keep their difference separate. No guessing/division recovery ofmproduct.',
    decomposition='Actualsavedmnorm-input minus actualupstream-output is decomposed into diagnosticmnormtransfer,inputprojectionGEMMtransfer,finitecoefficientrounding,SiLUtheoreticalsecantcurvature,nativeSiLUendpointsecant/outputrounding,bilinearinteraction,B2-to-B1endpointtransfer,nativeproductrounding,outputprojectionGEMMtransfer. B2control andexact3conditionalpoints,signs/groups preserved; closed1e-7.',
    budget={'model_loads':0,'DT_calls':0,'FT_calls':0,'native_score_calls':0,'generation_calls':0,'new_samples':0,
        'weight_tensor_get_calls':3,'compiled_diagnostic_graph_calls':1,'semantic_GEMMs_per_graph':3,
        'extra_one_GEMM_control':0,'finite_FA_calls':0,'finite_FLA_calls':0,'conditional_points':3,'B2_controls':1,
        'candidate_scans':0,'wall_time_seconds':300},
    acceptance='Hashactualprivateartifact/results; fixedcheckpointindexSHAand3explicitlayer6keys/shardstats; readonly3selectedtensors withbefore/aftertensorhash. No fullmodel instantiation. Reportactualrecomputedmnormdrift,neverrename itcaptured. LedgeractualendpointsmustmatchpriorMLPcombined1e-7 andtelescopesclose1e-7; no empiricalwinnerclaim orretrospectivedrifttolerance.',
    stop='One graph invocation only,includingfirstcallcompilation. Preservefirstfailure/partialCPUresults;no retry,warmup,sweep,extraGEMMcontrol,scorecall,otherlayer,globalcontent1 candidate,orframeworkedit.',
    costs='ThreeGEMMchosenoveronebecauseit separates recomputed-to-savedmnorm drift frominputprojectiontransfer. AllCPUartifactread/hash,threeweightreads,hosttodevice,compiler/graphwall andCPUalgebraincluded. No claim compiler performs only3physicalkernellaunches;3semanticGEMMs inoneexplicitgraph. Privatecoeffartifact clearlyrecomputed andexcludedfromreviewZIP.',
    files_sha256={name:sha(value) for name,value in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();protocol=A/'dt_mlp6_mechanism_protocol_20260908.json';launch=A/'launch_dt_mlp6_mechanism_20260908.json'
assert not protocol.exists() and not launch.exists();protocol.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(value).decode() for name,value in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_mlp6_mechanism_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
launch.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'launch_payload':str(launch),'remote_directory':directory,'budget':p['budget']}))
