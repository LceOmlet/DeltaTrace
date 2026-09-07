"""Check identity exposure in old trace manifests without using their scores."""
import hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent;root=A/'snapshot${FLASHTRACE_ROOT}-results/main-41da17d/traces'
release=A/'published_flashtrace/table1-data-v1/extracted/data'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={'scope':'Identity-only historical exposure audit. Old VJP experiment quality/correctness is not trusted or imported. Matching prompt/target records establish prior use, not validity of that implementation.','datasets':{},'model_calls':0}
for dataset,count in [('niah_mq_q2',100),('morehopqa',95)]:
    manifests=list((root/dataset/'qwen-8B').glob('VJP-signed*/manifest.jsonl'));assert len(manifests)==1
    f=manifests[0];rows=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    cache=release/(dataset+'.jsonl');original=[json.loads(x) for x in cache.read_text().splitlines() if x.strip()]
    assert len(original)==count
    by_identity={}
    for i,r in enumerate(original):
        identity=tuple(hashlib.sha1(r[k].encode()).hexdigest() for k in ['prompt','target'])
        by_identity.setdefault(identity,[]).append(i)
    matches=[];unmatched=[]
    for r in rows:
        key=r['prompt_sha1'],r['target_sha1'];indices=by_identity.get(key,[])
        item={'historical_index':r['example_idx'],'original_cache_indices':indices,'prompt_sha1':key[0],'target_sha1':key[1]}
        (matches if indices else unmatched).append(item)
    matched=sorted({j for r in matches for j in r['original_cache_indices']})
    out['datasets'][dataset]={'historical_manifest_sha256':sha(f),'original_cache_sha256':sha(cache),'historical_manifest_rows':len(rows),
        'original_cache_rows':count,'matched_original_indices':matched,'unmatched_original_indices':sorted(set(range(count))-set(matched)),
        'matches':matches,'unmatched_historical_rows':unmatched,'historical_metrics_used':False}
capacity_file=A/'snapshot${ARTIFACT_ROOT}/codex_qwen_official_cache_capacity_20260906_v1/results.json'
assert sha(capacity_file)=='4db8f66946de9ad66f3d1313a472f6d5e626173715d201a5fecac6c6a90dd250'
capacity=json.loads(capacity_file.read_text());assert capacity['status']=='complete' and capacity['native_forwards']==0
out['verified_later_experiments']=[]
for directory,digest in [('codex_finite_pair_remaining_20260906_v1','0a8f8cc4ae70c975fbb1fb99bc48ef9e7698d2c0ae65f08b705ad48c41784568'),('codex_qwen_secant_confirmation32_20260906_v1','623bcda088cddf631843b684ea15eddb7d2eecdbebf15939f4b42e80f8183c88')]:
    path=A/'snapshot/tmp'/directory/'results.json';assert sha(path)==digest
    data=json.loads(path.read_text());assert data['status']=='complete'
    item={'raw_sha256':digest,'matched_complete_indices':{ds:[] for ds in out['datasets']}}
    for r in data['records']:
        assert r.get('complete',r.get('row_complete',False))
        ds=r['dataset'];idx=r['idx'];cap=capacity['datasets'][ds]
        assert cap['source_sha256']==out['datasets'][ds]['original_cache_sha256']
        identity=next(x for x in cap['records'] if x['idx']==idx)
        assert identity['input_ids_sha256']==r['input_ids_sha256']==hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest()
        item['matched_complete_indices'][ds].append(idx)
    out['verified_later_experiments'].append(item)
    del data
for ds,r in out['datasets'].items():
    union=set(r['matched_original_indices'])
    for experiment in out['verified_later_experiments']:union.update(experiment['matched_complete_indices'][ds])
    r['all_proven_exposed_indices']=sorted(union)
    r['not_proven_exposed_by_these_sources']=sorted(set(range(r['original_cache_rows']))-union)
    assert not r['not_proven_exposed_by_these_sources']
out['conclusion']='Both released NIq2-100 and MH95 have identity-matched prior experimentation across preserved records. NI8-15 are exposed too; no pristine rows remain in these two caches. This does not retroactively validate old results, or establish whether other published tasks are independent.'
(A/'original_cache_exposure_summary_20260907.json').write_text(json.dumps(out,indent=2))
print(json.dumps({ds:{k:v for k,v in r.items() if k not in ['matches','unmatched_historical_rows']} for ds,r in out['datasets'].items()},indent=2))
