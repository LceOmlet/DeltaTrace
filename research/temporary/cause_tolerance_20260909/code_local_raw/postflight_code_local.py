"""Read-only checkpoint and source verification after all serial timing jobs."""
from pathlib import Path
import hashlib
import json
import time
import traceback


def main():
    root=Path(__file__).resolve().parent
    expected=json.loads((Path('/tmp/codex_short_b1_efficiency_20260910_v1')/'postflight_expected.json').read_bytes())
    manifest=json.loads((root/'production_release/deltatrace/accelerated/code_local_qwen35_sources.json').read_bytes())
    expected['files'] += [{'path':str(root/'production_release'/name),'sha256':digest,'kind':'new_code_local_source'} for name,digest in manifest['files'].items()]
    result={'status':'waiting_for_timing_jobs','model_calls':0,'GPU_calls':0,'files':[]}
    def save():
        p=root/'postflight.partial';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(root/'postflight.json')
    save()
    try:
        while not (root/'qwen35_author_short/results.json').exists() or json.loads((root/'qwen35_author_short/results.json').read_bytes())['status'] not in ('complete','failed'):
            time.sleep(5)
        time.sleep(2)
        result['started_unix']=time.time();result['status']='hashing';save()
        for record in expected['files']:
            p=Path(record['path']);before=p.stat();h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            after=p.stat()
            actual=h.hexdigest()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p)
            assert actual==record['sha256'],str(p)
            result['files'].append({**record,'actual_sha256':actual,'bytes':after.st_size,'matches':True});save()
        result['status']='verified';result['elapsed_seconds']=time.time()-result['started_unix']
        result['scope']='Post-run on-disk identity and unchanged-during-read stat checks. Does not hash live GPU parameters. Qwen3 matches the previous frozen local checkpoint; Qwen3.5 shards match the previously checked official LFS hashes.'
    except Exception:
        result['status']='failed';result['error']=traceback.format_exc()
    save();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
