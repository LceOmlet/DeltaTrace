"""Read-only checkpoint and source verification after all serial timing jobs."""
from pathlib import Path
import hashlib
import json
import time
import traceback


def main():
    root=Path(__file__).resolve().parent
    expected=json.loads((root/'postflight_expected.json').read_bytes())
    result={'status':'waiting_for_timing_jobs','model_calls':0,'GPU_calls':0,'files':[]}
    def save():
        p=root/'postflight.partial';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(root/'postflight.json')
    save()
    try:
        while any(json.loads((root/name).read_bytes())['status']!='complete' for name in ['queue_v2.json','queue_v3.json']):
            time.sleep(5)
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
