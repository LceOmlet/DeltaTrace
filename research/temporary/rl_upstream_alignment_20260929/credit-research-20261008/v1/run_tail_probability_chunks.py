"""Run the four frozen, read-only native diagnostic chunks sequentially.

This composes the existing actor diagnostic launcher and records its status.
It is not a training scheduler, sampler, retry loop or new model execution.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    out = Path(sys.argv[1])
    prepared = json.loads((out/'native-prepared.json').read_bytes())
    status = dict(scope=__doc__,pid=os.getpid(),birth=psutil.Process().create_time(),
        started_unix=time.time(),completed_jobs=[],phase='starting',
        sample_sha256=prepared['sample_sha256'],optimizer=0,DT=0,rollout=0)

    def save(**values):
        status.update(observed_unix=time.time(),**values)
        (out/'native-status.json').write_text(json.dumps(status,indent=2)+'\n')

    try:
        for job in prepared['jobs']:
            directory = Path(job['directory'])
            save(phase='launching',active_job=job)
            launched = subprocess.run([sys.executable,str(directory/'launch_native_head_collection.py'),
                job['task'],str(directory)],capture_output=True,check=True,text=True)
            (directory/'launcher.stdout').write_text(launched.stdout)
            (directory/'launcher.stderr').write_text(launched.stderr)
            launch = json.loads(launched.stdout)
            save(phase='native_collection',active_launch=launch)
            last_report = 0
            while True:
                try:
                    process = psutil.Process(launch['pid'])
                    alive = process.create_time() == launch['birth'] and process.status() != psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    alive = False
                if time.time()-last_report >= 30 or not alive:
                    ranks = []
                    for rank in (0,1):
                        path = directory/'results'/f'rank{rank}.json'
                        if path.exists():
                            try:
                                value = json.loads(path.read_bytes())
                            except json.JSONDecodeError:
                                # Optional live observation can overlap the
                                # owner's existing write. Do not fail its job.
                                continue
                            ranks.append({k:value.get(k) for k in ('rank','pid','birth','phase','batch','round',
                                'elapsed_seconds','peak_allocated','process_pss_bytes','operations','completed_points')})
                    save(driver_alive=alive,rank_status=ranks)
                    last_report = time.time()
                if not alive:
                    break
                time.sleep(2)
            completed = directory/'results/completed.json'
            assert completed.exists(), 'Diagnostic ended without original owner completion; no retry or missing-as-TN'
            for rank in (0,1):
                value = json.loads((directory/'results'/f'rank{rank}.json').read_bytes())
                assert value['phase'] == 'complete'
            status['completed_jobs'].append(job)
            save(phase='chunk_complete')
        save(phase='complete',seconds=time.time()-status['started_unix'])
    except BaseException:
        import traceback
        save(phase='failed',traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
