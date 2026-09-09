"""Run explicit promoted API checks serially; no overlapping GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    scripts=args.release/'research/temporary/cause_tolerance_20260909'
    report={'status':'running','jobs':[]}
    def save():(args.output/'controller.json').write_text(json.dumps(report,indent=2)+'\n')
    for family,python in [('qwen3','/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python'),
                          ('qwen35','/tmp/codex_qwen35_isolated_import_20260908_v1/env/bin/python')]:
        command=[python,'-B',str(scripts/f'verify_promoted_{family}.py'),'--release',str(args.release),
                 '--environment',str(args.environment),'--output',str(args.output/family)]
        if family=='qwen3':command+=['--reference','/tmp/codex_clean_development16_20260909_v1/qwen3/results.json',
                                    '--reference-sha256','dccfbf8d2f2ba33b0ff68c686031fb86b2ac76501164f63d91228be545984de6']
        row={'family':family,'command':command,'started':time.time()};report['jobs'].append(row);save()
        with (args.output/(family+'.log')).open('x') as stream:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                   env=dict(os.environ,MACA_PATH='/opt/maca'))
            row['pid']=child.pid;save();row['exit_code']=child.wait()
        row['ended']=time.time()
        if row['exit_code']!=0:report['status']='failed';save();raise RuntimeError('Promoted API check failed: '+family)
        assert json.loads((args.output/family/'results.json').read_bytes())['status']=='complete'
        save()
    report['status']='complete';save()


if __name__=='__main__':main()
