"""Read existing DT call receipts; no model execution and no replacement kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    raw=args.report.read_bytes(); report=json.loads(raw)
    if report['family']!='qwen35' or report['status']!='complete':
        raise ValueError('Requires a complete native Qwen3.5 execution receipt.')
    out={'source_sha256':hashlib.sha256(raw).hexdigest(),
         'scope':'timing diagnosis of existing smoke calls, not steady-state benchmarks','cases':[]}
    for row in report['cases']:
        detail=row['DT_details']; groups={}
        for call in detail['calls']:
            kind=call['kind']
            if kind.startswith(('finite_decoder_','native_replay_')):
                layer=detail['layers'][kind.rsplit('_',1)[1]]['block_type']
                kind=kind.rsplit('_',1)[0]+'_'+layer
            elif kind.startswith('public_FA_LSE_'):kind='public_FA_LSE'
            groups.setdefault(kind,[]).append(call['seconds'])
        out['cases'].append({'dataset':row['dataset'],'index':row['index'],
            'attribution_seconds_with_diagnostics':detail['complete_attribution_seconds_with_diagnostics'],
            'stages':{name:{'calls':len(values),'total_seconds':sum(values),'first_seconds':values[0],
                'later_median_seconds':statistics.median(values[1:]) if len(values)>1 else None}
                for name,values in groups.items()}})
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':main()
