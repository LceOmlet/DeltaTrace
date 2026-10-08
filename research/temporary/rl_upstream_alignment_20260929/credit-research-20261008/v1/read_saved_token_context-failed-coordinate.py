"""Decode all frozen TextCraft probe positions, without model or DT calls.

The original readout owns preparation. Decoding is display-only and never
round-trips into training inputs. All token IDs, masks and endpoint values
remain preserved separately.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import psutil
import torch
from transformers import AutoTokenizer


def ref(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    root=Path(args.root)
    plan=json.loads(Path(args.plan).read_bytes())
    source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
    assert ref(source_path)['sha256']=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
    source=json.loads(source_path.read_bytes())
    sys.path[:0]=source['pythonpath'].split(':')
    path=source['actual_CPU_imports']['reward_readout']['path']
    assert ref(path)['sha256']==source['actual_CPU_imports']['reward_readout']['sha256']
    spec=importlib.util.spec_from_file_location('saved_context_readout_owner',path)
    owner=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=owner
    spec.loader.exec_module(owner)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=True)
    grouped={}
    for entry in plan['tasks']['textcraft']['entries']:
        grouped.setdefault(entry['native']['path'],[]).append(entry)
    trajectories=[]
    peak_pss=0
    for filename,entries in grouped.items():
        identity=ref(filename)
        assert all(identity['sha256']==e['native']['sha256'] for e in entries)
        capture=torch.load(filename,map_location='cpu',weights_only=False)
        by_uid={r['traj_uid']:r for r in capture['rows']}
        for entry in entries:
            saved=by_uid[entry['traj_uid']]
            row=owner.DirectActionTargetReadout._prepare_row(saved['row'],saved['trajectory_index'])
            for key in ('selected','suffix_positions','policy','target','prior'):
                assert torch.equal(row[key],saved[key])
            ids=row['selected'].tolist()
            contexts=[]
            for query in entry['queries']:
                slot=query['packed_slot']
                assert ids[slot]==query['token_id'] and row['prior'][slot]
                left,right=max(0,slot-16),min(len(ids),slot+17)
                contexts.append(dict(packed_slot=slot,token_id=ids[slot],
                    tokenizer_piece=tokenizer.convert_ids_to_tokens(ids[slot]),
                    decoded_token=tokenizer.decode(ids[slot:slot+1],skip_special_tokens=False),
                    window_start=left,window_end_exclusive=right,window_ids=ids[left:right],
                    window_policy_mask=row['policy'][left:right].tolist(),
                    window_target_mask=row['target'][left:right].tolist(),
                    decoded_window=tokenizer.decode(ids[left:right],skip_special_tokens=False),
                    cohorts=query['cohorts']))
            target_ids=[ids[i] for i in row['target'].nonzero().flatten().tolist()]
            trajectories.append(dict(traj_uid=entry['traj_uid'],initial_state_sha256=entry['initial_state_sha256'],
                capture=identity,first_stage=entry['first_stage'],actual_target_offsets=row['target_offsets'],
                actual_joint_target_ids=target_ids,
                decoded_actual_joint_target=tokenizer.decode(target_ids,skip_special_tokens=False),
                contexts=contexts))
        peak_pss=max(peak_pss,psutil.Process().memory_full_info().pss)
        del capture
    assert sum(len(t['contexts']) for t in trajectories)==165
    assert not torch.cuda.is_initialized()
    result=dict(scope=__doc__,unix=time.time(),plan=ref(args.plan),readout=ref(path),
        source=ref(source_path),script=ref(__file__),tokenizer_class=type(tokenizer).__name__,
        model_path=args.model,trajectories=trajectories,points=165,
        elapsed_seconds=time.perf_counter()-started,sampled_peak_PSS_bytes=peak_pss,
        operations=dict(model=0,DT=0,optimizer=0,rollout=0),CUDA_initialized=False,
        display_only=True,production_modified=False)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
