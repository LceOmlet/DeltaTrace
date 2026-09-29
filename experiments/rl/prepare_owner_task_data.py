"""Store native task identities in VERL's existing Parquet dataset format.

The collector obtains every model prompt from the actual environment owner.
The AppWorld prompt column is an ID carrier for the native dataset, never an
LLM prompt; make_loop_environments supplies the native rendered token IDs.
"""
import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def prepare(task, destination):
    import site
    extra=os.environ.get('TEXTCRAFT_EXTRAS' if task=='TextCraft' else 'LOOP_EXTRAS')
    if extra:site.addsitedir(extra)
    destination.mkdir(parents=True,exist_ok=True)
    if task=='TextCraft':
        from agentenv.envs import TextCraftEnvClient
        source=Path(os.environ['TEXTCRAFT_DATA'])
        prompt=[dict(role=role,content=message['value']) for role,message in zip(
            ['user','assistant'],TextCraftEnvClient.conversation_start)]
        splits={phase:json.loads((source/relative).read_text()) for phase,relative in
            [('train','train/textcraft_train.json'),('validation','eval/textcraft_test.json')]}
        rows={phase:[dict(prompt=prompt,data_source=task,
            env_kwargs=dict(item_id=int(item['item_id'].split('_')[-1])),
            extra_info=dict(index=i,owner_item_id=item['item_id']))
            for i,item in enumerate(items)] for phase,items in splits.items()}
    else:
        from phi_agents.appworld.interface import load_task_ids
        from loop_owner_recipe import environment_configuration
        native=environment_configuration(Path(os.environ['LOOP_ROOT']))
        rows={}
        for phase,native_phase in [('train','training'),('validation','evaluation')]:
            split=native[f'{native_phase}_task_sampler']['dataset_name']
            rows[phase]=[dict(prompt=[dict(role='user',content=task_id)],data_source=task,
                env_kwargs=dict(task_id=task_id),extra_info=dict(index=i,owner_split=split))
                for i,task_id in enumerate(load_task_ids(split))]
    for phase,items in rows.items():
        pq.write_table(pa.Table.from_pylist(items),destination/f'{phase}.parquet')
    receipt=dict(task=task,counts={k:len(v) for k,v in rows.items()},
                 note='Representation only. Actual model inputs come from the native environment manager.')
    (destination/'source.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['TextCraft','AppWorld'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    prepare(args.task,args.output)
