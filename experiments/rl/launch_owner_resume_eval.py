"""Original VERL restore and validation on a labelled subset of native data.

Reuses a completed check's actual options. This is an interface/stability check,
not full benchmark evaluation. No model loading, checkpoint reader, inference
loop or metric implementation is provided here: VERL owns them all.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import pyarrow.parquet as pq
from owner_runtime_options import owner_command


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--examples',type=int,default=2)
    a=p.parse_args()
    options=json.loads((a.source/'launch.json').read_text())['options']
    marker=a.source/'checkpoints/latest_checkpointed_iteration.txt'
    step=int(marker.read_text().strip())
    a.output.mkdir(parents=True,exist_ok=True)
    original=Path(options['data.val_files'])
    subset=a.output/'validation-subset.parquet'
    table=pq.read_table(original).slice(0,a.examples)
    pq.write_table(table,subset)
    options.update({
        'data.val_files':str(subset), 'data.val_batch_size':a.examples,
        'trainer.resume_mode':'resume_path',
        'trainer.resume_from_path':str(a.source/'checkpoints'/f'global_step_{step}'),
        'trainer.default_local_dir':str(a.output/'unused-checkpoints'),
        'trainer.val_only':True, 'trainer.val_before_train':True,
        'trainer.validation_data_dir':str(a.output/'validation'),
        'trainer.rollout_data_dir':str(a.output/'unused-rollouts'),
        '+ray_init.runtime_env.env_vars.DT_VLLM_OBSERVE_DIR':str(a.output)})
    argv=owner_command(options)
    os.environ.update(DT_TASK=options['env.env_name'],DT_MAX_STEPS=str(options['env.max_steps']),
                      DT_MAX_LENGTH='32768')
    if '+env.sql' in options:
        os.environ['DT_SAMPLING_JSON']=json.dumps(options['+env.sql']['sampling'])
    (a.output/'launch.json').write_text(json.dumps(dict(argv=argv,options=options,
        source=str(a.source),checkpoint_step=step,original_validation_data=str(original),
        subset_rows=table.num_rows,scope=__doc__),indent=2))
    os.execv(sys.executable,argv)
