"""Reuse the existing native derivative launcher on the authorized GPU pair."""
from pathlib import Path

source = Path(__file__).with_name('launch_native_gradient_chain_v4.py')
text = source.read_text()
changes = {
    "'native-PG-driver-components-v38')": "'native-PG-driver-components-v38','native-trained-control-propagation-v78')",
    r"r'\|\s*[01]\s+\d+\s+'": r"r'\|\s*[23]\s+\d+\s+'",
    'GPU0/1 occupied': 'GPU2/3 occupied',
    "env['CUDA_VISIBLE_DEVICES']='0,1'": "env['CUDA_VISIBLE_DEVICES']='2,3'",
    "'physical_gpus':[0,1]": "'physical_gpus':[2,3]",
    "grpo=psutil.Process(3174448);assert grpo.create_time()==1791793629.73":
        "grpo=psutil.Process(3174448);assert grpo.create_time()==1791793629.73\ndt=psutil.Process(3140624);assert dt.create_time()==1791993570.92",
    "(dest/'job.json').write_text": """if dest.name=='native-trained-control-propagation-v78':
 record.update(discriminating_quantity=case_description['discriminating_quantity'],
  decision=case_description['decision'],counterfactual_scope=case_description['counterfactual_scope'],
  checkpoint=160,natural_target_PG_passes=1,fixed_seed_VJP_passes=1,
  GRPO_training_reruns=0,GRPO_algorithm_calls=0,
  trained_control_checkpoint_model_derivative_passes=2,
  protected_DTPO_v3=dict(pid=dt.pid,birth=dt.create_time()))
(dest/'job.json').write_text""",
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), 'exec'), globals())
