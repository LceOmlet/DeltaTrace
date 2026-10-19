"""Compose the original native DT interface/capacity harness for clipped v9.

Reuse its real parser payloads, B4 per rank, original FSDP and 32768 fixture.
No rollout generation, optimizer update or before/after policy validation.
"""
import hashlib
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5/source/run_native_behavior_v5.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='ae4ba91b1978d572c17a5ab4589c0d2804be1eefb8d59d1de308f17d22815541'
code=source.read_text().replace("'receipts/textcraft-reward-target-exponential-20261016-v5'",
    "'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1'")
old='expected=weighted*-torch.expm1(-torch.where(estimated,d,0.))'
assert code.count(old)==1
code=code.replace(old,"assert kwargs.get('clip_exponential_credit') is True\n                expected=weighted*(-torch.expm1(-torch.where(estimated,d,0.))).clamp(-1,1)")
code=code.replace("'per_event_exponential_then_credit_sum'","'per_event_clipped_exponential_then_credit_sum'")
code=code.replace('config.actor_rollout_ref.actor.optim.total_training_steps=330',
    'config.actor_rollout_ref.actor.optim.total_training_steps=200')
code=code.replace("_temp_dir='/tmp/ray-tc-exponential-v5'","_temp_dir='/tmp/ray-tc-clip-u-v9'")
code=code.replace("'TORCHINDUCTOR_CACHE_DIR')","'TORCHINDUCTOR_CACHE_DIR','PYTORCH_CUDA_ALLOC_CONF')")
# On the already observed capture error only, retain the failing native tensor
# identity. Both aliases still invoke the same unmodified capture owner.
anchor='            native_attribute=runner.attribute;native_credit=reward_readout.reward_event_token_credit\n'
assert code.count(anchor)==1
code=code.replace(anchor,anchor+'''            import native_attention_capture,qwen35_decoder_finite
            original_capture=native_attention_capture.copy_capture_tensor
            def observe_capture(value,destination,**kwargs):
                try:return original_capture(value,destination,**kwargs)
                except BaseException as error:
                    state['capture_failure']=dict(shape=list(value.shape),strides=list(value.stride()),
                        dtype=str(value.dtype),device=str(value.device),destination=str(destination),
                        kwargs=kwargs,error=repr(error),allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
                    write(path,state);raise
            native_attention_capture.copy_capture_tensor=observe_capture
            qwen35_decoder_finite.copy_capture_tensor=observe_capture
''')
anchor='            finally:runner.attribute=native_attribute;reward_readout.reward_event_token_credit=native_credit'
assert code.count(anchor)==1
code=code.replace(anchor,'''            finally:
                runner.attribute=native_attribute;reward_readout.reward_event_token_credit=native_credit
                native_attention_capture.copy_capture_tensor=original_capture
                qwen35_decoder_finite.copy_capture_tensor=original_capture''')
exec(compile(code,str(source),'exec'),globals())
