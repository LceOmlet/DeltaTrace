"""Reuse the existing read-only worker RPC and include the native critic memory binding."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT/'receipts/appworld-sql-native-speed-20261016-v2/source/inspect_worker_imports.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '77e4e0bdea4a3874a495e601b3feb251ad61fa8166a5d4417f126fa91c3d020c'
code = owner.read_text().replace('appworld-sql-native-speed-20261016-v2', 'appworld-native-PPO-32k-speed-20261016-v3')
anchor = "             'verl.utils.experimental.torch_functional', 'verl.models.transformers.qwen3_vl']"
assert code.count(anchor) == 1
code = code.replace(anchor, anchor+"\n    names += ['verl.workers.critic.dp_critic', 'verl.utils.activation_offload']")
anchor = "                DT_CREDIT_MODE=os.environ.get('DT_CREDIT_MODE'), DT_ROOT=os.environ.get('DT_ROOT'),"
replacement = """                critic_memory=[dict(activation_offload=item.config.model.enable_activation_offload,
                                    native_max_seq_len=item.config.model.get('activation_offload_native_max_seq_len',0),
                                    checkpoint_closure=list(item.critic_module.model.language_model.layers[0]._gradient_checkpointing_func.__code__.co_freevars))
                               for item in worker.worker_dict.values() if getattr(item,'critic',None) is not None],
                FA_deterministic=os.environ.get('FLASH_ATTENTION_DETERMINISTIC'),
"""+anchor
assert code.count(anchor) == 1
code = code.replace(anchor, replacement)
exec(compile(code, str(owner), 'exec'), globals())
