"""Version the existing native AppWorld PPO preparation with the v4 memory backend."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT/'receipts/appworld-native-PPO-32k-speed-20261016-v2/source/prepare_remote.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '09dfab05f5f4e273650949e81cba099973742d30464f791f240cb3e4ead1ca10'
code = source.read_text()
assert "VERSION='appworld-native-PPO-32k-speed-20261016-v2'" in code
code = code.replace('appworld-native-PPO-32k-speed-20261016-v2', 'appworld-native-PPO-32k-speed-20261016-v3')
code = code.replace("NEW=ROOT/'candidates/PPO-32k-native-efficiency-20261016-v2/verl-owner'",
                    "NEW=ROOT/'candidates/PPO-32k-native-efficiency-20261016-v4/verl-owner'")
code = code.replace('76f3efbf53e25bd9919ae79de061cd6a47401d8e93df3d5c64bf397e09aa94dd',
                    '5bb0211a5cf5bd355bb763da93d26e3902f1297c96273eddf5e3a6640d455dd3')
code = code.replace('f62bd44a203f8ee27480792769281f4259d299beb74db4a215031bd6b6d5b423',
                    '242f7919ef00bcf4dced4d4c856946d2e761dbf24297d434356194f03943cfcb')
anchor = "p['options']['critic.model.enable_activation_offload']=True"
assert code.count(anchor) == 1
code = code.replace(anchor, anchor+"\np['options']['+critic.model.activation_offload_native_max_seq_len']=8192")
code = code.replace('Verified v2 native critic activation offload', 'Verified v2 long-input offload plus v4 native short-input dispatch')
old = "assert all(item['median_ratio']<=1.0 for item in performance['comparisons']), 'No efficiency regression accepted'"
new = """assert next(item for item in performance['comparisons'] if item['context']==32768)['median_ratio']<=1.0
short=read(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v4/performance-exec3/status.json')
assert short['status']=='completed' and short['native_short_dispatch']=='passed'
assert short['comparison']['median_ratio']<=1.0, 'No short-input efficiency regression accepted'
"""
assert code.count(old) == 1
code = code.replace(old, new)
exec(compile(code, str(source), 'exec'))
