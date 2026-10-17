"""Apply the evidenced shared critic resource fix through the accepted tool."""
import hashlib
from pathlib import Path

legacy = Path(__file__).resolve().parent / 'patch_TextCraft_PPO_critic_cache_return_v2.py'
assert hashlib.sha256(legacy.read_bytes()).hexdigest() == '025299a2bc0dcf386956da14bc8946b055571ec4c90d4329ddd89448ba049f85'
code = legacy.read_text()
replacements = [
    ('critic-rpc-host-cache-release-20261017-v2', 'critic-rpc-host-cache-release-AppWorld-20261017-v1', 4),
    ('9dfd6ce795384b98faa962269e1eac98ad05ec65e42af2aac434c41db41641d3',
     '5bb0211a5cf5bd355bb763da93d26e3902f1297c96273eddf5e3a6640d455dd3', 1),
    ("'TextCraft'", "'AppWorld'", 3),
    ('3823147', '3823264', 1), ('1792200692.49', '1792200693.87', 1),
    ('3874156', '3843629', 2), ('1792201135.09', '1792200848.3', 1),
    ('3876616', '3845945', 2), ('1792201155.53', '1792200868.59', 1),
    ('native-critic-RPC-host-cache-release-20261017-v2.json',
     'native-critic-RPC-host-cache-release-AppWorld-20261017-v1.json', 1),
    ('textcraft-PPO-host-memory-recovery-20261017-v3', 'appworld-native-PPO-host-memory-formal-20261017-v4', 1),
]
for old, new, count in replacements:
    assert code.count(old) == count, old
    code = code.replace(old, new)
exec(compile(code, str(legacy), 'exec'), globals())
