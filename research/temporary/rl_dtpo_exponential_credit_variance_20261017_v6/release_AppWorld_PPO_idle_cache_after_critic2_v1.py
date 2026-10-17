"""Reuse the accepted native cache action in the natural critic2/actor1 gap."""
import hashlib
from pathlib import Path

legacy = Path(__file__).resolve().parent / 'idle_native_host_cache_release_v2.py'
assert hashlib.sha256(legacy.read_bytes()).hexdigest() == '6aa41a3eb9be054d2c79ba2173e0d4be62f568954c7b8f131b07a2d3cc285672'
code = legacy.read_text()
replacements = [
    ('native-idle-host-cache-release-after-critic103-20261017-v2.json',
     'native-idle-host-cache-release-after-critic2-AppWorld-20261017-v1.json', 1),
    ('idle-native-host-cache-release-v2', 'idle-native-host-cache-release-AppWorld-critic2-v1', 3),
    ("'TextCraft'", "'AppWorld'", 2),
    ('3823147', '3823264', 1), ('1792200692.49', '1792200693.87', 1),
    ('3874156', '3843629', 2), ('1792201135.09', '1792200848.3', 1),
    ('3876616', '3845945', 2), ('1792201155.53', '1792200868.59', 1),
]
for old, new, count in replacements:
    assert code.count(old) == count, old
    code = code.replace(old, new)
anchor = '    before_roles = roles()\n'
assert code.count(anchor) == 1
code = code.replace(anchor, anchor +
    '    assert {r["actor_scheduler_last_epoch"] for r in before_roles if "actor_scheduler_last_epoch" in r} == {1}, "Natural actor1/critic2 measurement window passed; no resource action"\n'
    '    assert {r["critic_scheduler_last_epoch"] for r in before_roles if "critic_scheduler_last_epoch" in r} == {2}, "Natural critic2 return is not verified; no resource action"\n')
exec(compile(code, str(legacy), 'exec'), globals())
