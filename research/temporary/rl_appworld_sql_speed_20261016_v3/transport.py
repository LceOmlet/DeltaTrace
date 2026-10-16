"""Use the existing strict transport with this version's source/output directory."""
from pathlib import Path

owner = Path(__file__).resolve().parent.parent / 'rl_appworld_sql_speed_20261016_v2/transport.py'
code = owner.read_text(encoding='utf-8').replace('appworld-sql-native-speed-20261016-v2', 'appworld-sql-native-speed-20261016-v3')
exec(compile(code, str(owner), 'exec'), globals())
