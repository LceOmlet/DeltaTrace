"""Display v7 queue identities using the pinned original parser and renderer."""
import hashlib
from pathlib import Path
source=Path(__file__).resolve().parent.parent/'rl_six_run_fresh_after_efficiency_20261017_v1/plot_both_tasks.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='d734380585f633284e6cfbe847e383e89862fb41083c59a3e37ce3d4769f026c'
code=source.read_text().replace("task+'-eight-gpu-sequential-20261017-v2'","task+'-eight-gpu-symmetric-probability-ratio-20261018-v7'")
# The intermediate owner generates the unchanged renderer source; replace only its label.
anchor="        exec(compile(code, str(source), 'exec'), globals())"
assert code.count(anchor)==1
code=code.replace(anchor,"        code=code.replace('DTPO v6（8卡）', 'DTPO v7 对称概率比（8卡）')\n"+anchor)
exec(compile(code,str(source),'exec'),globals())
