"""Use the pinned six-panel renderer; only identities and verified norm labels change."""
import hashlib
from pathlib import Path

source=Path(__file__).resolve().parent.parent/'rl_six_run_fresh_after_efficiency_20261017_v1/plot_both_tasks.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='d734380585f633284e6cfbe847e383e89862fb41083c59a3e37ce3d4769f026c'
code=source.read_text().replace("task+'-eight-gpu-sequential-20261017-v2'",
    "task+'-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8'")
anchor="        exec(compile(code, str(source), 'exec'), globals())"
assert code.count(anchor)==1
# The intermediate owner builds the renderer after its remote collector. Insert
# presentation changes at that existing boundary, without another parser/logger.
presentation = '''
code=code.replace('首 rank 分片梯度范数（裁剪前）','全局梯度范数（裁剪前）')
code=code.replace('梯度为首 rank 分片范数的三位小数日志，全局范数未补估。',
    '梯度为原生 DTensor full_tensor 实测全局裁剪前范数；原分片值另存 CSV，不与历史分片口径混比。')
code=code.replace("['algorithm','iteration','metric','value','original_sources']",
    "['algorithm','iteration','metric','value','original_sources','norm_scope']")
code=code.replace("json.dumps(m['sources'],ensure_ascii=False)])",
    "json.dumps(m['sources'],ensure_ascii=False), 'global_preclip_native_DTensor_full_tensor' if key=='actor/grad_norm' else ('first_returned_rank_local_preclip' if key=='actor/grad_norm_local' else '')])")
'''
injection = "        code=code.replace('DTPO v6（8卡）', 'DTPO v8 对称比·std(g·u)（8卡）')\n"
injection += "        code=code.replace('Original first-returned rank preclip shard norm; no missing global norm estimated', 'Native DTensor.full_tensor global preclip norm; actor/grad_norm_local retains original first-returned shard; historical shards not converted')\n"
injection += '        code=code.replace("exec(compile(code, str(source), \'exec\'), globals())", '+repr(presentation)+"+\"exec(compile(code, str(source), 'exec'), globals())\")\n"
code=code.replace(anchor,injection+anchor)
exec(compile(code,str(source),'exec'),globals())
