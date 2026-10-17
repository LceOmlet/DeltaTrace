"""Reuse the pinned original collectors, parsers and six-panel renderer for v6."""
import argparse
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=('TextCraft', 'AppWorld', 'both'), default='both')
requested_task = parser.parse_args().task
PARENT = Path(__file__).resolve().parent.parent

if requested_task in ('TextCraft', 'both'):
    source = PARENT / 'rl_textcraft_weighted_reward_targets_20261015_v4/plot_current_algorithms_v4.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '78a85a43a5e0d938e6143e4c03b32b79c7f8d10e343df73650f6053e0dd4ba87'
    code = source.read_text()
    lines = code.splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith('new="snapshot[\'DT_display_label\']')]
    assert len(matches) == 1
    label = "assert current_DT['version']=='textcraft-dt-reward-target-exponential-variance-20261017-v6'; snapshot['DT_display_label']='DTPO v6（指数信用仅方差，官方组优势）'"
    lines[matches[0]] = 'new=' + repr(label)
    code = '\n'.join(lines) + '\n'
    code = code.replace('DTPO v4尚无完成迭代，共同窗口为空', 'DTPO v6尚无完成迭代，共同窗口为空')
    exec(compile(code, str(source), 'exec'), globals())

if requested_task in ('AppWorld', 'both'):
    source = PARENT / 'rl_textcraft_gradient_rootcause_20261013_v1/plot_current_approved_DTPO_v63.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'f6ed1ff351b9f8ea7980577d4897acd59e953bc105defe47fe4cb3e700d46244'
    code = source.read_text()
    code = code.replace("j.get('task')=='TextCraft' and j.get('method') in ('dt','grpo')",
                        "j.get('task')=='AppWorld' and j.get('method') in ('dt','ppo')")
    code = code.replace("{'dt','grpo'}", "{'dt','ppo'}")
    start = code.index('job_path=OUT/\'native-PPO-control-v9-job.json\'')
    end = code.index("snapshot['source_label']=", start)
    # AppWorld PPO is already selected by the same original collector; it is
    # a fresh native run, not the historical TextCraft checkpoint resume chain.
    code = code[:start] + code[end:]
    old = "snapshot['DT_display_label']=('DTPO v3（组乘子，仅方差）' if current_DT.get('variance_standardization')\n                               else '新 DTPO（组乘子，无白化）')"
    assert old in code
    code = code.replace(old, "assert current_DT['version']=='appworld-dtpo-reward-target-exponential-variance-formal-20261017-v6'; snapshot['DT_display_label']='DTPO v6（指数信用仅方差，官方组优势）'")
    code = code.replace("assert last['dt']>0,'Current DTPO has no completed native iteration yet; source snapshot retained.'", '# Empty fresh native histories retain missing values.')
    legend_and_missing = ("        if ax.lines:ax.legend(frameon=False,fontsize=8)\n"
                          "        else:ax.text(.5,.5,'原日志无此指标',ha='center',va='center',transform=ax.transAxes)")
    assert code.count(legend_and_missing) == 1
    code = code.replace(legend_and_missing,
                        "        has_metric_data=bool(ax.lines)\n"
                        "        if limit is None:\n"
                        "            for empty_job in selected:\n"
                        "                if last[empty_job['method']]==0:\n"
                        "                    style=styles[empty_job['method']]\n"
                        "                    ax.plot([],[],color=style['color'],linestyle=style['linestyle'],label=style['label']+'（无完成迭代）')\n"
                        "        if ax.lines:ax.legend(frameon=False,fontsize=8)\n"
                        "        if not has_metric_data:\n"
                        "            if limit is not None and common==0:\n"
                        "                missing_label='尚无共同完成迭代；指标缺失'\n"
                        "            elif not any(last[j['method']]>0 for j in selected):\n"
                        "                missing_label='尚无已完成迭代；指标缺失'\n"
                        "            else:\n"
                        "                missing_label='原日志未记录此指标'\n"
                        "            ax.text(.5,.5,missing_label,ha='center',va='center',transform=ax.transAxes)")
    code = code.replace('TextCraft ·', 'AppWorld ·').replace('/330', '/200')
    code = code.replace('工具调用数沿用原 TextCraft 交互计数', '工具调用数沿用原 AppWorld 交互计数')
    code = code.replace('; original GRPO/PPO controls', '; original native PPO control')
    exec(compile(code, str(source), 'exec'), globals())
