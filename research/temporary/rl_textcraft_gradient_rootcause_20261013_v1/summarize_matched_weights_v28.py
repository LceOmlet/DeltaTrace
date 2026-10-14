"""Account the completed matched input/weight interaction, preserving scope."""
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
status=json.loads((HERE/'matched-history-late-weights-v28-status-final.json').read_bytes())['report']
assert status['job']['pid']==225395 and not status['alive']
assert status['state']['phase']=='complete_native_gradient_chain_localization_not_repair'
old=json.loads((HERE/'native-gradient-ratio-chain-summary-v23.json').read_bytes())
score5=json.loads((HERE/'account_native_target_scores_v27.json').read_bytes())['report']
score80=json.loads((HERE/'account_native_target_scores_v27-dt.json').read_bytes())['report']
cases=[]
for case in ('early','late'):
    chains=[r['value'] for r in status['chain_reports'] if r['value']['iteration']==case]
    assert len(chains)==2
    sums=defaultdict(lambda:defaultdict(float))
    checks=[]
    for rank in chains:
        assert rank['trainable_SHA256_before']==rank['trainable_SHA256_after']
        assert rank['scheduler_before']==rank['scheduler_after']==80
        counts=Counter((r['boundary'],r['micro']) for r in rank['records'])
        assert set(counts.values())=={1}
        for r in rank['records']:
            name=r['boundary'].split('base_model.model.')[-1]
            for which in ('input','output'):
                assert r.get(which+'_adjoint_nonfinite_positions',0)==0
                if which+'_adjoint_norm' in r:sums[name][which]+=r[which+'_adjoint_norm']**2
        pg=json.loads((HERE/'matched-history-late-weights-v28'/f'iteration{case}-rank{rank["rank"]}-PG.json').read_bytes())
        rows=pg['passes']['native_target']['microbatch_losses']
        assert len(rows)==8
        assert all(r['ppo_kl']==0 and r['pg_clipfrac']==0 for r in rows)
        checks.append(dict(rank=rank['rank'],native_microbatches=8,ppo_kl_all_zero=True,
            clipping_all_zero=True,native_clip_return=pg['passes']['native_target']['optimizer_boundary']['native_clip_return']))
    boundaries={name:{k:math.sqrt(v)/2 for k,v in row.items()} for name,row in sums.items()}
    observation=next(r for r in status['state']['observations'] if r['iteration']==case)
    G=observation['gradient_statistics']['norms']['native_target']
    H=boundaries['model.norm']['output']
    cases.append(dict(case=case,checkpoint=80,native_parameter_gradient_norm=G,
        final_hidden_adjoint_norm=H,G_over_H=G/H,boundaries=boundaries,
        unchanged_state=chains,forward_point_checks=checks))
a,b=cases
history_ratio=b['native_parameter_gradient_norm']/a['native_parameter_gradient_norm']
table=[]
for cp,ss,gs in [(5,score5,old['cases']),(80,score80,cases)]:
    for s,g in zip(ss['cases'],gs):
        table.append(dict(checkpoint=cp,history=s['case'],
            G=g.get('native_parameter_gradient_norm',g.get('native_global_sharded_parameter_norm')),
            H=g.get('final_hidden_adjoint_norm',g.get('native_global_final_hidden_adjoint_norm')),
            G_over_H=g['G_over_H'],score_norm_bounds=s['score_norm_bounds'],
            native_target_NLL=s['target_NLL'],coefficient_L2=s['coefficient_vector_L2']))
now=datetime.now().astimezone().isoformat()
report=dict(observed_local=now,version='matched-history-late-weights-v28',
    status='completed_native_matched_weight_interaction_not_repair',goal_status='active',job=status['job'],
    actual_worker_bindings=status['state']['bindings'],cases=cases,four_corner_table=table,
    history_gradient_ratio_at_checkpoint80=history_ratio,
    history_hidden_adjoint_ratio_at_checkpoint80=b['final_hidden_adjoint_norm']/a['final_hidden_adjoint_norm'],
    history_G_over_H_ratio_at_checkpoint80=b['G_over_H']/a['G_over_H'],
    history_score_norm_ratio_bounds_at_checkpoint80=score80['late_over_early_score_norm_bounds'],
    weight_parameter_gradient_ratios={case:table[2+i]['G']/table[i]['G'] for i,case in enumerate(('early','late'))},
    weight_G_over_H_ratios={case:table[2+i]['G_over_H']/table[i]['G_over_H'] for i,case in enumerate(('early','late'))},
    boundary_history_ratios_at_checkpoint80={name:{k:b['boundaries'][name][k]/v for k,v in row.items() if v} for name,row in a['boundaries'].items()},
    conclusions=[
        'At the original late weights, the same15 matched histories/common coefficients have parameter-gradient norms0.254532 versus1.105569, ratio4.34354, while their output-score norm bounds nearly overlap.',
        'At checkpoint80 final-hidden adjoint is slightly smaller for late histories, yet G/H is substantially larger. Internal propagation/parameter projection, rather than a larger output-score norm alone, explains this finite difference.',
        'Changing weights also changes the early-history gradient greatly. The older fixed-late-input1.35x contrast did not establish a small weight effect for other inputs; input/weight interaction is material.',
        'The checkpoint5 28.27x result remains valid within its original scope. It cannot be treated as a weight-independent shortening factor or the actual late-policy probability contrast.',
        'No demonstrated mismatch in DT Q/V/A transport or original whitening has yet explained the growing parameter-gradient sensitivity. No repair is accepted or deployed.'],
    limits=[
        'The same15 saved pairs are a bounded factual contrast, not a population-wide causal estimate or complete historical optimizer batch.',
        'G/H is an observed directional quotient combining propagation, projection and cancellation, not an operator norm or independent causal share.',
        'The checkpoint5 instrumented result28.95868 differs from the earlier uninstrumented28.27402; that measured variation remains recorded.',
        'Score bounds use native rounded probabilities and do not measure the full vocabulary-logit adjoint exactly.',
        'Both fixed-weight contrasts include one off-policy history set. They do not predict fresh training or identify the entire85-iteration root cause.'],
    optimizer_updates=0,scheduler_updates=0,DT_calls=0,rollout_calls=0,validation_calls=0,GRPO_calls=0,
    rootcause_resolved=False,accepted_repair=None,deployed_repair=None)
path=HERE/'matched-weight-interaction-summary-v28.json'
path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
entry=dict(observed_local=now,version=report['version'],status=report['status'],job=report['job'],
    four_corner_table=table,history_gradient_ratio_at_checkpoint80=history_ratio,
    weight_G_over_H_ratios=report['weight_G_over_H_ratios'],record=str(path),rootcause_resolved=False,accepted_repair=None,deployed_repair=None)
for name in ('current_runtime.json','results_textcraft_gradient_rootcause_20261013_v1.json'):
    p=ROOT/'experiments/rl'/name;value=json.loads(p.read_text(encoding='utf-8'))
    p.write_text(json.dumps({'20261014_matched_late_weights_v28_complete':entry,**value},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
p=ROOT/'experiments/rl/RUNTIME_RECORD.md'
p.write_text(f'''## {now} v28相同事实/系数的晚期权重对照完成，旧28倍解释收紧

225395/1791965846.14已完成退出；实际worker229037/1791965872.83、230740/1791965887.54，原actor3a65e173/corefc2f992b及checkpoint80/scheduler80/trainable摘要保持。源fb83514f/harness2d31b570。原输入/同15对事实/target系数不变，两侧各原8个B4的PPO KL、clipfrac全0；未执行optimizer/scheduler/DT/生成/验证/GRPO更新。

checkpoint80 early G0.254532/H0.000718323/GH354.342；late G1.105569/H0.000658709/GH1678.387，参数范数倍率{history_ratio:.6f}、hidden伴随量倍率约0.917、GH倍率约4.737。native输出score范数界early[0.000552020,0.000780674]、late[0.000562462,0.000795442]几乎重叠，倍率界[0.720483,1.440965]；target NLL6.04695/10.21582。此晚期差异主要定位到内部传播/投影/抵消，不能再用旧权重上低Action概率独自解释。最后decoder31输出伴随量late/early约1.219、输入约4.037，具体分支尚未分离，不据此声称该block有bug。

与v23固定checkpoint5相比，early G上升约22.26倍、late约3.338倍；GH分别上升约5.18和9.21倍。旧固定late输入的1.35倍结果不能推出所有输入上的权重影响都小，必须保留输入×权重交互。原28.274倍仍是旧权重/原输入下的有限对照，不能当成与权重无关的shortening固定倍率或85轮根因。原DT改编链Q/V/A运输及整批白化未见已证实对应算错，不调参、不改信用质量。旧DTPO/分母候选停止、GRPO不重做、小时检查PAUSED、goal active，accepted/deployed均无。回执{path}。

'''+p.read_text(encoding='utf-8'),encoding='utf-8')
p=ROOT/'experiments/rl/REMOTE_ENVIRONMENT.md'
p.write_text(f'## {now} v28完成退出，0/1无本诊断占用\n\n225395/1791965846.14已完成退出，原PPO2/3及GRPO4/5保持。源fb83514f、原owner及环境/缓存不变，无训练更新；后续设备操作仍重查mx-smi。小时检查PAUSED、goal active；v28输入×权重交互的实际数据及限制见RUNTIME_RECORD顶部。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps(dict(record=str(path),table=table,history_ratio_at_checkpoint80=history_ratio)))
