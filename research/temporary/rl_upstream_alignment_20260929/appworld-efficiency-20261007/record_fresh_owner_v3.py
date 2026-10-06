"""Record observed fresh runtime identity, preserving prior verification scopes."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = HERE / 'native-conv-canonical-owner-v3'


def read(name):
    return json.loads((SOURCE / name).read_bytes())


def ref(path):
    return dict(path=path.relative_to(REPO).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    submission=read('submission.json')
    job=submission['job']
    snapshot=read('current_runtime.json')
    imports=read('actual-worker-imports.json')
    launch=read('launch.json')
    assert job['pid']==imports['driver_pid']==1953903
    assert job['resume_mode']=='disable' and not job['checkpoint_restore_requested']
    assert job['lora_rank']==8 and job['lora_alpha']==16 and job['actor_microbatch']==4
    assert all(row['lora_rank']==8 and row['lora_alpha']==16
               and row['actor_microbatch']==4 and row['ppo_epochs']==2
               and row['entropy_coeff']==.001 and row['clip_ratio_c']==3
               for row in imports['workers'])
    # The new source receipt changes output/provenance identity, not any
    # owner source bytes; retain both parent and current receipt hashes.
    result_path=REPO/'experiments/rl/results_appworld_efficiency_20261007.json'
    result=json.loads(result_path.read_bytes())
    result['status']='fresh_v3_official_owner_sampling_started_no_completed_iteration_yet'
    result['fresh_job']=job
    result['fresh_formal']=submission
    result['native_conv_actual_worker_imports']=imports
    result['latest_observed_runtime']=ref(SOURCE/'current_runtime.json')
    result['fresh_v3_launch_identity_comparison']=ref(SOURCE/'launch-identity-comparison.json')
    result['fresh_v3_startup_observation']=ref(SOURCE/'startup-observation.json')
    result['fresh_v3_scope']='Same frozen source3cd/preparedf523 via original launcher, output identity source97e3; rank8/alpha16/B4/official budget/PPO unchanged. Official whole-collected-batch whitening remains in imported trainer. No old checkpoint or new candidate is loaded; initialization/sampling is not completed-update or learning-recovery evidence.'
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    current=REPO/'experiments/rl/current_runtime.json'
    previous=json.loads(current.read_bytes())
    assert snapshot['observed_unix']>previous['observed_unix']
    current.write_bytes((SOURCE/'current_runtime.json').read_bytes())
    ledger_path=REPO/'experiments/rl/RUNTIME_RECORD.md'
    ledger=ledger_path.read_bytes().decode('utf8')
    title='## 2026-10-07 05:23 AppWorld原冻结owner fresh v3已提交，采样开始'
    text='''
## 2026-10-07 05:23 AppWorld原冻结owner fresh v3已提交，采样开始

旧v2 actor中途退出保留在下方。复用原launch_appworld_native.py1223c007与原env
beb9c001，从原基础权重fresh启动；没有--resume-from，原resume_mode=disable。
新driver1953903/birth1791321793.72，GPU4/5；worker1959905/1961414。
运行目录/runs/appworld-fresh-native-conv-canonical-20261007-v3/appworld-dt。
新source97e3cb754f505b79a52d3dc3464b9027ae7bdc38e7074b866f80dc7624be3481；
继承原冻结3cd90b2d与preparedf5232ac1，95entry/396VERL/55LOOP/105DT文件逐项
SHA无差异。新source只变运行身份与来源记录，不是新的数值算法版本。
提交5155c9b3保留原guard，旧一次性v2脚本未原样重跑，不重prepare、不改参数。
本次完整launch对照v2，只有4个output/visibility目录字段变化。

实际原worker RPC已核LoRA8/alpha16、actor micro4、PPO epochs2、entropy.001、
dualclip3、sharedpadding1、canonical HF59f9、actor3a65/fsdp e5eb/torchfunction079a；
原trainer d35ddd26/373行整批masked_whiten保持。v2该白化实际已执行；本次
仍在采样，尚未执行DT/actor更新，不能冒称本次梯度或学习恢复。DT/vLLM lazy
模块没有出现在此时worker sys.modules，未将其冒称已执行的真实DT导入证据。
预算仍200×40groups×6、mini32、epoch2、40turns、32runners/rank、train32000/
上限32768，没改rank/alpha/B4，也没接storage或row-cut候选。

05:27原vLLM已初始化、LOOP任务采样服务开始；物理GPU4/5分别48501/48485MiB。
容器总122789081088B（114.36GiB），anonymous约89.98GiB；未相加fork RSS。
没有完整rollout/DT/actor时间、梯度或新检查点；进程存活不是训练健康证明。
不再live attach mcTracer；候选继续在隔离目录做原操作数、官方断言与表示对照。
本机snapshot更新为原只读collector独立输出e094acb7；collector33e1705c字节未改，
仅内存重定向保存目的地，IO重定向收据可查。来源都在
research/temporary/rl_upstream_alignment_20260929/appworld-efficiency-20261007/
native-conv-canonical-owner-v3/，旧snapshot/失败日志仍保留，不当新运行事实。

'''
    assert title not in ledger
    end=ledger.index('\n')+1
    ledger_path.write_bytes((ledger[:end]+text+ledger[end:]).encode('utf8'))
    print(json.dumps(dict(source=submission['source_sha256'],current_snapshot=ref(current),
                         result=ref(result_path),driver=job['pid']),ensure_ascii=False))


if __name__=='__main__':
    main()
