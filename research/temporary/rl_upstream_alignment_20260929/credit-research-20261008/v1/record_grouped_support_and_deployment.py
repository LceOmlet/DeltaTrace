"""Record completed grouped diagnostics and the separately accepted dtype fix."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def ref(path):
    raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    version=HERE/'accepted-fla-seed-range-v1'
    deployment=json.loads((version/'deployed.json').read_bytes())
    snapshot=json.loads((version/'post-deploy-snapshot.json').read_bytes())
    assert all(x['unchanged'] for x in snapshot['unchanged_numerical_files'])
    assert 'no process found' in snapshot['physical']
    assert ref(REPO/'deltatrace/clean/qwen35/qwen35_gdn_finite.py')['sha256']==deployment['owners'][0]['new_sha256']
    old=REPO/'experiments/rl/results_fla_range_owner_20261009.json'
    tests=json.loads(old.read_bytes())
    result=dict(scope='User-authorized promotion of the already verified FP16 conversion-range repair',
        observed_unix=time.time(),version=deployment['version'],source_commit=deployment['source_commit'],
        deployment=deployment,post_deploy_snapshot=ref(version/'post-deploy-snapshot.json'),
        original_verification_receipt=ref(old),
        official_FLA_checks=tests['official_checks'],
        version_mapping={row['task']:dict(base_sha256=row['original_sha256'],
            fixed_sha256=row['new_sha256'],active_path=row['path'],versioned_path=row['versioned_path'],
            actual_import=row['fresh_import'],historical_owner_unchanged=row['historical_owner_unchanged'])
            for row in deployment['owners']},
        unchanged_numerical_files=snapshot['unchanged_numerical_files'],
        local_default_source=ref(REPO/'deltatrace/clean/qwen35/qwen35_gdn_finite.py'),
        integration_scope='Actual fresh imports in original environments, exact three verified source edits and unchanged signatures. Original AppWorld failing B4 replay and official 22 checks are reused. No new whole TextCraft DT replay or whole-model tolerance is claimed.',
        deployment_verification_failures=[dict(stage='Initial probe omitted original Python import environment; TextCraft leaf was changed, AppWorld was not; partial deployment was not reported complete',
            stderr=ref(version/'failed-import-v1-deploy.stderr')),
            dict(stage='Second probe restored task environment but omitted provisioned shell MACA runtime. Fixed by sourcing the existing entry env, without reinstall.',
            stderr=ref(version/'failed-import-v2-deploy.stderr'))],
        formal_training_restarted=False,original_PPO_NaN_fixed=False,extreme_credit_fixed=False,
        QVA_whitening_PPO_changed=False,official_tolerance_changed=False)
    output=REPO/'experiments/rl/results_fla_seed_range_deployment_20261009.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    stats=REPO/'experiments/rl/results_grouped_negative_support_20261009.json'
    support=REPO/'experiments/rl/results_saved_FA_source_support_20261009.json'
    regions=REPO/'experiments/rl/results_saved_FA_PV_regions_20261009.json'
    current_path=REPO/'experiments/rl/current_runtime.json'
    current=json.loads(current_path.read_bytes())
    now=time.time();stamp=datetime.fromtimestamp(now,timezone.utc).isoformat()
    current.update(observed_unix=now,observed_utc=stamp,
        latest_readonly_observation=dict(textcraft='Formal training stopped',appworld='Formal training stopped',
            diagnostic='All 165 source-support and original-FA key-region observations completed; extreme credit not repaired',
            deployment='Accepted FP16 range fix promoted to both active owner seams and local default',receipt=ref(output)))
    current['accepted_fla_seed_range_20261009']=dict(receipt=ref(output),version=deployment['version'],
        source_commit=deployment['source_commit'],version_mapping=result['version_mapping'],
        verification='Original FLA 22 checks and AppWorld B4 replay; exact TextCraft edit transfer and fresh imports, not new TextCraft full replay',
        formal_restart=False,extreme_credit_repaired=False)
    current['latest_grouped_negative_support_20261009']=dict(receipt=ref(stats),
        source_support=ref(support),PV_key_regions=ref(regions),
        study='Frozen original cohorts, state-equal bounded frequencies with exploratory state-cluster intervals; raw tail moments not pooled',
        model_calls=0,DT_calls=0,original_public_FA_calls=480,production_credit_changed=False)
    current['latest_fla_range_owner_20261009']['historical_before_promotion']=True
    current['latest_fla_range_owner_20261009']['current_deployment']='See accepted_fla_seed_range_20261009'
    for job in current['jobs']:
        task=job.get('task','').lower()
        if task not in result['version_mapping']:continue
        job['accepted_stopped_owner_overlay']=dict(version=deployment['version'],
            source_commit=deployment['source_commit'],**result['version_mapping'][task],
            startup_source_json_unchanged=True,training_restarted=False)
    for metadata in snapshot['metadata']:
        p=Path(metadata['path']);local=version/'runtime-metadata'/p.name
        assert ref(local)['sha256']==metadata['sha256']
        value=dict(**metadata,local_path=str(local))
        if p.name in current['authoritative_remote']:current['authoritative_remote'][p.name]=value
        if p.name=='verified_runtime.json':current['baseline']=value
    current_path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    note='''
## 2026-10-09 分组负尾仍不可信；已接受的FP16溢出修复全面固定部署

FP16转换修复版本为fla-seed-range-20261009-v1，部署源码c1da84ea；本机默认及
两实际DT入口均已更新。TextCraft原ef55ce08→bc1a11d9，AppWorld原448ef32c→
33b169b3；只复用已经验证的三处数值表示改动，不交换两环境的owner。AppWorld
consume_captures保留，两个函数签名均不变。两独立新进程通过原启动环境与已有
metax-entry.env.sh读取实际文件；12项runner/head/FA等文件逐SHA不变。原已解析
历史owner文件和停止作业的source.json未覆盖，只有当前叶symlink指向新version。
根active-training/active-source和原verified_runtime均另记accepted overlay。

验收沿用原FLA22项实际dtype断言、原失败AppWorld B4两rank回放和31个未溢出
head逐值一致证据。TextCraft同三处diff逐字复用、逆补丁恢复原base，接口/导入已
核验；不声称新增TextCraft整网回放或全网官方容差。两个导入探针环境遗漏的失败
均保存：首次不全的PYTHONPATH只改变了TextCraft叶，第二次漏source既有MACA
shell环境；未重装依赖，完成状态只在最终两入口确认后发布。此修复不代表极端
归因或原PPO NaN修复，两个正式任务仍停止，无恢复/训练更新。

分组诊断复用冻结的两个环境各165位置，主体均匀128与原预测负尾37分开。
TextCraft的37负尾中9个在全部对照中为负、18个非负、10个跨零；只有1个的
A/r<-1由全部对照支撑，35个被全部对照排除该幅度。AppWorld的37位置中为负/
非负/跨零为21/13/3，支撑A/r<-1为6个、排除为23个；其中只有34个在DT重复中
保持大负，稳健反号11个。初始状态等权稳健反号率37.86%/26.56%，探索性状态
bootstrap95%区间21.19–53.33%/8.85–46.88%；不当作全任务错误率或数值容差。
原分层及已查看/未查看身份保留，TextCraft含最新PV运行的重复对照。

CPU原操作数检查165位置/60文件，44.63秒，采样PSS峰0.575GB，零模型/DT/GPU。
原FA分key区域检查165位置/60文件，104.11秒，480原public FA，零模型/DT/更新；
allocated/reserved峰1.186/3.064GB，采样PSS峰9.438GB。18稳健误报中15个最大
PV背景余项在source key，状态等权频率.7361；该区域绝对中位2.1055。全165
单删除均影响后续隐藏行，18稳健误报的source之前Q/K/V逐值相同。支持进一步
查source条件背景与多行交互，未接受新候选、未把张量能量当信用或扣除余项。
原FA完整PV重放差≤1.78e-15；分区域BF16闭合差另存，不是新容差或修复。

见results_grouped_negative_support_20261009.json、results_saved_FA_source_support_20261009.json、
results_saved_FA_PV_regions_20261009.json、results_fla_seed_range_deployment_20261009.json。
'''
    record=REPO/'experiments/rl/RUNTIME_RECORD.md';raw=record.read_bytes()
    marker='# 当前运行版本与修复记录'.encode();assert raw.count(marker)==1
    end=raw.index(b'\n',raw.index(marker))+1
    record.write_bytes(raw[:end]+note.encode('utf8')+raw[end:])
    print(json.dumps(dict(receipt=ref(output),runtime=ref(current_path),grouped_support=ref(stats))))


if __name__=='__main__':main()
