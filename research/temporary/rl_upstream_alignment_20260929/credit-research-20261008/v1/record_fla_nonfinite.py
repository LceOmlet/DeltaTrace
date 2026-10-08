"""Bind terminal diagnostics and preserved exact operands to actual source IDs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def ref(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def latest(folder):
    path=max((p for p in folder.glob('*.json') if p.stem.isdecimal()),key=lambda p:int(p.stem))
    return path,json.loads(path.read_bytes())


def main():
    result=dict(scope=__doc__,production_modified=False,formal_restart=False,credit_repaired=False,
        original_PPO_NaN_fixed=False,QVA_modified=False,official_tolerance_claim=False,diagnostics={})
    for revision in ('v1','v2','v3'):
        label='single-background-nonfinite-appworld'+('' if revision=='v1' else '-'+revision)
        launch=HERE/(label+'-launch.json')
        observation,snapshot=latest(HERE/(label+'-observations'))
        preserved=HERE/('single-background-nonfinite-appworld-'+revision)/'preserved-artifacts/manifest.json'
        ranks=[]
        for rank in (0,1):
            value=snapshot['files'].get(f'results/rank{rank}.json',{}).get('value',{})
            identity=snapshot['files'].get(f'results/rank{rank}-first-nonfinite.json',{})
            seed=snapshot['files'].get(f'results/rank{rank}-precast-seed.json',{})
            ranks.append(dict(rank=rank,phase=value.get('phase'),pid=value.get('pid'),birth=value.get('birth'),
                elapsed=value.get('elapsed_seconds'),operations=value.get('operations'),
                first_nonfinite=identity,precast_seed=seed))
        result['diagnostics'][revision]=dict(launch=ref(launch),actual_launch=json.loads(launch.read_bytes()),
            observation=ref(observation),observed_unix=snapshot['unix'],driver_alive=snapshot.get('driver_alive'),
            ranks=ranks,preservation=ref(preserved),physical=snapshot.get('physical'))
    result['input_analysis']=ref(HERE/'single-background-nonfinite-appworld-v2-inputs.json')
    result['SiLU_analysis']=ref(HERE/'single-background-nonfinite-appworld-cpu.json')
    result['actual_profiles']=ref(HERE/'single-background-nonfinite-actual-profiles.json')
    result['finding']='Original FP32 norm-gate callback is finite. Original BF16->FP16 conversion introduces one -Inf in the FLA seed; all saved native endpoints are finite. This contaminates the original FLA coefficients and subsequently conv-SiLU. This numerical defect is distinct from joint-reference attribution error.'
    folder=HERE/'fla-seed-range-observations'
    if folder.exists():
        path,snapshot=latest(folder)
        result['range_diagnostic']=dict(observation=ref(path),driver_alive=snapshot.get('driver_alive'),
            result=snapshot.get('files',{}).get('result.json'),
            launch=ref(HERE/'fla-seed-range-launch.json'),accepted=False,production_modified=False)
    folder=HERE/'fla-seed-range-v2-observations'
    if folder.exists():
        path,snapshot=latest(folder)
        result['minimal_range_diagnostic']=dict(observation=ref(path),driver_alive=snapshot.get('driver_alive'),
            result=snapshot.get('files',{}).get('result.json'),
            launch=ref(HERE/'fla-seed-range-v2-launch.json'),
            unmodified_heads=ref(HERE/'fla-seed-range-unchanged-heads.json'),
            exact_sources=ref(HERE/'fla-seed-range-preserved/manifest.json'),accepted=False,production_modified=False)
    result['prepared_owner_patch']=ref(HERE/'native-fla-range-owner-prepared/prepared.json')
    output=REPO/'experiments/rl/results_fla_seed_nonfinite_20261009.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=REPO/'experiments/rl/current_runtime.json';current=json.loads(path.read_bytes())
    current['latest_fla_seed_nonfinite_20261009']=dict(receipt=ref(output),
        status='localized_native_FP16_seed_overflow_not_fixed_in_production',
        production_modified=False,formal_restart=False,original_PPO_NaN_fixed=False)
    current['latest_single_background_nonfinite_replay_20261009']['status']='terminal_preserved'
    current['latest_readonly_observation']['diagnostic']='Extreme attribution investigation: joint-reference mismatch and separate native FP16 seed overflow; formal jobs stopped.'
    current['latest_readonly_observation']['receipt']=ref(output)
    if 'minimal_range_diagnostic' in result:
        _,physical_snapshot=latest(HERE/'fla-seed-range-v2-observations')
        current['observed_unix']=physical_snapshot['unix']
        current['observed_utc']=datetime.fromtimestamp(physical_snapshot['unix'],timezone.utc).isoformat()
    path.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(receipt=str(output),status='localized_and_preserved',credit_repaired=False)))


if __name__=='__main__':main()
