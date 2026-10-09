"""Read completed native target records through the existing packing/row owners.

CPU interface and saved-score audit only. Does not recompute model scores,
single-deletion truth, or introduce a numerical acceptance threshold.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import psutil
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
FORMAL=ROOT/'runs/textcraft-formal-stable-20261009-v1'
SOURCE_SHA='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.perf_counter()
    assert digest(FORMAL/'source.json')==SOURCE_SHA
    manifest=json.loads((FORMAL/'source.json').read_bytes())
    sys.path.insert(0,str(Path(manifest['dt_root'])/'clean/qwen35'))
    packing=importlib.import_module('qwen35_answer_finite')
    selector_owner=importlib.import_module('native_target_logit_rows')
    assert digest(packing.__file__)=='d47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e'
    assert digest(selector_owner.__file__)=='ebb5345e7c344168fae5d0014da16c2c30a07041842e2204d9d0c69a7af7c80d'
    install=json.loads((FORMAL/'runtime-overrides/native-target-records-20261010-v2.json').read_bytes())
    assert install['complete']
    files={};observations=[]
    logroot=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs')
    marker='[DT direct joint record] '
    for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
        process=psutil.Process(pid);assert process.create_time()==birth
        log=next(logroot.glob('*-'+str(pid)+'.out'))
        for line in log.read_text(errors='replace').splitlines():
            if marker not in line:continue
            record=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
            path=Path(record['path'])
            # The owner's completed-save callback supplies the exact artifact path.
            if int(path.stem.split('-')[1]) <= int(install['completed_unix']*1e9):continue
            assert path.is_relative_to(FORMAL/'credit-records')
            files[str(path)]=record
        observations.append(dict(pid=pid,birth=birth,phase=process.name(),log=str(log)))
    input_kind='completed_formal_owner_save_callbacks'
    if len(sys.argv)>2:
        path=Path(sys.argv[2])
        files={str(path):dict(bytes=path.stat().st_size,seconds=None)}
        input_kind='explicit_existing_native_serialization_artifact_not_formal_cohort'
    batches=[];rows=[]
    for filename,callback in sorted(files.items()):
        path=Path(filename);assert path.stat().st_size==callback['bytes']
        stored=torch.load(path,map_location='cpu',weights_only=True)
        native=stored['native_target_diagnostics'];batch=stored['rows']
        assert digest(stored['readout_source'])=='f0fbae610f1a8c22b65c6486cefb73eb5ff04c154178e0f822e679b600fa33bc'
        selected=packing.PackedAnswerTargets(
            [dict(prompt_length=row['prompt_length'],target_ids=row['input_ids'][row['prompt_length']:]) for row in batch],
            [row['target_offsets'] for row in batch],max(row['input_ids'].numel() for row in batch),'cpu')
        original=selected
        if native['native_row_prefix_lengths'] is not None:
            selected=selected.suffix_rows(native['native_row_prefix_lengths'],native['native_row_context_lengths'])
        elif native['native_shared_prefix_length']:
            selected=selected.suffix(native['native_shared_prefix_length'])
        selector=selector_owner.NativeTargetLogitRows(selected)
        reference=torch.as_tensor(native['target_logp0'],dtype=torch.float64)
        factual=torch.as_tensor(native['target_logp1'],dtype=torch.float64)
        ref_sum=selected.sample_sums(reference);fact_sum=selected.sample_sums(factual)
        checks=dict(actual_predictor_rows_equal_original_owner=selector.rows.tolist()==native['selected_predictor_rows'],
                    rebasing_preserves_target_labels=torch.equal(original.labels,selected.labels),
                    rebasing_preserves_endpoint_samples=torch.equal(original.paired_samples,selected.paired_samples),
                    native_head_batch_rows_match=len(native['actual_head_input_shapes'])==1 and native['actual_head_input_shapes'][0][:2]==[2*len(batch),len(selector.rows)],
                    reference_and_factual_target_scores_finite=bool(torch.isfinite(reference).all() and torch.isfinite(factual).all()),
                    reference_and_factual_target_scores_nonpositive=bool((reference<=0).all() and (factual<=0).all()),
                    saved_per_sample_endpoint_scores_present=all(row['factual_target_logp'] is not None and row['reference_target_logp'] is not None for row in batch))
        batches.append(dict(path=filename,sha256=digest(path),bytes=path.stat().st_size,original_rank_rows=len(batch),
          target_tokens=len(factual),selected_predictor_rows=len(selector.rows),checks=checks,
          completed_save_seconds=callback['seconds']))
        for index,row in enumerate(batch):
            rows.append(dict(path=filename,traj_uid=row['traj_uid'],rank_row=index,target_tokens=selected.counts[index],
              factual_score_from_native_targets=float(fact_sum[index]),factual_score_saved=row['factual_target_logp'],
              factual_sum_minus_saved=None if row['factual_target_logp'] is None else float(fact_sum[index])-row['factual_target_logp'],
              reference_score_from_native_targets=float(ref_sum[index]),reference_score_saved=row['reference_target_logp'],
              reference_sum_minus_saved=None if row['reference_target_logp'] is None else float(ref_sum[index])-row['reference_target_logp']))
    result=dict(scope=__doc__,unix=time.time(),seconds=time.perf_counter()-started,
      source_sha256=SOURCE_SHA,script_sha256=digest(__file__),installed_override_sha256=digest(FORMAL/'runtime-overrides/native-target-records-20261010-v2.json'),
      owners=[dict(path=x.__file__,resolved=str(Path(x.__file__).resolve()),sha256=digest(x.__file__)) for x in [packing,selector_owner]],
      live_workers=observations,input_kind=input_kind,completed_batches=len(batches),original_rank_rows=len(rows),unique_UIDs=len({x['traj_uid'] for x in rows}),
      failed_batches_by_exact_check={key:sum(not x['checks'][key] for x in batches) for key in batches[0]['checks']} if batches else {},
      maximum_absolute_factual_sum_residual=max((abs(x['factual_sum_minus_saved']) for x in rows if x['factual_sum_minus_saved'] is not None),default=None),
      maximum_absolute_reference_sum_residual=max((abs(x['reference_sum_minus_saved']) for x in rows if x['reference_sum_minus_saved'] is not None),default=None),
      batches=batches,rows=rows,PSS_bytes=psutil.Process().memory_full_info().pss,
      CUDA_initialized=torch.cuda.is_initialized(),model_DT_optimizer_calls=0,production_changes=0,
      limits=['No completed batches means pending, not passed.','Exact mapping checks and score-sum residuals do not measure individual deletion accuracy.','Residuals are reported as values without a new numerical tolerance or correction.'])
    assert not result['CUDA_initialized']
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+chr(10))
    print(json.dumps({k:v for k,v in result.items() if k not in ['batches','rows']}))


if __name__=='__main__':
    main()
