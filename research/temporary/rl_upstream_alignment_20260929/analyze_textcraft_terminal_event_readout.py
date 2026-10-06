"""Observe saved categorical endpoint LPs for actual successful final actions.

No probability is reconstructed from signed attribution. The original native
handler and exact response IDs identify terminal versus earlier responses;
old and equivalent-label head scores remain separate source-bound observations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

AUDIT = Path(__file__).resolve().parent
DEG = AUDIT / 'textcraft-degradation-20261005'
NATIVE = DEG / 'readout-quality-20261006/v2'
V4 = DEG / 'native-minibatch-v4'
LABEL = DEG / 'equivalent-label-gradient-20261006/v2'


def identity(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def describe(values):
    x = list(values)
    return dict(count=len(x), mean=math.fsum(x)/len(x), mean_absolute=math.fsum(map(abs,x))/len(x),
        minimum=min(x), maximum=max(x), population_standard_deviation=statistics.pstdev(x),
        positive=sum(v>0 for v in x), negative=sum(v<0 for v in x), zero=sum(v==0 for v in x))


def aggregate(rows):
    fields = ('source_step','source_tokens','old_factual_success_probability','old_EOS_success_probability',
              'old_factual_target_logp','old_reference_target_logp','old_endpoint_logp_contrast',
              'swapped_factual_success_probability','swapped_EOS_success_probability',
              'swapped_factual_target_logp','swapped_reference_target_logp','swapped_endpoint_logp_contrast')
    by_uid = {}
    for row in rows:
        by_uid.setdefault(row['traj_uid'],[]).append(row)
    return dict(responses=len(rows), UIDs=len(by_uid),
        per_response={key:describe(row[key] for row in rows) for key in fields},
        UID_balanced_means={key:math.fsum(math.fsum(row[key] for row in group)/len(group)
            for group in by_uid.values())/len(by_uid) for key in fields},
        old_factual_success_probability_below_half=sum(row['old_factual_success_probability']<.5 for row in rows),
        swapped_factual_success_probability_below_half=sum(row['swapped_factual_success_probability']<.5 for row in rows),
        mean_scope='Per-response statistics and equal-UID means are both shown; earlier responses are not reweighted in training.')


def build_analysis():
    paths = dict(metadata=NATIVE/'original-native-collect-metadata.json',
        first_cases=NATIVE/'original-first-response-cases.json',
        mapped=NATIVE/'original-readout-mapped-requests.json',
        mapping_receipt=V4/'native-minibatch-readout-mapping.json', original_log=V4/'diagnostic.log',
        original_job=V4/'job.json', original_source_identity=V4/'native-minibatch-source-identity.json',
        swapped_inspection=LABEL/'native-owner-inspection.json', swapped_completed=LABEL/'completed.json',
        **{f'swapped_rank{rank}':LABEL/f'rank{rank}-swapped-label-native-dt-report.json' for rank in (0,1)})
    data = {name:json.loads(path.read_bytes()) for name,path in paths.items() if name!='original_log'}
    assert identity(paths['mapped'])['sha256'] == data['mapping_receipt']['mapped_requests_artifact']['sha256']
    assert identity(paths['original_log'])['sha256'] == data['mapping_receipt']['inputs']['diagnostic_log']['sha256']
    original_records = data['mapped']['requests']
    first_cases = [row for rows in data['first_cases']['rank_cases'] for row in rows]
    handlers = data['metadata']['handlers']
    first_responses = {}
    for index,handler in enumerate(handlers):
        key = tuple(handler['turns'][0]['native_response_ids'])
        first_responses.setdefault(key,[]).append((index,handler))
    UID_handlers = {}
    for case in first_cases:
        response = tuple(case['selected_input_ids'][case['source_start']:case['source_end']])
        matches = first_responses[response]
        assert len(matches)==1
        UID_handlers[case['traj_uid']] = matches[0]
    assert len(handlers)==len(first_cases)==len(UID_handlers)==64
    swapped_traces = {}
    for rank in (0,1):
        raw = data[f'swapped_rank{rank}']
        assert raw['rank']==rank
        assert raw['original_query_source']['sha256']=='228afbc7a10841d482c3d73def59dfe9ef192c057a97d76dca57369502a10137'
        assert raw['producer_source']['sha256']=='0ad37a17aede30089fd2ac9a609a42689e6a3a68d00b601520db0a5cff8b8e6e'
        for index,trace in enumerate(raw['report']['traces']):
            swapped_traces[(rank,index)] = trace
    assert len(original_records)==len(swapped_traces)==192
    # Each already-bound original line is read as JSON transport; the original
    # mapper owns its rank/PID/request binding. No new rank inference is used.
    lines = paths['original_log'].read_text(encoding='utf-8',errors='replace').splitlines()
    original_reports = {}
    for line_number in {row['report_log_line'] for row in original_records}:
        line = lines[line_number-1]
        original_reports[line_number] = json.loads(line.split('[DeltaTrace readout] ',1)[1])
    grouped = {}
    for row in original_records:
        rank,index = row['rank'],row['request_index']
        old = original_reports[row['report_log_line']]['traces'][index]
        assert old == row['trace']
        new = swapped_traces[(rank,index)]
        for key in ('episode_index','source_step','observed_return','context_tokens',
                    'compute_tokens','query_tokens','owner_batch_index'):
            assert old[key]==new[key], (rank,index,key)
        assert old['observed_return']==new['observed_return']==1
        handler_index,handler = UID_handlers[row['traj_uid']]
        step = row['env_step']
        assert step==old['source_step']
        start,end = row['source_start'],row['source_end']
        assert row['selected_input_ids'][start:end] == handler['turns'][step]['native_response_ids']
        assert handler['score']==1 and handler['done'] and not handler['missing_fields']
        assert len(handler['turns'])==int(handler['task_rounds'])
        record = dict(traj_uid=row['traj_uid'],source_step=step,source_tokens=end-start,
            rank=rank,request_index=index,original_report_log_line=row['report_log_line'],
            official_handler_index=handler_index,official_item_id=handler['item_id'],
            original_context_tokens=old['context_tokens'],original_compute_tokens=old['compute_tokens'],
            official_final_score=handler['score'],official_final_done=handler['done'],
            official_recorded_turns=len(handler['turns']),
            is_final_success_response=step==len(handler['turns'])-1,
            old_factual_target_logp=old['factual_target_logp'],
            old_reference_target_logp=old['reference_target_logp'],
            old_factual_success_probability=math.exp(old['factual_target_logp']),
            old_EOS_success_probability=math.exp(old['reference_target_logp']),
            old_endpoint_logp_contrast=old['factual_target_logp']-old['reference_target_logp'],
            swapped_factual_target_logp=new['factual_target_logp'],
            swapped_reference_target_logp=new['reference_target_logp'],
            swapped_factual_success_probability=math.exp(new['factual_target_logp']),
            swapped_EOS_success_probability=math.exp(new['reference_target_logp']),
            swapped_endpoint_logp_contrast=new['factual_target_logp']-new['reference_target_logp'])
        grouped.setdefault((row['traj_uid'],step),[]).append(record)
    # Keep the first original transport slot as primary, preserving all replica
    # ranges; the source mapper's order is never regrouped for model execution.
    rows,replicas = [],[]
    score_fields = ('old_factual_target_logp','old_reference_target_logp',
                    'swapped_factual_target_logp','swapped_reference_target_logp')
    for key,records in grouped.items():
        primary = dict(records[0])
        primary['transport_replicas'] = len(records)
        rows.append(primary)
        if len(records)>1:
            replicas.append(dict(traj_uid=key[0],source_step=key[1],
                slots=[dict(rank=x['rank'],request_index=x['request_index']) for x in records],
                original_compute_tokens=[x['original_compute_tokens'] for x in records],
                endpoint_replica_ranges={name:dict(minimum=min(x[name] for x in records),
                    maximum=max(x[name] for x in records),spread=max(x[name] for x in records)-min(x[name] for x in records)) for name in score_fields}))
    terminal = [row for row in rows if row['is_final_success_response']]
    earlier = [row for row in rows if not row['is_final_success_response']]
    assert len(rows)==186 and len(terminal)==21 and len(earlier)==165
    assert len({row['traj_uid'] for row in terminal})==21
    return dict(scope='Existing categorical head endpoint LP observation for actual final-success versus earlier G1 responses; no new model, DT, credit or environment-counterfactual computation.',
        analysis_source=identity(Path(__file__)),sources={name:identity(path) for name,path in paths.items()},
        actual_original_sources=data['original_source_identity'],
        mapped_owner_sources=data['mapping_receipt']['sources'],
        swapped_original_query_source=data['swapped_rank0']['original_query_source'],
        swapped_producer_source=data['swapped_rank0']['producer_source'],
        original_checkpoint=data['original_job']['checkpoint'],
        swapped_checkpoint=data['swapped_completed']['checkpoint'],
        population=dict(transport_slots=192,unique_UID_responses=186,terminal_success_responses=21,
            earlier_success_trajectory_responses=165,successful_UIDs=21,official_rollout_UIDs=64,
            failed43_UIDs_not_emitted_by_G0_readout=43),
        absolute_probability_source='Saved original report.traces factual_target_logp/reference_target_logp, independently matched to original log JSON and exact original request mapping; equivalent-label fresh reports read their own same-named head fields. Never sum(d) or conservation inversion.',
        terminal_designation='Exact original first-response IDs identify each native handler. Each G1 mapped response IDs matches handler.turns[source_step]; final handler score=1 and done=True, and source_step=len(turns)-1 identifies its last submitted success response. The artifact contains final done/score, not a separately saved per-turn done/reward flag.',
        forecast_input_scope='Recorded request prompt/current response and original reward query, before successor observation; terminal grouping is observation metadata used only by this CPU report.',
        encoding_scope='Original query228 labels01 and equivalent-label query228 labels10 are reported separately; fresh query-clock94 results are not used.',
        terminal21=aggregate(terminal),earlier165=aggregate(earlier),
        rows=rows,transport_duplicate_records=replicas,
        duplicate_policy='Primary is first occurrence in exact original mapped request order. No silent replica averaging; all duplicated endpoint ranges retained.',
        limits=[
            'Only 21 successful trajectories have nonzero-return native readout; this does not measure predictions on all trajectories or early prefixes with random later outcomes.',
            'The factual final action was actually executed and its native handler ended successfully; this is distinct from a sampled early-action forecast of an uncertain continuation.',
            'The EOS-deleted action has not been executed in an environment. Its reported probability and endpoint contrast are not an environment oracle or a token-error rate.',
            'Neither these endpoint probabilities nor attribution conservation establishes overall DT ranking quality or the cause of historical training degradation.'],
        operations=dict(model_calls=0,DT_calls=0,environment_calls=0,backward_calls=0,optimizer_steps=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEG/'terminal-event-readout-source-observation-20261006.json')
    args=parser.parse_args()
    result=build_analysis()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),sha256=identity(args.output)['sha256'],
        population=result['population'],terminal21=result['terminal21'],earlier165=result['earlier165'])))
