"""Read completed raw/officially whitened native gradients; no model or reducer.

Original ownership-aware norm/dot records supply every gradient measurement.
The full native64 normalization receipt supplies coefficient/field identities.
"""
import argparse
import json
from pathlib import Path

from analyze_textcraft_label_gradients import gradient_pair, source

GROUPS = ('raw', 'white')
LABELS = ('dt_pg', 'weighted_entropy', 'weighted_kl')


def build_analysis(root):
    root = Path(root)
    paths = {name: root / filename for name, filename in {
        'completed': 'completed.json', 'inspection': 'native-owner-inspection.json',
        'prepared': 'prepared.json', 'job': 'job.json'}.items()}
    for rank in (0, 1):
        for branch in GROUPS:
            paths[f'rank{rank}_{branch}'] = root / f'rank{rank}-{branch}-gradients.json'
        paths[f'rank{rank}_cross'] = root / f'rank{rank}-cross-group-gradients.json'
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Completed native records are not all present: ' + ', '.join(missing))
    raw = {name: json.loads(path.read_bytes()) for name, path in paths.items()}
    inspection, completed = raw['inspection'], raw['completed']
    ranks = []
    for rank in (0, 1):
        before, white, cross = (raw[f'rank{rank}_{name}'] for name in ('raw', 'white', 'cross'))
        ranks.append(dict(rank=rank,
            raw_PG_H=gradient_pair(before['gradient_statistics'], 'dt_pg', 'weighted_entropy'),
            raw_PG_KL=gradient_pair(before['gradient_statistics'], 'dt_pg', 'weighted_kl'),
            white_PG_H=gradient_pair(white['gradient_statistics'], 'dt_pg', 'weighted_entropy'),
            white_PG_KL=gradient_pair(white['gradient_statistics'], 'dt_pg', 'weighted_kl'),
            raw_white_PG=gradient_pair(cross['gradient_statistics'], 'old_dt_pg', 'swapped_dt_pg'),
            raw_PG_white_H=gradient_pair(cross['gradient_statistics'], 'old_dt_pg', 'weighted_entropy'),
            original_raw_statistics=before['gradient_statistics'],
            original_white_statistics=white['gradient_statistics'],
            original_cross_statistics=cross['gradient_statistics'],
            cross_aliases=before['cross_group_aliases'],
            group_config_exact=before['effective_config'] == white['effective_config'],
            effective_config=before['effective_config'],
            input_shapes_and_dtypes_exact=before['input_fields'] == white['input_fields'],
            input_fields=before['input_fields'],
            rng_observations=dict(raw=before['rng_observation'], white=white['rng_observation']),
            group_sources=dict(raw=before['sources'], white=white['sources'],
                original_cross_helper=cross['original_statistics_source'],
                reused_paired_observer=cross['adapter_source']),
            operations=dict(
                actual_update_policy_backward_passes=sum(x['actual_backward_passes'] for x in (before, white)),
                actual_microbatch_backward_calls=sum(len(p['microbatch_losses'])
                    for x in (before, white) for p in x['passes'].values()),
                optimizer_step_executed=[x['optimizer_step_executed'] for x in (before, white)],
                scheduler_step_executed=[x['scheduler_step_executed'] for x in (before, white)],
                scheduler_no_op_calls=[x['scheduler_step_no_op_calls'] for x in (before, white)])))
    return dict(
        scope='Same saved native64/checkpoint25 raw-A versus full-collected-batch official whitening; gradient sensitivity only, not training convergence or DT quality acceptance.',
        source_files={name: source(path) for name, path in paths.items()},
        analysis_source=source(__file__),
        reused_description_source=source(__import__('inspect').getsourcefile(gradient_pair)),
        sources=completed['sources'], checkpoint=completed['checkpoint'],
        input_sha256=completed['input_sha256'],
        groups=list(GROUPS), labels=list(LABELS), rows=inspection['rows'],
        actor_config=inspection['actor_config'], config_source=inspection['config_source'],
        normalization=inspection['normalization'],
        white_carrier=inspection['white_carrier'], per_rank=ranks,
        operations=dict(
            native_backward_passes_per_rank=completed['native_backward_passes_per_rank'],
            sampling_calls=completed['sampling_calls'], DT_calls=completed['DT_calls'],
            optimizer_steps=completed['optimizer_steps'], scheduler_steps=completed['scheduler_steps'],
            analysis_model_forward_calls=0, analysis_DT_calls=0, analysis_backward_calls=0),
        interpretation_limits=[
            'Norm/dot values are read from the original ownership-aware helper; the two rank reports are not summed through another reducer.',
            'Raw/white PG cross direction is observed. Each branch has PG/H/KL norms and within-branch pairs; the reused cross-group report contains raw PG, white PG and white H only, not cross-group KL.',
            'Full native64 whitening occurs before original rank and B4 splitting; it can center formerly zero-reward positions, while original Q/V/A fields remain untouched.',
            'Whitening changes the actor advantage input in this isolated diagnostic; it does not prove an overall DT method repair or prescribe production deployment.',
            'A PG norm ratio is not a parameter-update ratio, success-rate change or causal explanation of historical degradation.',
            'All original PPO loss settings, masks, saved old/ref probabilities and native load/offload remain part of this bounded comparison. No numerical tolerance or quality gate is defined.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build_analysis(args.root)
    output = args.output or args.root / 'official-whitening-analysis.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=source(output), per_rank=[
        dict(rank=r['rank'], raw_white_PG=r['raw_white_PG'], white_PG_H=r['white_PG_H'],
             white_PG_KL=r['white_PG_KL']) for r in result['per_rank']])))
