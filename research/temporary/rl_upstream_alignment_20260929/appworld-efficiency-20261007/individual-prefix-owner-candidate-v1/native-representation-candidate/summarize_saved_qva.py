"""Compact stdlib view of the executed saved-vector CPU analysis."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent/'saved-qva-cpu-review'


def identity(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    source = ROOT/'real-b8-v3-qva-comparison.json'
    original = json.loads(source.read_bytes())
    ranks = []
    for rank in original['ranks']:
        comparisons = []
        for comparison in rank['comparisons']:
            fields = []
            for field in comparison['fields']:
                stats = field['action_difference']
                fields.append(dict(key=field['key'],dtype=field['dtype'],shape=field['shape'],
                    complete_literal_bytes_equal=field['complete_literal_bytes_equal'],
                    changed_action_values=field['changed_action_values'],
                    max_absolute_difference=max(abs(stats['min']),abs(stats['max'])),
                    mean_absolute_difference=stats['mean_absolute'],rms_difference=stats['rms'],
                    sign_counts=field['action_sign_counts'],
                    outside_action_difference=field['outside_action_difference']))
            endpoint = {key:max(abs(row[key]) for row in comparison['endpoint_differences']
                               if row[key] is not None)
                        for key in ('factual_target_logp','reference_target_logp','root_effect','signed_sum')}
            comparisons.append(dict(left=comparison['left'],right=comparison['right'],
                scope=comparison['scope'],fields=fields,
                native_endpoint_max_absolute_differences=endpoint,
                native_endpoint_per_row_differences=comparison['endpoint_differences']))
        ranks.append(dict(rank=rank['rank'],original_rank=rank['rank_json'],original_vectors=rank['vectors'],
            original_requests=rank['original_request_file'],valid_action_tokens=rank['valid_action_tokens'],
            outside_action_positions=rank['outside_action_positions'],selected_rows=rank['selected_rows'],
            original_variant_wall_seconds={key:value['original_wall_seconds'] for key,value in rank['variants'].items()},
            original_shared_native_prefix_preparation={key:value['per_row'][0]['native_shared_prefix_length']
                for key,value in rank['variants'].items()},comparisons=comparisons))
    result = dict(scope='Compact descriptive saved-vector/endpoint observations; no whole-network pass/fail tolerance.',
        script=identity(Path(__file__)),source=identity(source),CPU_analyzer=original['script'],
        original_prepared=original['prepared'],execution=original['execution'],ranks=ranks,
        limits=original['limitations']+[
            'The detailed analyzer field per_row.native_shared_prefix_length contains the original readout shared_native_prefix preparation dictionary, not a numeric native cut. This compact view gives it the correct preparation name; actual cut is not inferred from that dictionary.',
            'Reported sign differences count saved nonzero action coefficients; they are not token-error rates or an official numerical criterion.'])
    output = ROOT/'compact-qva-review.json'
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(output=str(output),sha256=identity(output)['sha256'])))


if __name__ == '__main__':
    main()
