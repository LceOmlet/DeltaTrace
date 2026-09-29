"""Account for the original runner's recorded residual; no acceptance threshold."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
record = json.loads(args.source.read_text())
owner = record['owner_details']
layers = []
previous = owner['seed_effect']
boundary_reduction = 0.0
for index, row in sorted(owner['layers'].items(), key=lambda item: int(item[0]), reverse=True):
    boundary = previous - row['root_output_effect']
    boundary_reduction += boundary
    layers.append(dict(layer=int(index), block_type=row['block_type'],
        root_output_effect=row['root_output_effect'],
        replay_output_effect=row['replay_output_effect'], input_effect=row['input_effect'],
        root_minus_replay=row['root_output_effect']-row['replay_output_effect'],
        replay_minus_input=row['replay_output_effect']-row['input_effect'],
        boundary_reduction_difference=boundary,
        replay_relative_L2=row['replay_relative_L2']))
    previous = row['input_effect']
parts = dict(head_and_final_norm=owner['root_effect']-owner['seed_effect'],
             root_minus_replay=sum(row['root_minus_replay'] for row in layers),
             replay_minus_input=sum(row['replay_minus_input'] for row in layers),
             boundary_reduction=boundary_reduction,
             final_reduction=previous-owner['signed_sum'])
residual = owner['root_effect']-owner['signed_sum']
result = dict(scope=__doc__, source=str(args.source),
              root_effect=owner['root_effect'], seed_effect=owner['seed_effect'],
              signed_sum=owner['signed_sum'], residual=residual,
              components=parts, accounted_residual=sum(parts.values()),
              accounting_difference=residual-sum(parts.values()),
              native_shared_prefix_length=owner['native_shared_prefix_length'],
              layers=layers,
              largest_replay_differences=sorted(layers, key=lambda row:abs(row['root_minus_replay']), reverse=True)[:5],
              largest_finite_differences=sorted(layers, key=lambda row:abs(row['replay_minus_input']), reverse=True)[:5])
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='layers'}, indent=2))
