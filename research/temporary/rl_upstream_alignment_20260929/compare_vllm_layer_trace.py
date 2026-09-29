"""Locate first differing saved owner boundary. No pass tolerance is invented."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('left', type=Path)
parser.add_argument('right', type=Path)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
left = torch.load(args.left, map_location='cpu', weights_only=False)
right = torch.load(args.right, map_location='cpu', weights_only=False)
report = dict(left=str(args.left), right=str(args.right), boundaries=[])
for boundary, inputs in left.items():
    for name, value in inputs.items():
        other = right.get(boundary, {}).get(name)
        row = dict(boundary=boundary, tensor=name, shape=list(value.shape), dtype=str(value.dtype))
        if other is None or other.shape != value.shape:
            row['different_shape'] = None if other is None else list(other.shape)
        else:
            delta = (value.float()-other.float()).abs()
            row.update(equal=torch.equal(value, other), max_abs=float(delta.max()) if delta.numel() else 0.,
                       changed=int((value != other).sum()), elements=value.numel(),
                       rms_difference=float(delta.square().mean().sqrt()) if delta.numel() else 0.)
        report['boundaries'].append(row)
report['first_difference'] = next((v for v in report['boundaries'] if not v.get('equal')), None)
args.out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(first_difference=report['first_difference'],
    differences=[v for v in report['boundaries'] if not v.get('equal')],out=str(args.out))))
