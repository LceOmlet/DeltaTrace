"""Prepare an isolated low-rank readout patch for the verified DT decoder owner.

Only _linear_weights/_linear_transpose source ranges are replaced. The base
map and original _mm backend remain unchanged. Nothing changes the default DT
tree, launcher, actor, PEFT forward, offload policy or numerical tolerance.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parents[2]
OWNER = REPO / "deltatrace/clean/qwen35/qwen35_decoder_finite.py"
OWNER_SHA = "047c38e6b180adb350083ff370693ed20b95e2df82e4a78cd59038bc0803a197"
MM_SHA = "f1555736d32974668676eff4e040a500f4c2b907e6ac1b298f01dc2ba87f7db6"

REPLACEMENTS = {
    "_linear_weights": '''def _linear_weights(module):
    """Read vanilla PEFT maps as base plus its existing low-rank factors.

    No full delta matrix is materialized or merged. PEFT retains ownership
    of active adapters, factors and scaling; disabled/merged/plain maps keep
    the original weight path. This candidate covers the current Qwen Linear
    adapters only, without fan-in/fan-out transposition or LoRA variants.
    """
    weight=module.weight
    if not hasattr(module,'get_delta_weight'):
        return weight
    if module.disable_adapters or module.merged:
        return weight
    if not isinstance(module.get_base_layer(),torch.nn.Linear) or module.fan_in_fan_out:
        raise ValueError('Low-rank DT candidate requires the original vanilla PEFT Linear orientation')
    weights=[weight]
    for adapter in module.active_adapters:
        if adapter in module.lora_A:
            if adapter in module.lora_variant:
                raise ValueError('Low-rank DT candidate does not support a PEFT LoRA variant')
            weights.append((module.lora_B[adapter].weight,
                            module.lora_A[adapter].weight,module.scaling[adapter]))
    return tuple(weights) if len(weights)>1 else weight
''',
    "_linear_transpose": '''def _linear_transpose(upstream,weight):
    if isinstance(weight,tuple):
        # The base consumes the unchanged owner BF16 operand/MM. Each active
        # vanilla adapter's transpose map is m @ B @ A times PEFT scaling.
        # Keep the original MM backend and its rounding boundaries visible.
        shape=upstream.shape
        operand=upstream.reshape(1,-1,shape[-1]).to(torch.bfloat16)
        result=_mm(operand,weight[0].unsqueeze(0))
        for adapter_B,adapter_A,scaling in weight[1:]:
            result=result+_mm(_mm(operand,adapter_B.unsqueeze(0)),
                              adapter_A.unsqueeze(0))*scaling
        return result.reshape(*shape[:-1],weight[0].shape[-1])
    shape=upstream.shape
    return _mm(upstream.reshape(1,-1,shape[-1]),weight.unsqueeze(0)).reshape(*shape[:-1],weight.shape[-1])
''',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patched_owner(original):
    """Splice only the two reviewed owner functions, preserving other bytes."""
    assert sha(original) == OWNER_SHA
    lines = original.splitlines(keepends=True)
    tree = ast.parse(original)
    ranges = []
    for name, replacement in REPLACEMENTS.items():
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        start = sum(map(len, lines[:node.lineno-1]))
        end = sum(map(len, lines[:node.end_lineno]))
        ranges.append((start, end, replacement.encode(), name))
    ranges.sort()
    pieces = []
    position = 0
    records = []
    for start, end, replacement, name in ranges:
        pieces.extend((original[position:start], replacement))
        records.append(dict(function=name,original_byte_start=start,original_byte_end=end,
            original_method_sha256=sha(original[start:end]),candidate_method_sha256=sha(replacement)))
        position = end
    pieces.append(original[position:])
    candidate = b"".join(pieces)
    parsed = ast.parse(candidate)
    unchanged_before = [ast.dump(n) for n in tree.body
                        if not (isinstance(n, ast.FunctionDef) and n.name in REPLACEMENTS)]
    unchanged_after = [ast.dump(n) for n in parsed.body
                       if not (isinstance(n, ast.FunctionDef) and n.name in REPLACEMENTS)]
    assert unchanged_before == unchanged_after
    return candidate, records


def prepare(output, *, owner_path=OWNER):
    assert not output.exists(), "Preserve immutable candidates; inspect an existing preparation"
    original = owner_path.read_bytes()
    candidate, ranges = patched_owner(original)
    output.mkdir(parents=True)
    before_path = output / "qwen35_decoder_finite.before.py"
    candidate_path = output / "qwen35_decoder_finite_low_rank_candidate.py"
    before_path.write_bytes(original)
    candidate_path.write_bytes(candidate)
    patch_path = output / "qwen35_decoder_finite.low-rank.patch"
    patch_path.write_text("".join(difflib.unified_diff(original.decode().splitlines(keepends=True),
        candidate.decode().splitlines(keepends=True),
        fromfile="c9cd147/clean/qwen35/qwen35_decoder_finite.py",
        tofile="isolated/clean/qwen35/qwen35_decoder_finite.py")),encoding="utf8",newline="\n")
    receipt = dict(status="isolated_prepared_only_not_deployed",source_owner=str(owner_path),
        baseline_dt_release="c9cd147",baseline_dt_reference="fc2e6c2",owner_before_sha256=sha(original),
        candidate=str(candidate_path),candidate_sha256=sha(candidate),ranges=ranges,
        other_owner_bytes_preserved=True,mm_owner_sha256=MM_SHA,
        supported_interface="Existing PEFT vanilla Linear, fan_in_fan_out=False, no LoRA variant",
        peft_actual_source=dict(version="0.18.1",
            path="/opt/conda/lib/python3.12/site-packages/peft/tuners/lora/layer.py",
            sha256="e8a47a49cf69f92ded68bf7ab286aeaf068f0ac809ab3465ad16f78a8f9f8b63"),
        unchanged=["base weight/operand/_mm", "plain/disabled/merged map path", "PEFT forward",
                   "FA/FLA finite formulas", "offload policy", "PPO", "LoRA8/16", "B4"],
        numerical_scope="Associative low-rank map preserves FP32 algebra; original BF16 MM stages change the adapter rounding path and require separate actual-dtype owner comparison. No FA/FLA acceptance claim.")
    (output / "prepared.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,
        default=AUDIT/"native-lora-factorization-owner-20261004")
    parser.add_argument("--owner-source",type=Path,default=OWNER,
        help="Exact c9cd147 owner source; its full SHA is fixed")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir,owner_path=args.owner_source),indent=2))


if __name__ == "__main__":
    main()
