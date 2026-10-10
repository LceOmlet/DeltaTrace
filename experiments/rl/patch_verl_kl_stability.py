"""Apply VERL's upstream low_var_kl exponential guard to the pinned owner.

Source: verl-project/verl@9058412fb308e9f2e31a794ad1d18d97d4286485,
verl/trainer/ppo/core_algos.py. This only edits the owning function; it does
not supply another loss or change DT advantages, masks, or PPO clipping.

Incident37's actual CPU/MetaX input comparisons are recorded in
results_textcraft_incident37_diagnosis_20261011.json. The old final clamp
leaves forward loss finite while exp backward can still produce NaN.
"""

OLD = "        kl = ref_logprob - logprob\n        ratio = torch.exp(kl)"
NEW = (
    "        kl = ref_logprob - logprob\n"
    "        # For numerical stability\n"
    "        kl = torch.clamp(kl, min=-20, max=20)\n"
    "        ratio = torch.exp(kl)"
)


def patch(text: str) -> str:
    """Keep an already patched owner intact; reject an incompatible source."""
    if NEW in text:
        return text
    if text.count(OLD) != 1:
        raise RuntimeError("Cannot find the pinned VERL low_var_kl exp anchor")
    return text.replace(OLD, NEW, 1)
