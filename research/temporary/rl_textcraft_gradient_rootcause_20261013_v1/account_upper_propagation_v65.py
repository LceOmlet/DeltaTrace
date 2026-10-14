"""Account upper residual gains from completed native VJP receipts, CPU only.

This reads existing measurements. It is not a model, optimizer, Jacobian or
residual-block implementation, and does not run any new derivative experiment.
"""
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sources = {
    "historical_5_80": HERE / "native-fixed-seed-propagation-summary-v33.json",
    "historical_40": HERE / "native-midpoint-fixed-seed-summary-v35.json",
    "original_branch_events_5_80": HERE / "native-fixed-seed-propagation-v33-status-final.json",
    "original_branch_events_40": HERE / "native-midpoint-fixed-seed-v35-status-final.json",
    "conditional_update": HERE / "completed-common-update-chain-v58.json",
}
data = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in sources.items()}
cases = {5: data["historical_5_80"]["cases"]["late5_seed80"],
         80: data["historical_5_80"]["cases"]["late80_native"],
         40: data["historical_40"]}
cases[40]["residuals"] = [dict(
    layer=r["layer"], output_norm=r["out"], MLP_branch_norm=r["MLP"],
    middle_total_norm=r["mid"], attention_branch_norm=r["mixer"], input_total_norm=r["inp"],
) for r in cases[40]["branches"]]
cases[40]["branch_modules"] = cases[40]["modules"]
incoming = {cp: c["boundaries"]["model.layers.31"]["output"] for cp, c in cases.items()}
assert len(set(incoming.values())) == 1


def residual_geometry(g, branch, total):
    a, gain = branch / g, total / g
    cross = gain * gain - 1.0 - a * a
    return dict(branch_norm_over_incoming=a, post_add_norm_gain=gain,
                normalized_branch_squared=a * a, normalized_cross_term_from_norms=cross,
                cosine_from_norm_identity=cross / (2.0 * a))


report = dict(
    recorded_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), scope=__doc__,
    source_receipts={k: dict(path=str(p), SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
                     for k, p in sources.items()},
    fixed_block31_incoming_adjoint_norm=incoming, cases={}, comparisons={},
    new_model_calls=0, new_backward_calls=0, optimizer_updates=0,
    rootcause_repair_verified=False,
    limits=[
        "The full incoming BF16 tensor was fixed at block31, not at every internal branch.",
        "Norm ratios are directional gains, not operator norms or independent causal shares.",
        "Cross terms and cosines are inferred from measured norms; they are not independently saved dot products. Native BF16 residual rounding remains.",
        "Per-B4 active forward RMS quantile ranges are not pooled token distributions.",
        "The historical propagation change and the late conditional native update have different layer profiles; the latter does not identify the entire old trajectory.",
        "The old late-input whole-PG 40-to-80 ratio of about1.074 remains counterevidence against scaling all policy gradients by the fixed-seed VJP ratio.",
    ],
)
for cp, case in sorted(cases.items()):
    boundaries = case["boundaries"]
    bands = {}
    for first in (0, 8, 16, 24, 29):
        last = first + 7 if first < 29 else 31
        bands[f"{first}-{last}"] = (
            boundaries[f"model.layers.{first}"]["input"] /
            boundaries[f"model.layers.{last}"]["output"]
        )
    layers = {}
    for r in case["residuals"]:
        layers[str(r["layer"])] = dict(
            MLP=residual_geometry(r["output_norm"], r["MLP_branch_norm"], r["middle_total_norm"]),
            mixer=residual_geometry(r["middle_total_norm"], r["attention_branch_norm"], r["input_total_norm"]),
            block_gain=r["input_total_norm"] / r["output_norm"],
        )
    report["cases"][str(cp)] = dict(band_gains=bands, upper_residuals=layers, upper_modules={})
    for layer in (29, 30, 31):
        for suffix in ("mlp", "post_attention_layernorm", "input_layernorm"):
            name = f"model.layers.{layer}.{suffix}"
            module = case["branch_modules"][name]
            report["cases"][str(cp)]["upper_modules"][name] = dict(
                directional_input_over_output=module["input"] / module["output"])

for old, new in ((5, 40), (40, 80), (5, 80)):
    b, a = (report["cases"][str(cp)] for cp in (old, new))
    report["comparisons"][f"{old}_to_{new}"] = dict(
        band_gain_ratios={k: a["band_gains"][k] / v for k, v in b["band_gains"].items()},
        residual_gain_ratios={layer: {
            branch: a["upper_residuals"][layer][branch]["post_add_norm_gain"] /
                    b["upper_residuals"][layer][branch]["post_add_norm_gain"]
            for branch in ("MLP", "mixer")}
            for layer in ("29", "30", "31")},
    )

wanted = {5: "late5_seed80", 40: "late40_seed80", 80: "late80_native"}
for key in ("original_branch_events_5_80", "original_branch_events_40"):
    for wrapper in data[key]["report"]["chain_reports"]:
        r = wrapper["value"]
        if r["iteration"] != wanted[r["checkpoint"]]:
            continue
        modules = report["cases"][str(r["checkpoint"])]["upper_modules"]
        for event in r["native_decoder_branch_backward"]:
            name = event["name"].removeprefix("base_model.model.")
            if name not in modules:
                continue
            for v in event["native_module_grad_input"]:
                q = v.get("forward_RMS_at_nonzero_branch_adjoint_quantiles")
                if v["index"] == 0 and q:
                    modules[name].setdefault("active_B4_input_RMS_quantiles", []).append(q)
for case in report["cases"].values():
    for module in case["upper_modules"].values():
        q = module.pop("active_B4_input_RMS_quantiles", [])
        module["active_B4_median_RMS_range"] = [min(v[2] for v in q), max(v[2] for v in q)] if q else None
        module["minimum_observed_active_RMS"] = min(v[0] for v in q) if q else None

conditional = data["conditional_update"]["cases"]
report["conditional_update_band_gain_ratios"] = {}
for arm in ("full_original", "without_constant_positive_component"):
    b, a = (conditional[arm + "_" + phase]["layers"] for phase in ("before", "after"))
    report["conditional_update_band_gain_ratios"][arm] = {
        f"{first}-{first+7}": (a[str(first)]["input_norm"] / a[str(first+7)]["output_norm"]) /
        (b[str(first)]["input_norm"] / b[str(first+7)]["output_norm"])
        for first in (0, 8, 16, 24)
    }
report["conclusions"] = [
    "The historical40-to80 increase concentrates in upper blocks; the final three alone multiply their directional transfer by about5.105.",
    "At block31 MLP residual merge, the norm-implied cross term changes sign while branch amplitude also rises. The unchanged incoming seed rules out explaining this measurement by a larger output seed.",
    "Observed block31 RMS values remain away from zero. Neither this observation nor a norm identity proves the optimizer cause of the change.",
    "The late conditional full update decreases upper24-to31 transfer, despite increasing the final fixed-seed parameter VJP. Its lower-layer response is not the same as the historical upper-layer change.",
    "Do not patch an official MLP, normalization or mixer based on these records; the remaining issue is the connection between the historical update component and the measured internal propagation change.",
]
curve_path = Path('F:/DeltaTraceBackups/plots/gradient-rootcause-20261014/native-gradient-growth-85.csv')
curve = list(csv.DictReader(curve_path.read_text(encoding='utf-8-sig').splitlines()))
assert [int(r['iteration']) for r in curve] == list(range(1, 86))
blocks = []
for first in range(1, 86, 10):
    selected = curve[first-1:min(first+9, 85)]
    last = int(selected[-1]['iteration'])
    blocks.append(dict(first=first, last=last, center=(first+last)/2,
                       native_grad_norm_median=median(float(r['native_grad_norm']) for r in selected)))
changes = []
for before, after in zip(blocks, blocks[1:]):
    width = after['center'] - before['center']
    old, new = before['native_grad_norm_median'], after['native_grad_norm_median']
    changes.append(dict(before=[before['first'], before['last']], after=[after['first'], after['last']],
                        center_distance=width, median_slope=(new-old)/width,
                        descriptive_geometric_change_per_iteration=(new/old)**(1/width)-1))
report['native_temporal_scale_check'] = dict(
    source=str(curve_path), source_SHA256=hashlib.sha256(curve_path.read_bytes()).hexdigest(),
    blocks=blocks, changes=changes,
    conditional_fixed_seed_VJP_change_per_native_iteration={
        arm: value['parameter_VJP_ratio']-1
        for arm, value in data['conditional_update']['arms'].items()},
    scope='Descriptive rate/scale comparison, not a fitted growth law or a prediction of the complete native gradient.',
    limits=[
        'Native console grad_norm is the original rounded pre-clip metric, averaged over four optimizer updates; the fixed-seed VJP is a different conditional observable.',
        'The final81-to85 block has five observations; center distances, not identical ten-iteration widths, are used.',
        'A comparable local percentage is an order-of-magnitude consistency check only. It does not identify a causal model, prove a quadratic/exponential law or justify extrapolating the late intervention.',
        'No confidence interval or independent-sample assumption is assigned to dependent training iterations.',
    ],
)
dest = HERE.parents[2] / "experiments/rl/results_textcraft_upper_propagation_20261015_v1.json"
dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("receipt", dest)
print("historical40to80", report["comparisons"]["40_to_80"]["band_gain_ratios"])
print("conditional", report["conditional_update_band_gain_ratios"])
