"""Render the complete fixed-policy comparison, with provenance and caveats."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
from common import HERE,TASKS,BASELINES,METHODS,sha

NAMES={'DT':'DeltaTrace','FT_K3':'FlashTrace K3','FT_K1':'FlashTrace K1 (diagnostic)',
    'Perturbation':'Perturbation (20 segments)','REAGENT':'REAGENT (20 segments)',
    'CLP':'CLP (20 segments)','IFR':'IFR','AttnLRP':'AttnLRP †'}
LABELS={'vt_h2_c3':'VT H2-C3','vt_h4_c1':'VT H4-C1','vt_h6_c1':'VT H6-C1','vt_h10_c1':'VT H10-C1',
    'vt_macro':'VT macro','hotpotqa_long':'HotpotQA'}
pct=lambda x:f'{100*x:.2f}%'
pp=lambda x:f'{100*x:+.2f}'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])

def main():
    analysis=json.loads((HERE/'analysis.json').read_bytes())
    assert analysis['status']=='verified_complete_all_baselines' and analysis['new_method_cases']==2240
    controller=json.loads((HERE/'audit/all_baselines_logs_v3/controller.json').read_bytes())
    assert controller['status']=='complete' and all(r['status']=='complete' for r in controller['phases'])
    controller_hours=(controller['end']-controller['start'])/3600
    primary={r['method']:r for r in analysis['primary']};summary=analysis['summaries'];costs=analysis['costs']
    primary_table=table(['Method',*[LABELS[t] for t in TASKS[:-1]],'VT macro','HotpotQA'],
        [[NAMES[m],*[pct(primary[m][t]) for t in TASKS[:-1]],pct(primary[m]['vt_macro']),pct(primary[m]['hotpotqa_long'])] for m in METHODS])
    comparisons=table(['Scope','Comparator','DT − comparator (pp)','Adjusted interval (pp)'],
        [[LABELS[r['scope']],NAMES[r['comparator']],pp(r['dt_minus_comparator']),
            f"[{pp(r['interval'][0])}, {pp(r['interval'][1])}]"] for r in analysis['comparisons']])
    main=[r for r in summary if r['aggregation']=='signed_sum']
    hp=[r for r in main if r['dataset']=='hotpotqa_long' and r['method'] in METHODS and r['budget_unit']=='all_body_tokens' and r['fraction']==.1]
    hp={r['method']:r for r in hp}
    hp_table=table(['Method','Precision','Recall','F1','Exact set','Complete support','Token Recall','Tokens used','Unused','Empty selections'],
        [[NAMES[m],*[pct(hp[m][k]) for k in ('precision','recall','f1','exact_match','complete_support','token_recall')],
            f"{hp[m]['spent_tokens']:.2f}",f"{hp[m]['unused_tokens']:.2f}",pct(hp[m]['empty_selection'])] for m in METHODS])
    budgets=table(['Task','Method','Recall @5%','Recall @10%','Recall @20%'],
        [[LABELS[t],NAMES[m],*[pct(next(r['recall'] for r in main if r['dataset']==t and r['method']==m and r['fraction']==f and
            r['view']==('native_sentence' if t=='hotpotqa_long' else 'raw') and r['budget_unit']!= 'sentences')) for f in (.05,.1,.2)]]
            for t in TASKS for m in METHODS])
    topk=table(['Method','Recall top 2','Recall top 4','Recall top 8'],
        [[NAMES[m],*[pct(next(r['recall'] for r in main if r['dataset']=='hotpotqa_long' and r['method']==m and
            r['budget_unit']=='sentences' and r['fraction']==k)) for k in (2,4,8)]] for m in METHODS])
    density=table(['Method',*[LABELS[t] for t in TASKS[:-1]]],
        [[NAMES[m],*[pct(next(r['recall'] for r in main if r['dataset']==t and r['method']==m and r['view']=='density' and r['fraction']==.1))
            for t in TASKS[:-1]]] for m in METHODS])
    diagnostic=table(['Method',*[LABELS[t] for t in TASKS[:-1]],'HotpotQA'],
        [[NAMES[m],*[pct(next(r['recall'] for r in summary if r['dataset']==t and r['method']==m and
            r['aggregation']=='native_positive_normalized' and r['fraction']==.1 and r['budget_unit']!='sentences' and
            r['view']==('native_sentence' if t=='hotpotqa_long' else 'raw'))) for t in TASKS]] for m in BASELINES])
    cost_table=table(['Method','Cases','Measured operation time (h)','Median time/case (s)','Max PyTorch allocation (GiB)','Qwen forwards'],
        [[NAMES[m],len([r for r in costs if r['method']==m]),
            f"{sum(r['seconds'] for r in costs if r['method']==m)/3600:.3f}",
            f"{statistics.median(r['seconds'] for r in costs if r['method']==m):.3f}",
            f"{max(r['peak_allocated_bytes'] for r in costs if r['method']==m)/2**30:.3f}",
            sum(r['calls']['model_forwards'] for r in costs if r['method']==m)] for m in BASELINES])
    repaired=sum(r.get('lrp_repaired_zero_ratios',0) for r in costs)
    text=f'''# All seven algorithms under the frozen VT and HotpotQA policies

Completed **448 paired inputs and 2,240 new baseline method-case results** on Qwen3-8B.
The five newly computed methods are Perturbation, REAGENT, CLP, IFR and AttnLRP.
DT and FT K3 are reused as requested; FT K1 is supplementary. Every main-table
baseline value is recomputed from raw vectors under the fixed policy.

The user selected HotpotQA `full` + `signed_sum` after viewing DT/FT results.
VT retains its preceding `answer_only` + raw positive-token policy. This is a
retrospective protocol choice over the complete released cases, including
development examples; it is not an independent holdout or evidence that the
measurement is neutral. See the [frozen protocol](PROTOCOL.md).

## Primary Recall at 10%

{primary_table}

VT measures eligible **body-token Recall**: remove the cached reasoning prefix,
construct the previously fixed generated-answer input, rank positive source-token
scores, and use ceil(10% × eligible body tokens). The VT macro weights the four
100-case tasks equally. This VT input is not context-preserving attribution of
the answer inside its original reasoning.

HotpotQA measures official **supporting-fact sentence Recall** over 48 cases.
Keep the entire reasoning-and-answer input and explain its full response with
the same saved output weights. Sum signed source scores over each original
body sentence, rank by that sum, and retrieve the longest initial ranking
prefix costing at most ceil(10% × all body tokens). Charge punctuation and
whitespace as well. Stop at the first sentence that does not fit. Titles and
document IDs are provided metadata outside this stated body-only budget.
The two tasks' denominators differ; no VT-plus-Hotpot raw-Recall average is used.

† AttnLRP includes an explicitly documented FP16 zero-division repair and
lossless saved-tensor offload. See the technical record below.

## Fixed family of paired comparisons

{comparisons}

Intervals use 10,000 paired bootstrap draws, seed 73, with Bonferroni confidence
99.583333% across the fixed family of 12 comparisons (DT against six alternatives
on HotpotQA and VT macro). VT resampling is independent within each task before
the equal-task average. Intervals are descriptive for this complete benchmark.

## HotpotQA supporting-fact and budget diagnostics at 10%

{hp_table}

All values are case means. Exact set requires retrieving exactly the gold fact
set; complete support permits extra sentences. Empty selections occur when the
top-ranked whole sentence exceeds the allowed prefix budget.

## Supplementary budgets

{budgets}

HotpotQA's integer sentence budgets use the same ranking:

{topk}

## Diagnostic VT sentence-density token ordering

{density}

This unchanged diagnostic ranks eligible tokens using shared newline/punctuation
segment means, then token score and position. Its 10% budget still counts tokens;
these columns do not represent an integer number of complete retrieved sentences.

## Diagnostic positive normalization of baseline output rows

{diagnostic}

These values retain the native-style positive row normalization before summing
selected output rows. For AttnLRP this is the positive normalized aggregate.
They do not replace the signed-sum primary table, and no winning view was chosen
after inspecting baseline scores. The complete [summary CSV](summary.csv) contains
all budgets, and [case scores](per_case.csv) and [selections](selections.json) make
the exact retrieval decisions inspectable.

FT K1 supplementary Recall@10%: {', '.join(LABELS[t]+' '+pct(primary['FT_K1'][t]) for t in TASKS)};
VT macro {pct(primary['FT_K1']['vt_macro'])}.

## Implementation identity and technical corrections

The author source snapshot is commit `075e7e44ae4d5acd2ed76e0d2aced57107d02736`;
all nine dependencies are copied in [source_snapshot](source_snapshot) and bound
by [source hashes](baseline_source_identity.json). The main protocol was frozen
in commit `2bf98cb` before the new baseline quality scores were examined.

Perturbation, CLP and REAGENT use the authors' published **20-source-segment fast
approximation**, with generation-sentence sink groups. This is not exhaustive
single-token perturbation. Native causal-prefix and source-intervention calls
are retained and checked against the same frozen tokenized response. REAGENT
uses the unchanged Longformer replacement and 4,096-token auxiliary input limit;
the [expected auxiliary model ledger](mlm_identity.json) records revision and full hashes,
and the [uploaded-asset receipt](mlm_verification.json) confirms all five actual files.
The uploaded 597,257,159-byte weight file matched SHA-256
`06c56757f0510de87c231acd03650ca204c5ed65c281cd1081b98288183f2468`.

IFR uses the native all-position primitive with chunk sizes 128/32. AttnLRP uses
the native generated-logit weighted aggregate, with normalization disabled and
the same saved 0/1 output weights. The other methods' raw matrices are summed
over those selected output rows. The older result wrapper is bypassed because
it clips signs, normalizes rows and can restrict the sink to the cached answer.
Attribution quantities still differ: perturbation log-probability differences,
the author's KL-like CLP score, IFR proximity, AttnLRP logits and DT's finite
log-probability contrast are not numerically equivalent quantities.

The first AttnLRP pilot failed because its FP16 `output/(input+1e-10)` saved ratio
became 0/0 at zero activations. The [numeric amendment](NUMERIC_FIX.md), frozen
in commit `6f967e4`, re-evaluates only those exact zero/zero ratios with the
existing epsilon in FP32. Forward outputs and all finite ratios are untouched;
other nonfinite cases still fail. The complete run records {repaired:,} repaired
ratios. Original author source files are unchanged, while the runtime adapter
is explicitly amended and labeled †.

HotpotQA then exceeded 64 GiB when retaining the full AttnLRP backward graph.
The [storage amendment](STORAGE_FIX.md) offloads saved activations while preserving
their entire storage, shape, strides and offsets, retaining model parameters on
the GPU. Generic contiguous offload failed the bitwise control and was rejected.
The accepted version reproduced all six stored arrays bit for bit; its input,
target weights and attribution values matched the existing VT H2-C3 pilot.
See the [control receipt](storage_control/verification.json).

All 23 successful pre-offload pilots remain immutable and are accepted only by
their exact registered result hashes in [execution compatibility](execution_compatibility.json).
The three first-pilot Perturbation/CLP/IFR results also match their repeated pilot
vectors bit for bit. Original numerical failures, OOM and the rejected storage
control are retained. DT/FT attribution is never rerun in this experiment.

## Measured run costs

{cost_table}

Total successful attribution-operation time is {analysis['gpu_operation_seconds']/3600:.3f} hours.
The accepted controller's elapsed wall time, including its pilot-resume and full
phases, is {controller_hours:.3f} hours. Its wall interval excludes earlier pilot
attempts and storage controls, whereas the operation total above includes all
successful records, including reused pilots. The two totals cover different
execution intervals and should not be subtracted as an overhead estimate.
These are the observed execution costs of this implementation, including CPU
transfers for offloaded AttnLRP and the auxiliary-model work inside REAGENT.
Peak allocations include models already resident in the shared process. Model
startup, failed attempts, controls, orchestration, transfer and CPU evaluation
are additional. This table is not a separately standardized efficiency benchmark;
the pre-offload VT pilots and subsequent offloaded cases use different storage.

## Artifacts and reproduction

- [Protocol JSON](protocol.json), [input audit](input_audit.json), [all-input CPU preflight](inputs_preflight.json).
- [Analysis and hashes](analysis.json), [full primary CSV](primary_recall10.csv), [raw per-case records](raw).
- [Independent verification](verification.json), [execution controller record](audit/all_baselines_logs_v3/controller.json).
- [Frozen previous VT/DT/FT run](../source_v2_gpu_20260910/full_recall/RESULTS.md), [native HotpotQA v3 evaluation](../hotpot_context_v3_20260910/RESULTS.md).

From the repository root, with NumPy and the recorded dependencies installed:

```bash
python research/temporary/all_baselines_20260910/analyze_all.py --run research/temporary/all_baselines_20260910/raw
python research/temporary/all_baselines_20260910/build_report.py
python research/temporary/all_baselines_20260910/verify_outputs.py
```

Attribution runs require the frozen author environment and checkpoint receipt:

```bash
python research/temporary/all_baselines_20260910/run_controller.py --environment /path/environment.json --preflight /path/inputs_preflight.json --output /path/raw --logs /path/new_logs --methods Perturbation CLP IFR AttnLRP REAGENT --mlm /path/longformer-base-4096
```

The driver refuses unregistered completed records, altered inputs, mismatched
target weights, changed source/checkpoint files, incomplete vectors and any new
generation call. Inputs passed to the GPU runner do not contain retrieval gold.
The independent verifier reconstructs rankings, whole-sentence prefixes, all-token
costs, official fact sets, paired intervals and the unchanged DT/FT evidence.
'''
    (HERE/'RESULTS.md').write_text(text,encoding='utf-8')
    readme=f'''# Frozen VT and HotpotQA: all seven algorithms

The complete result covers 448 paired inputs: 400 VT cases and 48 HotpotQA cases.
Five baselines were newly run for 2,240 method-case records; DT and FT K3 reuse
their verified vectors. The HotpotQA scope is full-response signed sentence sums
under full body-token prefix budgets. VT keeps generated-answer reconstruction
and raw positive eligible-token ranking. The scope choice is retrospective.

{primary_table}

† AttnLRP includes the documented FP16 zero-ratio repair and lossless saved-tensor
offload. Perturbation/CLP/REAGENT use the author's 20-segment approximation.

Read [full results and technical history](RESULTS.md), [protocol](PROTOCOL.md),
[primary CSV](primary_recall10.csv), [case scores](per_case.csv),
[raw vectors](raw), and [independent verification](verification.json).
'''
    (HERE/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps(dict(status='report_written',results_sha256=sha(HERE/'RESULTS.md'))))

if __name__=='__main__':main()
