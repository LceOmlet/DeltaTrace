"""Record one bounded composition hypothesis before any model query.

This records research evidence, not a second RL method specification.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def file(path):
    path=Path(path)
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    before=HERE/'conditional-gdn-collection-v1/textcraft-analysis.json'
    result=json.loads(before.read_bytes())
    assert result['complete'] and not result['candidate_accepted']
    overlap=result['existing_candidate_overlap']
    assert overlap['original_robust_spurious_points']==18
    scripts=[HERE/'conditional_gdn_candidate.py',
        HERE/'conditional-attention-owner-v1/inspect_conditional_collection.py',
        HERE/'conditional-attention-owner-v1/submit_conditional_collection.py',
        HERE/'observe_conditional_gdn_collection.py',
        HERE/'preserve_conditional_gdn_collection.py',HERE/'analyze_conditional_gdn_collection.py']
    for path in scripts:ast.parse(path.read_text(encoding='utf8'))
    value=dict(status='derived_unaccepted_research_only',
        git_at_preparation=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        method_owner=file(REPO/'experiments/rl/PLAN.md'),
        PPO_debug=file(REPO/'experiments/rl/results_preserved_actor_debug_20261008.json'),
        observable_need='Reduce inaccurate extreme negative token deletion effects while preserving the original author cumulative-deletion quality and bounded all-source DT cost.',
        observed_evidence=[file(REPO/'experiments/rl/results_single_background_completed_20261009.json'),
            file(before),file(HERE/'conditional-attention-owner-v1/textcraft-v3-analysis.json')],
        existing_overlap={k:v for k,v in overlap.items() if k not in ('points','sources')},
        interpretation='Exact single-background diagnostics remove the robust false tails; FA-only and GDN-only conditional rules affect different positions but each failed the whole collection. The overlap is motivation for one coherent composition, not a predicted combined improvement or permission to select output per token.',
        construction=[
            'For a mixer M and source position i, use the existing conditional finite rule anchored at factual other positions: contract the incoming DT cotangent with M(x_factual)-M(x_factual with its local i input replaced).',
            'Apply that same factual-context convention in both existing FA and GDN owner seams throughout the preserved runner, rather than mixing a conditional mixer with the other joint-background mixer.',
            'Pointwise norms, gate/projection, head seed, residual rules and incoming DT cotangent propagation remain existing owners. This is a composed approximation, not a claim of exact global single deletion.',
            'Build the FA candidate using the preserved builder, then bind only the GDN callback on a private copy of its exact attribute function globals. Preserve every other global object and the same code object; reimporting the runner would silently lose the FA seam.',
            'Use both seams for all source tokens and layers, independent of observed sign/magnitude. Never pick whichever candidate agrees with a native single deletion.'
        ],
        ownership=dict(native_model='Unchanged HF/PEFT and capture/model objects',
            FA='Previously checked public FA conditional endpoint composition and existing compiled finite FA',
            GDN='Previously checked original FLA readout, public causal convolution/l2norm and conditional width-four memory composition',
            RL='Unchanged PLAN Q/V/A, original whole-batch whitening and VERL PPO; no training executed'),
        numeric_verification=[file(REPO/'experiments/rl/results_conditional_attention_composition_20261008.json'),
            file(REPO/'experiments/rl/results_conditional_gdn_20261009.json')],
        numeric_scope='The same already checked primitive sources and original dtype thresholds are reused unchanged. Binding identities and paired factual model scores are checked separately; no official tolerance for a whole finite-attribution scalar is invented.',
        frozen_comparison=dict(task='textcraft',primary_trajectories=32,initial_states=16,
            extra_tail_trajectories=13,DT_B4_calls_per_rank=6,native_author_forward_calls_per_rank=84,
            native_single_deletions_reused=True,existing_worker_wall_budget_seconds=1800,
            heldout_used=False,AppWorld_launched=False,optimizer=0,rollout=0,checkpoint_restore=0),
        cost='One DT per B4. No new model queries beyond the frozen original author diagnostic. Added work is the sum of the two already measured mixer seams; GDN alone was about 2.02x DT. This candidate may therefore remain too costly; quality and observed cost are both reported without an invented threshold.',
        decision='Run this one frozen composition to test the complete sequence-mixer-background hypothesis. If it fails, preserve its result and reject the construction; do not tune it on the same set or launch AppWorld/heldout automatically.',
        sources=[file(p) for p in scripts],production_modified=False)
    output=HERE/'conditional-mixers-derivation.json'
    output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(file(output)))


if __name__=='__main__':main()
