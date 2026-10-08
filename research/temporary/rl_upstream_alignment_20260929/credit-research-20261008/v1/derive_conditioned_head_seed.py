"""Research derivation only: condition the existing finite log-softmax seed.

No new training plan, model call, kernel, deployment or claimed accuracy gain.
This composition is not an official DeepLIFT implementation. The owning DT
seed remains the primitive; RevealCancel motivates averaging two group orders.
"""
import hashlib
import json
from pathlib import Path
import sympy as sp

HERE = Path(__file__).resolve().parent


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    # Symbols are the four evaluations of the SAME output function. This is
    # an algebraic identity, not synthetic evidence of attribution quality.
    a,b,c,d = sp.symbols('f0 fPlus fMinus f1')
    positive = (b-a+d-c)/2
    negative = (c-a+d-b)/2
    closure = sp.expand(positive+negative-(d-a))
    assert closure == 0
    m = sp.symbols('m0:3');delta=sp.symbols('d0:3')
    mean_m=sum(m)/3;mean_delta=sum(delta)/3
    gauge_identity=sp.expand(sum((v-mean_m)*x for v,x in zip(m,delta))-
                             sum(v*(x-mean_delta) for v,x in zip(m,delta)))
    assert gauge_identity == 0
    protocol = json.loads((HERE/'suboperation-protocol.json').read_bytes())
    sources = json.loads((HERE/'actual-head-seed-owner.json').read_bytes())
    seed_owner = next(f for f in sources['files'] if 'compiled_logprob_seed' in f['path'])
    rule_owner = next(f for f in sources['files'] if 'signed_secant_rules' in f['path'])
    analysis = HERE/'layer-suboperations-textcraft-v2-analysis.json'
    assert json.loads(analysis.read_bytes())['complete']
    value = dict(scope=__doc__,status='derived_only_no_GPU_candidate',
        evidence=[ref(analysis),ref(HERE/'suboperation-protocol.json')],
        owners=dict(seed={k:v for k,v in seed_owner.items() if k!='source'},
                    existing_group_rule={k:v for k,v in rule_owner.items() if k!='source'}),
        references=[dict(url='https://proceedings.mlr.press/v70/shrikumar17a/shrikumar17a.pdf',
                         scope='Section3.5.3: average positive/negative contributions in two orders; not a theorem about token deletion or this vector seed'),
                    dict(url='https://github.com/kundajelab/deeplift/blob/master/README.md',
                         scope='Official implementation distinguishes RevealCancel from Rescale; its Keras framework is not imported or recreated')],
        observed_need='Measured opposite large FA/head terms on the frozen collection, with head background residuals much larger than projection rounding. A consistent new finite seed must be propagated through the entire original reverse computation; never subtract the measured residual.',
        composition=dict(
            target='Same log_softmax(z)[actual_target] and same original paired logits',
            canonical_coordinates='x0=z0-mean(z0), x1=z1-mean(z1); linear projection removes the common-logit null direction',
            intermediate_states='xPlus=maximum(x0,x1), xMinus=minimum(x0,x1); only local scalar-rule evaluations, no model forward',
            positive_coordinates='0.5*(seed(x0,xPlus)+seed(xMinus,x1))',
            negative_coordinates='0.5*(seed(x0,xMinus)+seed(xPlus,x1))',
            joined_seed='Select each coordinate by sign(x1-x0), then transpose the explicit centering map: m=mJoined-mean(mJoined)',
            pullback='Original _linear_transpose exactly once, followed by original DT propagation',
            probabilities='Do not alter native logits, target labels, event probability, reward, d-to-Q/V/A map or PPO'),
        algebra=dict(two_group_endpoint_identity=str(closure),centering_transpose_identity=str(gauge_identity),
            general_centering_identity='sum((m-mean(m))*delta)=sum(m*(delta-mean(delta))) for any dimension',
            endpoint_completeness='The two masked group contractions add to f(x1)-f(x0)=f(z1)-f(z0); no multiplier is fitted to an observed residual',
            shift_invariance='Both endpoints are first projected to the same zero-mean-logit space and the covector is pulled back through that exact linear map',
            pair_symmetry='Exchanging endpoints swaps Plus/Minus and reverses paths; the existing symmetric seed yields the same covector',
            coincident_limit='All four owner seeds approach the original log-softmax gradient; its mean is zero',
            class_permutation_equivariance='Mean, maximum/minimum and the original seed commute with a common class permutation'),
        rejected_naive_design='Uncentered coordinate group composition need not have sum(seed)=0 and can attribute an irrelevant common-logit translation. Do not implement that version or repair token credits afterward.',
        work=dict(existing_DT_calls=1,additional_DT_calls=0,additional_native_forwards=0,
            finite_logsoftmax_seed_evaluations=4,original_head_reverse_GEMMs=1,
            row_readout_chunk=128,scope='Constant extra scalar-head operations. No source-by-source model replay or source-times-target DT requests.',
            per_chunk_vocabulary_FP32_bytes=128*protocol['model_config']['vocab']*4,
            memory='Sequentially consume seed pairs; observe actual memory before any deployment. This derivation is not a capacity test.'),
        research_decision=dict(
            candidate_not_ready=True,
            prerequisite='Bind completed AppWorld suboperation results, then review whether the same measured head mismatch supports this one hypothesis',
            first_quality_set='Unchanged frozen32 TextCraft trajectories/16 states, original cumulative deletion, signed RISE and positive MAS; separate frozen uniform/tail cells',
            avoid='No hand-selected examples as method evidence, no held-out tuning, no retry of rejected FA candidate, no extra top-k deletion rescue',
            rejection='A mathematically conservative seed is not sufficient. If whole-collection author quality worsens, discard this hypothesis rather than compensate or change the credit formula'),
        exact_oracle_premise_unchanged=True,global_token_accuracy_proved=False,
        original_tolerance_unchanged=True,official_tolerance_test=False,
        accepted_candidate=False,production_modified=False,formal_restart=False,
        CUDA_initialized=False,model_calls=0,DT_calls=0,recorder=ref(Path(__file__)))
    output=HERE/'conditioned-head-derivation.json'
    output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),algebra_identities=[str(closure),str(gauge_identity)],
                         GPU_calls=0,candidate_ready=False)))


if __name__=='__main__':
    main()
