"""Algebra review of an isolated DT log-softmax finite-rule hypothesis.

No model/DT calls. These identities are not attribution-quality evidence,
an official tolerance, or permission to deploy an untested rule.
"""
import hashlib
import json
from pathlib import Path
import mpmath as mp
import sympy as sp

HERE = Path(__file__).resolve().parent


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    # KL identities for exact normalized endpoint distributions. The mixture
    # weight is determined by the secant, not fitted to attribution residuals.
    lse, dot0, dot1 = sp.symbols('deltaLSE p0_dot_delta p1_dot_delta')
    kl0, kl1 = lse-dot0, dot1-lse
    weight = kl0/(kl0+kl1)
    closure = sp.cancel((1-weight)*dot0+weight*dot1-lse)
    symmetry = sp.cancel(weight+kl1/(kl0+kl1)-1)
    assert closure == symmetry == 0
    a = sp.symbols('a', positive=True)
    fraction = 1/a-1/(sp.exp(a)-1)
    series = sp.series(fraction, a, 0, 10)
    # This is a scalar series approximation domain, not clipping or a new
    # training parameter. Omitted term at a=1/2 is below FP32 epsilon.
    omitted = sp.Rational(1, 2)**9/sp.Integer(47900160)
    assert omitted < sp.Rational(1, 2)**24

    # Reject our prior coordinate-group composition before a GPU experiment.
    # An exact local invariant counterexample is sufficient to refute that
    # rule, but is not population evidence about DeepLIFT or DT quality.
    mp.mp.dps = 70
    def lse_mp(z):
        return mp.log(sum(mp.exp(x) for x in z))
    def old_seed(z0, z1, target=0):
        p0 = [mp.exp(x-lse_mp(z0)) for x in z0]
        p1 = [mp.exp(x-lse_mp(z1)) for x in z1]
        mean = [(v-u)/(mp.log(v)-mp.log(u)) if v != u else u
                for u, v in zip(p0, p1)]
        total = sum(mean)
        return [(1 if i == target else 0)-v/total for i, v in enumerate(mean)]
    z0 = list(map(mp.mpf, [0, 0, 0, -100]))
    z1 = list(map(mp.mpf, [16, -8, -8, -100]))
    plus = [max(u, v) for u, v in zip(z0, z1)]
    minus = [min(u, v) for u, v in zip(z0, z1)]
    seeds = [old_seed(z0, plus), old_seed(minus, z1),
             old_seed(z0, minus), old_seed(plus, z1)]
    joined = [(seeds[0][i]+seeds[1][i])/2 if z1[i] >= z0[i]
              else (seeds[2][i]+seeds[3][i])/2 for i in range(len(z0))]
    centered = [v-sum(joined)/len(joined) for v in joined]
    assert centered[3] > 0  # A competing logit must have nonpositive seed.
    counterexample = dict(z0=list(map(str, z0)), z1=list(map(str, z1)), target=0,
        competing_class=3, centered_seed=str(centered[3]),
        conclusion='Our centered coordinate-group composition can assign a positive coefficient to a competing logit; reject this hypothesis without GPU testing.',
        scope='Refutes a universal local monotonicity claim; not a benchmark or overall attribution claim.')

    owners = json.loads((HERE/'actual-head-seed-owner.json').read_bytes())
    seed_owner = next(f for f in owners['files'] if f['module'] == 'compiled_logprob_seed')
    value = dict(scope=__doc__, status='derived_only_unaccepted',
        owner={k:v for k,v in seed_owner.items() if k!='source'},
        evidence=[ref(HERE/('layer-suboperations-'+task+'-v2-analysis.json'))
                  for task in ('textcraft','appworld')],
        observed_need='On both fixed collections, the nonlinear log-softmax background residual is a major head term, while seed recomputation and linear projection differences are small. This does not establish that replacing this rule alone fixes attribution.',
        original_rule='q_j=LogMean(p0_j,p1_j)/sum_k LogMean(p0_k,p1_k). This is a valid conservative finite rule, not a missing-normalization bug.',
        hypothesis='When endpoints differ, the sum of unnormalized logarithmic means is below one. Normalizing can increase even a class with unchanged endpoint probability. Test a conservative rule whose class weights stay within actual endpoint probabilities.',
        proposed_rule=dict(p0='softmax(z0)', p1='softmax(z1)',
            D0='KL(p0||p1)', D1='KL(p1||p0)',
            mixture_weight='D0/(D0+D1); 1/2 at equal endpoint distributions',
            q='(1-mixture_weight)*p0+mixture_weight*p1',
            seed='one_hot(actual_target)-q'),
        algebra=dict(endpoint_identity=str(closure), pair_symmetry=str(symmetry),
            derivation=['D0=deltaLSE-p0.dot(delta_z)', 'D1=p1.dot(delta_z)-deltaLSE',
                        'q.dot(delta_z)=deltaLSE', 'seed.dot(delta_z)=delta log p(target)'],
            properties=['q is a probability distribution', 'each q_j lies between p0_j and p1_j',
                        'competing logits have nonpositive seed', 'sum(seed)=0',
                        'endpoint exchange symmetry', 'class permutation equivariance',
                        'coincident limit equals native log-softmax gradient'],
            uniqueness='Unique on the line segment between p0 and p1 when they differ. No claim of globally optimal token deletion attribution.',
            exact_token_oracle_premise_unchanged=True),
        stable_evaluation=dict(u='logp1-logp0', a='abs(u)',
            J='exp(max(logp0,logp1))*a*(-expm1(-a))',
            fraction='1/a-exp(-a)/(-expm1(-a))',
            small_fraction=str(series), small_domain='a<=1/2',
            first_omitted_at_half=str(omitted),
            D0_terms='J*fraction if u>=0; J*(1-fraction) otherwise',
            D1_terms='J*(1-fraction) if u>=0; J*fraction otherwise',
            numerics='Sum nonnegative terms; no signed KL cancellation, exp(+abs(u)), credit clipping, or residual-dependent multiplier.'),
        prior_coordinate_group_candidate=dict(status='rejected_before_GPU', counterexample=counterexample,
            receipt=ref(HERE/'conditioned-head-derivation.json')),
        cost=dict(DT_calls_per_trajectory=1, extra_model_forwards=0, head_reverse_GEMMs=1,
                  scalar_work='O(number_of_actual_target_rows * vocabulary); constant factor, not source-by-target calls',
                  bounded_row_temporary=128,
                  not_a_measured_speed_or_capacity_claim=True),
        limits=['Head change must propagate through the complete original reverse computation.',
                'Decoder/head residuals interact; no subtraction of diagnostic terms.',
                'Mathematical identities do not predict whole-collection RISE/MAS improvement.',
                'No FA/FLA kernel change, no new tolerance, no Q/V/A or PPO change.'],
        quality_set='Same frozen TextCraft32 trajectories/16 states, original author cumulative deletion/signed RISE/positive MAS; same uniform/tail cells separately. No held-out tuning.',
        production_modified=False, formal_restart=False, model_calls=0, DT_calls=0,
        accepted_candidate=False, recorder=ref(Path(__file__)))
    out = HERE/'endpoint-head-derivation.json'
    out.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    prior_path = HERE/'conditioned-head-derivation.json'
    prior = json.loads(prior_path.read_bytes())
    prior.update(status='rejected_before_GPU', rejection=counterexample,
                 candidate_ready=False, accepted_candidate=False)
    prior['research_decision']['candidate_not_ready'] = True
    prior_path.write_text(json.dumps(prior, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(out), identities=[str(closure), str(symmetry)],
                         prior_rejected=True, model_calls=0)))


if __name__ == '__main__':
    main()
