"""Design-based statistics for the frozen finite-population diagnostic sample.

SciPy owns the hypergeometric probabilities. These are sampling intervals,
never FA/FLA tolerances or claims about counterfactual estimation accuracy.
"""
import math


def hypergeom_interval(N, n, x, alpha=.05):
    """Invert the exact finite-population distribution of a SRSWOR count."""
    from scipy.stats import hypergeom
    assert 0 < n <= N and 0 <= x <= n and 0 < alpha < 1
    if n == N:
        return [x, x]
    first, last = x, N-n+x
    lo, hi = first, last
    while lo < hi:
        mid = (lo+hi)//2
        if hypergeom.sf(x-1, N, mid, n) >= alpha/2:
            hi = mid
        else:
            lo = mid+1
    lower = lo
    lo, hi = first, last
    while lo < hi:
        mid = (lo+hi+1)//2
        if hypergeom.cdf(x, N, mid, n) >= alpha/2:
            lo = mid
        else:
            hi = mid-1
    return [lower, lo]


def positive_bounds(d_interval, threshold):
    """Observed reference sensitivity bounds, NOT confidence probabilities."""
    lo, hi = d_interval
    cut = -math.log(threshold)
    return int(hi < cut), int(lo < cut)


def summarize_task(spec, observations, threshold=2, domain_state=None, alpha=.05):
    """A single confusion matrix with known design weights and FN bounds.

    All sampled outcomes must be present. Missing/incomplete work cannot be
    treated as random missingness or as TN. A state domain uses indicators for
    that state under the same global probability sample, not a new sample.
    """
    assert threshold in (2,10,100)
    entries = {e['traj_uid']:e for e in spec['entries']}
    expected = {(u,q['packed_slot']):q for u,e in entries.items() for q in e['queries']}
    observed = {(o['traj_uid'],o['packed_slot']):o for o in observations}
    assert len(observed) == len(observations) and set(observed) == set(expected)
    cells = {p+':'+r:0. for p in ('predicted_tail','predicted_not_tail')
        for r in ('reference_tail','reference_not_tail','reference_unresolved')}
    equal_state_cells = dict(cells)
    sampled_by_stratum = {}
    TP = [0,0]
    FN_census = [0,0]
    sampled_reference_positives = 0
    sampled_positive_states = set()
    sample_rows = 0
    for key, query in expected.items():
        o = observed[key]
        assert o['token_id'] == query['token_id']
        entry = entries[key[0]]
        inside = domain_state is None or entry['initial_state_sha256'] == domain_state
        low, high = positive_bounds(o['native_d_interval'],threshold)
        low, high = low*inside, high*inside
        stratum = query['prediction_stratum']
        if stratum in ('ratio_le_1','ratio_1_to_2'):
            count = sampled_by_stratum.setdefault(stratum,[0,0])
            count[0] += low
            count[1] += high
        if not inside:
            continue
        sample_rows += 1
        pred = query['saved_d'] < -math.log(threshold)
        status = 'reference_tail' if low else 'reference_unresolved' if high else 'reference_not_tail'
        cell = ('predicted_tail' if pred else 'predicted_not_tail')+':'+status
        cells[cell] += query['token_total_weight']
        equal_state_cells[cell] += query['state_trajectory_source_weight']
        if low:
            sampled_reference_positives += 1
            sampled_positive_states.add(entry['initial_state_sha256'])
        if stratum not in ('ratio_le_1','ratio_1_to_2'):
            bounds = TP if pred else FN_census
            bounds[0] += low
            bounds[1] += high
    strata = [h for h in spec['strata'] if h['name'] in ('ratio_le_1','ratio_1_to_2') and h['population']]
    FN_sampling = [0,0]
    confidence_components = []
    for h in strata:
        N, n = h['population'],h['sample']
        low, high = sampled_by_stratum[h['name']]
        # One lower and one upper tail per stratum: Bonferroni total alpha.
        a = alpha/len(strata)
        lower = hypergeom_interval(N,n,low,a)[0]
        upper = hypergeom_interval(N,n,high,a)[1]
        domain_size = sum(e['source_strata_populations'][h['name']] for e in entries.values()
            if domain_state is None or e['initial_state_sha256'] == domain_state)
        upper = min(upper,domain_size)
        assert lower <= upper
        FN_sampling[0] += lower
        FN_sampling[1] += upper
        confidence_components.append(dict(stratum=h['name'],population=N,sample=n,
            sampled_certain_positives=low,sampled_possible_positives=high,
            native_positive_population_interval=[lower,upper]))
    FN = [a+b for a,b in zip(FN_census,FN_sampling)]
    recall_lo = TP[0]/(TP[0]+FN[1]) if TP[0]+FN[1] else None
    recall_hi = TP[1]/(TP[1]+FN[0]) if TP[1]+FN[0] else None
    predicted_positive_count = sum(e['source_strata_populations'][h['name']]
        for e in entries.values() for h in spec['strata']
        if h['name'] in (('ratio_2_to_10','ratio_10_to_100','ratio_gt_100') if threshold == 2 else
            ('ratio_10_to_100','ratio_gt_100') if threshold == 10 else ('ratio_gt_100',))
        and (domain_state is None or e['initial_state_sha256'] == domain_state))
    certain_FN = cells['predicted_not_tail:reference_tail']
    possible_FN = certain_FN+cells['predicted_not_tail:reference_unresolved']
    estimate_lo = TP[0]/(TP[0]+possible_FN) if TP[0]+possible_FN else None
    estimate_hi = TP[1]/(TP[1]+certain_FN) if TP[1]+certain_FN else None
    return dict(threshold=threshold,domain_initial_state=domain_state,
        complete_probability_sample=True,sampled_positions_in_domain=sample_rows,
        sampled_certain_native_positives=sampled_reference_positives,
        sampled_positive_initial_states=len(sampled_positive_states),
        weighted_token_total_confusion=cells,
        equal_state_trajectory_source_confusion_mass=equal_state_cells,
        TP_reference_range=TP,FN_sampling_and_reference_interval=FN,
        precision_reference_range=[x/predicted_positive_count for x in TP] if predicted_positive_count else None,
        recall_HT_estimate_reference_range=[estimate_lo,estimate_hi],
        recall_sampling_and_reference_interval=[recall_lo,recall_hi],
        confidence_level=1-alpha,
        confidence_scope='Finite completed saved frame, pointwise for this threshold/domain. '
            'Not simultaneous across reported thresholds/states; reference sensitivity bounds are conditional assumptions, not statistical coverage.',
        confidence_components=confidence_components,
        interpretation='Weighted counts estimate one common finite population. Predicted-tail census contributes TP/FP; '
            'probability samples supply FN/TN. None denotes an undefined ratio, not zero recall.')
