"""CPU tests of sampling arithmetic, not substitute DT/FA/FLA evidence."""
from itertools import combinations, product
import json
import math
import time

from prepare_tail_probability_sample import allocate
from tail_probability_statistics import hypergeom_interval, positive_bounds


def main():
    from scipy.stats import hypergeom
    begin = time.perf_counter()
    checked = 0
    worst = 1.
    # Exhaustive finite distributions, including zero observed positives and
    # census cases. All probabilities come from the installed SciPy owner.
    for N in range(1,13):
        for n in range(1,N+1):
            intervals = [hypergeom_interval(N,n,x) for x in range(n+1)]
            for K in range(N+1):
                coverage = sum(float(hypergeom.pmf(x,N,K,n)) for x,(lo,hi) in enumerate(intervals) if lo <= K <= hi)
                assert coverage >= .95-1e-12
                worst = min(worst,coverage)
                checked += 1
    zero = hypergeom_interval(10000,128,0)
    assert zero[0] == 0 and zero[1] > 0
    assert hypergeom_interval(100,100,3) == [3,3]
    assert positive_bounds([-.8,-.7],2) == (1,1)
    assert positive_bounds([-.8,-.6],2) == (0,1)
    assert positive_bounds([-.6,.2],2) == (0,0)
    assert allocate([98948,22263],1024) == [836,188]
    assert allocate([166304,38524],1024) == [831,193]
    # Enumerate the whole two-stratum sampling design. Weighted TP/FP/FN/TN
    # totals, unlike concatenated raw counts, recover the known population.
    strata = [[('TP',),('FP',),('FP',)], [('FN',),('TN',),('TN',),('FN',),('TN',)]]
    samples = [list(combinations(range(len(h)),2)) for h in strata]
    totals = {c:0. for c in ('TP','FP','FN','TN')}
    for a,b in product(*samples):
        for h,indices in zip(strata,(a,b)):
            for i in indices:
                totals[h[i][0]] += len(h)/2
    designs = math.prod(len(s) for s in samples)
    assert {c:v/designs for c,v in totals.items()} == dict(TP=1.,FP=2.,FN=2.,TN=3.)
    print(json.dumps(dict(scope=__doc__,status='passed',finite_distributions=checked,
        minimum_coverage=worst,zero_observed_positive_interval=zero,
        inverse_inclusion_enumeration_designs=designs,seconds=time.perf_counter()-begin,
        operations=dict(model=0,DT=0,FA=0,FLA=0,GPU=0,optimizer=0))))


if __name__ == '__main__':
    main()
