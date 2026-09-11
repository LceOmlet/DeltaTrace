"""FP64 dense algebra check; separate from native GPU finite-kernel execution."""
import hashlib
import json
from pathlib import Path

import numpy as np


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def coefficients(q0, k0, v0, q1, k1, u):
    scale = q0.shape[1] ** -.5
    p0, p1 = softmax(scale*q0@k0.T), softmax(scale*q1@k1.T)
    dlog = np.log(p1)-np.log(p0)
    lm = np.divide(p1-p0, dlog, out=p0.copy(), where=abs(dlog)>1e-12)
    edge = u@v0.T
    ds = lm*(edge-(lm*edge).sum(axis=1,keepdims=True)/lm.sum(axis=1,keepdims=True))
    return (scale*ds@(.5*(k0+k1)), scale*ds.T@(.5*(q0+q1)), p1.T@u)


def main():
    rng = np.random.default_rng(73)
    max_product = max_contraction = max_direct = 0.
    for case in range(100):
        q0,k0,v0,q1,k1,v1,u = [rng.normal(size=(7,5)) for _ in range(7)]
        if case % 5 == 0:
            q1,k1 = q0.copy(),k0.copy()
        if case % 5 == 1:
            v1 = v0.copy()
        scale = 5**-.5
        p0,p1 = softmax(scale*q0@k0.T),softmax(scale*q1@k1.T)
        actual = p1@v1-p0@v0
        decomposed = (p1-p0)@(.5*(v0+v1)) + (.5*(p0+p1))@(v1-v0)
        max_product = max(max_product,float(abs(actual-decomposed).max()))
        fwd = coefficients(q0,k0,v0,q1,k1,u)
        rev = coefficients(q1,k1,v1,q0,k0,u)
        averaged = [.5*(a+b) for a,b in zip(fwd,rev)]
        direct_qk = coefficients(q0,k0,.5*(v0+v1),q1,k1,u)
        direct = [direct_qk[0],direct_qk[1],(.5*(p0+p1)).T@u]
        max_direct = max(max_direct,max(float(abs(a-b).max()) for a,b in zip(averaged,direct)))
        contraction = sum(float((a*d).sum()) for a,d in zip(averaged,[q1-q0,k1-k0,v1-v0]))
        max_contraction = max(max_contraction,abs(contraction-float((u*actual).sum())))
    assert max(max_product,max_direct,max_contraction)<1e-10
    report = dict(status='passed',cases=100,seed=73,precision='float64',
        max_PV_identity_error=max_product,max_symmetric_multiplier_error=max_direct,
        max_QKPV_contraction_error=max_contraction,
        scope='Dense algebra only. Native kernel, half precision, full model and quality require separate GPU checks.',
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (Path(__file__).parent/'symmetric_algebra_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
