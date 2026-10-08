"""Independent small regression checks for continuing square weights past k=3."""
import numpy as np
import pytest
from GPU_REML.blup_effect_weighting import effect_power_variance
from GPU_REML.iterative_weighted_ridge import profiled_validation


@pytest.mark.parametrize('seed', [31, 94])
def test_resume_from_third_endpoint_matches_ten_dense_primal_updates(seed):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(29,18));x[:,4]=x[:,3]
    groups=np.arange(18)%4;c=np.array([.02,.03,0.,.01])[groups]
    norms=np.sum(x*x,axis=0);t=.7;r=rng.normal(size=len(x))
    original=c*(x.T@np.linalg.solve((x*c)@x.T+t*np.eye(len(x)),r))
    oracle=original.copy();chain=[]
    for iteration in range(1,11):
        d=np.zeros_like(c)
        for group in range(4):
            use=groups==group
            if c[use][0]==0:continue
            raw=c[use]*oracle[use]**2
            d[use]=c[use] if raw@norms[use]==0 else raw*(c[use]@norms[use])/(raw@norms[use])
        active=d>0;xx=x[:,active]
        following=np.zeros_like(c)
        # Scale eta=sqrt(d)*q: the precision is I+sqrt(D)X'Xsqrt(D)/t.
        # This remains finite when repeated squaring makes d subnormal.
        root=np.sqrt(d[active]);design=xx*root
        following[active]=root*np.linalg.solve(
            design.T@design/t+np.eye(len(root)),design.T@r/t)
        chain.append((d.copy(),following.copy()));oracle=following
    pilot=chain[2][1].copy()
    for iteration in range(4,11):
        d,_=effect_power_variance(pilot,c,norms,groups,2.)
        actual=d*(x.T@np.linalg.solve((x*d)@x.T+t*np.eye(len(x)),r))
        # Nonlinear differences can amplify; compare independent paths at
        # each finite endpoint, not a claim of a converged stationary point.
        np.testing.assert_allclose(actual,chain[iteration-1][1],atol=2e-12,rtol=2e-9)
        np.testing.assert_allclose(d,chain[iteration-1][0],atol=2e-12,rtol=2e-9)
        assert np.all(actual[c==0]==0)
        for group in range(4):
            use=groups==group
            assert d[use]@norms[use]==pytest.approx(c[use]@norms[use],rel=2e-14)
        pilot=actual


def test_calibrate_each_endpoint_against_original_anchor_not_previous_endpoint():
    base=np.array([1.,2.,-1.]);y=np.array([1.1,1.8,-.7])
    directions=[np.array([.2,-.1,.4]),np.array([.8,-.3,.9]),np.array([0.,0.,0.])]
    for direction in directions:
        result=profiled_validation(y,base,direction)
        expected=0. if not np.any(direction) else np.clip(direction@(y-base)/(direction@direction),0,1)
        assert result['rho']==pytest.approx(expected,abs=2e-15)
        assert result['mse']<=np.mean((y-base)**2)+1e-15
