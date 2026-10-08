import numpy as np
import pytest
from GPU_REML.iterative_weighted_ridge import (
    ValidationStopper, POLICIES, fit_iterative_background, build_profiled_information)


def test_selection_order_and_early_stopping_are_distinct():
    y=np.array([1.,1.,-1.,-1.]);orth=np.array([1.,-1.,1.,-1.])
    directions=[.9*y+.5*orth,3*y+.5*orth,6*y+.2*orth,3*y+2*orth]
    rules=[]
    for policy in POLICIES:
        stopper=ValidationStopper(y,np.zeros(4),policy)
        for k,direction in enumerate(directions,1):
            if not stopper.active:break
            stopper.observe(k,direction)
        rules.append(stopper.rule())
    assert [r['iteration'] for r in rules]==[1,3]
    assert [r['evaluated_iterations'] for r in rules]==[2,4]
    for rule in rules:
        d=directions[rule['iteration']-1]
        assert rule['rho']==pytest.approx(np.clip(d@y/(d@d),0,1))


def test_tie_stops_and_keeps_earlier_endpoint():
    stopper=ValidationStopper(np.ones(3),np.zeros(3),POLICIES[0])
    stopper.observe(1,np.ones(3)*.8);stopper.observe(2,np.ones(3)*1.2)
    assert not stopper.active and stopper.rule()['iteration']==1
    with pytest.raises(ValueError):stopper.observe(3,np.ones(3))


@pytest.mark.parametrize('seed',[38,92])
def test_shared_path_and_transferred_refit_match_independent_dense_equations(seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=(19,12));xv=rng.normal(size=(8,12))
    groups=np.arange(12)%4;c=np.array([.02,.01,0.,.04])[groups];noise=.9
    norms=np.sum(x*x,axis=0);r=rng.normal(size=19);yv=rng.normal(size=8)
    u0=np.linalg.solve((x*c)@x.T+noise*np.eye(19),r);eta0=c*(x.T@u0);calls=[]
    certificate=dict(original_system_verified=True,maximum_relative_residual=0.)
    def solve(d,warm):
        calls.append(d.copy());u=np.linalg.solve((x*d)@x.T+noise*np.eye(19),r)
        return u,d*(x.T@u),certificate
    result=fit_iterative_background(eta0,c,norms,groups,solve,initial_dual=u0,
        validation_y=yv,validation_base=xv@eta0,validation_products=lambda e:xv@e)
    oracle=eta0.copy();effects=[];raw_losses=[];cal_losses=[];rhos=[]
    for k in range(len(calls)):
        d=np.zeros(12)
        for g in range(4):
            use=groups==g;budget=c[use]@norms[use]
            if budget==0:continue
            raw=c[use]*oracle[use]**2
            d[use]=c[use] if raw@norms[use]==0 else raw*budget/(raw@norms[use])
        np.testing.assert_allclose(d,calls[k],atol=2e-14,rtol=2e-13)
        active=d>0;root=np.sqrt(d[active]);design=x[:,active]*root
        oracle=np.zeros(12);oracle[active]=root*np.linalg.solve(
            np.eye(len(root))+design.T@design/noise,design.T@r/noise)
        effects.append(oracle.copy());direction=xv@(oracle-eta0);base=xv@eta0
        rho=float(np.clip(direction@(yv-base)/(direction@direction),0,1))
        rhos.append(rho);raw_losses.append(np.mean((yv-base-direction)**2))
        cal_losses.append(np.mean((yv-base-rho*direction)**2))
    for p,losses in zip(POLICIES,[raw_losses,cal_losses]):
        best=0
        for i in range(1,len(losses)):
            if losses[i]<losses[best]-1e-12:best=i
            else:break
        rule=result['rules'][p];assert rule['iteration']==best+1
        np.testing.assert_allclose(result['effects'][p],eta0+rhos[best]*(effects[best]-eta0),atol=2e-13)
    calls.clear()
    final=fit_iterative_background(eta0,c,norms,groups,solve,initial_dual=u0,rules=result['rules'])
    assert len(calls)==max(r['iteration'] for r in result['rules'].values())
    for p in POLICIES:np.testing.assert_array_equal(final['effects'][p],result['effects'][p])
    with pytest.raises(ValueError):
        fit_iterative_background(eta0,c,norms,groups,solve,rules=result['rules'],validation_y=yv)


@pytest.mark.parametrize('empty',[False,True])
def test_direct_profiled_information_matches_dense_projection(empty):
    rng=np.random.default_rng(154);x=rng.normal(size=(17,13));c=rng.uniform(.01,.04,17);t=.7
    y=rng.normal(size=13);cov=np.column_stack((np.ones(13),rng.normal(size=13)))
    s=np.array([],int) if empty else np.array([1,4,8,15]);old=rng.normal(size=len(s))
    v=(x.T*c)@x+t*np.eye(13)
    class Backend:
        def solve_precise(self,rhs,variance,noise):
            return np.linalg.solve(v,rhs),dict(original_system_verified=True,maximum_relative_residual=0.)
    z,record=build_profiled_information(x,s,old,y,cov,c,t,Backend(),block_size=2)
    w=np.linalg.inv(v);p=w-w@cov@np.linalg.solve(cov.T@w@cov,cov.T@w)
    np.testing.assert_allclose(z['gram'],x[s]@p@x[s].T,atol=2e-14)
    np.testing.assert_allclose(z['score'],x[s]@p@y,atol=2e-14)
    assert record['numeric_accepted'] and 'vinv_x' not in z


def test_uncertified_endpoint_is_rejected():
    def bad(d,warm):return np.ones(3),np.ones(4),dict(original_system_verified=False)
    with pytest.raises(ArithmeticError):
        fit_iterative_background(np.ones(4),np.ones(4),np.ones(4),np.arange(4),bad,
            validation_y=np.ones(3),validation_base=np.zeros(3),validation_products=lambda e:np.ones(3))
