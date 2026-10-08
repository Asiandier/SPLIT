import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from GPU_REML.matrix_free_inverse_quadratics import estimate_inverse_quadratics


class DenseOracle:
    def __init__(self, x):
        self.x=x
    def solve_precise(self, rhs, variance, noise):
        v=(self.x.T*variance)@self.x+noise*np.eye(self.x.shape[1])
        result=np.linalg.solve(v,rhs)
        error=np.linalg.norm(v@result-rhs,axis=0)/np.linalg.norm(rhs,axis=0)
        return result,dict(original_system_verified=True,maximum_relative_residual=float(error.max()))
    def scores(self, rhs):
        return self.x@rhs


def problem():
    rng=np.random.default_rng(784)
    x=rng.normal(size=(19,11));x[5]=0
    c=rng.uniform(.01,.1,len(x));c[::4]=0;noise=.7
    norms=np.sum(x*x,axis=1);groups=np.arange(len(x))%3
    reference=norms/(noise+np.dot(norms,c)/x.shape[1])
    return x,c,noise,norms,groups,reference


def test_streamed_estimate_matches_dense_equations_and_exact_audit():
    x,c,noise,norm,groups,ref=problem();seed=71
    got,record=estimate_inverse_quadratics(x,DenseOracle(x),c,noise,norm,groups,ref,
        design_identity='toy',draws=32,block_draws=8,batch_draws=4,audit_per_group=2,seed=seed)
    inverse=np.linalg.inv((x.T*c)@x+noise*np.eye(x.shape[1]))
    expected=[]
    for block in range(4):
        values=[]
        for offset in [0,4]:
            rng=np.random.default_rng(np.random.SeedSequence([seed,x.shape[1],block,offset]))
            e=rng.standard_normal((x.shape[1],4),dtype=np.float32).astype(float)
            center=np.divide(ref,norm,out=np.zeros_like(ref),where=norm>0)
            values.append(ref[:,None]+(x@e)*(x@inverse@e-center[:,None]*(x@e)))
        expected.append(np.column_stack(values).mean(axis=1))
    assert_allclose(got['h_blocks'],expected,rtol=1e-13,atol=1e-13)
    exact=np.diag(x@inverse@x.T)
    assert_allclose(got['exact_h'],exact[got['audit_indices']],rtol=1e-13)
    assert_array_equal(got['h_blocks'][:,5],0)
    assert record['new_solve_calls']==9 and record['newly_computed_draws']==32
    assert not record['audit_estimates_replaced']


def test_prefix_extension_is_identical_and_rejects_changed_or_refined_inputs():
    x,c,noise,norm,groups,ref=problem();backend=DenseOracle(x)
    args=(x,backend,c,noise,norm,groups,ref)
    options=dict(design_identity='toy',block_draws=8,batch_draws=4,audit_per_group=2,seed=72)
    prefix,_=estimate_inverse_quadratics(*args,draws=16,**options)
    extended,record=estimate_inverse_quadratics(*args,draws=32,previous=prefix,**options)
    direct,_=estimate_inverse_quadratics(*args,draws=32,**options)
    for key in direct:
        assert_array_equal(extended[key],direct[key])
    assert record['reused_draws']==16 and record['new_solve_calls']==4
    for replacement in [dict(seed=73),dict(design_identity='different design')]:
        with pytest.raises(ValueError,match='prefix inputs changed'):
            estimate_inverse_quadratics(*args,draws=32,previous=prefix,**dict(options,**replacement))
    with pytest.raises(ValueError,match='unrefined'):
        estimate_inverse_quadratics(*args,draws=32,previous=dict(prefix,exact_indices=np.array([1])),**options)


def test_bad_covariance_solve_cannot_be_accepted():
    x,c,noise,norm,groups,ref=problem()
    class Bad(DenseOracle):
        def solve_precise(self,*args):
            value,record=super().solve_precise(*args)
            record['maximum_relative_residual']=1e-7
            return value,record
    with pytest.raises(ArithmeticError,match='original-system residual'):
        estimate_inverse_quadratics(x,Bad(x),c,noise,norm,groups,ref,
            design_identity='toy',draws=16,block_draws=8,batch_draws=4,audit_per_group=2)
