import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from scipy.special import logsumexp
from GPU_REML.grouped_slab_variational import fit_grouped_slab


@pytest.mark.parametrize('scope',['population','support'])
def test_fixed_group_variances_match_exact_orthogonal_posterior_and_evidence(scope):
    d=np.array([20.,80.,50.,30.]);b=np.array([.5,30.,-8.,20.]);groups=np.arange(4)
    omega=np.array([.02,.8,.1,1.2]);pi=np.array([[.95,.05],[.8,.2],[.9,.1],[.7,.3]])
    pop=np.array([100,200,80,90]);pseudo=.5
    fit=fit_grouped_slab(np.diag(d),b,np.zeros(4),groups=groups,population_sizes=pop,
        initial_variances=omega,weights=pi,learn_weights=False,learn_variances=False,tolerance=1e-9,
        inclusion_scope=scope)
    v=omega/(1+d*omega);m=v*b
    logw=np.column_stack((np.log(pi[:,0]),np.log(pi[:,1])-.5*np.log1p(d*omega)+.5*b*m))
    q=np.exp(logw-logsumexp(logw,axis=1)[:,None])
    assert fit['converged'];assert_allclose(fit['responsibility'],q,atol=1e-12)
    assert_allclose(fit['mean'],q[:,1]*m,atol=1e-12)
    expected=logsumexp(logw,axis=1).sum()+pseudo*np.log(pi).sum()
    if scope=='population':expected+=(pop-1)@np.log(pi[:,0])
    assert_allclose(fit['elbo'][-1],expected,atol=1e-11)


@pytest.mark.parametrize('scope',['population','support'])
def test_group_m_steps_match_independent_marginal_optimizers(scope):
    d=np.full(16,80.);b=np.array([0.,1.,-2.,.5,30.,-40.,55.,4.,0.,2.,-1.,1.,70.,-100.,120.,3.])
    groups=np.repeat(np.arange(2),8);pop=np.array([100,200])
    fit=fit_grouped_slab(np.diag(d),b,b/d,groups=groups,population_sizes=pop,tolerance=1e-7,
        inclusion_scope=scope)
    assert fit['converged'];optimum=0.
    for group in range(2):
        ids=groups==group;size=pop[group] if scope=='population' else int(ids.sum())
        def objective(v):
            pi=1/(1+np.exp(-v[0]));omega=np.exp(v[1]);slab=-.5*np.log1p(d[ids]*omega)+.5*b[ids]**2*omega/(1+d[ids]*omega)
            return -(np.logaddexp(np.log1p(-pi),np.log(pi)+slab).sum()
                     +(size-8)*np.log1p(-pi)+.5*(np.log(pi)+np.log1p(-pi)))
        ref=minimize(objective,[-3.,0.],method='BFGS',options={'gtol':1e-7});optimum-=ref.fun
        assert_allclose(fit['slab_variances'][group],np.exp(ref.x[1]),rtol=2e-5)
        q=fit['responsibility'][ids,1];v=fit['component_variance'][ids,1];m=fit['component_mean'][ids,1]
        assert_allclose(fit['slab_variances'][group],np.sum(q*(v+m*m))/sum(q),rtol=1e-12)
        assert_allclose(fit['weights'][group,1],(sum(q)+.5)/(size+1),atol=1e-13)
    assert_allclose(fit['elbo'][-1],optimum,atol=2e-8)
    assert np.min(np.diff(fit['elbo']))>=-1e-8
    assert fit['slab_variances'][1]>3*fit['slab_variances'][0]


def test_correlated_gaussian_limit_has_group_specific_precision():
    g=np.array([[4.,3.5],[3.5,5.]]);b=np.array([5.,7.]);omega=np.array([.2,1.5])
    fit=fit_grouped_slab(g,b,[.1,.05],groups=[0,1],population_sizes=[1,1],
        weights=[[0.,1.],[0.,1.]],pseudocount=0,initial_variances=omega,
        learn_weights=False,learn_variances=False,tolerance=1e-8)
    assert fit['converged']
    assert_allclose(fit['mean'],np.linalg.solve(g+np.diag(1/omega),b),atol=1e-8)
    assert_allclose(fit['variance'],1/(np.diag(g)+1/omega),atol=1e-12)


def test_empty_group_flag_and_group_label_equivariance():
    d=np.full(6,80.);b=np.array([1.,30.,-40.,2.,70.,-100.]);labels=np.array([0,0,0,2,2,2]);pop=np.array([100,70,200])
    fit=fit_grouped_slab(np.diag(d),b,b/d,groups=labels,population_sizes=pop)
    assert fit['converged'] and fit['diagnostics']['omega_unidentified_groups']==[1]
    assert fit['slab_variances'][1]==fit['diagnostics']['initial_slab_variances'][1]
    assert_allclose(fit['weights'][1],[70.5/71,.5/71],atol=1e-12)
    perm=np.array([2,0,1]);other=fit_grouped_slab(np.diag(d),b,b/d,groups=perm[labels],population_sizes=pop[np.argsort(perm)])
    assert other['converged'];assert_allclose(other['mean'],fit['mean'],atol=1e-12)
    assert_allclose(other['slab_variances'][perm],fit['slab_variances'],atol=1e-12)


def test_group_boundary_is_not_falsely_converged_and_inputs_checked():
    fit=fit_grouped_slab([[100.]],[0.],[0.],groups=[0],population_sizes=[100,100],maximum_sweeps=50)
    assert not fit['converged'] and fit['variance_log_step']>1e-6
    with pytest.raises(ValueError,match='per group'):
        fit_grouped_slab([[4.]],[3.],[.1],groups=[0],population_sizes=[10,10],initial_variances=[.3])
    with pytest.raises(ValueError,match='positive semidefinite'):
        fit_grouped_slab([[1.,2.],[2.,1.]],[1.,1.],[0.,0.],groups=[0,1],population_sizes=[10,10])


@pytest.mark.parametrize('scope',['population','support'])
def test_grouped_ld_elbo_below_enumerated_evidence(scope):
    g=np.array([[8.,5.,-1.],[5.,6.,.2],[-1.,.2,4.]]);b=np.array([10.,8.,1.])
    labels=np.array([0,1,0]);omega=np.array([.04,1.]);pi=np.array([[.9,.1],[.8,.2]]);pop=np.array([100,100])
    fit=fit_grouped_slab(g,b,[.1,0,0],groups=labels,population_sizes=pop,initial_variances=omega,
        weights=pi,learn_weights=False,learn_variances=False,inclusion_scope=scope)
    terms=[]
    for z in itertools.product(range(2),repeat=3):
        z=np.asarray(z);active=np.flatnonzero(z);term=np.log(pi[labels,z]).sum()
        if len(active):
            var=omega[labels[active]];jj=g[np.ix_(active,active)]
            term-=.5*np.linalg.slogdet(np.eye(len(active))+np.sqrt(var[:,None]*var)*jj)[1]
            term+=.5*b[active]@np.linalg.solve(jj+np.diag(1/var),b[active])
        terms.append(term)
    exact=logsumexp(terms)+.5*np.log(pi).sum()
    if scope=='population':exact+=np.array([98,99])@np.log(pi[:,0])
    assert fit['converged'] and fit['elbo'][-1]<=exact+1e-9


def test_default_alpha_mean_matches_independent_initial_objective_without_zero_retry():
    g=np.array([[8.,3.,-1.],[3.,6.,.2],[-1.,.2,4.]])
    b=np.array([10.,8.,1.]);old=np.array([.3,-.1,.2]);labels=np.array([0,1,0])
    omega=np.array([.4,1.]);weights=np.array([[.9,.1],[.8,.2]]);population=np.array([100,90])
    fit=fit_grouped_slab(g,b,old,groups=labels,population_sizes=population,
        initial_variances=omega,weights=weights,maximum_sweeps=1)
    q=weights[labels,1];v=1/(np.diag(g)+1/omega[labels]);m=old/q
    second=q*(v+m*m)
    expected=b@old-.5*old@g@old-.5*np.diag(g)@(second-old*old)
    expected-=.5*np.sum(q*((v+m*m)/omega[labels]-1+np.log(omega[labels]/v)))
    expected+=(population-np.bincount(labels))@np.log(weights[:,0])+.5*np.log(weights).sum()
    assert_allclose(fit['elbo'][0],expected,rtol=1e-13,atol=1e-12)
    assert not fit['converged']  # One sweep cannot satisfy three consecutive checks.
    assert len(fit['diagnostics']['starts'])==1
    assert fit['diagnostics']['mean_initialization']=='original_alpha_only'
    assert fit['diagnostics']['selected_start_label']=='original_alpha'


def test_support_scope_is_invariant_to_outside_snp_counts_including_initialization():
    g=np.array([[80.,12.,2.],[12.,80.,9.],[2.,9.,80.]])
    b=np.array([40.,-60.,2.]);old=np.array([.4,-.6,.02]);options=dict(groups=[0,1,0],inclusion_scope='support')
    a=fit_grouped_slab(g,b,old,population_sizes=[2,1],**options)
    c=fit_grouped_slab(g,b,old,population_sizes=[158064,158064],**options)
    assert a['converged'] and c['converged']
    for key in ['mean','pip','weights','slab_variances','elbo']:assert_allclose(a[key],c[key],rtol=0,atol=0)
    assert c['diagnostics']['inclusion_prior_sizes']==[2,1]
    assert c['diagnostics']['outside_prior_counts']==[0,0]
    assert c['diagnostics']['outside_fixed_zero_counts']==[158062,158063]
    assert c['diagnostics']['initial_inclusion_probabilities']==[.5,.5]
    with pytest.raises(ValueError,match='inclusion_scope'):
        fit_grouped_slab(g,b,old,population_sizes=[2,1],groups=[0,1,0],inclusion_scope='other')


@pytest.mark.parametrize('pseudo',[0.,.5])
def test_support_empty_group_has_no_data_counts(pseudo):
    d=np.full(6,80.);b=np.array([1.,30.,-40.,2.,70.,-100.]);labels=np.array([0,0,0,2,2,2])
    fit=fit_grouped_slab(np.diag(d),b,b/d,groups=labels,population_sizes=[100,70,200],
        inclusion_scope='support',pseudocount=pseudo,weights=[[.8,.2],[.7,.3],[.8,.2]])
    assert fit['converged'] and fit['diagnostics']['pi_prior_only_groups']==[1]
    assert_allclose(fit['weights'][1],[.5,.5] if pseudo else [.7,.3],atol=1e-12)
    assert fit['slab_variances'][1]==fit['diagnostics']['initial_slab_variances'][1]
