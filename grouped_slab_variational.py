"""One learned spike-and-Gaussian slab per LD group, retaining the full LD J.

The default mean initialization is the original COHERIT sparse alpha only.
The four groups start at the same original-data-derived variance scale. Groups
with no active support have unidentified omega, kept at its initial value and
explicitly flagged; their sparse effects are constrained to zero by Q.
"""
from __future__ import annotations
import numpy as np
from numba import njit
def _inputs(gram, score):
    g,h=np.asarray(gram,float),np.asarray(score,float)
    if g.shape!=(len(h),len(h)) or not len(h) or not np.isfinite(g).all() or not np.isfinite(h).all():
        raise ValueError('Invalid marginal sufficient statistics')
    if np.any(np.diag(g)<=0) or not np.allclose(g,g.T,rtol=1e-9,atol=1e-9):
        raise ValueError('Positive diagonal and symmetric Gram required')
    return np.asfortranarray(g),h

def _initial_slab_variances(old, score, diagonal, components):
    """Choose training-only starts without duplicating exchangeable slabs.

    Equal starting variances with equal weights remain equal under EM. Keep
    the original-effect initialization whenever it already distinguishes the
    components; otherwise use marginal training effect sizes as well. If the
    training data also have tied sizes, exponential midpoint quantiles spread
    the same data-derived mean scale without introducing a scale-grid model.
    These are initial values only: every slab variance is subsequently free.
    """
    quantiles = (np.arange(components)+.5)/components
    scale = old**2+1/diagonal
    tau = np.quantile(scale, quantiles)
    source = 'original_effect_quantiles'
    if np.unique(tau).size < components:
        scale = np.maximum(old**2, (score/diagonal)**2)+1/diagonal
        tau = np.quantile(scale, quantiles)
        source = 'training_effect_quantiles'
        if np.unique(tau).size < components:
            tau = np.mean(scale)*(-np.log1p(-quantiles))
            source = 'training_scale_exponential_quantiles'
    return tau, source

@njit(cache=True)
def _targets(score, diagonal, weights, tau):
    components = len(tau)+1
    logw = np.empty(components)
    location = np.zeros(components); variance = np.zeros(components)
    for k in range(components):
        logw[k] = np.log(weights[k]) if weights[k] > 0 else -np.inf
        if k:
            variance[k] = tau[k-1]/(1.+diagonal*tau[k-1])
            location[k] = variance[k]*score
            logw[k] += -.5*np.log1p(diagonal*tau[k-1])+.5*score*location[k]
    probability = np.exp(logw-np.max(logw))
    probability /= np.sum(probability)
    return probability, location, variance



@njit(cache=True)
def _bound(g,b,mean,q,m,v,weights,omega,groups,outside,pseudocount):
    second=np.sum(q*(v+m*m),axis=1)
    value=b@mean-.5*mean@(g@mean)-.5*np.diag(g)@(second-mean*mean)
    for j in range(len(mean)):
        group=groups[j]
        for k in range(2):
            if q[j,k]>0:
                value-=q[j,k]*(np.log(q[j,k])-np.log(weights[group,k]))
                if k:
                    value-=.5*q[j,k]*((v[j,k]+m[j,k]**2)/omega[group]-1.+np.log(omega[group]/v[j,k]))
    for group in range(len(outside)):
        if outside[group]>0:value+=outside[group]*np.log(weights[group,0])
        if pseudocount>0:value+=pseudocount*np.log(weights[group]).sum()
    return value


@njit(cache=True)
def _solve(g,b,initial,groups,outside,weights,omega,pseudo,maximum,tolerance,learn_weights,learn_variances,order):
    n=len(b);mean=initial.copy();weights=weights.copy();omega=omega.copy()
    q=weights[groups].copy();m=np.zeros((n,2));v=np.zeros((n,2))
    for j in range(n):
        m[j,1]=initial[j]/q[j,1] if q[j,1]>0 else 0.
        v[j,1]=omega[groups[j]]/(1.+g[j,j]*omega[groups[j]])
    previous=_bound(g,b,mean,q,m,v,weights,omega,groups,outside,pseudo)
    history=np.empty(maximum+1);history[0]=previous;residual=b-g@mean
    stable=0;ok=False;decreased=False;error=np.inf;step=np.inf;hyper_step=np.inf
    for sweep in range(maximum):
        oldmean=mean.copy();oldomega=omega.copy()
        for index in range(n):
            j=order[index];group=groups[j]
            qq,mm,vv=_targets(residual[j]+g[j,j]*mean[j],g[j,j],weights[group],omega[group:group+1])
            updated=qq@mm;change=updated-mean[j];mean[j]=updated;q[j],m[j],v[j]=qq,mm,vv
            for i in range(n):residual[i]-=g[i,j]*change
        residual=b-g@mean
        if learn_weights:
            previous_weights=weights.copy()
            weights[:]=pseudo
            for group in range(len(outside)):weights[group,0]+=outside[group]
            for j in range(n):weights[groups[j]]+=q[j]
            for group in range(len(outside)):
                total=np.sum(weights[group])
                if total>0:weights[group]/=total
                else:weights[group]=previous_weights[group]
        if learn_variances:
            for group in range(len(omega)):
                indices=groups==group
                mass=np.sum(q[indices,1])
                if mass>0:omega[group]=np.sum(q[indices,1]*(v[indices,1]+m[indices,1]**2))/mass
        bound=_bound(g,b,mean,q,m,v,weights,omega,groups,outside,pseudo);history[sweep+1]=bound
        if not np.isfinite(bound) or bound<previous-1e-9*max(1.,abs(previous)):
            decreased=True;break
        error=0.
        for j in range(n):
            group=groups[j]
            qq,mm,vv=_targets(residual[j]+g[j,j]*mean[j],g[j,j],weights[group],omega[group:group+1])
            error=max(error,np.max(np.abs(qq-q[j])),abs(qq@mm-mean[j])*np.sqrt(g[j,j]))
            if q[j,1]>0:
                ratio=v[j,1]/vv[1]
                kl=.5*q[j,1]*((m[j,1]-mm[1])**2/vv[1]+ratio-1.-np.log(ratio))
                error=max(error,np.sqrt(max(kl,0.)))
        step=np.max(np.abs(mean-oldmean)*np.sqrt(np.diag(g)))
        hyper_step=np.max(np.abs(np.log(omega/oldomega))) if learn_variances else 0.
        good=(error<=tolerance and step<=tolerance and hyper_step<=tolerance
              and abs(bound-previous)<=1e-10*max(1.,abs(bound)))
        stable=stable+1 if good else 0;previous=bound
        if stable>=3:ok=True;break
    return mean,q,m,v,weights,omega,history[:sweep+2],ok,decreased,error,step,hyper_step


def fit_grouped_slab(gram,score,old,*,groups,population_sizes,pseudocount=.5,
                     maximum_sweeps=20000,tolerance=1e-6,initial_variances=None,
                     weights=None,starts=None,learn_weights=True,learn_variances=True,
                     inclusion_scope='population'):
    """Variational EM for alpha_j ~ (1-pi_g) delta0 + pi_g N(0,omega_g).

    omega_g <- sum_{j in S_g} q_j(v_j+m_j^2)/sum_{j in S_g} q_j.
    pi_g <- (sum_{j in S_g} q_j+pseudocount)/(N_g+2*pseudocount).
    N_g is p_g for population scope, or |S_g| for support scope. In support
    scope outside effects are deterministic zeros, with no Bernoulli prior
    counts or ELBO contribution. Empty groups retain a prior-only pi (or its
    initial value without regularization) and an unidentified omega.
    Default to one mean start, old (the original COHERIT sparse alpha).
    Explicit starts remain available for controlled comparisons. The full J
    and numerical convergence gates are unchanged; no implicit zero retry.
    """
    g,b=_inputs(gram,score);old=np.asarray(old,float);labels=np.asarray(groups);population=np.asarray(population_sizes)
    if (old.shape!=b.shape or not np.isfinite(old).all() or labels.shape!=b.shape
        or not np.isfinite(labels).all() or np.any(labels!=np.floor(labels)) or np.any(labels<0)
        or population.ndim!=1 or not len(population) or not np.isfinite(population).all()
        or np.any(population!=np.floor(population)) or np.any(population<=0) or np.any(labels>=len(population))
        or not np.isfinite(pseudocount) or pseudocount<0 or maximum_sweeps<1 or int(maximum_sweeps)!=maximum_sweeps
        or not np.isfinite(tolerance) or tolerance<=0):
        raise ValueError('Aligned finite inputs, valid groups and positive controls required')
    labels=labels.astype(np.int64);population=population.astype(np.int64)
    if inclusion_scope not in ('population','support'):
        raise ValueError('inclusion_scope must be population or support')
    included=np.bincount(labels,minlength=len(population));fixed_zero=population-included
    if np.any(fixed_zero<0):raise ValueError('Population cannot be smaller than active support')
    outside=fixed_zero if inclusion_scope=='population' else np.zeros_like(fixed_zero)
    diagonal=np.diag(g);roundoff=64*np.finfo(float).eps*len(b)*float(diagonal.max())
    try:np.linalg.cholesky(g+roundoff*np.eye(len(b)))
    except np.linalg.LinAlgError as exc:raise ValueError('Information must be positive semidefinite') from exc
    if initial_variances is None:
        scale,_=_initial_slab_variances(old,b,diagonal,1)
        omega=np.full(len(population),scale[0]);initialization='same_global_original_effect_scale_for_each_group'
    else:omega=np.asarray(initial_variances,float);initialization='provided'
    if omega.shape!=(len(population),) or not np.isfinite(omega).all() or np.any(omega<=0):
        raise ValueError('One positive finite variance per group required')
    if weights is None:
        inclusion=(np.clip(included/population,1e-6,.5) if inclusion_scope=='population'
                   else np.full(len(population),.5))
        weights=np.column_stack((1-inclusion,inclusion))
    weights=np.asarray(weights,float)
    if (weights.shape!=(len(population),2) or not np.isfinite(weights).all() or np.any(weights<0)
        or not np.allclose(weights.sum(axis=1),1,rtol=1e-12,atol=1e-12)
        or np.any((outside>0)&(weights[:,0]==0)) or (pseudocount>0 and np.any(weights==0))):
        raise ValueError('Valid group spike and slab probabilities required')
    initial_values=[old] if starts is None else starts
    if not len(initial_values):raise ValueError('At least one initialization required')
    order=np.argsort(-np.abs(b/np.sqrt(diagonal)),kind='stable').astype(np.int64);fits=[]
    for initial in initial_values:
        initial=np.asarray(initial,float)
        if initial.shape!=b.shape or not np.isfinite(initial).all():raise ValueError('Finite aligned initialization required')
        if np.any((weights[labels,1]==0)&(initial!=0)):raise ValueError('A spike-only prior requires zero initial mean')
        result=_solve(np.asfortranarray(g),b,initial,labels,outside,weights,omega,float(pseudocount),
            int(maximum_sweeps),float(tolerance),learn_weights,learn_variances,order)
        mean,q,m,v,pi,variances,elbo,ok,decreased,error,step,hyper_step=result
        if decreased:raise ArithmeticError('Grouped variational EM lost finite ELBO ascent')
        fits.append(dict(mean=mean,pip=1-q[:,0],variance=np.sum(q*(v+m*m),axis=1)-mean*mean,
            responsibility=q,component_mean=m,component_variance=v,weights=pi,slab_variances=variances,
            omega_has_support=included>0,elbo=elbo,converged=bool(ok),coordinate_error=float(error),
            mean_step=float(step),variance_log_step=float(hyper_step)))
    qualified=[i for i,f in enumerate(fits) if f['converged']]
    selected=max(qualified or list(range(len(fits))),key=lambda i:fits[i]['elbo'][-1]);out=fits[selected]
    start_labels=['original_alpha' if np.array_equal(value,old) else
                  'zero' if np.all(np.asarray(value)==0) else 'custom' for value in initial_values]
    out['diagnostics']=dict(selected_start=selected,selected_start_label=start_labels[selected],
        start_labels=start_labels,
        mean_initialization='original_alpha_only' if start_labels==['original_alpha'] else 'explicit',
        positive_components_per_group=1,variance_scope='group',
        inclusion_scope=inclusion_scope,inclusion_prior_sizes=(included+outside).tolist(),
        initial_inclusion_probabilities=weights[:,1].tolist(),outside_prior_counts=outside.tolist(),
        pi_prior_only_groups=np.flatnonzero(included+outside==0).tolist(),
        population_sizes=population.tolist(),outside_fixed_zero_counts=fixed_zero.tolist(),
        support_counts=included.tolist(),omega_unidentified_groups=np.flatnonzero(included==0).tolist(),
        pseudocount=float(pseudocount),scale_grid_used=False,initial_slab_variances=omega.tolist(),
        variance_initialization=initialization,
        starts=[dict(converged=f['converged'],sweeps=len(f['elbo'])-1,elbo=float(f['elbo'][-1]),
            coordinate_error=f['coordinate_error'],mean_step=f['mean_step'],variance_log_step=f['variance_log_step']) for f in fits],
        maximum_start_mean_difference=float(max(np.max(abs(f['mean']-out['mean'])) for f in fits)))
    return out
