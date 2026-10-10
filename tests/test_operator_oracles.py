"""Independent dense references for public covariance operators and GWAS."""
import csv
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from bed_reader import to_bed
from scipy.stats import t as student_t

from GPU_REML.gwas import run_continuous_gwas
from GPU_REML.reml_model import FitConfig, InfinitesimalREMLFitter


def cohort(tmp_path):
    rng = np.random.default_rng(546)
    x = rng.binomial(2, .32, size=(36, 24)).astype(np.int8)
    x[:5, 3] = -127
    x[:, 7] = 1
    prefix = tmp_path/'genotype'
    to_bed(str(prefix)+'.bed', x, properties={
        'iid':[f's{i}' for i in range(len(x))],
        'sid':[f'v{i}' for i in range(x.shape[1])]})
    mask = x != -127
    counts = mask.sum(axis=0)
    means = np.where(mask, x, 0).sum(axis=0)/counts
    centered = np.where(mask, x-means, 0.)
    variance = (centered**2).sum(axis=0)/counts
    scale = np.divide(1., np.sqrt(variance), out=np.zeros_like(variance), where=variance>0)
    return str(prefix), x, centered*scale, rng


@pytest.mark.parametrize('mode', ['standard', 'partitioned', 'smile', 'admixed'])
def test_streamed_covariances_and_diagonals_match_dense(tmp_path, mode):
    prefix, x, z, rng = cohort(tmp_path)
    kwargs = dict(bed_prefix=prefix, device='cpu', call_width=8, cpu_threads=2,
                  precond_rank=0, keep_host_stats=True, verbose=False)
    eff = np.count_nonzero(np.sum(z*z, axis=0)>0)
    if mode == 'standard':
        kernels = [z@z.T/eff]
    elif mode == 'partitioned':
        groups = [np.arange(0, 24, 2), np.arange(1, 24, 2)]
        kwargs['component_variant_indices'] = groups
        kernels = [z[:, g]@z[:, g].T/np.count_nonzero(np.sum(z[:, g]**2, axis=0)>0)
                   for g in groups]
    elif mode == 'smile':
        w = []
        total = np.zeros((len(x), len(x)))
        for lo in [0, 12]:
            a = rng.normal(size=(12, 5))
            weight = a@a.T/5
            w.append(weight)
            total += z[:, lo:lo+12]@weight@z[:, lo:lo+12].T
        kwargs.update(smile_weight_matrices=w, smile_diag_mode='full')
        kernels = [total/(np.trace(total)/len(x))]
    else:
        q = rng.dirichlet([2., 3.], size=len(x))
        kwargs['admix_weights'] = q
        k = z@z.T/eff
        kernels = [np.sqrt(q[:, a])[:, None]*k*np.sqrt(q[:, a])[None, :]
                   for a in range(2)]
    fitter = InfinitesimalREMLFitter(FitConfig(**kwargs))
    try:
        ops = fitter._assemble_reml_operators()
        rhs = rng.normal(size=(len(x), 3)).astype(np.float32)
        for mv, diag, kernel in zip(ops.K_mvs, ops.diag_list, kernels, strict=True):
            np.testing.assert_allclose(mv(jnp.asarray(rhs)), kernel@rhs, rtol=3e-5, atol=3e-6)
            atom = np.asarray(diag)
            expected = np.trace(kernel)/len(x) if atom.ndim == 0 else np.diag(kernel)
            np.testing.assert_allclose(atom, expected, rtol=3e-5, atol=3e-6)
        theta = np.linspace(.2, .4, len(kernels)).astype(np.float32)
        hv = fitter._make_hv(ops, jnp.asarray(theta), jnp.asarray(.5, dtype=jnp.float32))
        dense = sum(t*k for t,k in zip(theta,kernels,strict=True))+.5*np.eye(len(x))
        np.testing.assert_allclose(hv(jnp.asarray(rhs)), dense@rhs, rtol=3e-5, atol=3e-6)
    finally:
        fitter.close()


@pytest.mark.parametrize('partitioned', [False, True])
def test_gwas_source_order_and_fwl_match_dense_ols(tmp_path, partitioned):
    prefix, x, z, rng = cohort(tmp_path)
    kwargs = dict(bed_prefix=prefix, device='cpu', call_width=8, cpu_threads=2,
                  precond_rank=0, keep_host_stats=True, verbose=False)
    if partitioned:
        kwargs['component_variant_indices'] = [np.arange(0, 24, 2), np.arange(1, 24, 2)]
    cov = np.column_stack([np.ones(len(x)), rng.normal(size=len(x))]).astype(np.float32)
    y = (.3*z[:, 3]-.2*z[:, 15]+cov@np.array([.5, .2])+rng.normal(size=len(x))).astype(np.float32)
    fitter = InfinitesimalREMLFitter(FitConfig(**kwargs))
    try:
        result = run_continuous_gwas(fitter, y, cov, out_prefix=str(tmp_path/'result'))
        with open(result.out_path) as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        assert [r['variant_id'] for r in rows] == [f'v{i}' for i in range(24)]
        for j,row in enumerate(rows):
            if j == 7:
                assert np.isnan(float(row['beta']))
                continue
            # Mean imputation uses the same analysis population for each SNP.
            raw = np.where(x[:, j] != -127, x[:, j], x[x[:, j] != -127, j].mean())
            design = np.column_stack([cov.astype(float), raw])
            coef, _, rank, _ = np.linalg.lstsq(design, y.astype(float), rcond=None)
            residual = y-design@coef
            sigma2 = residual@residual/(len(x)-rank)
            se = np.sqrt(sigma2*np.linalg.solve(design.T@design, np.eye(rank))[-1, -1])
            statistic = coef[-1]/se
            p = 2*student_t.sf(abs(statistic), len(x)-rank)
            np.testing.assert_allclose([float(row[k]) for k in ['beta','se','t','p']],
                                      [coef[-1],se,statistic,p], rtol=2e-5, atol=2e-6)
    finally:
        fitter.close()


def test_gwas_rejects_nonfinite_phenotype_before_writing():
    fitter = SimpleNamespace(_has_sparse=False, streamers=[object()], _n_dense_streamers=1)
    with pytest.raises(ValueError, match='phenotype must be finite'):
        run_continuous_gwas(fitter, np.array([np.nan]), out_prefix='unused')
