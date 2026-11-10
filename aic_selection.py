"""Training-only Gaussian AIC for a native weighted Lasso path.

Covariance coefficients stay fixed within each path; the native outer loop
updates them after selecting alpha. Only lambda-dependent likelihood terms
are compared. This is not joint maximum-likelihood optimization over lambda.
"""
import numpy as np
import scipy.linalg as la
import jax
import jax.numpy as jnp

from .sparse_information import solve_sparse_pcg


def _selected_df(z, c, qc):
    """Whitening by a positive-definite H preserves rank([C,Z])."""
    q = c.shape[1]
    if not z.shape[1]:
        return q
    residual = z - qc @ (qc.T @ z)
    gram = residual.T @ residual
    try:
        factor = la.cho_factor(gram, lower=True, check_finite=False)
        rcond, info = la.lapack.dpocon(factor[0], np.linalg.norm(gram, 1), uplo='L')
        if not info and rcond > max(z.shape)*np.finfo(float).eps:
            return q + z.shape[1]
    except np.linalg.LinAlgError:
        pass
    singular = la.svdvals(residual, check_finite=False)
    cutoff = max(z.shape)*np.finfo(float).eps*singular[0]
    return q + int(np.count_nonzero(singular > cutoff))


class TrainingAIC:
    """A scalar callback; never constructs a held-out prediction context."""
    def __init__(self, y, covar):
        self.y = np.asarray(y, float).reshape(-1)
        self.c = np.empty((len(self.y), 0)) if covar is None else np.asarray(covar, float)
        if self.c.ndim != 2 or self.c.shape[0] != len(self.y):
            raise ValueError('AIC covariates and phenotype do not align')
        if self.c.shape[1] and np.linalg.matrix_rank(self.c) != self.c.shape[1]:
            raise ValueError('AIC nuisance covariates must have full column rank')
        self.qc_host = la.qr(self.c, mode='economic', check_finite=False)[0]
        self.rank_cache = {}
        self._hv = None

    def _prepare(self, hv, precond):
        if self._hv is hv:
            return
        self._hv = hv
        with jax.experimental.enable_x64():
            qc = jnp.asarray(self.qc_host, dtype=jnp.float64)
            qc32 = qc.astype(jnp.float32)

        def project(value):
            q = qc32 if value.dtype == jnp.float32 else qc
            return value-q @ (q.T @ value)

        def projected_hv(value):
            w = project(value)
            return project(hv(w))+(value-w)

        def projected_precond(value):
            w = project(value)
            return project(precond(w) if precond is not None else w)+(value-w)

        self.project = project
        self.operator = projected_hv
        self.preconditioner = projected_precond

    def evaluate(self, *, candidate, train_candidate, beta_cov_path,
                 beta_snp_path, hinv_residual_path, training_score_path,
                 hv, precond, sigma, pcg_tolerance=1e-5, maxiter=400):
        z = np.asarray(train_candidate, float)
        beta = np.asarray(beta_snp_path, float)
        dual = np.asarray(hinv_residual_path, float)
        markers = np.asarray(candidate, dtype=np.int64)
        if (z.ndim != 2 or z.shape[0] != len(self.y)
                or markers.shape != (z.shape[1],)):
            raise ValueError('AIC selected design and marker identities do not align')
        if dual.ndim == 1:
            dual = dual[:, None]
        if beta.shape != (1, z.shape[1]) or dual.shape != (len(self.y), 1):
            raise ValueError('Training AIC requires a single certified residual-path point')
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Training AIC requires positive residual covariance')
        active = np.flatnonzero(beta[0] != 0)
        signature = np.sort(markers[active]).tobytes()
        if signature not in self.rank_cache:
            selected = z if len(active) == z.shape[1] else z[:, active]
            self.rank_cache[signature] = _selected_df(selected, self.c, self.qc_host)
        df = self.rank_cache[signature]
        self._prepare(hv, precond)
        with jax.experimental.enable_x64():
            raw = self.y[:, None]-z @ beta.T
            target = self.project(jnp.asarray(raw, dtype=jnp.float64))
            value = self.project(jnp.asarray(dual, dtype=jnp.float64))
            product = self.operator(value)
            error = target-product
            scale = float(jnp.linalg.norm(target)) or 1.
            relative = float(jnp.linalg.norm(error))/scale
            tolerance = min(float(pcg_tolerance), 1e-5)
            iterations = extra = 0
            if relative > tolerance:
                value, relative, iterations = solve_sparse_pcg(
                    self.operator, target, M=self.preconditioner, X0=value,
                    tol=tolerance, maxiter=maxiter, log_diagnostics=False)
                extra = 1
                product = self.operator(value)
                error = target-product
                relative = float(jnp.linalg.norm(error))/scale
            if not np.isfinite(relative) or relative > 1.05*tolerance:
                raise FloatingPointError('Training likelihood solve failed verified residual')
            # Exact RSS minus this variational expression is e'A^-1 e.
            # A=Pi H Pi+QQ' has lambda_min >= min(theta_e,1).
            rss = float(jnp.sum(value*(2*target-product)))
            error_bound = float(jnp.sum(error*error))/min(float(sigma), 1.)
        if not np.isfinite(rss) or rss < 0:
            raise ValueError('Invalid training Gaussian quadratic')
        row = dict(aic=float(rss+2*df), weighted_rss=rss, df=int(df), n_train=len(self.y),
            data_role='training_only', predictive_r2=None,
            dual_source='verified_reused_projected_residual_solve',
            weighted_rss_error_bound=error_bound, likelihood_pcg_residual=relative,
            additional_covariance_solves=extra, additional_pcg_iterations=int(iterations))
        return dict(metrics=[row], pcg_rel_res=relative, pcg_iters=int(iterations))


def stopping_decision(metrics, *, stopping_lag):
    """Apply the existing native nonimprovement lag to lower-is-better AIC."""
    values = np.array([row['aic'] for row in metrics], float)
    lag = int(stopping_lag)
    if lag < 1 or not np.isfinite(values).all():
        raise ValueError('Invalid AIC path monitor')
    best = int(np.argmin(values)) if len(values) else None
    stopped = bool(len(values) > lag and values[:-lag].min() < values[-lag:].min())
    return dict(stopped=stopped, stopping_lag=lag, n_evaluated=len(values),
        best_path_index=best, best_aic=None if best is None else float(values[best]),
        reason='earlier_AIC_minimum_beats_recent_window' if stopped else 'AIC_prefix_continues',
        heuristic=True)


def select_index(rows, metrics):
    if len(rows) != len(metrics) or not rows:
        raise ValueError('AIC path points and likelihood metrics must align')
    eligible = [i for i, (row, metric) in enumerate(zip(rows, metrics)) if
        row.get('converged') and row.get('kkt_passed') and row.get('global_kkt_passed')
        and np.isfinite(metric['aic'])]
    if not eligible:
        raise RuntimeError('No KKT-certified AIC path point')
    return min(eligible, key=lambda i: (metrics[i]['aic'], rows[i]['k'], -rows[i]['lam_ratio']))
