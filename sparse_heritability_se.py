"""Observed-derivative local sandwich at a saved SPLIT endpoint; no optimizer.

Random vectors evaluate Gaussian quadratic-form moments and matrix traces.
They never trigger phenotype refitting, support selection, or REML updates.
"""
from __future__ import annotations

import logging
import time
import numpy as np
import scipy.linalg as sla

logger = logging.getLogger(__name__)


class Backend:
    def __init__(self, gpu=False):
        self.gpu = gpu
        if gpu:
            import jax.numpy as xp
            import jax.scipy.linalg as la
            self.xp, self.la, self.dtype = xp, la, xp.float32
        else:
            self.xp, self.la, self.dtype = np, sla, np.float64

    def array(self, x):
        return self.xp.asarray(x, dtype=self.dtype)

    def host(self, x):
        return np.asarray(x, dtype=np.float64)

    def chol(self, x):
        return self.la.cholesky(x, lower=True)

    def solve(self, factor, x):
        return self.la.cho_solve((factor, True), x)

    def inner(self, a, b):
        return self.xp.sum(a * b, axis=0, dtype=self.xp.float64)


class KernelBank:
    def __init__(self, kernels, gpu=False, streamed=False):
        self.b = Backend(gpu)
        self.streamed = bool(streamed)
        self.k = list(kernels) if streamed else [self.b.array(k) for k in kernels]
        self.n = self.k[0].shape[0]
        self._fused_covariance = None
        if gpu and not streamed:
            import jax
            xp = self.b.xp
            # Fuse the elementwise sum so a 55k cohort does not retain several
            # additional 12-GB intermediate covariance matrices at once.
            @jax.jit
            def build(theta, *kernels):
                h = theta[0] * kernels[0]
                for i, kernel in enumerate(kernels[1:], 1):
                    h = h + theta[i] * kernel
                return h + theta[-1] * xp.eye(h.shape[0], dtype=h.dtype)
            self._fused_covariance = build

    def mv(self, i, x):
        if i < len(self.k) and self.streamed:
            k = self.b.array(self.k[i])
            out = k @ x
            if self.b.gpu:out.block_until_ready()
            del k
            return out
        return self.k[i] @ x if i < len(self.k) else x

    def covariance(self, theta):
        if self.streamed:
            # Host RAM holds the cohort cache; at most one kernel is transferred
            # for a product. This changes memory use, not the covariance model.
            h = np.zeros((self.n,self.n), dtype=np.float32)
            for t,k in zip(theta[:-1],self.k):
                for start in range(0,self.n,1024):
                    h[start:start+1024] += np.float32(t)*k[start:start+1024]
            h.flat[::self.n+1] += np.float32(theta[-1])
            return self.b.array(h)
        if self._fused_covariance is not None:
            return self._fused_covariance(self.b.array(theta), *self.k)
        xp = self.b.xp
        h = theta[-1] * xp.eye(self.n, dtype=self.b.dtype)
        for t, k in zip(theta[:-1], self.k):
            h = h + float(t) * k
        return h


    def factor(self, theta):
        return self.b.chol(self.covariance(theta))

    def solve(self, factor, rhs, *, initial=None):
        return self.b.solve(factor, rhs)

    def inverse_draw(self, factor, rng, count):
        normal = self.b.array(rng.normal(size=(self.n, count)))
        return self.b.la.solve_triangular(factor.T, normal, lower=False)

    def draw(self, factor, rng, count):
        return factor @ self.b.array(rng.normal(size=(self.n, count)))


def calculate(bank, theta, y, z, c, alpha, ratio, anchor, anchor_sign,
              q_saved=None, atoms=None, trace_probes=64, moment_probes=256,
              seed=581104, return_details=False, penalty_signs=None,
              anchor_candidates=None, progress=None, cached_solutions=None,
              mean_trace_probes=None, allow_projected=True, _verified_basis=False):
    """Compute free-all and fixed-boundary SEs without changing the endpoint.

    Inputs must use exactly the saved sample order and SNP standardization.
    alpha is the native Lasso effect, NOT the later prediction refinement.
    """
    started = time.perf_counter()
    b, xp = bank.b, bank.b.xp
    n, p, nc = len(y), z.shape[1], len(theta)
    theta = np.asarray(theta, dtype=float)
    if (not np.all(np.isfinite(theta)) or np.any(theta[:-1] < 0)
            or theta[-1] <= 0 or not 0 <= ratio <= 1
            or moment_probes < 16 or moment_probes % 8 or trace_probes < 0):
        raise ValueError('Invalid covariance, lambda ratio or SE probe configuration')
    if (z.shape != (n, len(alpha)) or c.ndim != 2 or c.shape[0] != n
            or any(not np.all(np.isfinite(v)) for v in [y, z, c, alpha])):
        raise ValueError('Nonfinite or misaligned SE endpoint inputs')
    if c.shape[1] and np.linalg.matrix_rank(c) != c.shape[1]:
        raise ValueError('SE covariates must have full column rank')
    if p >= n-c.shape[1]:
        raise ValueError('Selected model has no residual degrees of freedom')
    contrasts = None
    use_projected = allow_projected and mean_trace_probes is not None and hasattr(bank, 'selected_contrasts')
    if hasattr(bank, 'verify_selected_workspace'):
        bank.verify_selected_workspace(p, projected=use_projected)
    if use_projected:
        contrasts = bank.selected_contrasts(z, c)
    # A change of nuisance basis leaves the projection invariant.
    cache = {} if cached_solutions is None else cached_solutions
    hc_initial = cache.get('hinv_covar')
    if c.shape[1]:
        c, rc = sla.qr(c, mode='economic')
        if hc_initial is not None:
            hc_initial = sla.solve_triangular(rc.T, np.asarray(hc_initial).T, lower=True,
                                              check_finite=False).T
    auto_anchor = anchor is None
    y, c, alpha = [b.array(a) for a in (y, c, alpha)]
    z = b.array(z) if contrasts is None else contrasts.b64
    anchor = b.array(np.zeros(n) if auto_anchor else anchor)
    signs = xp.sign(alpha) if penalty_signs is None else b.array(penalty_signs)
    factor = bank.factor(theta)
    if progress is not None:
        if b.gpu and factor is not None:factor.block_until_ready()
        progress('covariance_factor',time.perf_counter()-started)
    hc = (bank.solve(factor, c) if hc_initial is None else
          bank.solve(factor, c, initial=hc_initial))
    ci = b.array(np.linalg.inv(b.host(c.T @ hc)))

    def project_hinv(v):
        return v-hc @ (ci @ (c.T @ v))

    def project(v):
        return project_hinv(bank.solve(factor, v))

    if contrasts is None:
        hz_initial = cache.get('hinv_selected')
        hz = (bank.solve(factor, z) if hz_initial is None else
              bank.solve(factor, z, initial=hz_initial))
        c_hz = c.T @ hz
        pz = hz-hc @ (ci @ c_hz)
        m = b.host(z.T @ pz)
        if p:
            mf = sla.cho_factor((m+m.T)/2, lower=True)
            mi_host = sla.cho_solve(mf, np.eye(p))
            rcond, info = sla.lapack.dpocon(mf[0], np.linalg.norm(m, 1), uplo='L')
            if info or rcond <= 1e-12:
                raise ValueError('Selected mean information is numerically singular')
            mean_condition = 1/rcond
        else:
            mi_host, mean_condition = np.empty((0, 0)), 1.
        mi = b.array(mi_host)

        def selected_project(v):
            pv = project(v)
            return pv-pz @ (mi @ (z.T @ pv))

        def selected_inverse(directions):
            inverse = mi @ directions
            return pz @ inverse, inverse

        def direct_branches(v, initial=None):
            hv = (bank.solve(factor, v) if initial is None else
                  bank.solve(factor, v, initial=initial))
            pv = project_hinv(hv)
            score_a = anchor @ pv
            av = mi @ (z.T @ pv-float(ratio*anchor_sign)*signs[:, None]*score_a[None, :])
            gv = z @ av
            dv = pv-pz @ av
            gamma = ci @ (c.T @ hv-c_hz @ av)
            rv = v-gv-c @ gamma
            return gv, rv, dv
    else:
        selected_project = contrasts.solve
        selected_inverse = contrasts.inverse_information
        # The overlapping covariate subspace is smaller than the selected SNP
        # subspace. A sufficient condition bound uses only its small Gram.
        if atoms is None:
            raise ValueError('Projected SE requires fitted kernel traces')
        norm_bound = float(theta[-1]+n*(theta[:-1] @ np.asarray(atoms)))
        mean_condition = contrasts.condition_bound(norm_bound, theta[-1])
        if mean_condition >= 1e12:
            raise ValueError('Projected selected-mean information condition cannot be verified')

    if progress is not None:progress('selected_information',time.perf_counter()-started)
    hy_initial = cache.get('hinv_response')
    hy = (bank.solve(factor, y) if hy_initial is None else
          bank.solve(factor, y, initial=hy_initial))
    py = project_hinv(hy)
    anchor_index = -1
    if auto_anchor and p and hasattr(bank, 'resolve_anchor'):
        anchor, anchor_sign, anchor_index = bank.resolve_anchor(py)
        anchor = b.array(anchor)
    elif auto_anchor and p:
        candidates = z if anchor_candidates is None else b.array(anchor_candidates)
        anchor_scores = b.host(candidates.T @ py)
        anchor_index = int(np.argmax(abs(anchor_scores)))
        anchor = candidates[:,anchor_index]
        anchor_sign = float(np.sign(anchor_scores[anchor_index]))
    g = z @ alpha
    # Linearity supplies a cheap initial solution, but PCG still verifies it
    # against the original residual RHS; cancellation never weakens accuracy.
    hinv_g = None if contrasts is not None else hz @ alpha
    u = project_hinv(bank.solve(factor, y-g, initial=None if hinv_g is None else hy-hinv_g))
    residual = y-g-c @ (ci @ (hc.T @ (y-g)))
    a = project(anchor)
    # Differentiate the SAME selected-mean trace as the point estimate. For
    # stochastic endpoints z is the native orthonormal basis and these are
    # its fixed Rademacher probes. Standalone endpoints use the exact trace,
    # including Z'Z when their selected coordinates are not orthonormal.
    if mean_trace_probes is None:
        if _verified_basis:
            directions = xp.eye(p, dtype=b.dtype)
        else:
            gram = b.host(z.T @ z)
            directions = b.array(sla.cholesky((gram+gram.T)*.5, lower=True))
        normalization = 1.
    else:
        directions = np.asarray(mean_trace_probes)
        if (directions.ndim != 2 or directions.shape[0] != p or directions.shape[1] == 0
                or not np.isfinite(directions).all()
                or (not _verified_basis and not np.allclose(
                    b.host(z.T @ z), np.eye(p), atol=1e-8, rtol=1e-8))):
            raise ValueError('Endpoint trace probes require the matching orthonormal selected basis')
        normalization = float(directions.shape[1])
        directions = b.array(directions)
    if contrasts is None:
        trace_weights, inverse_directions = selected_inverse(directions)
    else:
        # Share the small solve for the penalty direction with the trace
        # directions. Only these M^-1 actions are needed for the sandwich.
        columns = directions.shape[1]
        weights, inverse = selected_inverse(xp.column_stack([directions, signs]))
        trace_weights, inverse_directions = weights[:, :columns], inverse[:, :columns]
        penalty_dual, penalty_coef = weights[:, columns:], inverse[:, columns:]
        penalty_residual = bank.hv(penalty_dual)

        def projected_branches(v, initial=None, penalty=None):
            av, rv, dv = contrasts.unpenalized(v)
            if penalty is None:
                penalty = float(ratio*anchor_sign)*(a @ v)
            return (z @ (av-penalty_coef*penalty),
                    rv+penalty_residual*penalty, dv+penalty_dual*penalty)

    mean_trace = float(b.host(xp.sum(directions*inverse_directions)))/normalization
    if progress is not None:progress('mean_trace_actions',time.perf_counter()-started)
    q_reconstructed = float(b.host(g@g+2*g@residual)/n)-mean_trace/n
    q = q_reconstructed if q_saved is None else float(q_saved)
    lam = ratio*float(b.host(anchor @ py))*anchor_sign
    kkt = b.host(z.T @ u)-lam*b.host(signs)
    if contrasts is None:
        branch_alpha = b.host(mi @ (z.T @ py-lam*signs))
    else:
        _, adjustment = selected_inverse(b.array(kkt)[:, None])
        branch_alpha = b.host(alpha+adjustment[:, 0])

    rng = np.random.default_rng(seed)
    # A PSD trace estimator: Cov(psi)=P_S, then E[(K_i psi)'P_S(K_j psi)].
    if trace_probes:
        if contrasts is None:
            inv_draw = bank.inverse_draw(factor, rng, trace_probes)
            psi = project_hinv(inv_draw)
            psi = psi-pz @ (mi @ (z.T @ psi))
            del inv_draw
        else:
            psi = selected_project(bank.draw(factor, rng, trace_probes))
        ki_psi = [bank.mv(i, psi) for i in range(nc)]
        ps_ki = selected_project(xp.concatenate(ki_psi, axis=1))
        ps_ki = [ps_ki[:, i*trace_probes:(i+1)*trace_probes] for i in range(nc)]
        tr = np.array([[float(b.host(b.inner(x, v)).mean()) for v in ps_ki] for x in ki_psi])
        tr = (tr+tr.T)/2
        del psi, ki_psi, ps_ki
    else:
        # Small-n independent algebra checks only; production uses matrix probes.
        ps = selected_project(xp.eye(n, dtype=b.dtype))
        pski = [bank.mv(i, ps) for i in range(nc)]
        tr = np.array([[float(b.host(xp.sum(x*v.T))) for v in pski] for x in pski])
        del ps, pski
    if progress is not None:progress('information_traces',time.perf_counter()-started)

    ku = [bank.mv(i, u) for i in range(nc)]
    kpy = [bank.mv(i, py) for i in range(nc)]
    if contrasts is None:
        pku = project(xp.column_stack(ku))
        dalpha = [mi @ (-pz.T @ ku[j]+float(ratio*anchor_sign)*signs*(a @ kpy[j])) for j in range(nc)]
        du = xp.column_stack([-pku[:, j]-pz @ dalpha[j] for j in range(nc)])
        dg = z @ xp.column_stack(dalpha)
        dr = -dg+c @ (ci @ (hc.T @ xp.column_stack(ku)+c_hz @ xp.column_stack(dalpha)))
    else:
        kuv = xp.column_stack(ku)
        levels = float(ratio*anchor_sign)*(a @ xp.column_stack(kpy))
        mean, remainder, dual = projected_branches(kuv, penalty=levels)
        dg, dr, du = -mean, kuv-remainder, -dual
    jac = b.host(xp.column_stack(ku).T @ du)+.5*tr
    derivative_trace = np.array([float(b.host(xp.sum(trace_weights*bank.mv(j, trace_weights))))
                                 for j in range(nc)])/normalization
    dq = (b.host(2*(g+residual) @ dg+2*g @ dr)-derivative_trace)/n
    if progress is not None:progress('observed_derivative',time.perf_counter()-started)
    del ku, kpy, du, dg, dr, trace_weights, inverse_directions, directions

    # Antithetic quadratic/linear decomposition of Gaussian moments.
    # This is trace estimation at fixed theta, not a parametric refitting bootstrap.
    xi = bank.draw(factor, rng, moment_probes)
    branches = direct_branches if contrasts is None else projected_branches
    gx, rx, dx = branches(xi)
    gm, rm, dm = branches(g[:, None], initial=None if hinv_g is None else hinv_g[:, None])
    quad, linear = [], []
    for i in range(nc):
        kd = bank.mv(i, dx)
        quad.append(b.host(.5*b.inner(dx, kd)))
        linear.append(b.host(dm[:, 0] @ kd))
    quad.append(b.host(b.inner(gx, gx)+2*b.inner(gx, rx))/n)
    linear.append(b.host(2*(gm[:, 0] @ gx+gm[:, 0] @ rx+rm[:, 0] @ gx))/n)
    quad, linear = np.asarray(quad), np.asarray(linear)
    meat = np.cov(quad, ddof=1)+np.cov(linear, ddof=1)
    if progress is not None:progress('gaussian_moments',time.perf_counter()-started)
    atoms = (np.array([float(b.host(xp.trace(k)))/n for k in bank.k]) if atoms is None else np.asarray(atoms))
    bg = atoms @ theta[:-1]
    total = q+bg+theta[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError('SE endpoint has nonpositive total phenotypic variance')
    h2 = (q+bg)/total
    gq = theta[-1]/total**2
    gt = np.r_[gq*atoms, -(q+bg)/total**2]
    ans = dict(h2=float(h2),q_reconstructed=q_reconstructed,q_saved=float(q),
               q_reconstruction_error=float(q_reconstructed-q),
               kkt_max_error=float(np.max(abs(kkt), initial=0)),
               lambda_reconstructed=float(lam),
               alpha_branch_max_difference=float(np.max(abs(branch_alpha-b.host(alpha)), initial=0)),
               information_rank=int(p), mean_uncertainty=mean_trace/n,
               endpoint_trace_correction=('exact' if mean_trace_probes is None else 'projected_hutchinson'),
               anchor_index_within_support=anchor_index,
               mean_information_condition=float(mean_condition),
               mean_information_condition_kind=('estimate_1norm' if contrasts is None else 'upper_bound_1norm'),
               selected_information_backend=('inverse_columns' if contrasts is None else 'projected_actions'),
               trace_probes=trace_probes,moment_probes=moment_probes,
               fitting_information_reused=bool(cache),
               boundary_count=int(np.sum(theta[:-1]<=1e-8)))
    for policy, free in [('all',np.arange(nc)),('active',np.r_[np.flatnonzero(theta[:-1]>1e-8), nc-1])]:
        try:
            bread = -jac[np.ix_(free, free)]
            condition = float(np.linalg.cond(bread))
            if condition > 1e10:
                raise ValueError(f'Ill-conditioned observed derivative ({condition:g})')
            weights = np.linalg.solve(bread.T, (gt+gq*dq)[free])
            inf = np.r_[weights, gq]
            idx = np.r_[free, nc]
            variance = float(inf @ meat[np.ix_(idx,idx)] @ inf)
            if variance < 0 or not np.isfinite(variance):
                raise ValueError('Invalid propagated variance')
            ans['se_'+policy] = float(np.sqrt(variance))
            ans['derivative_condition_'+policy] = condition
            # Approximate moment-only Monte Carlo error; trace/Jacobian error is separate.
            qinf, linf = inf @ quad[idx], inf @ linear[idx]
            batch_se = [np.sqrt(np.var(qq,ddof=1)+np.var(ll,ddof=1))
                        for qq,ll in zip(np.array_split(qinf,8),np.array_split(linf,8))]
            ans['moment_mc_se_'+policy] = float(np.std(batch_se,ddof=1)/np.sqrt(8))
        except (ValueError, np.linalg.LinAlgError) as exc:
            if policy == 'active':
                raise
            ans['se_all'] = None
            ans['all_directions_error'] = str(exc)
    ans['seconds'] = time.perf_counter()-started
    if return_details:
        ans.update(jac=jac,dq=dq,meat=meat,traces=tr)
    return ans


def add_h2_se_arguments(parser):
    """Shared options for the production runner and its low-level fit stage."""
    import argparse
    parser.add_argument('--h2-se', action=argparse.BooleanOptionalAction, default=True,
                        help='Compute a local approximate total-h2 SE at the final sparse endpoint (default: enabled).')
    parser.add_argument('--h2-se-trace-probes', type=int, default=64)
    parser.add_argument('--h2-se-moment-probes', type=int, default=256)
    parser.add_argument('--h2-se-seed', type=int, default=581104)
    parser.add_argument('--h2-se-pcg-tol', type=float, default=1e-5)
    parser.add_argument('--h2-se-batch-size', type=int, default=0,
                        help='SE RHS batch width; 0 reuses the native sparse solve memory plan (default).')


def validate_h2_se_arguments(args):
    if (args.h2_se_trace_probes < 1 or args.h2_se_moment_probes < 16
            or args.h2_se_moment_probes % 8 or args.h2_se_seed < 0
            or args.h2_se_batch_size < 0 or not 0 < args.h2_se_pcg_tol <= 1e-3):
        raise ValueError('SE requires positive trace count, nonnegative batch width, moments >=16 divisible by 8, '
                         'nonnegative seed and PCG tolerance in (0, 1e-3]')


def selected_coordinates(z, alpha, *, basis=None):
    """Rank-reduce selected SNPs while preserving the signed L1 functional."""
    z, alpha = np.asarray(z, dtype=float), np.asarray(alpha, dtype=float)
    if not len(alpha):
        return z, alpha, alpha.copy()
    if basis is not None:
        # The native information object already computed this same immutable
        # selected span. Retain its basis so its H^-1 basis solves are reusable.
        # A support-sized transform preserves the mean and signed L1 term.
        basis = np.asarray(basis, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != z.shape[0]:
            raise ValueError('Cached selected basis has incompatible dimensions')
        transform = basis.T @ z
        signs, _, rank, _ = sla.lstsq(transform.T, np.sign(alpha), lapack_driver='gelsy')
        if rank != basis.shape[1] or np.linalg.norm(transform.T @ signs-np.sign(alpha)) > 1e-5:
            raise ValueError('Selected L1 penalty is not a unique mean-space functional')
        return basis, transform @ alpha, signs
    basis, r = sla.qr(z, mode='economic', check_finite=False)
    u, singular, vt = sla.svd(r, full_matrices=False, check_finite=False)
    cutoff = max(z.shape) * np.finfo(float).eps * singular[0]
    ranks = [int((singular > k*cutoff).sum()) for k in (.1, 1., 10.)]
    if len(set(ranks)) != 1:
        raise ValueError('Numerically ambiguous selected mean-space rank')
    rank = ranks[1]
    if rank < len(alpha):
        sign = np.sign(alpha)
        if np.linalg.norm(sign - vt[:rank].T @ (vt[:rank] @ sign)) > 1e-5:
            raise ValueError('Selected L1 penalty is not a unique mean-space functional')
        return (basis @ u[:, :rank], singular[:rank]*(vt[:rank] @ alpha),
                (vt[:rank] @ sign)/singular[:rank])
    return basis, r @ alpha, sla.solve_triangular(r.T, np.sign(alpha), lower=True)


def endpoint_h2_se(bank, *, theta, y, z, covar, alpha, ratio, lam, q, atoms,
                   h2, trace_probes=64, moment_probes=256, seed=581104,
                   solve_tol=1e-5, mean_information=None, progress=None):
    """Public fixed-endpoint API. Caller supplies the analysis-scale inputs.

    JAX callers must enable x64 in a scoped context, as the pipeline does.
    A failed endpoint/anchor/solve check raises instead of returning a valid SE.
    """
    cache = None
    cached_basis = None
    verified_basis = False
    if mean_information is not None:
        from .sparse_information import SparseMeanInformation
        from .sparse_reml import SparseProjectedInformation

        info = mean_information
        c_input = np.empty((len(y), 0)) if covar is None else np.asarray(covar)
        if info.n != len(y) or not np.array_equal(info.covar, c_input):
            raise ValueError('Cached fitting information does not match the SE endpoint')
        cached_basis = info.basis
        # Both native information paths already constructed an orthonormal
        # basis by QR/SVD. Rechecking its full Gram would cost O(n rank^2).
        verified_basis = isinstance(info, (SparseMeanInformation, SparseProjectedInformation))
        # Projected fits no longer build the wide endpoint information state.
        # Its immutable selected basis is still useful; covariance solutions
        # are reused only when evaluate() has an exact matching cache.
        if info._state is not None and np.array_equal(np.asarray(theta), info._theta):
            cache = dict(hinv_covar=info._state['hinv_covar'], hinv_selected=info.hinv_warm)
    basis, coef, signs = selected_coordinates(z, alpha, basis=cached_basis)
    verified_basis = verified_basis and basis is cached_basis
    if cached_basis is not None:
        original_mean = np.asarray(z, float) @ np.asarray(alpha, float)
        represented_mean = basis @ coef
        if not np.allclose(represented_mean, original_mean, rtol=solve_tol, atol=solve_tol):
            raise ValueError('Cached selected span does not preserve the fitted sparse mean')
    c = np.empty((len(y), 0)) if covar is None else np.asarray(covar)
    mean_probes = None
    if mean_information is not None:
        state = info._endpoint_state
        if state is None or not np.array_equal(np.asarray(theta), info._endpoint_theta):
            raise ValueError('Missing trace calculation at the supplied SE covariance')
        if state['endpoint_trace_correction'] == 'projected_hutchinson':
            from .sparse_trace import trace_probes as make_trace_probes
            mean_probes = make_trace_probes(info.rank, state['trace_probes'], state['trace_seed'])
    def evaluate(allow_projected):
        estimate = calculate(bank, theta, np.asarray(y), basis, c, coef, ratio, None, 1.,
            q_saved=q, atoms=atoms, trace_probes=trace_probes, moment_probes=moment_probes,
            seed=seed, penalty_signs=signs, anchor_candidates=z, cached_solutions=cache,
            mean_trace_probes=mean_probes, progress=progress, allow_projected=allow_projected,
            _verified_basis=verified_basis)
        if not np.isclose(estimate['h2'], h2, rtol=0, atol=1e-7):
            raise ValueError('SE endpoint h2 does not match the reported native h2')
        if abs(estimate['q_reconstruction_error']) > max(1e-5, 20*solve_tol*max(1., abs(q))):
            raise ValueError('SE endpoint does not reconstruct the native sparse variance Q')
        anchor_error = abs(estimate['lambda_reconstructed']-lam)/max(abs(lam), 1e-20)
        if len(alpha) and anchor_error > .03:
            raise ValueError(f'SE lambda-max branch mismatch: relative error {anchor_error:g}')
        return estimate, anchor_error

    projected = mean_probes is not None and hasattr(bank, 'selected_contrasts')
    if projected:
        try:
            estimate, anchor_error = evaluate(True)
        except (ValueError, FloatingPointError, RuntimeError, MemoryError, np.linalg.LinAlgError) as error:
            fallback_reason = str(error)
            logger.warning('[h2 SE fallback] projected actions failed, trying inverse columns: %s', error)
        else:
            fallback_reason = None
        if fallback_reason is not None:
            estimate, anchor_error = evaluate(False)
            estimate['projected_fallback_reason'] = fallback_reason
    else:
        estimate, anchor_error = evaluate(False)
    return dict(estimate, status='available', method='local_observed_sandwich', approximate=True,
        se_recommended=estimate['se_active'], boundary_policy='fixed_current_zero_variance_components',
        boundary_threshold=1e-8, seed=int(seed), anchor_lambda_relative_error=float(anchor_error),
        model_refits=0, selection_inference='local fixed support/sign/tuning-ratio branch',
        gaussian_working_moments=True, se_multiplier=1)


def _endpoint_batch_size(args, theta, z, covar, mean_information):
    """Reuse the RHS width already budgeted by the native sparse planner.

    This is a numerical blocking choice, not a probe-count or model change.
    Standalone callers without a native memory plan retain the bounded fallback.
    """
    if args.h2_se_batch_size:
        return int(args.h2_se_batch_size)
    planned = (getattr(mean_information, 'batch_size', None) or
               getattr(args, 'candidate_pcg_rhs_batch_size', 32))
    largest_rhs = max(z.shape[1], 0 if covar is None else covar.shape[1],
                      args.h2_se_moment_probes, len(theta)*args.h2_se_trace_probes)
    return max(1, min(int(planned), int(largest_rhs)))


def pipeline_h2_se(*, args, fitter, grm_index, valid, theta, y, z, covar,
                   alpha, ratio, lam, q, atoms, h2, mean_information=None,
                   outer_converged=True):
    """Best-effort uncertainty output; an unavailable SE never changes h2."""
    if not args.h2_se:
        return dict(status='disabled', se_recommended=None, model_refits=0)
    if not valid:
        return dict(status='unavailable', reason='invalid_sparse_endpoint',
                    se_recommended=None, model_refits=0)
    if not outer_converged:
        return dict(status='unavailable', reason='unconverged_sparse_endpoint',
                    se_recommended=None, model_refits=0)
    import jax
    from .sparse_se_operator import OperatorKernelBank
    started = time.perf_counter()
    try:
        batch_size = _endpoint_batch_size(args, theta, z, covar, mean_information)
        with jax.experimental.enable_x64(), jax.default_device(grm_index.streamer.dev):
            bank = OperatorKernelBank(fitter, grm_index, tol=args.h2_se_pcg_tol,
                maxiter=args.max_pcg_iters, batch_size=batch_size)
            estimate = endpoint_h2_se(bank, theta=theta, y=y, z=z, covar=covar,
                alpha=alpha, ratio=ratio, lam=lam, q=q, atoms=atoms, h2=h2,
                trace_probes=args.h2_se_trace_probes, moment_probes=args.h2_se_moment_probes,
                seed=args.h2_se_seed, solve_tol=args.h2_se_pcg_tol,
                mean_information=mean_information,
                progress=lambda phase, elapsed: logger.info(
                    '[h2 SE phase] %s completed, elapsed=%.2fs', phase, elapsed))
            estimate.update(backend='matrix_free_native_kernels', batch_size=batch_size,
                linear_solve_arithmetic='fp32_corrections_verified_fp64',
                batch_size_requested=args.h2_se_batch_size,
                pcg_tolerance=args.h2_se_pcg_tol, max_solve_residual=bank.max_solve_residual,
                solve_batches=bank.solve_batches,
                final_solve_batch_size=bank.solve_batch_size,
                allocation_retries=bank.allocation_retries,
                rhs_columns=bank.rhs_columns, projected_rhs_columns=bank.projected_rhs_columns,
                pcg_iterations=getattr(bank, 'pcg_iterations', None),
                fp64_rescue_batches=bank.fp64_rescue_batches,
                fp64_rescue_iterations=bank.fp64_rescue_iterations,
                warm_solve_batches=getattr(bank, 'warm_solve_batches', None))
    except (ValueError, FloatingPointError, RuntimeError, MemoryError, np.linalg.LinAlgError) as exc:
        estimate = dict(status='unavailable', reason=f'{type(exc).__name__}: {exc}',
                        se_recommended=None, model_refits=0)
    estimate['elapsed_sec'] = time.perf_counter()-started
    return estimate
