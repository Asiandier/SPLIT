"""Matrix-free shifted sample solves and cached Onsager trace quadrature."""
from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np

from .pcg import pcg_solve
from .slq import _set_row, _step


class MatrixFreeSampleSolver:
    """Solve (A A' + shift I)u=b and estimate tr(G/(G+shift I))/p.

    Fixed, randomly signed Walsh probes are sampled without replacement.
    A complete small probe population gives the trace without sampling error;
    large cohorts use a fixed-size sample. Only bounded Lanczos workspaces and
    small quadrature nodes survive setup. No sample Gram is constructed.
    """
    def __init__(self, products, active, *, pcg_tolerance=1e-9,
                 pcg_max_iterations=400, probes=32, lanczos_steps=64,
                 workspace_bytes=256*1024**2, seed=20261011,
                 exact_probe_limit=128, trace_tolerance=1e-6):
        if not jax.config.jax_enable_x64:
            raise RuntimeError('Enable jax_enable_x64 before matrix-free SS setup')
        active = np.asarray(active, bool)
        self.n, self.p = int(products.n), int(active.sum())
        if self.n < 1 or active.shape != (products.p,) or self.p < 1:
            raise ValueError('Need a nonempty active SNP mask')
        for name, value in [('probes', probes), ('lanczos_steps', lanczos_steps),
                            ('pcg_max_iterations', pcg_max_iterations),
                            ('workspace_bytes', workspace_bytes),
                            ('exact_probe_limit', exact_probe_limit)]:
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not 0 < pcg_tolerance < 1 or not 0 < trace_tolerance < 1:
            raise ValueError('Invalid matrix-free numerical tolerances')
        self.matvec = products.gram_operator(active)
        self.active = active.copy()
        self.diagonal = jnp.asarray(products.gram_diagonal(active), jnp.float64)
        if (self.diagonal.shape != (self.n,)
                or not bool(jnp.all(jnp.isfinite(self.diagonal) & (self.diagonal >= 0)))):
            raise ValueError('Invalid sample Gram diagonal')
        self.pcg_tolerance = pcg_tolerance
        self.pcg_max_iterations = pcg_max_iterations
        self.trace_tolerance = trace_tolerance
        self._warm = None
        self.last_diagnostics = {}
        self.solve_calls = self.total_pcg_iterations = 0
        self.maximum_pcg_residual = self.maximum_quadrature_gap = 0.
        self.maximum_trace_relative_standard_error = 0.
        started = time.monotonic()
        population = 1 << (self.n-1).bit_length()
        count = min(population, probes)
        if population <= exact_probe_limit:
            count = population
        if count == 1 and population > 1:
            raise ValueError('Need at least two SS trace probes to report sampling uncertainty')
        depth = min(lanczos_steps, self.n)
        # Two-pass reorthogonalization retains one basis for the current
        # batch. Account for matvec, recurrence and small eigensystem buffers.
        per_probe = 8*(self.n*(2*depth+32)+8*(depth+1)**2)
        if per_probe > workspace_bytes:
            raise MemoryError('SS trace workspace cannot hold one Lanczos probe; '
                              'reduce Lanczos steps or increase workspace_bytes')
        width = min(count, 8, max(1, workspace_bytes//per_probe))
        rng = np.random.default_rng(seed)
        columns = rng.choice(population, size=count, replace=False).astype(np.uint64)
        signs = rng.choice(np.array([-1., 1.]), size=self.n)
        rows = np.arange(self.n, dtype=np.uint64)
        nodes, weights, radau_nodes, radau_weights = [], [], [], []
        for start in range(0, count, width):
            indices = columns[start:start+width]
            parity = rows[:, None] & indices[None, :]
            for shift in (32, 16, 8, 4, 2, 1):
                parity ^= parity >> np.uint64(shift)
            vectors = signs[:, None]*(1.-2.*(parity & np.uint64(1)))
            vectors = jnp.asarray(vectors, jnp.float64)/np.sqrt(self.n)
            basis = jnp.zeros((depth+1, self.n, len(indices)), jnp.float64)
            basis = _set_row(basis, 0, vectors)
            previous = jnp.zeros(len(indices), jnp.float64)
            alpha, beta = [], []
            for index in range(depth):
                product = jnp.asarray(self.matvec(basis[index]), jnp.float64)
                nxt, a, b = _step(basis, product, previous, jnp.asarray(index))
                basis = _set_row(basis, index+1, nxt)
                alpha.append(np.asarray(a)); beta.append(np.asarray(b))
                previous = b
                if np.all(beta[-1] == 0):
                    break
            del basis, product, vectors
            for c in range(len(indices)):
                diagonal = np.array([a[c] for a in alpha])
                off = np.array([b[c] for b in beta])
                # Lanczos breakdown disconnects padded rows; keep only the
                # actual Krylov chain so padding cannot contribute a fake mode.
                zeros = np.flatnonzero(off == 0)
                actual = int(zeros[0]+1) if len(zeros) else depth
                t = np.diag(diagonal[:actual])
                if actual > 1:
                    t += np.diag(off[:actual-1], 1)+np.diag(off[:actual-1], -1)
                eig, vec = np.linalg.eigh(t)
                scale = max(1., float(np.max(np.abs(eig))))
                if eig.min() < -1e-10*scale:
                    raise ArithmeticError('SS sample operator is not positive semidefinite')
                eig = np.maximum(eig, 0.)
                nodes.append(eig); weights.append(vec[0]**2)
                if actual < depth or off[actual-1] == 0:
                    radau_nodes.append(eig); radau_weights.append(vec[0]**2)
                else:
                    # Gauss-Radau anchored at the known spectral lower bound
                    # zero brackets the Stieltjes resolvent. Transforming
                    # 1/(x+s) to x/(x+s) reverses the bounds.
                    inverse_last = np.sum(vec[-1]**2/np.maximum(eig, np.finfo(float).tiny))
                    if not np.isfinite(inverse_last*off[actual-1]**2):
                        raise ArithmeticError('SS Radau quadrature unresolved near a zero Ritz value; '
                                              'increase lanczos_steps')
                    augmented = np.zeros((actual+1, actual+1))
                    augmented[:actual, :actual] = t
                    augmented[-1, -2] = augmented[-2, -1] = off[actual-1]
                    augmented[-1, -1] = off[actual-1]**2*inverse_last
                    re, rv = np.linalg.eigh(augmented)
                    radau_nodes.append(np.maximum(re, 0.)); radau_weights.append(rv[0]**2)
        self.nodes, self.weights = nodes, weights
        self.radau_nodes, self.radau_weights = radau_nodes, radau_weights
        self.probe_count, self.population = count, population
        self.setup_diagnostics = dict(method='matrix_free_pcg_cached_lanczos',
            gram_materialized=False, genotype_materialized=hasattr(products, 'x'),
            trace_estimator='randomly_signed_Walsh_without_replacement',
            trace_probes=count, trace_population=population,
            trace_sampling_exact=count == population, lanczos_steps=depth,
            trace_batch_width=width, trace_workspace_bytes=workspace_bytes,
            trace_workspace_estimated_bytes=width*per_probe, seed=int(seed),
            setup_seconds=time.monotonic()-started)
        if hasattr(products, 'chunk_size'):
            self.setup_diagnostics['genotype_chunk_snps'] = int(products.chunk_size)

    def trace(self, shifts):
        shifts = np.asarray(shifts, float)
        if (shifts.ndim != 1 or not len(shifts) or not np.all(np.isfinite(shifts))
                or np.any(shifts <= 0)):
            raise ValueError('SS shifts must be a nonempty finite positive vector')
        upper = np.array([np.sum(w[:, None]*e[:, None]/(e[:, None]+shifts), axis=0)
                          for e, w in zip(self.nodes, self.weights)])*self.n/self.p
        lower = np.array([np.sum(w[:, None]*e[:, None]/(e[:, None]+shifts), axis=0)
                          for e, w in zip(self.radau_nodes, self.radau_weights)])*self.n/self.p
        estimate = upper.mean(axis=0)
        if not np.all(np.isfinite(lower)) or np.any(lower > upper+1e-10*np.maximum(upper, 1.)):
            raise ArithmeticError('Invalid SS Gauss-Radau trace bounds')
        gap = np.maximum(upper-lower, 0.).mean(axis=0)/np.maximum(estimate, 1e-300)
        if not np.all(np.isfinite(estimate)) or np.any(estimate <= 0) or np.any(estimate >= 1):
            raise ArithmeticError('Invalid SS linear Onsager trace; increase trace probes')
        if np.any(gap > self.trace_tolerance):
            raise ArithmeticError(f'SS Lanczos trace quadrature unresolved: relative gap {gap}; '
                                  'increase lanczos_steps')
        correction = (self.population-self.probe_count)/(self.population-1) if self.population > 1 else 0.
        error = (np.std(upper, axis=0, ddof=1)*np.sqrt(correction/self.probe_count)
                 if self.probe_count > 1 else np.zeros_like(estimate))
        return estimate, gap, error

    def solve_and_trace(self, rhs, shifts):
        rhs, shifts = jnp.asarray(rhs, jnp.float64), np.asarray(shifts, float)
        if (rhs.ndim != 2 or rhs.shape[0] != self.n or rhs.shape[1] < 1
                or shifts.shape != (rhs.shape[1],)):
            raise ValueError('Invalid SS linear solve dimensions')
        if not np.all(np.isfinite(shifts)) or np.any(shifts <= 0):
            raise ValueError('SS shifts must be finite and positive')
        estimate, gap, error = self.trace(shifts)
        shift = jnp.asarray(shifts)
        def operator(value):
            return self.matvec(value)+value*shift[None, :]
        def precondition(value):
            return value/(self.diagonal[:, None]+shift[None, :])
        warm = self._warm if self._warm is not None and self._warm.shape == rhs.shape else None
        solved, relative, iterations = pcg_solve(operator, rhs, M=precondition,
            tol=self.pcg_tolerance, maxiter=self.pcg_max_iterations, X0=warm)
        relative = float(relative)
        if not np.isfinite(relative) or relative > self.pcg_tolerance*1.05:
            raise ArithmeticError(f'SS PCG did not converge: true relative residual {relative:.3e}')
        self._warm = solved
        self.solve_calls += 1
        self.total_pcg_iterations += iterations
        self.maximum_pcg_residual = max(self.maximum_pcg_residual, relative)
        self.maximum_quadrature_gap = max(self.maximum_quadrature_gap, float(gap.max()))
        if np.all(np.isfinite(error)):
            self.maximum_trace_relative_standard_error = max(
                self.maximum_trace_relative_standard_error, float(np.max(error/estimate)))
        self.last_diagnostics = dict(pcg_true_relative_residual=relative,
            pcg_iterations=iterations, trace_fraction=estimate.tolist(),
            trace_quadrature_relative_gap=gap.tolist(),
            trace_sampling_standard_error=error.tolist())
        return solved, jnp.asarray(estimate)

    def diagnostics(self):
        return dict(self.setup_diagnostics, pcg_tolerance=self.pcg_tolerance,
            pcg_max_iterations=self.pcg_max_iterations,
            solve_calls=self.solve_calls, total_pcg_iterations=self.total_pcg_iterations,
            maximum_pcg_true_relative_residual=self.maximum_pcg_residual,
            maximum_trace_quadrature_relative_gap=self.maximum_quadrature_gap,
            maximum_trace_relative_standard_error=self.maximum_trace_relative_standard_error,
            last_solve=self.last_diagnostics)
