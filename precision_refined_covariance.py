"""Preconditioned covariance solves with FP64-certified mixed corrections.

The callbacks must implement the same symmetric positive definite operator,
in reference and approximate arithmetic respectively. Right-hand-side scaling
protects small columns from the inner solver's absolute normalization floor.
Neither preconditioning nor correction precision changes the original system.
"""
from __future__ import annotations
import time
import numpy as np
def block_pcg(product, rhs, *, initial=None, tolerance=1e-8, max_iter=500):
    """Independent CG systems batched as columns; verified true residuals."""
    rhs = np.asarray(rhs, float)
    solution = np.zeros_like(rhs) if initial is None else np.asarray(initial, float).copy()
    residual = rhs - product(solution) if initial is not None else rhs.copy()
    direction = residual.copy()
    rr = np.sum(residual**2, axis=0)
    norms = np.maximum(np.linalg.norm(rhs, axis=0), 1e-30)
    for iteration in range(max_iter + 1):
        relative = np.linalg.norm(residual, axis=0) / norms
        if np.max(relative) <= tolerance:
            residual = rhs - product(solution)
            relative = np.linalg.norm(residual, axis=0) / norms
            if np.max(relative) <= tolerance:
                return solution, dict(iterations=iteration, residual_max=float(relative.max()))
            direction, rr = residual.copy(), np.sum(residual**2, axis=0)
        if iteration == max_iter:
            break
        active = relative > tolerance
        applied = product(direction)
        denominator = np.sum(direction * applied, axis=0)
        if np.any(active & (denominator <= 0)):
            raise ArithmeticError("PCG operator is not numerically positive definite")
        step = np.divide(rr, denominator, out=np.zeros_like(rr), where=active)
        solution += direction * step
        residual -= applied * step
        refresh = (iteration + 1) % 30 == 0
        if refresh:
            residual = rhs - product(solution)
        new_rr = np.sum(residual**2, axis=0)
        ratio = np.divide(new_rr, rr, out=np.zeros_like(rr), where=rr > 0)
        direction = residual.copy() if refresh else residual + direction * ratio
        rr = new_rr
    raise ArithmeticError(f"PCG did not converge: relative residual {relative.max():.3g}")

def refined_pcg(exact_product, approximate_product, rhs, *, initial=None,
                tolerance=1e-8, max_iter=500, correction_tolerance=1e-3,
                max_refinements=8):
    """Batched correction solves with an explicitly checked FP64 residual."""
    rhs = np.asarray(rhs, float)
    if (rhs.ndim != 2 or not np.isfinite(rhs).all()
            or not 0 < tolerance < correction_tolerance < 1 or max_refinements < 1):
        raise ValueError("Invalid iterative-refinement input or tolerances")
    solution = np.zeros_like(rhs) if initial is None else np.asarray(initial, float).copy()
    if solution.shape != rhs.shape or not np.isfinite(solution).all():
        raise ValueError("Invalid iterative-refinement initial value")
    norms = np.maximum(np.linalg.norm(rhs, axis=0), 1e-30)
    residual = rhs.copy() if initial is None else rhs-exact_product(solution)
    inner_iterations, corrections = 0, []
    for step in range(max_refinements+1):
        relative = np.linalg.norm(residual, axis=0)/norms
        if float(relative.max()) <= tolerance:
            return solution, dict(iterations=inner_iterations, residual_max=float(relative.max()),
                                  refinement_steps=step, correction_solves=corrections,
                                  verification='FP64 true residual')
        if step == max_refinements:
            break
        active_rhs = residual.copy()
        active_rhs[:, relative <= tolerance] = 0.
        try:
            correction, diagnostic = block_pcg(approximate_product, active_rhs,
                                                tolerance=correction_tolerance, max_iter=max_iter)
            diagnostic['arithmetic'] = 'FP32 correction'
        except ArithmeticError as failure:
            # Only the cheap correction solver is replaced. The original
            # FP64 system and its final residual requirement remain identical.
            correction, diagnostic = block_pcg(exact_product, active_rhs,
                                                tolerance=correction_tolerance, max_iter=max_iter)
            diagnostic.update(arithmetic='FP64 correction fallback', approximate_failure=str(failure))
        solution += correction
        residual = rhs-exact_product(solution)
        inner_iterations += diagnostic['iterations']
        corrections.append(diagnostic)
    raise ArithmeticError(f"FP64-verified iterative refinement failed: residual {relative.max():.3g}")



def solve_refined_covariance(exact_product, approximate_product, rhs, diagonal, *,
                             initial=None, tolerance=1e-10, correction_tolerance=1e-3,
                             max_iter=500, max_refinements=8):
    began = time.monotonic(); b = np.asarray(rhs, float)
    d = np.asarray(diagonal, float)
    if (b.ndim not in (1, 2) or b.shape[0] == 0 or d.shape != (b.shape[0],)
            or not np.isfinite(b).all() or not np.isfinite(d).all() or np.any(d <= 0)
            or not 0 < tolerance < correction_tolerance < 1
            or not isinstance(max_iter, (int, np.integer)) or max_iter < 1
            or not isinstance(max_refinements, (int, np.integer)) or max_refinements < 1):
        raise ValueError('Finite aligned RHS, positive preconditioner and valid tolerances required')
    vector = b.ndim == 1
    if vector: b = b[:, None]
    if b.shape[1] == 0: raise ValueError('At least one RHS required')
    scale = np.max(np.abs(b), axis=0)
    nonzero = scale > 0; scale = np.where(nonzero, scale, 1.)
    normalized_rhs = b/scale
    root = np.sqrt(d)
    def wrap(product):
        def apply(value):
            result = np.asarray(product(value/root[:, None]), float)
            if result.shape != value.shape or not np.isfinite(result).all():
                raise ArithmeticError('Invalid covariance product')
            return result/root[:, None]
        return apply
    first = None
    if initial is not None:
        first = np.asarray(initial, float)
        if vector: first = first[:, None]
        if first.shape != b.shape or not np.isfinite(first).all():
            raise ValueError('Invalid initial solution')
        first = first/scale*root[:, None]
        # V is positive definite, so the exact solution to a zero RHS is zero.
        first[:, ~nonzero] = 0.
    inner_tolerance = tolerance*np.sqrt(d.min()/d.max())*.5
    solution, diagnostic = refined_pcg(wrap(exact_product), wrap(approximate_product),
        normalized_rhs/root[:, None], initial=first, tolerance=inner_tolerance,
        correction_tolerance=correction_tolerance, max_iter=max_iter,
        max_refinements=max_refinements)
    normalized_solution = solution/root[:, None]
    normalized_solution[:, ~nonzero] = 0.
    residual = normalized_rhs-np.asarray(exact_product(normalized_solution), float)
    norm = np.linalg.norm(normalized_rhs, axis=0)
    relative = np.divide(np.linalg.norm(residual, axis=0), norm,
        out=np.zeros_like(norm), where=nonzero)
    if not np.isfinite(residual).all() or relative.max() > tolerance:
        raise ArithmeticError(f'Original FP64 covariance residual failed: {relative.max()}')
    answer = normalized_solution*scale
    if not np.isfinite(answer).all(): raise ArithmeticError('Nonfinite covariance solution')
    diagnostic = dict(diagnostic, maximum_relative_residual=float(relative.max()),
        relative_residuals=relative.tolist(), scaled_tolerance=inner_tolerance,
        seconds=time.monotonic()-began, original_system_verified=True,
        rhs_scaling=True, explicit_grm=False, mcmc=False)
    return (answer[:, 0] if vector else answer), diagnostic
