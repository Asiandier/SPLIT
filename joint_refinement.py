"""Joint sparse/background refinement with native-pair failure recovery.

No held-out outcomes or precomputed sample Gram matrix enter this module.
Products and the Gaussian solver are supplied by the calling pipeline.
"""

from dataclasses import asdict, dataclass, replace
import time
import numpy as np
import jax
import jax.numpy as jnp

from .background_ss import StreamedGenotypeProducts
from .ss_linear import MatrixFreeSampleSolver
from .sparse_planner import sparse_packed_device_cache_budget

from .grouped_ss_vamp import VAMPConfig, validate_vamp_config
from .joint_vamp import fit_joint


@dataclass
class JointRefinement:
    alpha: np.ndarray
    background: np.ndarray
    total: np.ndarray
    status: str
    attempts: list
    posterior: dict | None
    summary: dict


def refine_joint(
    *,
    products,
    response,
    groups,
    support,
    original_alpha,
    original_background,
    q_sparse,
    alpha_normalizer,
    theta,
    normalizers,
    noise_variance,
    global_trace_normalizer,
    solver,
    config: VAMPConfig,
    callback=None,
):
    """Fit one joint model; never return a partial, unconverged predictor.

    response is fitting y minus the unchanged covariate prediction. Sparse
    coefficients are in support order; background coefficients span all SNPs.
    The trace must be tr(X'X)/n for these fitting individuals and this scaling.
    Q_s is truncated at zero as in the native pipeline. It constrains the
    genome-wide alpha prior moment: pi_a * v_a * tr(X'X)/n = Q_s. The
    approximate posterior restricts alpha to the unchanged LASSO support.
    Original heritability estimates remain fixed inputs. The prior moment
    constraint does not identify realized posterior component variances.

    Both numerical attempts use the same iteration budget; no test-driven
    choice is made here. A retry starts from the same native total posterior mean.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("FP64 required")
    p, n = int(products.p), int(products.n)
    support_input, groups_input = np.asarray(support), np.asarray(groups)
    if (
        support_input.ndim != 1
        or groups_input.shape != (p,)
        or not np.issubdtype(support_input.dtype, np.integer)
        or not np.issubdtype(groups_input.dtype, np.integer)
    ):
        raise ValueError("Support and group labels must be integer vectors")
    support = support_input.astype(int, copy=True)
    groups = groups_input.astype(int, copy=True)
    alpha0 = np.asarray(original_alpha, float).copy()
    background0 = np.asarray(original_background, float).copy()
    theta = np.asarray(theta, float).copy()
    normalizers = np.asarray(normalizers, float).copy()
    y = np.asarray(response, float)
    if (
        p < 1
        or n < 1
        or y.shape != (n,)
        or theta.ndim != 1
        or not len(theta)
        or normalizers.shape != theta.shape
        or alpha0.shape != support.shape
        or background0.shape != (p,)
        or len(np.unique(support)) != len(support)
        or np.any(support < 0)
        or np.any(support >= p)
        or np.any(groups < 0)
        or np.any(groups >= len(theta))
        or any(
            not np.isfinite(v).all()
            for v in [alpha0, background0, theta, normalizers, y]
        )
        or np.any(theta < 0)
        or np.any(normalizers < 0)
        or np.any((theta > 0) & (normalizers <= 0))
    ):
        raise ValueError("Invalid joint refinement inputs")
    scalars = np.array(
        [q_sparse, alpha_normalizer, noise_variance, global_trace_normalizer], float
    )
    if (
        not np.isfinite(scalars).all()
        or alpha_normalizer < 0
        or noise_variance <= 0
        or global_trace_normalizer <= 0
    ):
        raise ValueError("Invalid variance or training trace")
    q = max(float(q_sparse), 0.0)
    if q > 0 and (not len(support) or alpha_normalizer <= 0):
        raise ValueError("Positive Q_s needs a nonempty informative support")
    validate_vamp_config(config)
    initial_total = background0.copy()
    initial_total[support] += alpha0
    active = theta[groups] > 0
    if q > 0:
        active[support] = True
    initial_pi = max(len(support) / p, config.pi_min)
    initial_slab = float(q / (global_trace_normalizer * initial_pi))
    summary = dict(
        model="joint_global_budget_p",
        alpha_em_count=p,
        support_size=len(support),
        Q_s=q,
        global_trace_normalizer=float(global_trace_normalizer),
        initial_alpha_slab_variance=initial_slab,
        alpha_slab_learned=False,
        alpha_prior_constraint="pi_a * v_a * global_trace_normalizer = Q_s",
        support_trace_normalizer=float(alpha_normalizer),
        original_variance_components=theta.tolist(),
        original_noise_variance=float(noise_variance),
        heritability_estimates_unchanged=True,
        posterior_scope="support-truncated genome-wide alpha prior",
    )
    if not np.any(active):
        summary.update(
            reason="All supplied genetic variance budgets are zero",
            total_seconds=0.0,
            pi=[0.0] * (len(theta) + 1),
            slab_variance=[0.0] * (len(theta) + 1),
            genome_alpha_prior_variance=0.0,
        )
        return JointRefinement(
            np.zeros_like(alpha0),
            np.zeros(p),
            np.zeros(p),
            "converged_zero",
            [],
            None,
            summary,
        )
    if solver is None or not np.array_equal(solver.active, active):
        raise ValueError("Linear solver active mask does not match the model")

    attempts = []
    result = None
    started = time.monotonic()
    configs = [
        config,
        replace(
            config,
            damping=min(config.damping, 0.1),
            em_damping=min(config.em_damping, 0.025),
        ),
    ]
    for number, current in enumerate(configs, 1):
        attempt_start = time.monotonic()

        def progress(row):
            if callback is not None:
                callback(dict(attempt=number, **row))

        try:
            result = fit_joint(
                products,
                y,
                groups,
                support,
                q,
                global_trace_normalizer,
                theta,
                normalizers,
                noise_variance,
                solver,
                config=current,
                initial_mean=initial_total,
                callback=progress,
            )
            details = {k: v for k, v in result.items() if k != "posterior"}
        except ArithmeticError as error:
            result = None
            details = dict(
                status="arithmetic_failure",
                error=str(error),
                seconds=time.monotonic() - attempt_start,
            )
        details.update(attempt=number, config=asdict(current))
        attempts.append(details)
        if result is not None and result["status"] == "converged":
            break
    summary["total_seconds"] = time.monotonic() - started
    if result is None or result["status"] != "converged":
        summary["reason"] = "Joint fit failed; both original components restored"
        return JointRefinement(
            alpha0,
            background0,
            initial_total,
            "fallback_native",
            attempts,
            None,
            summary,
        )
    posterior = result["posterior"]
    alpha = np.asarray(posterior["mean_alpha"])[support, 0].copy()
    background = np.asarray(posterior["mean_background"])[:, 0].copy()
    total = background.copy()
    total[support] += alpha
    np.testing.assert_allclose(total, posterior["mean"][:, 0], rtol=1e-12, atol=1e-14)
    summary.update(
        pi=result["pi"].tolist(),
        slab_variance=result["slab"].tolist(),
        genome_alpha_prior_variance=float(
            result["pi"][0] * result["slab"][0] * global_trace_normalizer
        ),
    )
    return JointRefinement(
        alpha, background, total, "converged", attempts, posterior, summary
    )


@dataclass
class NativeJointRefinement:
    fit: JointRefinement
    background_effects: object
    training_prediction: np.ndarray
    metadata: dict


def training_trace(products, active=None):
    """Compute tr(X_active' X_active)/n on the exact FP64 product scale.

    Only a genotype block and an n-vector are needed. The host-packed branch
    avoids the native REML diagonal's final FP32 cast. No phenotype is read.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("FP64 required")
    mask = np.ones(products.p, bool) if active is None else np.asarray(active)
    if mask.shape != (products.p,) or mask.dtype != np.dtype(bool):
        raise ValueError("Trace mask must be a Boolean vector over fitting SNPs")
    if products._device_packed is not None:
        diagonal = np.asarray(products.gram_diagonal(mask), dtype=np.float64)
    elif products.packed_calls is not None:
        from .geno_stream import _accumulate_packed_standardized_diag_numba

        diagonal = np.zeros(products.n, dtype=np.float64)
        means = np.asarray(products.means, dtype=np.float64)
        inverse = np.asarray(products.inverse, dtype=np.float64)
        for k, (_, _, _, start, width) in enumerate(products.packed_calls):
            selected = mask[start : start + width]
            if np.any(selected):
                _accumulate_packed_standardized_diag_numba(
                    products._packed(k),
                    width,
                    means[k, :width],
                    inverse[k, :width] * selected,
                    int(products._missing),
                    diagonal,
                )
    else:
        diagonal = np.zeros(products.n, dtype=np.float64)
        for start, stop, block in products._blocks(mask):
            selected = np.asarray(block, dtype=np.float64)[:, mask[start:stop]]
            diagonal += np.einsum("ij,ij->i", selected, selected)
    value = float(np.sum(diagonal, dtype=np.float64) / products.n)
    if not np.isfinite(value) or value < 0:
        raise ArithmeticError("Invalid fitting-genotype trace")
    return value


def refine_native_fit(
    *,
    fitter,
    grm_index,
    y,
    covariates,
    gamma,
    support,
    original_alpha,
    theta,
    q_sparse,
    background_effects,
    config: VAMPConfig,
    callback=None,
):
    """Refine a finalized sparse fit without refitting its h2 or covariates.

    theta contains the genetic components followed by residual variance. The
    caller supplies the same fixed VAMPConfig used for the validated candidate.
    On numerical nonconvergence BOTH original components are returned; input,
    resource and I/O errors are not converted into successful fallback results.
    No files are written. The caller owns publication and held-out scoring.
    """
    started = time.monotonic()
    counts = np.asarray(grm_index.m_per_grm)
    p = int(grm_index.m_total)
    y = np.asarray(y, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    support = np.asarray(support)
    theta = np.asarray(theta, dtype=np.float64)
    alpha0 = np.asarray(original_alpha, dtype=np.float64)
    n = len(y)
    indexed_samples = getattr(getattr(grm_index, "streamer", None), "n", n)
    if (
        y.shape != (n,)
        or n < 1
        or counts.ndim != 1
        or not len(counts)
        or int(indexed_samples) != n
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 1)
        or counts.sum() != p
        or covariates.ndim != 2
        or covariates.shape[0] != n
        or gamma.shape != (covariates.shape[1],)
        or theta.shape != (len(counts) + 1,)
        or support.ndim != 1
        or not np.issubdtype(support.dtype, np.integer)
        or len(np.unique(support)) != len(support)
        or np.any(support < 0)
        or np.any(support >= p)
        or alpha0.shape != support.shape
        or any(not np.isfinite(a).all() for a in [y, covariates, gamma, theta, alpha0])
        or np.any(theta[:-1] < 0)
        or theta[-1] <= 0
        or not np.isfinite(q_sparse)
    ):
        raise ValueError("Invalid finalized sparse-fit inputs")
    if len(background_effects.snp_effects) != len(counts):
        raise ValueError("Original BLUP components do not match the SNP partition")
    parts = [np.asarray(b, dtype=np.float64) for b in background_effects.snp_effects]
    if any(
        b.shape != (int(m),) or not np.isfinite(b).all() for b, m in zip(parts, counts)
    ):
        raise ValueError("Original BLUP SNP order or component sizes are invalid")
    background0 = np.concatenate(parts)
    old_score = np.asarray(background_effects.random_effect, dtype=np.float64)
    if old_score.shape != (n,) or not np.isfinite(old_score).all():
        raise ValueError("Invalid original training BLUP score")
    partitioned = getattr(fitter, "_partitioned_streamer", None)
    if partitioned is not None:
        normalizers = np.asarray(partitioned._component_eff_m_host, dtype=np.float64)
    else:
        normalizers = np.array(
            [float(np.asarray(s._eff_m_const)) for s in fitter.streamers]
        )
    if (
        normalizers.shape != counts.shape
        or not np.isfinite(normalizers).all()
        or np.any(normalizers < 0)
        or np.any((theta[:-1] > 0) & (normalizers <= 0))
    ):
        raise ValueError("Invalid GRM normalizers")
    groups = np.repeat(np.arange(len(counts)), counts)
    support_mask = np.zeros(p, bool)
    support_mask[support] = True
    active = theta[:-1][groups] > 0
    if q_sparse > 0:
        active[support] = True
    # Match native sparse-path memory planning; these are workspace limits,
    # not model parameters or a change to ordinary REML planning.
    budget = getattr(getattr(fitter, "cfg", None), "gpu_budget_bytes", None)
    block_bytes = 512 * 1024**2
    if budget is not None:
        block_bytes = min(block_bytes, max(1, int(budget) // 16))
        depth = min(config.trace_lanczos_steps, n)
        minimum = 8 * (n * (2 * depth + 32) + 8 * (depth + 1) ** 2)
        workspace = min(
            max(config.trace_workspace_bytes, minimum), max(1, int(budget) // 8)
        )
        config = replace(config, trace_workspace_bytes=workspace)
    cache_bytes = getattr(
        getattr(fitter, "cfg", None), "packed_device_cache_bytes", None
    )
    if cache_bytes is None:
        cache_bytes = sparse_packed_device_cache_budget(budget)
    with jax.experimental.enable_x64():
        for streamer in getattr(fitter, "streamers", ()):
            release = getattr(streamer, "release_packed_device_cache", None)
            if release is not None:
                release()
        products = StreamedGenotypeProducts(
            grm_index,
            n,
            normalizers,
            block_bytes=block_bytes,
            device_cache_bytes=cache_bytes,
        )
        global_trace = training_trace(products)
        support_trace = training_trace(products, support_mask)
        solver = None
        if np.any(active):
            solver = MatrixFreeSampleSolver(
                products,
                active,
                pcg_tolerance=config.linear_pcg_tolerance,
                pcg_max_iterations=config.linear_pcg_max_iterations,
                probes=config.trace_probes,
                lanczos_steps=config.trace_lanczos_steps,
                workspace_bytes=config.trace_workspace_bytes,
                seed=config.trace_seed,
                exact_probe_limit=config.trace_exact_probe_limit,
                trace_tolerance=config.trace_tolerance,
            )
        nuisance = covariates @ gamma
        fitted = refine_joint(
            products=products,
            response=y - nuisance,
            groups=groups,
            support=support,
            original_alpha=alpha0,
            original_background=background0,
            q_sparse=q_sparse,
            alpha_normalizer=support_trace,
            theta=theta[:-1],
            normalizers=normalizers,
            noise_variance=theta[-1],
            global_trace_normalizer=global_trace,
            solver=solver,
            config=config,
            callback=callback,
        )
        if fitted.status == "fallback_native":
            updated = background_effects
        else:
            components = []
            for first in range(0, len(counts), 8):
                ids = np.arange(first, min(first + 8, len(counts)))
                effects = np.where(
                    groups[:, None] == ids[None, :], fitted.background[:, None], 0.0
                )
                scores = products.forward(jnp.asarray(effects))
                scores.block_until_ready()
                components.extend(scores[:, k] for k in range(len(ids)))
            background_score = sum(components, jnp.zeros(n, dtype=jnp.float64))
            updated = replace(
                background_effects,
                random_effect=background_score,
                random_effect_components=tuple(components),
                snp_effects=tuple(
                    jax.device_put(fitted.background[groups == g])
                    for g in range(len(counts))
                ),
            )
        alpha = np.zeros(p, dtype=np.float64)
        alpha[support] = fitted.alpha
        sparse_score = np.asarray(products.forward(jnp.asarray(alpha[:, None])))[:, 0]
        prediction = nuisance + sparse_score + np.asarray(updated.random_effect)
    metadata = dict(
        status=fitted.status,
        model="joint_global_budget_p",
        total_refinement_seconds=time.monotonic() - started,
        fitting_samples=n,
        snps=p,
        support_size=len(support),
        global_trace_normalizer=global_trace,
        support_trace_normalizer=support_trace,
        grm_normalizers=normalizers.tolist(),
        original_h2_and_variance_components_unchanged=True,
        original_pair_restored=fitted.status == "fallback_native",
        packed_device_cache_bytes=products.device_cache_bytes,
        blup_pcg_metadata_describes_initial_fit=True,
        sample_gram_materialized=False,
        joint_summary=fitted.summary,
    )
    return NativeJointRefinement(fitted, updated, prediction, metadata)
