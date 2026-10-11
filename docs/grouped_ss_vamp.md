# Fixed-foreground grouped spike-and-slab EM-VAMP

`grouped_ss_vamp.py` implements background inference with a fixed LASSO
foreground. It does not update alpha, gamma, the REML components or noise.
The sparse workflow consumes this kernel directly after its native fit.
Native output metadata records the initialization, numerical attempts,
variance constraint and fixed-parameter declarations.

## Model

For a SNP-major genotype array `x` of shape `(p,n)`, let `A=x.T` and
`r=y-A@alpha-C@gamma`. All groups share the likelihood
`r | beta ~ N(A@beta, sigma_e_squared I)`. Group g has
`beta_j ~ (1-pi_g) delta_0 + pi_g N(0,v_g)` with no hyperpriors.
Native GRMs use `K_g=A_g@A_g.T/M_g`. The constraint is
`pi_g*v_g=d_g=theta_g/M_g`, giving the full prior covariance
`Cov(A_g beta_g | A)=theta_g K_g`. This does not fix the variance of the
posterior mean prediction or of the approximate posterior distribution.
Zero-theta groups have exactly zero background effects and are omitted from
Onsager averages and the sample-space operator. Other groups still use **all group
SNPs** as the Bernoulli denominator, including the original LASSO support.

## Constrained EM

The scalar Gaussian channel has pseudo-observation `r1_j` and precision
`gamma1`. The denoiser evaluates the exact scalar SS posterior, PIP, first
and second moments, and its derivative `gamma1*posterior_variance`.
With `a=sum(PIP)`, `b=sum(E_q beta^2)`, and full group count P,

```
Q(pi) = 1.5*a*log(pi) + (P-a)*log(1-pi) - b*pi/(2*d) + constant
k = b/(2*d)
c = P + a/2 + k
pi_new = 3*a / (c + sqrt(c*c - 6*a*k))
v_new = d/pi_new
```

The rationalized root handles cancellation and the all-PIP-one boundary.
Positive d uses numerical floor `pi_min=1e-8`; hitting it is reported and
the variance equality still holds. It is not a Beta prior or pseudocount.
Inner EM can use SQUAREM extrapolation. Every proposed extrapolation is
polished by the constrained EM map and accepted only if the exact current
scalar-channel marginal likelihood is at least that after two ordinary EM
steps (floating point allowance `1e-10` per group). Otherwise the step is
halved, eventually falling back to the two EM steps. Convex EM damping also
retains the constraint. Extrapolation must follow the current EM direction
and is limited to `[0.8*pi, 1.25*pi]` per inner step. Once the relative
ordinary EM step is below `1e-9`, curvature extrapolation is disabled to
avoid amplifying floating point noise; ordinary EM continues. Default outer
hyperparameter damping is 0.05 and message damping is 0.2. These safeguards
alter the numerical trajectory, not the eventual admissible prior family.
The large step cap allows weakly identified pi to
move without thousands of additional full-genotype products.
This scalar objective is **not** the full regression likelihood or a VAMP
ELBO. Neither full-likelihood monotonicity nor an exact joint posterior is
claimed. The method uses a scalar precision approximation; genomic LD can
violate the rotational assumptions of VAMP theory.

## GPU computation

`ss_linear.py` implements a matrix-free linear module, reused per active group
pattern and numerical configuration. The native adapter exposes streamed
`A @ v`, `A.T @ u`, `G @ u` and diagonal products using the existing packed
or sparse genotype cache; it does not construct the standardized full design.
With `G=A_active@A_active.T`, `lambda=gamma2*sigma_e_squared`,

```
u = (G + lambda I)^(-1) @ (r - A @ r2)
m2 = r2 + A.T @ u
t = trace(G @ (G + lambda I)^(-1))/p_active
gamma1_candidate = gamma2*t/(1-t)
r1_candidate = r2 + A.T@u/t
```

PCG computes `u` in FP64 with a Jacobi preconditioner, a warm start and
verification against the true residual of the same streamed operator.
The default relative tolerance is `1e-9`, with at most 400 iterations. An
unresolved solve raises an error rather than accepting inexact coefficients.

The Onsager trace uses fixed randomly signed Walsh columns sampled without
replacement from a power-of-two population enclosing the sample dimension.
Each sign probe has norm squared `n`, and the complete population satisfies
`mean(z z.T)=I`. A complete population is used when its size is at most 128;
larger cohorts default to 32 probes. This is an unbiased trace sampling scheme
before quadrature error. A fixed seed makes the approximation a consistent
function of the shift throughout EM-VAMP.

One Lanczos pass per probe caches only small nodes and weights. Shift
invariance lets every later VAMP iteration evaluate `x/(x+lambda)` on these
nodes without additional trace-specific genotype scans. Two-pass
reorthogonalization stabilizes the recurrence. Gauss quadrature and
zero-anchored Gauss-Radau bracket the positive-shift trace functional. Their
mean relative gap must be at most `1e-6`; otherwise the fit rejects the result
and asks for more Lanczos steps. The finite-population sampling standard error
is reported separately: a small quadrature gap does not establish an exact
trace or remove sampling uncertainty. Increase probes to reduce that source
of error. The default Lanczos depth is 64.

Trace setup batches at most eight probes, retaining only one batch's basis.
Its additional workspace is `O(n * depth * batch_width)`; PCG uses `O(n * cases)`
vectors, and SS messages and moments use `O(p * cases)` vectors. Small cached
quadratures cost `O(probes * depth)`; their eigendecompositions have dimension
at most `depth+1`. No `n*n` matrix, sample eigendecomposition or `p*p` matrix is
formed. Native packed genotype calls are subdivided when needed to bound FP64
decode buffers. The trace workspace defaults to 256 MiB, adjusted to hold one
probe and capped at one eighth of the native GPU budget. An insufficient
one-probe budget fails explicitly. The original packed/sparse host cache is
still required; matrix-free inference does not eliminate genotype I/O.

Messages are damped in natural coordinates `(gamma1, gamma1*r1)`.
Products decode bounded genotype blocks with FP64 arithmetic and retain
native training standardization statistics and logical SNP order. Enable
`jax_enable_x64` before using the standalone inference API.

Stopping requires undamped message residual, agreement of the nonlinear
and linear posterior means, precision residual, the constrained EM-map
residual and actual pi change, each below `1e-5` for five consecutive
iterations. The native adapter rejects unconverged posterior results before
publishing effects or predictions. Finished cases in a batch freeze both
messages and pi.

## BLUP initialization

`fit_grouped_ss_vamp(..., initial_mean=beta_blup)` accepts a finite `(p,cases)`
array (or `(p,)` for one case). Its default remains zero initialization.
Constrained zero-theta groups mask every supplied effect to zero, and remain
inactive throughout inference. Initial pi is still 0.1, `v=theta/(M*pi)`, and
the initial scalar precision is `n/(sigma_e_squared+sum(theta))`.

VAMP stores a pseudo-observation, not the posterior coefficient mean. We
invert the scalar SS denoiser to initialize a message whose first posterior
mean equals the supplied BLUP. With `a=gamma*v/(1+gamma*v)`,

```
m(r) = PIP(r)*a*r
PIP(0) = sigmoid(log(pi)-log(1-pi)-0.5*log(1+gamma*v))
|beta_blup|/a <= |r_initial| <= |beta_blup|/(a*PIP(0))
```

The positive SS mean is strictly increasing; 64 batched GPU bisection steps
in log-message coordinates within this bracket obtain the unique root.
Log coordinates retain accuracy for very small pi and wide brackets. Signs
are restored afterwards; zero targets and inactive groups return exact zero.
No genotype product or extra likelihood fit is needed for this conversion.
The runner checks the first mean against the supplied effects and records
relative L2 and maximum absolute errors, initial prior parameters and precision.
The existing 10-iteration pi warmup, damping, EM and stopping rules are retained.
This changes initialization only; it need not reduce runtime or change the
eventual fixed point.

## Native sparse pipeline integration

Both sparse CLI entry points accept `--background-ss` and
`--background-ss-max-iterations` (default 2000), plus
`--background-ss-pcg-tol`, `--background-ss-trace-probes` and
`--background-ss-lanczos-steps`. After the final valid
LASSO/variance fit, the pipeline computes its native BLUP and calls
`background_ss.refine_sparse_fit_background` with the in-memory final
support, prediction alpha, native gamma, theta, training phenotype and component
BLUP vectors. With `--progressive-alpha`, validation selection and the
combined-sample OLS refit both freeze their cohort's native conditional BLUP.
Only the final combined fit runs SS, after OLS, on the residual formed with
the updated alpha. It reuses the existing native BLUP for initialization.
The adapter reuses native training-standardized operators in logical GRM
order and calculates the residual itself. A scoped FP64 context preserves
the surrounding pipeline precision setting.

The Bernoulli group count P includes all modeled group SNPs. The covariance
normalizer M comes from the native GRM streamer and can exclude constant
variants; P and M need not be equal. A constant-only group with M=0 is
accepted only when its theta is zero. Zero-theta groups remain exactly zero
even when a supplied initialization is nonzero. All-zero background models
return analytically, with no iterative linear solve or trace setup.

The returned `EffectEstimates` contains SS background coefficients and
training predictions, so existing effect exports and held-out genotype
prediction consume the refinement directly. Native LASSO coefficients and
heritability estimates stay fixed. The native PCG fields describe only the
BLUP initialization; SS convergence is recorded separately. An unconverged
SS fit raises an error and is not published as a valid final prediction.
Before rejecting a max-iteration fit, the native adapter retries from the
same BLUP using message damping at most 0.1, EM damping at most 0.025 and a
curvature-extrapolation floor equal to the EM stopping tolerance. This
avoids extrapolating weakly identified pi after the ordinary EM step is
already below the required accuracy. The model, noise, variance constraints,
stopping tolerances and per-attempt iteration budget are unchanged. All
attempts and their numerical settings are recorded in the SS metadata.

Additional output files:

- `<prefix>.background_ss.npz`: original BLUP, SS means/PIPs/variances,
  pi/slab variances, theta, group assignments, M and source SNP indices.
- `<prefix>.background_ss.json`: method, initialization errors, iteration
  count, fit time, fixed-parameter declarations, linear-solver residuals,
  quadrature gaps, probe settings and sampling uncertainty.
- The native summary, effect metadata and prediction metadata identify the
  estimator as `constrained_ss_em_vamp`. The historical prediction column
  `lasso_background_blup` holds the SS background when this flag is enabled.

For repeated models on one cohort, construct and retain a
`GroupedSSBackgroundContext` with a reusable streamed product object. A small
resident SNP-major array remains supported by the numerical API. The inputs
below use native training-standardized phenotype and SNP scales. Batches must
share their active group pattern.

```python
import jax
from GPU_REML import GroupedSSBackgroundContext, refine_lasso_background_ss
from GPU_REML.background_ss import StreamedGenotypeProducts

jax.config.update('jax_enable_x64', True)
products = StreamedGenotypeProducts(grm_index, n_train, native_grm_normalizers)
context = GroupedSSBackgroundContext(products, groups, native_grm_normalizers)
result = refine_lasso_background_ss(
    context, y=y_train, alpha=lasso_alpha, gamma=lasso_gamma,
    covariates=c_train, blup=native_blup_beta,
    variance_components=theta_with_noise_last,
)
assert result.posterior.status == ['converged']
beta_background = result.posterior.mean[:, 0]
```

The present BLUP initialization is designed for p much larger
than n; initialization outside the admissible VAMP divergence range is
rejected. No universal convergence guarantee is claimed for arbitrary LD
or dense models with p smaller than n.

On a fresh cohort, Lanczos trace setup scans the streamed genotype operator
before VAMP iterations. Each PCG iteration then applies that operator, so this
implementation can cost more genotype passes than a small-cohort dense solver.
Memory scalability alone does not establish biobank throughput. The native
metadata separates total refinement time from posterior fit time, and
records setup plus foreground/component products separately. Repeated
models on the same cohort can reuse a context and cached trace quadrature.

## Verification

`tests/test_grouped_ss_vamp.py` checks scalar posteriors against mixture
densities and numerical derivatives; constrained roots against an independent
optimizer; chunk products and sample solves against dense CPU references;
the trace identity; the Gaussian multigroup limit against exact BLUP;
inactive groups; a sparse batch fixed point; the non-Gaussian orthogonal
case against its exact posterior with both zero and BLUP initializations;
initial message inversion against independent SciPy roots; and rejection of
invalid initial effect arrays. The native adapter checks all prior constraints
and rejects unconverged results. Real-file end-to-end tests independently
check that SS and OLS coefficients reproduce held-out prediction components
using training-only normalization, and verify the completed outputs and resume
contract without supplying test phenotypes.
`tests/test_ss_linear.py` independently checks positive-shift solves and
complete-population traces against dense references, stochastic trace errors,
bounded block requests, missing-value forward/transpose adjoints for dense and
sparse sources, and explicit rejection of unresolved PCG or quadrature.

References:

- VAMP: https://arxiv.org/abs/1610.03082
- EM-VAMP: https://arxiv.org/abs/1602.08207
- Genomic VAMP: https://pmc.ncbi.nlm.nih.gov/articles/PMC13174228/
- Genomic SLQ: https://pmc.ncbi.nlm.nih.gov/articles/PMC6668092/
- SQUAREM: https://onlinelibrary.wiley.com/doi/10.1111/j.1467-9469.2007.00585.x

This is a new JAX implementation with a grouped constrained prior, not a
port of all gVAMP corrections. No validation-adaptive damping, noise-variance
learning, causal effect inputs or test-score-based iteration selection are used.
