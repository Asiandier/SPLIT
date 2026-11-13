# Local approximate SE for sparse total heritability

The sparse pipeline emits `h2_se` by default at its final Lasso/covariance
endpoint, before prediction refinement. It uses the same analysis phenotype,
sample order, standardized SNP columns, kernel partition and native Lasso
coefficients as the reported `h2`. Prediction-stage VAMP effects are not inputs.
No Lasso, REML or tuning fit is repeated to calculate uncertainty.

## Outputs and controls

`<prefix>.summary.json` contains:

- `h2_se`: the recommended approximate SE, or JSON `null` when unavailable.
- `h2_se_status`: `available`, `unavailable` or `disabled`.
- `heritability_uncertainty`: the method, both boundary variants, numerical
  checks, random seed, probe counts, solve residuals and elapsed time.

When enabled, `<prefix>.h2_se.json` contains the same uncertainty diagnostics
and is included in the completion manifest. The production runner also includes
the SE and its status under `final_refit` in `<prefix>.pipeline.json`.
An uncertainty failure records its reason and leaves the fitted point estimate
available; a missing SE never represents zero uncertainty. An explicitly disabled
run removes any stale SE artifact at the same output prefix.

The production `gpu-reml-sparse` runner computes the SE only for the final
combined-sample refit, in both fixed and adaptive mode. Intermediate validation
and partition-selection fits skip it. Direct low-level sparse runs enable it by
default. The additive fields retain sparse point-estimator schema version 11.

```text
--no-h2-se                  disable uncertainty postprocessing
--h2-se-trace-probes 64     probes for derivative traces
--h2-se-moment-probes 256   Gaussian moment probes (>=16, divisible by 8)
--h2-se-seed 581104         nonnegative random seed
--h2-se-pcg-tol 1e-5        checked covariance-solve tolerance (<=1e-3)
--h2-se-batch-size 0        inherit the native sparse solve memory plan
```

All these settings are included in the runner configuration/resume signature.
Probe count, seed, batch width and numerical backend determine reproducibility.
The default precision is intended for an approximate SE; increasing probe counts
reduces numerical integration noise. `moment_mc_se_active` estimates numerical
moment-integration error in the SE, not the statistical uncertainty in h2.

## Estimating relationships and local propagation

Let `Z` contain the selected SNPs, `C` the ordinary covariates, and

$$
H=\sum_{j=1}^m\theta_j K_j+\theta_e I,\qquad
P=H^{-1}-H^{-1}C(C^TH^{-1}C)^{-1}C^TH^{-1},\qquad M=Z^TPZ.
$$

For the fitted support, coefficient signs `s`, selected tuning ratio `rho`,
and maximum absolute-score SNP `x_*` with score sign `t`,

$$
\lambda=\rho t x_*^TPy,\quad
\alpha=M^{-1}(Z^TPy-\lambda s),\quad
L=M^{-1}(Z^T-\rho t s x_*^T)P,\quad T=ZL,
\quad D=P(I-T),\quad R=HD.
$$

The maximum-score anchor is found using all fitted SNPs in the streamer's cache
order. The absolute penalty changes with data and covariance even though its
selected ratio is fixed. Rank-deficient selected columns are reduced with QR/SVD;
the signed L1 functional is transformed with them. An ambiguous rank or penalty
functional is rejected rather than regularized by an arbitrary ridge.

For each covariance coefficient, including the residual direction,

$$
F_i=\tfrac12\{y^TD^TK_iDy-\operatorname{tr}(P_SK_i)\},\quad
P_S=P-PZM^{-1}Z^TP,
$$

where the residual kernel is `I`. The sparse variance is

$$
Q=y^TA_Qy-\operatorname{tr}\{(B^TPB)^{-1}\}/n,\qquad
A_Q=(T^TT+T^TR+R^TT)/n.
$$

The endpoint subtracts selected-mean covariance uncertainty. Its derivative
is subtracted in `b=dQ/dtheta` as well, using the same fixed trace probes as
the point estimator (or the exact trace when available). This constant at
fixed theta does not change the conditional Gaussian moment covariance below.
The REML score F is unchanged. See [trace identity](sparse_numerics.md).

At the fitted endpoint, compute the observed Jacobian `A=-dF/dtheta` and
`b=dQ/dtheta`, retaining changes in alpha and the absolute lambda. With
`a_j=tr(K_j)/n`, `B=sum_j a_j theta_j`, `E=theta_e` and `V=Q+B+E`,

$$
h=(Q+B)/V,\quad d_Q=E/V^2,\quad
d_\theta=(d_Qa_1,\ldots,d_Qa_m,-(Q+B)/V^2)^T.
$$

The joint local sandwich/delta approximation is

$$
w=A^{-T}(d_\theta+d_Qb),\quad W=\operatorname{Cov}(F,Q),\qquad
\operatorname{Var}(\hat h)\approx(w^T,d_Q)W(w^T,d_Q)^T.
$$

`se_active` and `se_recommended` restrict the derivative to current nonzero
genetic variance directions (threshold `1e-8`) and the residual direction;
currently zero genetic components stay fixed. `se_all` retains every direction
as a sensitivity result. An ill-conditioned all-direction sensitivity calculation
does not suppress an otherwise valid recommended SE.

The covariance is computed under the Gaussian working moments with mean
`mu=Z alpha` and covariance `H`. For symmetric quadratic forms,

$$
\operatorname{Cov}(y^TA_i y,y^TA_j y)
=2\operatorname{tr}(A_i H A_j H)+4\mu^TA_i H A_j\mu.
$$

Separate quadratic and linear contributions of Gaussian draws estimate these
moments with less noise than retaining their zero-expectation cross terms.
All sparse/background/residual cross-covariances are retained.

## Matrix-free implementation

`sparse_heritability_se.py` implements the derivatives and propagation, with a
dense-kernel public reference backend for saved-endpoint use.
`sparse_se_operator.py` connects the production path to the native fitter.
It uses checked PCG solves with the existing covariance operator/preconditioner.
FP32 correction solves use FP32 covariance coefficients and native FP32
preconditioner factors, with both operators preserving the input precision.
The endpoint coefficients remain fixed; the FP64 operator uses their original
values to verify every returned solution against the requested residual
tolerance. This avoids promoting correction directions through an FP64
covariance accumulator or preconditioner. It changes numerical execution,
not the SE formula, random probes or acceptance standard.
`linear_solve_arithmetic` records `fp32_corrections_verified_fp64`.
If a finite mixed-precision result misses the requested accuracy, a full-FP64
solve may use the remaining iterations of that batch's original budget.
`fp64_rescue_batches` and `fp64_rescue_iterations` record this work. Loose
requested tolerances also retain a final FP64 check.
Covariance probes are sampled as

$$
\xi=\sum_g\sqrt{\theta_g/m_{\mathrm{eff},g}}\,X_g z_g
       +\sqrt{\theta_e}\,e,
\qquad z_g,e\ \text{independent standard Gaussian}.
$$

Thus `Cov(xi)=H` with the exact native empirical-variance standardization,
mean-imputation and effective SNP denominators. Applying `P_S` to independent
covariance draws gives probes with covariance `P_S`; these evaluate the required
PSD information traces. Probes are only matrix/trace evaluations: no model is fit
to a randomized phenotype. Storage grows with sample/marker counts times bounded
probe widths, without constructing sample-by-sample kernels or Cholesky factors.

Reported endpoint checks include reconstruction of Q and the lambda anchor.
The routine runs in a scoped JAX x64 context and restores the caller's precision
configuration. The supplied covariance coefficients and Lasso effects are not
updated. Explicit numerical/endpoint failures produce an unavailable SE status.

The final fitting stage supplies its immutable selected-space basis. SE reuses
that basis after checking the sample/covariate scope and fitted-mean mapping.
Projected fits need not have computed selected-column covariance solutions;
SE computes the solutions required for its own local derivative. If the direct
REML likelihood already has a cache at the exact covariance coefficients, SE
uses its solutions as checked warm starts. Every column is checked against the
current FP64 covariance operator and refined if necessary. Tighter requested SE
solve tolerances still trigger refinement. `fitting_information_reused`,
`warm_solve_batches`, and `pcg_iterations` record the reuse and extra solve work.
Standalone saved endpoints without a cache use the independent cold path.

The default `--h2-se-batch-size 0` reuses the selected-information/native sparse
RHS batch width already included in the fitting memory plan, capped by the
largest SE RHS block. It does not impose a separate 32-column restriction on a
large device. Standalone callers without fitting-plan metadata fall back to 32.
An explicit positive value overrides this numerical blocking choice. Probe
counts stay fixed; as with the earlier implementation, changing batch width can
change the random-draw realization, so the effective width is saved alongside
the seed and counts.

A finite point estimate from a nonconverged outer loop does not establish the
estimating-equation endpoint required by this local approximation. Such fits
keep their point estimate and report the SE as unavailable with reason
`unconverged_sparse_endpoint`.

## Interpretation

This is a local Gaussian working-model approximation with fixed support,
coefficient signs, tuning ratio, anchor branch and variance boundary. It is not
full inference for the support/tuning selection event, and it does not remove
point-estimator bias. A sandwich expression here does not imply robustness to
arbitrary model misspecification.

The initial saved-endpoint study tested the earlier trace-subtracted estimator
on 900 simulation fits and
12 real traits. On 868 cases also successful for GREML-LDMS, equal-setting mean
95% interval coverage was approximately 94.5% for the fixed-boundary SPLIT
approximation and 93.7% for GREML-LDMS. SPLIT was somewhat more conservative by
SE/empirical-SD comparison; low-h2 High-LD settings retained point-estimator bias.
The boundary recommendation was exploratory after comparing both variants.
The current endpoint restores that trace subtraction using a lightweight
numerical approximation. These earlier coverage results are supporting
evidence, not a new coverage study of the current full implementation.
Random-probe realizations also differ from the original dense backend.

Use `h2 +/- h2_se` for a one-SE display, and identify SPLIT's SE as approximate.
It is not an exact confidence interval or a guarantee of coverage in every setting.
