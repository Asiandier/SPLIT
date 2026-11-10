# Sparse fitting: numerical contracts

The sparse fit alternates weighted Lasso and information-corrected covariance
updates. Prediction refinement is a subsequent model described in
[joint_vamp.md](joint_vamp.md). The routines below accelerate the sparse fit;
they do not change that prediction prior or its variance budgets.

## Weighted Lasso

For fixed covariance $H$, response $y$, genotype matrix $Z$ and ordinary
covariates $C$, define

$$
P_C=H^{-1}-H^{-1}C(C^T H^{-1}C)^{-1}C^T H^{-1}.
$$

Both path solvers minimize the same profiled objective:

$$
f_\lambda(b)=\tfrac12(y-Zb)^T P_C(y-Zb)+\lambda\lVert b\rVert_1.
$$

`basil` computes candidate columns of $H^{-1}Z$ and solves a weighted Gram
problem. `residual` instead solves for $P_C(y-Zb)$ and computes the full-marker
score $g=Z^T P_C(y-Zb)$. Its preconditioned Gram and BFGS matrices propose
steps; they do not replace the covariance in the fitted objective. Exact
quadratic changes along a proposed direction, together with propagated solve
error bounds, control line-search acceptance.

The residual solver retains FP64 proposal geometry and score certificates.
Previous outer coefficients may initialize a quadratic subproblem only when
they lower that subproblem's current objective. Old gradients are never reused
after the covariance changes.

Validation reuses the already computed training score $Zb$. It still checks
the covariance solve against $y-C\gamma-Zb$ and refines it when needed. Only
the final selected design is materialized for the following information
correction; the entire path's candidate union is not needed for that step.
These changes retain the lambda grid, validation stopping rule and full-marker
KKT conditions. Finite-tolerance iterates can differ with warm starts.

The sparse entry points default to 50 fixed SLQ probes. The maximum Lanczos
depth remains 50, with its existing
quadrature tolerance. Probe count controls Monte Carlo trace precision;
reducing it from 100 does not imply an identical numerical likelihood.
Explicit `--slq-samples` values remain supported and are recorded in the run
configuration. Ordinary REML code and its settings are unchanged.

The acceptance conditions for every path point are

$$
g_j=\lambda\operatorname{sign}(b_j)\quad(b_j\ne0),
\qquad |g_j|\le\lambda\quad(b_j=0),
$$

up to the configured KKT tolerance. The residual solver adds the score-error
bound to the observed maximum violation before accepting. Refining an
uncertain score consumes the same iteration budget as other iterations;
exhausting that budget raises an error rather than returning an uncertified
path. The lambda grid and validation stopping/selection rules are shared.
Fixed-lambda final refits continue to use the weighted Gram solver.

## Sparse covariance update

Let $B$ be an orthonormal basis for the selected SNP span and keep the fitted
Lasso mean fixed, with $r=y-Zb$. The corrected objective is

$$
\ell(\theta)=-\tfrac12\left[
r^TP_Cr+\log|H|+\log|C^T H^{-1}C|+\log|B^TP_CB|
\right].
$$

For $D=[C,B]=QR$ and $\Pi=I-QQ^T$, the determinant identity

$$
\log|H|+\log|C^TH^{-1}C|+\log|B^TP_CB|
=\log|\Pi H\Pi+QQ^T|+\log|D^TD|
$$

removes the need to solve for every selected column at each line-search point.
Only the determinant uses the enlarged span $D$; the quadratic retains $P_C$.
With $H_j=\partial H/\partial\theta_j$, the score remains

$$
\frac{\partial\ell}{\partial\theta_j}
=\tfrac12\left[(P_Cr)^TH_j(P_Cr)
-\operatorname{tr}(P_DH_j)\right].
$$

The positive semidefinite AI matrix supplies a search direction; acceptance
uses the corrected objective. SLQ differentiates its own fixed-probe
quadrature. The exact determinant identities agree, but their finite-probe
approximations need not be identical. Numerical trajectories and boundary
coefficients can therefore differ without a change of statistical objective.

The endpoint sparse-variance correction still computes

$$
\operatorname{tr}\{B(B^TP_CB)^{-1}B^T\}
=\operatorname{tr}\{(B^TP_CB)^{-1}\}.
$$

Intermediate pre-update traces may be deferred; every h² convergence decision
and final h² output uses the calibrated trace. The direct information routine
remains necessary for these endpoints and for low-memory fallback. Its
presence alongside the projected routine is intentional.

## Precision, reuse and device execution

- Mixed-precision PCG solves the same operator throughout. An FP32 correction
  $H\delta\approx b-Hx$ is added in FP64. Acceptance requires an independently
  evaluated residual at the requested precision and within the original total
  iteration budget. Ordinary REML keeps its existing solver policy.
- Lasso inverse recycling stores $U,HU$ in one immutable covariance scope.
  Its balanced inverse preconditioner is
  $UU^T+(I-U(HU)^T)M(I-(HU)U^T)$, which is positive definite for positive
  definite $M$ and full-rank coarse directions. It changes convergence speed,
  not the linear system or its acceptance test.
- Design caches are local to one sample/standardization scope. Weighted Gram
  caches additionally require a fixed covariance; both cross blocks are
  computed because separately converged inverse columns need not be exactly
  symmetric. Projected-likelihood device buffers are released before the
  wide endpoint solves.
- Optional private cuBLAS and packed CUDA kernels are enabled only on sparse
  GPU streamers. They preserve $KV=Z(Z^TV)/m_{\mathrm{eff}}$, missing-value
  imputation and variant order. Native FP32 and dynamic-precision FP64 use
  bounded workspace; smaller tiles or native arithmetic handle insufficient
  workspace. Backend selection is logged. Speed depends on shape, precision
  and workspace, so backend availability alone is not a speed guarantee.
- Ordinary REML retains its planner, operator dispatch and SLQ setup. Sparse
  projected likelihood skips the ordinary determinant cache/reference that it
  does not consume. Neither path binds CPU cores.

Validate changes against dense objectives, gradients, projections and
predictions in an external workspace, as specified in
[CONTRIBUTING.md](../CONTRIBUTING.md). Operator microbenchmarks and complete
pipeline timing answer different questions and should be reported separately.
