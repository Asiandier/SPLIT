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

The sparse entry points default to 100 fixed SLQ probes. Rademacher draws use
an explicit FP32 Bernoulli probability, preserving the original FP32 probe
sequence across JAX precision scopes. Explicit integer output alone is not
sufficient: JAX's default Bernoulli probability changes precision when x64 is
enabled. FP64 covariance solves must not implicitly change the random probes.
Changing the probe count still changes the split-key sequence; estimates from
different counts are not expected to be bitwise identical. The maximum Lanczos
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

## Endpoint sparse-variance correction

For the fitted mean $g=Zb$ and GLS-profiled residual $r=y-C\hat\gamma-g$,

$$
Q_s=\frac{g^Tg+2g^Tr-T}{n},\qquad T=\operatorname{tr}\{(B^TP_CB)^{-1}\}.
$$

This restores the selected-mean covariance subtraction at outer h² checks,
final h² and the prediction-stage variance budget. Under a fixed selected
space, known covariance and unpenalized GLS, T is the fitted-mean noise
variance. This interpretation does not establish unbiasedness for adaptive
Lasso selection and shrinkage. The REML determinant/score above are unchanged.

The trace uses a covariance Schur identity. Let $Q_C$ span C, $W=B^TQ_C$,
$F=Q_C-BW$, $E=F^TF$ and

$$
U=B-FE^{-1}W^T,\quad \Pi=I-BB^T-FE^{-1}F^T,\quad
A=\Pi H\Pi+(I-\Pi).
$$

Then $U^TB=I$, $U^TC=0$ and

$$
(B^TP_CB)^{-1}=U^T(H-H\Pi A^{-1}\Pi H)U.
$$

For a Rademacher vector z in the selected space, set $u=Uz$, $b=\Pi Hu$,
solve $Ax=b$ and average $u^THu-b^Tx$. The expectation is T. No sample-sized
square covariance or selected-column inverse is needed. An approximate solve
is evaluated with FP64 operator products:

$$
q_U=u^THu-2b^Tx+x^TAx,\quad
0\le q_U-z^T(B^TP_CB)^{-1}z
\le\frac{\lVert b-Ax\rVert^2}{\min(\theta_e,1)}.
$$

The bound concerns linear solves in exact arithmetic, not Monte Carlo error
or floating-point interval certification. Sixteen fixed, reproducible probes
are initially used. Probes increase to 32, 64 or 128 until two empirical MC
SEs plus the solve bound are at most 0.5% of T; otherwise the exact calculation
is used. These are numerical accuracy controls, not phenotype-dependent model
parameters. The empirical SE check is not a finite-sample confidence guarantee.
PCG starts at tolerance 0.01 and tightens if its quadratic error exceeds a tenth
of this target. Packed GPU products use four-column batches; blocking does not
change the probe sequence. Supports of rank at most 16 use the direct method.

An already valid exact likelihood cache is reused before estimating T. Endpoint
trace caches are separate from likelihood caches and are invalidated by changes
of covariance, selected span or sample/standardization scope. Covariate profiles
are recomputed for each new response. An empirical uncertainty band touching
zero Q, a nonpositive denominator or the outer h² stopping boundary requests
exact evaluation. Deferred pre-update diagnostics do not require a trace.

SE uses the same fixed probes and basis. With $M=B^TP_CB$ and
$w=P_CB M^{-1}z$, the derivative is

$$
\frac{\partial T}{\partial\theta_j}=E[w^TK_jw],\qquad K_e=I.
$$

SE subtracts this derivative divided by n from dQ/dtheta. Its selected-space
solves already supply $P_CB$ and M; no new contrast solves are needed. This
conditions on the numerical probe realization. MC trace uncertainty is reported
separately from the local statistical SE.

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
- The residual Lasso's final FP64 covariance product can also return the
  unweighted standardized transpose $Z^T w$ used in its global score. This is
  retained only for the identical verified solution and its covariate-projected
  vector, then discarded at the next solve. Component weights affect $Hw$,
  not this score; zero-variance components still receive their SNP scores.
  The separate transpose remains the fallback for other operator backends.
- Quadratic subproblems reuse the already evaluated $B b$ when their starting
  coefficient is $b$. A different warm start computes its own matching product.
- Design caches are local to one sample/standardization scope. Weighted Gram
  caches additionally require a fixed covariance; both cross blocks are
  computed because separately converged inverse columns need not be exactly
  symmetric. Projected-likelihood device buffers are released before the
  subsequent endpoint processing.
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
