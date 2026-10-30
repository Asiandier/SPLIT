# Joint sparse and background prediction refinement

`run_sparse_pipeline --alpha-vamp` applies one joint EM-VAMP model after the
final combined-sample refit. It replaces the former two conditional rounds.
The LASSO support, covariate coefficients, sparse variance estimate, GRM
variance components and residual variance stay fixed. Only prediction effects
and inclusion probabilities change. This step uses the final refit's fitting
phenotypes, including any former validation individuals merged into that fit;
it adds no validation-based tuning and never uses held-out test labels.

## Model and variance constraints

Let X be the fitting-standardized genotype matrix, S the fixed LASSO support,
p the total SNP count, and g(j) the GRM group of SNP j. Write

$$
y-C\widehat\gamma=X(\alpha+b)+\varepsilon,
\qquad \varepsilon\sim N(0,\widehat\sigma_e^2 I).
$$

Independent generating priors are defined for every SNP:

$$
\alpha_j\sim(1-\pi_a)\delta_0+\pi_a N(0,v_a),
\qquad b_j\sim(1-\pi_{g(j)})\delta_0+\pi_{g(j)}N(0,v_{g(j)}).
$$

The approximate posterior fixes alpha and its inclusion indicator to zero
outside S. Background effects can be nonzero throughout active GRM groups.
There is one alpha inclusion probability and one background probability per
group. Q_s is the nonnegative part of the original sparse variance estimate.
With L=tr(X'X)/n and the original GRM definition K_g=X_g X_g'/M_g,

$$
\pi_a v_a L=Q_s,\qquad
\pi_g v_g M_g=\widehat\theta_g.
$$

L is calculated from the current fitting genotypes, not from a constant or a
research cache. Bernoulli counts remain p and p_g; they are not replaced by
L or M_g. Each group retains its original normalization, including constant
SNP handling. These constraints preserve the generating prior's second
moments, not the variance of the posterior prediction.

In particular, the support-only alpha prior budget is Q_s L_S/L, where
L_S=tr(X_S'X_S)/n. It is **not** Q_s. For unit-mean-square columns, the new slab
variance is Q_s/(p*pi_a), whereas the former conditional model used
Q_s/(s*pi_a). The change in prior scale is part of this model, not solely a
change to the iteration order. Restricting the posterior to a data-selected
support remains an approximation and supplies no selection-adjusted PIP/FDR
guarantee.

## Joint scalar posterior

For a VAMP pseudo-observation r=alpha+b+eta with eta~N(0,tau), enumerate the
four indicator states (u,v) in {0,1}². Set A=u*v_a, B=v*v_g and D=tau+A+B.
The state posterior weight is proportional to

$$
\omega_{uv}\propto
\pi_a^u(1-\pi_a)^{1-u}\pi_g^v(1-\pi_g)^{1-v}\phi(r;0,D).
$$

Given a state, the two means are Ar/D and Br/D. Their covariance matrix is

$$
\begin{pmatrix}
A-A^2/D&-AB/D\\
-AB/D&B-B^2/D
\end{pmatrix}.
$$

Average the conditional moments over states. The total posterior variance
includes twice the alpha/background cross moment; adding only the two
marginal variances would be incorrect. Outside S only the u=0 states remain.
The constant factor (1-pi_a) cancels from each such local posterior, but its
log probability remains in the hyperparameter objective.

## Constrained EM and VAMP

For each prior let N be its Bernoulli count, U its summed PIP, V its summed
second moment, and d its fixed per-SNP marginal variance. Alpha uses
N=p and d=Q_s/L, with U and V summed on S. Background group g uses
N=p_g and d=theta_g/M_g. Under v=d/pi, the fixed-posterior EM objective is

$$
Q(\pi)=\frac{3U}{2}\log\pi+(N-U)\log(1-\pi)-\frac{V\pi}{2d}+\mathrm{constant}.
$$

The extra (p-s)log(1-pi_a) is contributed by the restricted outside-support
indicators. The concave one-dimensional M step uses the stable quadratic
root shared with the background-only implementation. No Beta hyperprior is
added. Safeguarded SQUAREM acceptance uses the **summed joint scalar-channel
evidence**, including the outside-support term; it does not guarantee a
monotone full-regression objective or monotone test R².

VAMP exchanges messages for the total w=alpha+b. If m_1 and V_1 are its scalar
posterior moments, the first mean derivative is chi_1=kappa_1*sum(V_1)/m,
where m is the number of active total-effect coordinates. This denominator
is distinct from the EM counts. The external message is

$$
\kappa_2=\kappa_1(1-\chi_1)/\chi_1,
\qquad r_2=(m_1-\chi_1r_1)/(1-\chi_1).
$$

The linear module evaluates

$$
m_2=r_2+X_{\mathcal A}'
(X_{\mathcal A}X_{\mathcal A}'+\widehat\sigma_e^2\kappa_2 I)^{-1}
(y-C\widehat\gamma-X_{\mathcal A}r_2).
$$

Its derivative chi_2 supplies the return message by the same extrinsic
formula. Damping acts on Gaussian natural parameters (precision and
precision times mean). The initial message is found by inverting the joint
denoiser to reproduce the original alpha+BLUP total posterior mean.

## Numerical policy and memory

- `--alpha-vamp-max-iterations` bounds each joint attempt, default 10000.
  `--background-ss-max-iterations` applies only to background-only mode.
- Initial probabilities are max(s/p,1e-8) for alpha and 0.1 for active background
  groups. Zero variance budgets have exactly zero prior probability and effect.
- Message/EM damping is 0.2/0.05. A failed attempt restarts from the original
  total effects with damping at most 0.1/0.025 and the same iteration budget.
- All five undamped diagnostics must pass: message, module agreement,
  precision, constrained EM map and probability change. Tolerances are 1e-5;
  at least 30 iterations and five consecutive passing checks are required.
- If both attempts fail, restore **both** original LASSO alpha and BLUP.
  There is no EC or partial-update fallback. Invalid inputs, allocation
  failures and I/O failures are not silently accepted as numerical fallback.

The production path reuses native packed/streamed genotype operators, verified
PCG solves and cached Lanczos trace quadrature. It does not construct a dense
sample Gram, a full FP64 genotype matrix or a SNP-by-SNP LD matrix. The solver
and trace setup are reused across a retry; component scoring is batched in
at most eight groups. The sparse planner's memory budget applies without
changing ordinary REML planning. Inference uses scoped FP64.

`joint_vamp.py` contains the four-state denoiser and joint solver.
`joint_refinement.py` validates inputs, handles retries and joins the native
fit. The independent `--background-ss` mode remains available with alpha fixed.

## Outputs, timing and resume

The established `.alpha_vamp.npz/.json` and `.background_ss.npz/.json` names
are retained. Metadata identifies `joint_global_budget_p`,
`joint_refinement_schema=1` and `joint_inference=true`; it does not report two
conditional rounds. Native alpha, original BLUP, group/SNP order, fixed
covariates and variance components remain available for audit. Background
output also stores total posterior variance and alpha/background covariance
on S. The `conditioning_*` arrays identify the paired predictor components;
they do not imply a separate conditional background fit.

The update is one shared timed stage. `total_prediction_refinement_seconds`
is its full duration, including trace/solver setup, attempts and training
component scores. For additive consumers it is charged once to
`alpha_vamp.total_refinement_seconds`; `background_ss.total_refinement_seconds`
is zero and points to the shared metadata. `fit_seconds` is the sum of attempt
times. These fields exclude earlier LASSO/variance estimation and held-out
scoring. Do not add two copies of the shared duration.

On fallback, both posterior NPZ files are removed and JSON records describe
the failed joint attempts and restored original pair. The high-level resume
validator rejects old conditional-model results and checks matching joint
statuses, timing and output manifests. Use a new final-refit output/work
location when migrating an existing completed experiment; old scientific
results are not automatically rewritten.

The scalar posterior and M step are analytic; high-dimensional VAMP remains
approximate. Neither convergence nor a prediction gain is guaranteed for
arbitrary LD matrices. Numerical verification belongs in the separate
validation workspace, including exact mixture/Gaussian references, original
candidate equivalence, actual BED integration, failure recovery and GPU replay.
