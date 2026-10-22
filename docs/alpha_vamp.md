# Prior-Q SS EM-VAMP alpha after the final refit

`run_sparse_pipeline --alpha-vamp` replaces validation-selected alpha refitting
with a single SS model on the final combined-sample LASSO support S. The normal
LASSO lambda and covariance selection still use the existing validation stage.
Alpha refinement itself uses only the final refit's training data.

1. Finish the native combined-sample LASSO/variance refit, obtaining support S,
   gamma, background/noise components, Q_s, and native conditional BLUP beta.
2. Fix gamma and beta, and update all support alpha on
   `r_alpha = y - C gamma - X beta_native` with Gaussian noise variance sigma_e².
3. Fix the updated alpha and gamma, and run constrained background SS once on
   `y - C gamma - X_S alpha`, initialized from native conditional BLUP.
4. Run one more conditional round: update alpha on
   `y - C gamma - X beta_SS[1]`, then update background SS on
   `y - C gamma - X_S alpha[2]`.
5. If all conditional fits converge, export and predict with
   `C gamma + X_S alpha[2] + X beta_SS[2]`; otherwise restore the original
   `C gamma + X_S alpha_native + X beta_native`.

Up to two rounds are run after the final refit, stopping at nonconvergence. Both alpha fits have the
same single SS prior, fixed Q_s and noise, zero-message initialization and
all-SNP pi count. Both background fits start from the same native conditional
BLUP with pi_g=.1. Gamma, support and native variance components remain fixed.
Round two conditions on round one's posterior mean; it does not propagate
posterior uncertainty or carry learned pi/slab values into initialization.
Round count is fixed, without validation/test selection or adaptive stopping.

The native LASSO, covariance/CHIVE estimates, KKT certificate and reusable
warm state describe the original inferential fit. Refined coefficients enter
the existing effect export and prediction paths, with separate method metadata.

## Single SS prior and constraint

For j in S, `alpha_j ~ (1-pi) delta_0 + pi N(0, tau²)`. Outside S, both alpha
and its inclusion indicator are fixed zero. There are no Beta or variance
hyperpriors. Let P be the total retained SNP count and
`L = tr(X_S.T X_S)/n`. The constraint is

`pi * tau² * L = Q_s`.

This is the prior expected genetic variance. It does not constrain the squared
training score of the posterior mean, or force prediction corr² to equal Q_s.
The M step jointly updates pi and tau². For A=sum(PIP), B=sum(E[alpha_j²]),
d=Q_s/L, H=B/(2d), and c=P+A/2+H:

`pi_new = 3 A / (c + sqrt(c² - 6 A H))`, `tau²_new = d/pi_new`.

The known-zero indicators contribute `(P-s) log(1-pi)` to scalar evidence.
They enter EM counts but not the VAMP Onsager average, which uses s coordinates.
The shared implementation is `grouped_ss_vamp.fit_grouped_ss_vamp`, with
`em_counts=[P]`; background retains its usual group counts and constraints.

Initialization is a zero VAMP message, pi=s/P (subject to the numerical floor),
tau²=Q_s/(L*pi), and message precision n/(sigma_e²+Q_s). Native LASSO alpha
selects support and supplies diagnostics; it is not the initial posterior mean.
Negative unbiased Q_s estimates are projected to the nonnegative boundary,
which yields an all-zero sparse prior. Metadata records the raw and used Q_s.
A positive Q_s with an empty or constant-only support is rejected.

## Computation and convergence

`ss_linear.SupportPrecisionSolver` builds an s-by-s SNP Gram in sample chunks
and retains its FP64 eigensystem, including null directions. It constructs no
n-by-n sample Gram. Its storage is O(s²), in addition to the already extracted
O(ns) support design. Background continues to use streamed, matrix-free PCG
and cached Lanczos trace quadrature.
Support VAMP iterations run in compiled blocks between log points. Each step
still checks the undamped residuals and precision safeguard, freezes completed
batch members and stops immediately on convergence or invalid numerics. The
host reads small diagnostics at block boundaries; messages remain on device.
This removes per-step Python/device synchronization without changing the
iteration budget, update equations or acceptance criteria. Logging every step
still works, at the cost of that synchronization.
The background adapter retains a `GroupedSSBackgroundContext` between the two
rounds, reusing packed genotype products and the small trace quadrature cache.
It discards the first background's coefficient posterior after retaining its
training score, keeping the intermediate storage proportional to sample count.

Natural precision messages handle nonlinear SS divergence greater than one
when the Gaussian linear posterior remains positive definite. A transient
precision safeguard is recorded; a safeguarded iterate cannot satisfy stopping.
Message, module, precision and constrained EM fixed-point errors must satisfy
1e-4 for five consecutive iterations. Message and EM damping are both .2.
`--alpha-vamp-max-iterations` defaults to 10000 damped iterations, an iteration
ceiling rather than a mandatory count. If ordinary alpha VAMP does not
converge, prediction immediately returns to the native weighted-LASSO alpha
and its original conditional BLUP. There is no EC solver. If background VAMP
remains unconverged after its existing numerical damping retry, the same
fallback applies. This decision depends only on training-side convergence,
never validation or test prediction accuracy.

The two-round update is transactional: failure in either round discards all
conditional updates, including a successful first round. It does not combine
native alpha with a refined background, or retain a partially refined pair.
Native support, covariate coefficients, variance components and heritability
estimates are unchanged. Input errors, missing files, resource failures and
other exceptions are not silently treated as nonconvergence.

Convergence establishes a fixed point of the approximate inference equations;
it does not guarantee better held-out prediction than the native fit.

The method adds no alpha tuning stage. The two conditional rounds do not
constitute an outer convergence test or imply joint posterior convergence.
Both CLI entry points accept `--alpha-vamp`, which also enables background SS.
The high-level runner applies it only to the final combined-sample refit; the
low-level runner requires `--lasso-fixed-lam-ratio` for this mode.

## Outputs

- `<prefix>.alpha_vamp.npz`: support/source indices, native/refined alpha,
  posterior variances/PIPs, pi/tau², Q_s/L, fixed gamma/theta, alpha residual,
  training sparse score and conditioning background SS score. `old_alpha`
  remains the native LASSO; `previous_alpha`/`round1_alpha` are the first VAMP
  estimate. The first alpha residual/score and native BLUP score are retained
  to audit the conditional sequence. `alternating_rounds=2` identifies the fit.
- `<prefix>.alpha_vamp.json`: prior/constraint definition, initialization,
  convergence diagnostics, native and new squared scores, final full squared
  score/MSE after background SS, and elapsed times. `rounds` records both
  alpha/background fits. Top-level iterations and times sum the two alpha
  fits; final parameters and squared scores describe the second round.
- Existing background, effect and prediction metadata identify the updated
  alpha and final background. The completion manifest covers both new files.
  Background metadata also records both rounds and their summed times.
  Completion validation accepts either both converged rounds or a documented
  restoration of both native components. An old EC result cannot satisfy the
  new failure policy.
- On fallback, both JSON files have `status=fallback_native`, record the failed
  component/round, diagnostics and attempted refinement time, and name the
  actual native estimators. Refined posterior NPZs are not emitted and stale
  NPZs are removed. Effects and predictions use the restored native pair.
  The completion manifest includes the JSON records; fallback is a completed
  prediction, not a failed native fit.
